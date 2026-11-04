#!/usr/bin/env python3
"""Offline: calibrate and evaluate the ZED depth against lidar (e.g. the Livox on Spot).

    lidar_vs_zed.py extract --bag <bag> --out <dir> [--robot hamilton] [--scan-stride 10]
                            [--lidar-rpy-deg 180 0 0]
    lidar_vs_zed.py fit --out <dir>

extract (one pass over the bag, slow): every n-th lidar scan is deskewed point by point to the
stamp of the nearest ZED depth image using odom, kept if the camera can see it (z-buffer,
nominal extrinsic), and cached with the edge-masked depth image and the ground plane (gpe) in
the camera frame. -> <dir>/cache.npz

fit (fast): fits depth models on alternating 60 s chunks of the bag and evaluates on the
others, with and without a small correction to the lidar->camera extrinsic, plus the
ground-plane calibration (GroundCalibrator) on the same frames for comparison.
-> <dir>/summary.json, *.png
"""

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from depth_processing_ros.calibration import GroundCalibrator  # noqa: E402
from depth_processing_ros.lidar_reference import (FitResult, LidarFitParams, LidarZedData,  # noqa: E402
                                                  apply_model, evaluate, fit, make_T, mask_depth_edges,
                                                  project, transform, visible_mask)

CHUNK_S = 60.0                  # train/test split: alternating chunks of this length
WINDOW_S = 120.0                # per-window fits for stability over time
MAX_DEPTH_DT_S = 0.05           # lidar scan mid-time to depth stamp
SCAN_DURATION_S = 0.1
IMAGE_MARGIN_PX = 15            # keep points this far outside the image (extrinsic may move them in)
MIN_GPE_HEIGHT = 0.4            # as in depth_scale_correction_node


# ---------------------------------------------------------------------------------- extract

def ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def tf_to_T(tr) -> np.ndarray:
    q, p = tr.rotation, tr.translation
    return make_T(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z]))


def chain(edges: dict, target: str, source: str) -> np.ndarray:
    """T_target_source from static edges {(parent, child): T_parent_child} (BFS, either direction)."""
    adj = {}
    for (a, b), T in edges.items():
        adj.setdefault(a, []).append((b, T))
        adj.setdefault(b, []).append((a, np.linalg.inv(T)))
    frontier, seen = deque([(target, np.eye(4))]), {target}
    while frontier:
        f, T = frontier.popleft()                     # T = T_target_f
        if f == source:
            return T
        for g, Tfg in adj.get(f, []):
            if g not in seen:
                seen.add(g)
                frontier.append((g, T @ Tfg))
    raise KeyError(f"no static transform {target} <- {source}")


class PoseSeries:
    """Interpolated T_parent_child(t) from a stream of stamped transforms."""

    def __init__(self, t, T):
        order = np.argsort(t)
        t, T = np.asarray(t)[order], np.asarray(T)[order]
        uniq = np.concatenate([[True], np.diff(t) > 0])
        self.t, self.T = t[uniq], T[uniq]
        self.slerp = Slerp(self.t * 1e-9, Rotation.from_matrix(self.T[:, :3, :3]))

    def covers(self, t, max_gap_ns=0.1e9) -> bool:
        i = int(np.searchsorted(self.t, t))
        return 0 < i < len(self.t) and self.t[i] - self.t[i - 1] < max_gap_ns

    def at(self, t):
        t = np.atleast_1d(np.asarray(t, np.float64))
        tc = np.clip(t, self.t[0], self.t[-1])
        R = self.slerp(tc * 1e-9).as_matrix()
        i = np.clip(np.searchsorted(self.t, tc), 1, len(self.t) - 1)
        w = ((tc - self.t[i - 1]) / (self.t[i] - self.t[i - 1]))[:, None]
        p = (1 - w) * self.T[i - 1, :3, 3] + w * self.T[i, :3, 3]
        return R, p


def read_cloud(msg):
    fields = {f.name: f for f in msg.fields}
    names = ["x", "y", "z", "timestamp"]
    fmt = {f: ("<f4" if fields[f].datatype == 7 else "<f8") for f in names}
    arr = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
    cols = [arr[:, fields[f].offset:fields[f].offset + np.dtype(fmt[f]).itemsize].copy().view(fmt[f]).ravel()
            for f in names]
    xyz = np.stack(cols[:3], axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.1)
    return xyz[ok], cols[3][ok].astype(np.float64)


def extract(args):
    from rosbags.highlevel import AnyReader

    r = args.robot
    zed = f"/{r}/{r}_zed"
    topics = dict(lidar=f"/{r}/livox/lidar", depth=f"{zed}/depth/depth_registered",
                  info=f"{zed}/depth/camera_info", odom=f"/{r}/odom", tf="/tf", tf_static="/tf_static")
    body, lidar_frame, kin, gpe = f"{r}/body", None, f"{r}/kinematic", f"{r}/gpe"
    role = {v: k for k, v in topics.items()}

    static, K, cam_frame = {}, None, None
    odom_t, odom_T, kin_t, kin_T, gpe_t, gpe_T = [], [], [], [], [], []
    scans, pending, depth_buf = [], [], deque(maxlen=8)        # depth_buf: (stamp, msg)
    n_scans, next_report = 0, 100

    def match(scan_mid):
        best = min(depth_buf, key=lambda d: abs(d[0] - scan_mid))
        return best if abs(best[0] - scan_mid) < MAX_DEPTH_DT_S * 1e9 else None

    with AnyReader([args.bag]) as reader:
        conns = [c for c in reader.connections if c.topic in role]
        t0 = reader.start_time
        for conn, t, raw in reader.messages(connections=conns):
            kind = role[conn.topic]
            if kind == "lidar":
                n_scans += 1
                if (n_scans - 1) % args.scan_stride:
                    continue
            msg = reader.deserialize(raw, conn.msgtype)
            if kind == "tf_static":
                for tr in msg.transforms:
                    static[(tr.header.frame_id, tr.child_frame_id)] = tf_to_T(tr.transform)
            elif kind == "tf":
                for tr in msg.transforms:
                    if tr.header.frame_id == body and tr.child_frame_id == kin:
                        kin_t.append(ns(tr.header.stamp)); kin_T.append(tf_to_T(tr.transform))
                    elif tr.header.frame_id == kin and tr.child_frame_id == gpe:
                        gpe_t.append(ns(tr.header.stamp)); gpe_T.append(tf_to_T(tr.transform))
            elif kind == "odom":
                q, p = msg.pose.pose.orientation, msg.pose.pose.position
                odom_t.append(ns(msg.header.stamp))
                odom_T.append(make_T(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z]))
            elif kind == "info" and K is None:
                K, cam_frame = np.array(msg.k).reshape(3, 3), msg.header.frame_id
            elif kind == "lidar":
                lidar_frame = msg.header.frame_id
                xyz, ts = read_cloud(msg)
                pending.append((ns(msg.header.stamp) + SCAN_DURATION_S / 2 * 1e9, xyz, ts))
            elif kind == "depth":
                depth_buf.append((ns(msg.header.stamp), msg))
                still = []
                for sc in pending:
                    if depth_buf[-1][0] < sc[0] + MAX_DEPTH_DT_S * 1e9:
                        still.append(sc)                  # a closer depth image may still come
                        continue
                    m = match(sc[0])
                    if m is not None:
                        d = np.frombuffer(m[1].data, np.float32).reshape(m[1].height, m[1].width)
                        scans.append(dict(t_cam=m[0], t_mid=sc[0], xyz=sc[1], ts=sc[2],
                                          depth=mask_depth_edges(d).astype(np.float16)))
                pending[:] = still
                if len(scans) >= next_report:
                    next_report += 100
                    print(f"  {(t - t0) * 1e-9:7.1f} s  scans matched: {len(scans)}", flush=True)

    # the bag's TF may not include how the lidar is mounted (e.g. upside down): apply --lidar-rpy-deg
    T_fix = make_T(Rotation.from_euler("xyz", args.lidar_rpy_deg, degrees=True).as_matrix(), np.zeros(3))
    T_body_lidar = chain(static, body, lidar_frame) @ T_fix
    T_cam_body = chain(static, cam_frame, body)
    T_cam_lidar = T_cam_body @ T_body_lidar
    print(f"T_cam_lidar t = {np.round(T_cam_lidar[:3, 3], 3)}  "
          f"rpy = {np.round(Rotation.from_matrix(T_cam_lidar[:3, :3]).as_euler('xyz', degrees=True), 1)}")
    odom = PoseSeries(odom_t, odom_T)
    kin_s = PoseSeries(kin_t, kin_T) if kin_t else None
    gpe_s = PoseSeries(gpe_t, gpe_T) if gpe_t else None
    T_lidar_body = np.linalg.inv(T_body_lidar)
    h, w = scans[0]["depth"].shape

    depths, pts, frame, t_cam, dts, speed, yaw_rate, g_n, g_p = [], [], [], [], [], [], [], [], []
    for sc in scans:
        tc = sc["t_cam"]
        if not (odom.covers(tc) and odom.covers(sc["ts"].min()) and odom.covers(sc["ts"].max())):
            continue
        # deskew: lidar point at its own time -> body at camera time -> lidar frame at camera time
        Ri, pi = odom.at(sc["ts"])
        Rc, pc = odom.at(tc)
        pb = transform(T_body_lidar, sc["xyz"])
        pw = np.einsum("nij,nj->ni", Ri, pb) + pi
        pbc = (pw - pc[0]) @ Rc[0]
        pl = transform(T_lidar_body, pbc)
        u, v, z = project(transform(T_cam_lidar, pl), K)
        near = (z > 0.3) & (u > -IMAGE_MARGIN_PX) & (u < w + IMAGE_MARGIN_PX) & \
               (v > -IMAGE_MARGIN_PX) & (v < h + IMAGE_MARGIN_PX)
        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (z > 0.3)
        vis = visible_mask(u, v, z, (h, w))
        keep = near & (vis | ~inside)                    # drop occluded in-image points only
        if keep.sum() < 50:
            continue
        f = len(depths)
        depths.append(sc["depth"])
        pts.append(pl[keep].astype(np.float32))
        frame.append(np.full(keep.sum(), f, np.int32))
        t_cam.append(tc)
        dts.append((tc - sc["t_mid"]) * 1e-9)
        Ra, pa = odom.at([tc - 5e7, tc + 5e7])
        speed.append(float(np.linalg.norm(pa[1] - pa[0]) / 0.1))
        yaw_rate.append(float(np.linalg.norm(Rotation.from_matrix(Ra[0].T @ Ra[1]).as_rotvec()) / 0.1))
        n, p = np.full(3, np.nan), np.full(3, np.nan)
        if kin_s is not None and gpe_s is not None and kin_s.covers(tc) and gpe_s.covers(tc):
            Rk, pk = kin_s.at(tc)
            Rg, pg = gpe_s.at(tc)
            T_cam_gpe = T_cam_body @ make_T(Rk[0], pk[0]) @ make_T(Rg[0], pg[0])
            n, p = T_cam_gpe[:3, 2], T_cam_gpe[:3, 3]
        g_n.append(n)
        g_p.append(p)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "cache.npz", depths=np.stack(depths), points=np.concatenate(pts),
             frame=np.concatenate(frame), t_cam=np.array(t_cam), dt=np.array(dts), speed=np.array(speed),
             yaw_rate=np.array(yaw_rate), gpe_normal=np.array(g_n), gpe_point=np.array(g_p), K=K,
             T_cam_lidar=T_cam_lidar, lidar_rpy_deg=np.array(args.lidar_rpy_deg), bag=str(args.bag))
    print(f"cached {len(depths)} frames, {sum(len(p) for p in pts)} lidar points -> {args.out / 'cache.npz'}")


# ---------------------------------------------------------------------------------- fit

def ground_fit(c, frames) -> FitResult:
    cal = GroundCalibrator()
    used = 0
    for f in frames:
        n, p = c["gpe_normal"][f], c["gpe_point"][f]
        if not np.isfinite(n).all() or abs(float(n @ p)) < MIN_GPE_HEIGHT:
            continue
        ok, _ = cal.add_frame(float(c["t_cam"][f]) * 1e-9, c["depths"][f].astype(np.float32), c["K"], n, p)
        used += ok
    if used < 5:
        return None
    res = cal.fit()
    print(f"ground-plane fit: {used} frames, a={res.a:.4f} b={res.b:+.4f} accepted={res.accepted} ({res.reason})")
    return FitResult("inverse", dict(a=res.a, b=res.b), n_points=res.n_pixels)


def run_fit(args):
    c = dict(np.load(args.out / "cache.npz"))
    data = LidarZedData(c["depths"].astype(np.float32), c["points"].astype(np.float64), c["frame"],
                        c["K"], c["T_cam_lidar"])
    t = (c["t_cam"] - c["t_cam"][0]) * 1e-9
    chunk = (t // CHUNK_S).astype(int)
    train, test = np.flatnonzero(chunk % 2 == 0), np.flatnonzero(chunk % 2 == 1)
    p = LidarFitParams()
    dtr, dte = data.subset(train), data.subset(test)
    print(f"{len(t)} frames ({t[-1]:.0f} s), {len(data.points)} points; train {len(train)} / test {len(test)} frames")
    print(f"scan-to-depth dt: median {np.median(np.abs(c['dt'])) * 1e3:.1f} ms; "
          f"speed median {np.median(c['speed']):.2f} m/s")

    fits = {
        "identity": fit(dtr, "identity", p=p),
        "scale": fit(dtr, "scale", p=p),
        "inverse": fit(dtr, "inverse", p=p),
        "scale+extrinsic": fit(dtr, "scale", refine_extrinsic=True, p=p),
        "inverse+extrinsic": fit(dtr, "inverse", refine_extrinsic=True, p=p),
    }
    g = ground_fit(c, train)
    if g is not None:
        fits["ground-plane (existing)"] = g
    ext = fits["inverse+extrinsic"]
    ext_desc = dict(rot_deg=np.degrees(ext.delta_rotvec).round(3).tolist(), trans_m=ext.delta_trans.round(4).tolist())
    print(f"extrinsic correction (camera frame): {ext_desc}")

    rows = {}
    for name, fr in fits.items():
        nominal = evaluate(dte, fr, p, use_extrinsic=False)
        refined = evaluate(dte, FitResult(fr.model, fr.depth_params, ext.delta_rotvec, ext.delta_trans), p)
        rows[name] = dict(params=fr.depth_params, extrinsic=(dict(rot_deg=np.degrees(fr.delta_rotvec).tolist(),
                          trans_m=fr.delta_trans.tolist()) if "extrinsic" in name else None),
                          test_nominal_extrinsic=nominal, test_refined_extrinsic=refined)

    # stability: inverse model per window (nominal and refined extrinsic)
    win = (t // WINDOW_S).astype(int)
    stab = []
    for wi in np.unique(win):
        fr_w = np.flatnonzero(win == wi)
        if len(fr_w) < 10:
            continue
        d = data.subset(fr_w)
        sc = fit(d, "scale", p=p)
        iv = fit(d, "inverse", p=p)
        stab.append(dict(t=float(t[fr_w].mean()), s=sc.depth_params["s"], **iv.depth_params,
                         speed=float(np.median(c["speed"][fr_w]))))

    summary = dict(bag=str(c["bag"]), frames=len(t), points=int(len(data.points)), chunk_s=CHUNK_S,
                   extrinsic_correction=ext_desc, models=rows, windows=stab)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'model':26s} {'params':28s} | test, nominal extr.: med|err|  <5%   <10%  ratio"
          f" | refined extr.: med|err|  <5%   <10%  ratio")
    for name, rw in rows.items():
        ps = " ".join(f"{k}={v:.4f}" for k, v in rw["params"].items()) or "-"
        a, b = rw["test_nominal_extrinsic"], rw["test_refined_extrinsic"]
        print(f"{name:26s} {ps:28s} |            {np.expm1(a['median_abs_log']):7.2%} {a['within_5pct']:5.1%} "
              f"{a['within_10pct']:5.1%} {a['median_ratio']:.3f} |         {np.expm1(b['median_abs_log']):7.2%} "
              f"{b['within_5pct']:5.1%} {b['within_10pct']:5.1%} {b['median_ratio']:.3f}")
    print("\nmedian ZED/lidar ratio by lidar range (test, refined extrinsic):")
    hdr = "  ".join(f"{b['lo']:>4}-{b['hi']:<3}" for b in rows["identity"]["test_refined_extrinsic"]["bins"])
    print(f"{'':26s} {hdr}")
    for name, rw in rows.items():
        vals = "  ".join(f"{b.get('median_ratio', float('nan')):8.3f}" for b in rw["test_refined_extrinsic"]["bins"])
        print(f"{name:26s} {vals}")
    print("\nper-window fits:")
    for s in stab:
        print(f"  t={s['t']:6.0f}s  s={s['s']:.4f}  a={s['a']:.4f}  b={s['b']:+.4f}  speed={s['speed']:.2f}")

    plots(args.out, rows, stab, dte, fits, ext)
    print(f"\nwrote {args.out}")


def plots(out, rows, stab, dte, fits, ext):
    # 1. ratio vs range, per model (test frames, refined extrinsic)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, title in ((axs[0], "test_nominal_extrinsic", "nominal extrinsic"),
                           (axs[1], "test_refined_extrinsic", "refined extrinsic")):
        for name, rw in rows.items():
            bs = [b for b in rw[key]["bins"] if "median_ratio" in b]
            ax.plot([(b["lo"] + b["hi"]) / 2 for b in bs], [b["median_ratio"] for b in bs], "o-", label=name)
        ax.axhline(1, color="k", lw=1)
        ax.set(xlabel="lidar depth (m)", title=f"held-out frames, {title}")
    axs[0].set_ylabel("median corrected ZED / lidar")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ratio_vs_range.png", dpi=130)
    plt.close(fig)

    # 2. before/after 2D histograms (test, refined extrinsic)
    zz, zl, _, valid = dte.pairs(ext.delta_rotvec, ext.delta_trans)
    k = valid & (zl > 0.5) & (zl < 20)
    show = [("identity", fits["identity"]), ("inverse+extrinsic", ext)]
    if "ground-plane (existing)" in fits:
        show.append(("ground-plane (existing)", fits["ground-plane (existing)"]))
    fig, axs = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.8))
    for ax, (name, fr) in zip(axs, show):
        zc = apply_model(fr.model, fr.depth_params, zz[k])
        ax.hist2d(zl[k], zc, bins=150, range=[[0, 20], [0, 20]], cmin=1, norm=matplotlib.colors.LogNorm())
        ax.plot([0, 20], [0, 20], "k-", lw=1)
        ax.set(xlabel="lidar depth (m)", ylabel="ZED depth (m)", title=name)
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=130)
    plt.close(fig)

    # 3. per-window parameters
    if stab:
        fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
        tt = [s["t"] for s in stab]
        axs[0].plot(tt, [s["s"] for s in stab], "o-")
        axs[0].set_ylabel("scale s")
        axs[1].plot(tt, [s["a"] for s in stab], "o-", label="a")
        axs[1].plot(tt, [s["b"] for s in stab], "s-", label="b")
        axs[1].legend(fontsize=8)
        axs[1].set_ylabel("1/z = a/z_zed + b")
        axs[2].plot(tt, [s["speed"] for s in stab], "o-", color="tab:gray")
        axs[2].set(ylabel="speed (m/s)", xlabel="t (s)")
        fig.tight_layout()
        fig.savefig(out / "windows.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--bag", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--robot", default="hamilton")
    e.add_argument("--scan-stride", type=int, default=10, help="use every n-th lidar scan")
    e.add_argument("--lidar-rpy-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="extra rotation of the lidar in its TF frame, e.g. 180 0 0 for an upside-down "
                        "Mid-360 whose TF says identity (hamilton)")
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else run_fit(args)


if __name__ == "__main__":
    main()
