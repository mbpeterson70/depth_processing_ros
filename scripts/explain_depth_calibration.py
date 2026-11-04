#!/usr/bin/env python3
"""Explanatory figures for the ZED depth correction, from a lidar_vs_zed.py cache + summary.

    explain_depth_calibration.py --out <lidar_vs_zed dir> [--bag <bag>] [--frame N]

Compares three versions of the ZED depth: raw, the existing ground-plane calibration and the
lidar-optimized correction (both fitted on the training chunks; figures use held-out frames).
The lidar is projected with the refined extrinsic in every panel, so only the depth differs.

  fig1_image_errors.png     lidar points on the camera image, colored by ZED/lidar
  fig2_top_down.png         ZED depth as a point cloud vs lidar, seen from above
  fig3_error_vs_range.png   median ZED/lidar (and spread) vs range, all held-out frames
  fig4_correction_shape.png multiplier each correction applies vs what the lidar says is needed
  fig5_online.png           how fast each estimator converges, and drift over the session
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from depth_processing_ros.calibration import GroundCalibrator  # noqa: E402
from depth_processing_ros.lidar_reference import (LidarZedData, apply_model, fit, perturb,  # noqa: E402
                                                  project, transform)

# reference palette (dataviz skill): categorical slots 1-3 in fixed order, ink, surfaces
RAW, GROUND, LIDAR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# diverging blue <-> red with a neutral grey midpoint: blue = ZED too short, red = too long
DIVERGING = LinearSegmentedColormap.from_list("zed_ratio", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"])
RATIO_RANGE = 0.25              # +-25% on the colour scale
ZED_MAX_DEPTH = 10.0            # m, driver max depth: nothing beyond is comparable
CHUNK_S = 60.0                  # must match lidar_vs_zed.py

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(out: Path):
    c = dict(np.load(out / "cache.npz"))
    s = json.loads((out / "summary.json").read_text())
    data = LidarZedData(c["depths"].astype(np.float32), c["points"].astype(np.float64), c["frame"],
                        c["K"], c["T_cam_lidar"])
    m = s["models"]
    ext = m["inverse+extrinsic"]["extrinsic"]
    rv, tr = np.radians(ext["rot_deg"]), np.array(ext["trans_m"])
    versions = [("raw ZED", RAW, "identity", {}),
                ("existing ground-plane calibration", GROUND, "inverse", m["ground-plane (existing)"]["params"]),
                ("lidar-optimized", LIDAR, "inverse", m["inverse+extrinsic"]["params"])]
    t = (c["t_cam"] - c["t_cam"][0]) * 1e-9
    test = np.flatnonzero((t // CHUNK_S).astype(int) % 2 == 1)
    return c, data, versions, rv, tr, t, test


def frame_pairs(data, f, rv, tr):
    """(u, v, z_lidar, z_zed, p_cam) for lidar points of frame f that hit valid ZED depth."""
    sel = data.frame == f
    p = transform(perturb(data.T_cam_lidar, rv, tr), data.points[sel])
    u, v, z = project(p, data.K)
    h, w = data.depths.shape[1:]
    ok = (z > 0.5) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    zz = np.full(len(z), np.nan, np.float32)
    zz[ok] = data.depths[f, np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
    ok &= np.isfinite(zz)
    return u[ok], v[ok], z[ok], zz[ok], p[ok]


def pick_frame(data, test, rv, tr):
    """Held-out frame with many lidar/ZED pairs spread over 2-10 m."""
    best, score = test[0], -1
    for f in test[::3]:
        _, _, zl, _, _ = frame_pairs(data, f, rv, tr)
        hist = np.histogram(zl, bins=[1, 2.5, 4, 6, 8, 10])[0]
        s = np.minimum(hist, 150).sum()
        if s > score:
            best, score = f, s
    return int(best)


def rgb_at(bag: Path, robot: str, stamp_ns: int):
    from rosbags.highlevel import AnyReader
    topic = f"/{robot}/{robot}_zed/rgb/image_rect_color"
    with AnyReader([bag]) as r:
        con = [c for c in r.connections if c.topic == topic]
        for c, t, raw in r.messages(connections=con, start=int(stamp_ns - 3e8)):
            m = r.deserialize(raw, c.msgtype)
            if abs(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec - stamp_ns) < 2e7:
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 4]
                return cv2.cvtColor(img.reshape(m.height, m.width, 4), cv2.COLOR_BGRA2RGB)
            if t > stamp_ns + 1e9:
                break
    return None


# ---------------------------------------------------------------------------------- figures

def fig1_image(out, data, versions, f, rv, tr, rgb):
    u, v, zl, zz, _ = frame_pairs(data, f, rv, tr)
    keep = zz < ZED_MAX_DEPTH * 0.98
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.9), constrained_layout=True)
    for ax, (name, color, model, params) in zip(axs, versions):
        ratio = apply_model(model, params, zz[keep]) / zl[keep]
        bg = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) if rgb is not None else \
            np.full((*data.depths.shape[1:], 3), 40, np.uint8)
        ax.imshow((bg * 0.75).astype(np.uint8))
        sc = ax.scatter(u[keep], v[keep], c=np.log(ratio), s=7, cmap=DIVERGING, linewidths=0,
                        vmin=-np.log(1 + RATIO_RANGE), vmax=np.log(1 + RATIO_RANGE))
        within = np.mean(np.abs(np.log(ratio)) < np.log(1.05))
        ax.set_title(f"{name}\nmedian ZED/lidar {np.median(ratio):.2f} · {within:.0%} of points within 5%",
                     loc="left", color=INK)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.01)
    ticks = [0.8, 0.9, 1.0, 1.1, 1.2]
    cb.set_ticks(np.log(ticks), labels=[f"{x:.1f}" for x in ticks])
    cb.set_label("ZED / lidar depth  (blue: ZED too short)", color=INK2)
    fig.suptitle(f"Each dot is a lidar point, coloured by how far the ZED depth at that pixel is off "
                 f"(held-out frame {f})", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(out / "fig1_image_errors.png", dpi=140)
    plt.close(fig)


def fig2_top_down(out, c, data, versions, f, rv, tr):
    K = data.K
    d = data.depths[f]
    vv, uu = np.mgrid[0:d.shape[0]:2, 0:d.shape[1]:2]
    z = d[vv, uu]
    ok = np.isfinite(z) & (z < ZED_MAX_DEPTH * 0.98)
    vv, uu, z = vv[ok], uu[ok], z[ok]

    def cloud(zc):
        return np.stack([(uu - K[0, 2]) / K[0, 0] * zc, (vv - K[1, 2]) / K[1, 1] * zc, zc], 1)

    # ground pixels are chosen once, from the lidar-optimized cloud, and dropped in every panel:
    # judged per version, the raw (too short) ground would lift off the plane and stay in
    n, pt = c["gpe_normal"][f], c["gpe_point"][f]
    best = versions[-1]
    if np.isfinite(n).all():
        not_ground = (cloud(apply_model(best[2], best[3], z)) - pt) @ n > 0.15
    else:
        not_ground = np.ones(len(z), bool)
    _, _, _, _, pl = frame_pairs(data, f, rv, tr)
    if np.isfinite(n).all():
        pl = pl[(pl - pt) @ n > 0.15]
    pl = pl[pl[:, 2] < 14]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, color, model, params) in zip(axs, versions):
        P = cloud(apply_model(model, params, z))[not_ground]
        ax.scatter(P[:, 0], P[:, 2], s=1.5, c=color, alpha=0.35, linewidths=0)
        ax.scatter(pl[:, 0], pl[:, 2], s=5, c=INK, linewidths=0)
        ax.plot([0], [0], marker="^", ms=10, color=INK2, ls="none", clip_on=False)
        ax.set_title(f"ZED depth: {name}", loc="left", color=color)
        ax.set_aspect("equal")
        ax.set_xlabel("right of camera (m)")
    axs[0].set_ylabel("in front of camera (m)")
    axs[0].set_xlim(-7, 7), axs[0].set_ylim(0, 14)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=6, color=col, label=f"ZED, {nm}")
               for nm, col, *_ in versions] + \
              [plt.Line2D([], [], ls="none", marker="o", ms=6, color=INK, label="lidar (reference)"),
               plt.Line2D([], [], ls="none", marker="^", ms=8, color=INK2, label="camera")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    fig.suptitle(f"Seen from above, ground removed (frame {f}): raw ZED pulls every surface toward the camera",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(out / "fig2_top_down.png", dpi=140)
    plt.close(fig)


def all_pairs(data, frames, rv, tr):
    sub = data.subset(frames)
    zz, zl, _, valid = sub.pairs(rv, tr)
    k = valid & (zl > 0.5) & (zl < 20)
    return zz[k], zl[k]


def fig3_error_vs_range(out, data, versions, test, rv, tr):
    zz, zl = all_pairs(data, test, rv, tr)
    edges = np.arange(0.5, 16.01, 0.5)
    mid = (edges[1:] + edges[:-1]) / 2
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axvspan(ZED_MAX_DEPTH, edges[-1], color=GRID, alpha=0.6, lw=0)
    ax.text(ZED_MAX_DEPTH + 0.2, 1.27, "ZED depth is capped at 10 m:\nnot comparable here",
            color=INK2, fontsize=9, va="top")
    ax.axhline(1, color=INK2, lw=1)
    for name, color, model, params in versions:
        r = apply_model(model, params, zz) / zl
        q = np.array([np.percentile(r[(zl >= a) & (zl < b)], [25, 50, 75]) if ((zl >= a) & (zl < b)).sum() > 100
                      else [np.nan] * 3 for a, b in zip(edges[:-1], edges[1:])])
        ax.fill_between(mid, q[:, 0], q[:, 2], color=color, alpha=0.15, lw=0)
        ax.plot(mid, q[:, 1], color=color, label=name)
        last = np.flatnonzero(np.isfinite(q[:, 1]) & (mid < ZED_MAX_DEPTH))[-1]
        ax.annotate(f"{q[last, 1]:.2f}", (mid[last], q[last, 1]), xytext=(4, 7), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set(xlabel="true depth from lidar (m)", ylabel="ZED depth / lidar depth", ylim=(0.6, 1.3),
           xlim=(0.5, edges[-1]))
    ax.set_title(f"Held-out frames ({len(test)}, {len(zl) / 1e6:.1f} M lidar points): "
                 "line = median, band = middle 50%", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(out / "fig3_error_vs_range.png", dpi=140)
    plt.close(fig)


def fig4_shape(out, data, versions, test, rv, tr):
    zz, zl = all_pairs(data, test, rv, tr)
    edges = np.arange(0.5, ZED_MAX_DEPTH, 0.5)
    mid = (edges[1:] + edges[:-1]) / 2
    need = np.array([np.median(zl[(zz >= a) & (zz < b)] / zz[(zz >= a) & (zz < b)])
                     if ((zz >= a) & (zz < b)).sum() > 100 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    x = np.linspace(0.5, ZED_MAX_DEPTH, 200)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axvspan(1, 8, color=GROUND, alpha=0.07, lw=0)
    ax.text(1.1, 1.375, "ground-plane calibration only\nsees ground pixels at 1-8 m", color=INK2, fontsize=9, va="top")
    ax.scatter(mid, need, s=40, color=INK, zorder=3, label="needed (median lidar / ZED, held-out)")
    for name, color, model, params in versions:
        ax.plot(x, apply_model(model, params, x) / x, color=color, label=name)
    lp = versions[2][3]
    gp = versions[1][3]
    ax.annotate(f"1/z = a/z_zed + b\nlidar:  a={lp['a']:.3f}, b={lp['b']:+.3f}\n"
                f"ground: a={gp['a']:.3f}, b={gp['b']:+.3f}", (0.98, 0.12), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color=INK2, family="monospace")
    ax.set(xlabel="raw ZED depth (m)", ylabel="multiplier applied to ZED depth", ylim=(0.98, 1.4))
    ax.set_title("What each correction does: corrected depth = multiplier x raw depth", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86))
    fig.savefig(out / "fig4_correction_shape.png", dpi=140)
    plt.close(fig)


def scale_at(params, z=5.0):
    """Multiplier the inverse model applies at z m raw depth: one number to compare fits."""
    return 1.0 / (params["a"] + params["b"] * z)


def fig5_online(out, c, data, t, rv, tr):
    starts = np.arange(30, t[-1] - 130, 150.0)
    durations = [5, 10, 20, 30, 60, 120]

    def lidar_est(frames):
        return scale_at(fit(data.subset(frames), "inverse").depth_params) if len(frames) >= 3 else np.nan

    def ground_est(frames):
        cal, used = GroundCalibrator(), 0
        for f in frames:
            n, p = c["gpe_normal"][f], c["gpe_point"][f]
            if np.isfinite(n).all() and abs(float(n @ p)) >= 0.4:
                used += cal.add_frame(float(t[f]), data.depths[f], data.K, n, p)[0]
        return scale_at(dict(a=(r := cal.fit()).a, b=r.b)) if used >= 3 else np.nan

    conv = {"lidar": np.full((len(starts), len(durations)), np.nan),
            "ground": np.full((len(starts), len(durations)), np.nan)}
    for i, s0 in enumerate(starts):
        for j, d in enumerate(durations):
            fr = np.flatnonzero((t >= s0) & (t < s0 + d))
            conv["lidar"][i, j] = lidar_est(fr)
            conv["ground"][i, j] = ground_est(fr)
    win = np.arange(0, t[-1], 60.0)
    drift = {k: [] for k in ("lidar", "ground")}
    for w0 in win:
        fr = np.flatnonzero((t >= w0) & (t < w0 + 60))
        drift["lidar"].append(lidar_est(fr))
        drift["ground"].append(ground_est(fr))
    speed = [np.median(c["speed"][(t >= w0) & (t < w0 + 60)]) for w0 in win]

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    ax = axs[0]
    for key, color, name in (("lidar", LIDAR, "lidar"), ("ground", GROUND, "ground plane")):
        for row in conv[key]:
            ax.plot(durations, row, color=color, alpha=0.25, lw=1)
        ax.plot(durations, np.nanmedian(conv[key], axis=0), color=color, marker="o", ms=6,
                label=f"{name} (median over {len(starts)} start times)")
    ax.set_xscale("log")
    ax.set_xticks(durations, labels=[f"{d}s" for d in durations])
    ax.set(xlabel="seconds of data used for the fit", ylabel="correction at 5 m (multiplier)")
    ax.set_title("How much data an online fit needs (thin lines: individual start times)", loc="left")
    ax.legend(loc="lower right")
    ax = axs[1]
    ax.plot(win + 30, drift["lidar"], color=LIDAR, marker="o", ms=5, label="lidar, 60 s windows")
    ax.plot(win + 30, drift["ground"], color=GROUND, marker="o", ms=5, label="ground plane, 60 s windows")
    for w0, sp in zip(win, speed):
        if sp < 0.1:
            ax.axvspan(w0, w0 + 60, color=GRID, alpha=0.6, lw=0)
    ax.set(xlabel="time in bag (s)   (grey: robot standing)", ylabel="correction at 5 m (multiplier)")
    ax.set_title("Is the bias stable over the session?", loc="left")
    ax.legend(loc="lower right")
    lo = np.nanmin([np.nanmin(conv["ground"]), np.nanmin(drift["ground"]), np.nanmin(drift["lidar"])])
    hi = np.nanmax([np.nanmax(conv["lidar"]), np.nanmax(drift["lidar"]), np.nanmax(drift["ground"])])
    for a in axs:
        a.set_ylim(max(0.9, lo - 0.03), min(1.6, hi + 0.03))
    fig.savefig(out / "fig5_online.png", dpi=140)
    plt.close(fig)
    return dict(durations=durations, lidar_median=np.nanmedian(conv["lidar"], 0).tolist(),
                lidar_spread=np.nanstd(conv["lidar"], 0).tolist(),
                ground_median=np.nanmedian(conv["ground"], 0).tolist(),
                ground_spread=np.nanstd(conv["ground"], 0).tolist(),
                drift_lidar=drift["lidar"], drift_ground=drift["ground"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bag", type=Path, help="for the camera image in fig1 (else a plain background)")
    ap.add_argument("--robot", default="hamilton")
    ap.add_argument("--frame", type=int, help="frame for fig1/fig2 (default: picked from held-out frames)")
    args = ap.parse_args()
    c, data, versions, rv, tr, t, test = load(args.out)
    f = args.frame if args.frame is not None else pick_frame(data, test, rv, tr)
    print(f"frame {f} (t = {t[f]:.0f} s, speed {c['speed'][f]:.2f} m/s)")
    rgb = rgb_at(args.bag, args.robot, int(c["t_cam"][f])) if args.bag else None
    fig1_image(args.out, data, versions, f, rv, tr, rgb)
    fig2_top_down(args.out, c, data, versions, f, rv, tr)
    fig3_error_vs_range(args.out, data, versions, test, rv, tr)
    fig4_shape(args.out, data, versions, test, rv, tr)
    stats = fig5_online(args.out, c, data, t, rv, tr)
    (args.out / "fig5_online.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: np.round(v, 3).tolist() for k, v in stats.items()}, indent=None))


if __name__ == "__main__":
    main()
