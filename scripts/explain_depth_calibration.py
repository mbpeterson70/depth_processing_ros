#!/usr/bin/env python3
"""Explanatory figures for the ZED depth correction, from a lidar_vs_zed.py cache + summary.

    explain_depth_calibration.py --out <lidar_vs_zed dir> [--bag <bag>] [--frame N]

Compares versions of the ZED depth (figures use held-out frames):
  raw ZED
  ground plane / lidar fitted on all the training chunks (lidar also refines the extrinsic)
  ground plane / lidar fitted once at the start, as depth_scale_correction_node does: the node's
    own calibrators replayed from the first frame (ground: first 30 s of usable frames; lidar:
    first 10 s), then kept fixed
  with --zed-extrinsic, the ground plane on all training chunks with a corrected ZED mount
Each version is calibrated and scored with its own camera extrinsic: raw and ground-plane with
the nominal one from the bag, lidar-optimized with the nominal one plus its lidar refinement,
corrected-extrinsic with the given mount (ground plane and lidar projection both moved).

  fig1_image_errors.png     lidar points on the camera image, colored by ZED/lidar
  fig2_top_down.png         ZED depth as a point cloud vs lidar, seen from above
  fig3_error_vs_range.png   median ZED/lidar (and spread) vs range, all held-out frames
  fig4_correction_shape.png multiplier each correction applies vs what the lidar says is needed
  fig5_online.png           how fast each estimator converges, and drift over the session
"""

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from depth_processing_ros.calibration import CalibrationParams, GroundCalibrator  # noqa: E402
from depth_processing_ros.lidar_calibration import LidarCalibrator, depth_pairs  # noqa: E402
from depth_processing_ros.lidar_reference import (LidarZedData, apply_model, fit, perturb,  # noqa: E402
                                                  project, transform)


@dataclasses.dataclass
class Version:
    key: str                            # short name for annotations / lookups
    name: str
    color: str
    model: str
    params: dict
    T_cam_lidar: np.ndarray             # projection of the lidar used to score this version
    D: np.ndarray                       # T_cam_camnominal: moves the gpe plane into this camera frame

    def plane(self, c, f):
        n, p = c["gpe_normal"][f], c["gpe_point"][f]
        return self.D[:3, :3] @ n, self.D[:3, :3] @ p + self.D[:3, 3]

# reference palette (dataviz skill): categorical slots 1-5 in fixed order, slot 7 (violet) for the
# corrected extrinsic so it keeps its purple from earlier figures; ink, surfaces
RAW, GROUND, LIDAR, GROUND_START, LIDAR_START, GROUND_FIXED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# diverging blue <-> red with a neutral grey midpoint: blue = ZED too short, red = too long
DIVERGING = LinearSegmentedColormap.from_list("zed_ratio", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"])
RATIO_RANGE = 0.25              # +-25% on the colour scale
ZED_MAX_DEPTH = 10.0            # m, driver max depth: nothing beyond is comparable
CHUNK_S = 60.0                  # must match lidar_vs_zed.py
MIN_GPE_HEIGHT = 0.4            # as in depth_scale_correction_node
# depth_scale_correction_node defaults, for replaying its start-up calibration
NODE_FRAME_PERIOD_S = 0.5
NODE_GROUND_DURATION_S = 30.0
NODE_MAX_CALIBRATION_TIME_S = 300.0
# zed_camera_link -> left optical frame (ZED wrapper's static chain, as in the bag)
T_LINK_OPT = np.eye(4)
T_LINK_OPT[:3, :3] = Rotation.from_quat([0.5, -0.5, 0.5, -0.5]).as_matrix()   # x y z w
T_LINK_OPT[:3, 3] = [-0.01, 0.06, 0.015]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def mount_T(q_wxyz, xyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([*q_wxyz[1:], q_wxyz[0]]).as_matrix()
    T[:3, 3] = xyz
    return T


def camera_correction(old, new):
    """T_camnew_camold for T_frontleft_zed_camera_link old -> new. The ground and lidar are fixed to
    the body, so only the camera moves: T_opt_link inv(T_fl_new) T_fl_old T_link_opt."""
    return np.linalg.inv(T_LINK_OPT) @ np.linalg.inv(new) @ old @ T_LINK_OPT


def ground_fit_corrected(c, frames, D):
    """Ground-plane calibration with the gpe plane moved into the corrected camera frame D."""
    print(f"corrected ZED mount, camera frame: t = {np.round(D[:3, 3], 3)} m, "
          f"rot = {np.round(np.degrees(Rotation.from_matrix(D[:3, :3]).as_rotvec()), 2)} deg")
    cal, used = GroundCalibrator(), 0
    t = c["t_cam"] * 1e-9
    for f in frames:
        n, p = c["gpe_normal"][f], c["gpe_point"][f]
        if not np.isfinite(n).all():
            continue
        n, p = D[:3, :3] @ n, D[:3, :3] @ p + D[:3, 3]
        if abs(float(n @ p)) < MIN_GPE_HEIGHT:
            continue
        used += cal.add_frame(float(t[f]), c["depths"][f].astype(np.float32), c["K"], n, p)[0]
    r = cal.fit()
    print(f"corrected-extrinsic ground-plane fit: {used} frames, a={r.a:.4f} b={r.b:+.4f} "
          f"accepted={r.accepted} ({r.reason})")
    return dict(a=r.a, b=r.b)


def ground_at_start(c):
    """The node's ground calibration replayed from the first frame: (a, b, seconds into the bag)."""
    cal, t = GroundCalibrator(CalibrationParams()), c["t_cam"] * 1e-9
    last = -np.inf
    for f in range(len(t)):
        if t[f] - t[0] > NODE_MAX_CALIBRATION_TIME_S:
            break
        if t[f] - last < NODE_FRAME_PERIOD_S:
            continue
        last = t[f]
        n, p = c["gpe_normal"][f], c["gpe_point"][f]
        if not np.isfinite(n).all() or abs(float(n @ p)) < MIN_GPE_HEIGHT:
            continue
        if not cal.add_frame(float(t[f]), c["depths"][f].astype(np.float32), c["K"], n, p)[0]:
            continue
        if cal.span < NODE_GROUND_DURATION_S:
            continue
        r = cal.fit()
        if r.accepted:
            return dict(a=r.a, b=r.b), float(t[f] - t[0])
        cal.drop_before(cal.frames[-1][0] - NODE_GROUND_DURATION_S / 2)
    return dict(a=1.0, b=0.0), None                      # node falls back to identity


def lidar_at_start(c, data, T_cam_lidar):
    """The node's start-up lidar calibration (range-gated, fit once) replayed from the first frame:
    (a, b, seconds into the bag)."""
    cal, t = LidarCalibrator(), c["t_cam"] * 1e-9
    rng = np.random.default_rng(0)
    bounds = np.searchsorted(data.frame, np.arange(len(t) + 1))   # points are grouped by frame
    last = -np.inf
    for f in range(len(t)):
        if t[f] - last < NODE_FRAME_PERIOD_S:
            continue
        last = t[f]
        pts = transform(T_cam_lidar, data.points[bounds[f]:bounds[f + 1]])
        zz, zl = depth_pairs(data.depths[f], data.K, pts, cal.params, rng)
        if not cal.add_pairs(float(t[f]), zz, zl)[0]:
            continue
        fit_now, covered = cal.ready()
        if not fit_now:
            continue
        r = cal.fit()
        if r.accepted:
            print(f"lidar gate: fitted over {cal.span:.0f} s, pairs {cal.coverage_text()}"
                  + ("" if covered else " (TIMED OUT: range not covered)"))
            return dict(a=r.a, b=r.b), float(t[f] - t[0])
        cal.drop_before(cal.frames[-1][0] - cal.span / 2)
    return None, None


def load(out: Path, zed_extrinsic=None, old_extrinsic=None):
    c = dict(np.load(out / "cache.npz"))
    s = json.loads((out / "summary.json").read_text())
    data = LidarZedData(c["depths"].astype(np.float32), c["points"].astype(np.float64), c["frame"],
                        c["K"], c["T_cam_lidar"])
    m = s["models"]
    ext = m["inverse+extrinsic"]["extrinsic"]
    rv, tr = np.radians(ext["rot_deg"]), np.array(ext["trans_m"])
    T0, I = c["T_cam_lidar"], np.eye(4)
    versions = [Version("raw", "raw ZED", RAW, "identity", {}, T0, I),
                Version("ground", "ground plane, all training chunks", GROUND, "inverse",
                        m["ground-plane (existing)"]["params"], T0, I),
                Version("lidar", "lidar, all training chunks (+ extrinsic refinement)", LIDAR, "inverse",
                        m["inverse+extrinsic"]["params"], perturb(T0, rv, tr), I)]
    t = (c["t_cam"] - c["t_cam"][0]) * 1e-9
    test = np.flatnonzero((t // CHUNK_S).astype(int) % 2 == 1)
    gs, gs_t = ground_at_start(c)
    print(f"ground at start: a={gs['a']:.4f} b={gs['b']:+.4f} "
          + (f"fixed at t={gs_t:.0f} s" if gs_t is not None else "(no fit accepted: identity fallback)"))
    versions.append(Version("ground start", f"ground plane, start only ({NODE_GROUND_DURATION_S:.0f} s, as on robot)",
                            GROUND_START, "inverse", gs, T0, I))
    ls, ls_t = lidar_at_start(c, data, T0)
    if ls is not None:
        print(f"lidar at start: a={ls['a']:.4f} b={ls['b']:+.4f} fixed at t={ls_t:.0f} s")
        versions.append(Version("lidar start", "lidar, start only (range-gated, as on robot)",
                                LIDAR_START, "inverse", ls, T0, I))
    else:
        print("lidar at start: no fit accepted")
    # start-only fits see everything up to when they were fixed: score every version only on
    # held-out frames after that, so no version is scored on frames it calibrated on
    t_fixed = max(x or 0 for x in (gs_t, ls_t))
    dropped = int(np.sum(t[test] <= t_fixed))
    test = test[t[test] > t_fixed]
    if dropped:
        print(f"held-out frames before t={t_fixed:.0f} s (start-up calibration) dropped: {dropped}; {len(test)} left")
    if zed_extrinsic is not None:
        train = np.flatnonzero((t // CHUNK_S).astype(int) % 2 == 0)
        D = camera_correction(mount_T(old_extrinsic[:4], old_extrinsic[4:]),
                              mount_T(zed_extrinsic[:4], zed_extrinsic[4:]))
        versions.append(Version("corrected extr.", "corrected extrinsics, ground plane, all training chunks",
                                GROUND_FIXED, "inverse",
                                ground_fit_corrected(c, train, D), D @ T0, D))
    return c, data, versions, rv, tr, t, test


def panel_grid(n, w, h, **kw):
    """Up to 4 panels in a row, more in rows of 3. Returns (fig, axes for the n panels)."""
    ncols = n if n <= 4 else 3
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), constrained_layout=True,
                            squeeze=False, **kw)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    return fig, list(axs.flat[:n]), axs


def frame_pairs(data, f, T_cam_lidar):
    """(u, v, z_lidar, z_zed, p_cam) for lidar points of frame f that hit valid ZED depth."""
    sel = data.frame == f
    p = transform(T_cam_lidar, data.points[sel])
    u, v, z = project(p, data.K)
    h, w = data.depths.shape[1:]
    ok = (z > 0.5) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    zz = np.full(len(z), np.nan, np.float32)
    zz[ok] = data.depths[f, np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
    ok &= np.isfinite(zz)
    return u[ok], v[ok], z[ok], zz[ok], p[ok]


def pick_frame(data, test, T_cam_lidar):
    """Held-out frame with many lidar/ZED pairs spread over 2-10 m."""
    best, score = test[0], -1
    for f in test[::3]:
        _, _, zl, _, _ = frame_pairs(data, f, T_cam_lidar)
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

def fig1_image(out, data, versions, f, rgb):
    fig, axs, grid = panel_grid(len(versions), 5.3, 3.9)
    for ax, ver in zip(axs, versions):
        u, v, zl, zz, _ = frame_pairs(data, f, ver.T_cam_lidar)
        keep = zz < ZED_MAX_DEPTH * 0.98
        name = ver.name
        ratio = apply_model(ver.model, ver.params, zz[keep]) / zl[keep]
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
                 f"(held-out frame {f}; lidar projected with each version's own extrinsic)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(out / "fig1_image_errors.png", dpi=140)
    plt.close(fig)


def fig2_top_down(out, c, data, versions, f):
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
    has_plane = np.isfinite(c["gpe_normal"][f]).all()
    best = by_key(versions, "lidar")
    if has_plane:
        n, pt = best.plane(c, f)
        not_ground = (cloud(apply_model(best.model, best.params, z)) - pt) @ n > 0.15
    else:
        not_ground = np.ones(len(z), bool)

    fig, axs, grid = panel_grid(len(versions), 5, 6.2, sharex=True, sharey=True)
    for ax, ver in zip(axs, versions):
        name, color = ver.name, ver.color
        _, _, _, _, pl = frame_pairs(data, f, ver.T_cam_lidar)     # lidar in this version's camera frame
        if has_plane:
            n, pt = ver.plane(c, f)
            pl = pl[(pl - pt) @ n > 0.15]
        pl = pl[pl[:, 2] < 14]
        P = cloud(apply_model(ver.model, ver.params, z))[not_ground]
        ax.scatter(P[:, 0], P[:, 2], s=1.5, c=color, alpha=0.35, linewidths=0)
        ax.scatter(pl[:, 0], pl[:, 2], s=5, c=INK, linewidths=0)
        ax.plot([0], [0], marker="^", ms=10, color=INK2, ls="none", clip_on=False)
        ax.set_title(f"ZED depth: {name}", loc="left", color=color)
        ax.set_aspect("equal")
        ax.set_xlabel("right of camera (m)")
    for ax in grid[:, 0]:
        ax.set_ylabel("in front of camera (m)")
    axs[0].set_xlim(-7, 7), axs[0].set_ylim(0, 14)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=6, color=col, label=f"ZED, {nm}")
               for nm, col in ((v.name, v.color) for v in versions)] + \
              [plt.Line2D([], [], ls="none", marker="o", ms=6, color=INK, label="lidar (reference)"),
               plt.Line2D([], [], ls="none", marker="^", ms=8, color=INK2, label="camera")]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4))
    fig.suptitle(f"Seen from above, ground removed (frame {f}): raw ZED pulls every surface toward the camera",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(out / "fig2_top_down.png", dpi=140)
    plt.close(fig)


def all_pairs(data, frames, T_cam_lidar):
    sub = dataclasses.replace(data.subset(frames), T_cam_lidar=T_cam_lidar)
    zz, zl, _, valid = sub.pairs()
    k = valid & (zl > 0.5) & (zl < 20)
    return zz[k], zl[k]


def fig3_error_vs_range(out, data, versions, test):
    edges = np.arange(0.5, 16.01, 0.5)
    mid = (edges[1:] + edges[:-1]) / 2
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axvspan(ZED_MAX_DEPTH, edges[-1], color=GRID, alpha=0.6, lw=0)
    ax.text(ZED_MAX_DEPTH + 0.2, 1.27, "ZED depth is capped at 10 m:\nnot comparable here",
            color=INK2, fontsize=9, va="top")
    ax.axhline(1, color=INK2, lw=1)
    for ver in versions:
        zz, zl = all_pairs(data, test, ver.T_cam_lidar)
        name, color = ver.name, ver.color
        r = apply_model(ver.model, ver.params, zz) / zl
        q = np.array([np.percentile(r[(zl >= a) & (zl < b)], [25, 50, 75]) if ((zl >= a) & (zl < b)).sum() > 100
                      else [np.nan] * 3 for a, b in zip(edges[:-1], edges[1:])])
        ax.fill_between(mid, q[:, 0], q[:, 2], color=color, alpha=0.15, lw=0)
        ax.plot(mid, q[:, 1], color=color, label=name)
        last = np.flatnonzero(np.isfinite(q[:, 1]) & (mid < ZED_MAX_DEPTH))[-1]
        ax.annotate(f"{q[last, 1]:.2f}", (mid[last], q[last, 1]), xytext=(4, 7), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set(xlabel="true depth from lidar (m)", ylabel="ZED depth / lidar depth", ylim=(0.6, 1.3),
           xlim=(0.5, edges[-1]))
    ax.set_title(f"Held-out frames ({len(test)}, {len(zl) / 1e6:.1f} M lidar pts; own extrinsic per version): "
                 "median, middle 50%", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(out / "fig3_error_vs_range.png", dpi=140)
    plt.close(fig)


def fig4_shape(out, data, versions, test):
    edges = np.arange(0.5, ZED_MAX_DEPTH, 0.5)
    mid = (edges[1:] + edges[:-1]) / 2

    def needed(T):
        zz, zl = all_pairs(data, test, T)
        return np.array([np.median(zl[(zz >= a) & (zz < b)] / zz[(zz >= a) & (zz < b)])
                         if ((zz >= a) & (zz < b)).sum() > 100 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    x = np.linspace(0.5, ZED_MAX_DEPTH, 200)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.axvspan(1, 8, color=GROUND, alpha=0.07, lw=0)
    ax.text(1.1, 1.375, "ground-plane calibration only\nsees ground pixels at 1-8 m", color=INK2, fontsize=9, va="top")
    ax.scatter(mid, needed(by_key(versions, "lidar").T_cam_lidar), s=40, color=INK, zorder=3,
               label="needed (median lidar / ZED, held-out; lidar-refined extrinsic)")
    ax.scatter(mid, needed(by_key(versions, "raw").T_cam_lidar), s=40, facecolors="none", edgecolors=INK2,
               zorder=3, label="needed, nominal extrinsic")
    corrected = by_key(versions, "corrected extr.")
    if corrected is not None:
        ax.scatter(mid, needed(corrected.T_cam_lidar), s=40, facecolors="none", edgecolors=GROUND_FIXED,
                   zorder=3, label="needed, corrected extrinsic")
    for ver in versions:
        ax.plot(x, apply_model(ver.model, ver.params, x) / x, color=ver.color, label=ver.name)
    w = max(len(v.key) for v in versions)
    txt = "1/z = a/z_zed + b\n" + "\n".join(f"{v.key + ':':{w + 1}s} a={v.params['a']:.3f}, b={v.params['b']:+.3f}"
                                            for v in versions if v.model == "inverse")
    ax.annotate(txt, (0.98, 0.04), xycoords="axes fraction", multialignment="left",
                ha="right", va="bottom", fontsize=9, color=INK2, family="monospace")
    ax.set(xlabel="raw ZED depth (m)", ylabel="multiplier applied to ZED depth", ylim=(0.98, 1.4))
    ax.set_title("What each correction does: corrected depth = multiplier x raw depth", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(out / "fig4_correction_shape.png", dpi=140)
    plt.close(fig)


def by_key(versions, key):
    return next((v for v in versions if v.key == key), None)


def scale_at(params, z=5.0):
    """Multiplier the inverse model applies at z m raw depth: one number to compare fits."""
    return 1.0 / (params["a"] + params["b"] * z)


def fig5_online(out, c, data, t):
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
    ap.add_argument("--zed-extrinsic", type=float, nargs=7, metavar=("QW", "QX", "QY", "QZ", "X", "Y", "Z"),
                    help="corrected T_frontleft_zed_camera_link: adds a ground-plane calibration with it")
    ap.add_argument("--old-zed-extrinsic", type=float, nargs=7, metavar=("QW", "QX", "QY", "QZ", "X", "Y", "Z"),
                    default=[0.5860445702207298, -0.1518423673416495, -0.7689097936324478, 0.20560492808050715,
                             -0.1377625424350767, 0.021380403878896944, 0.057657655316766976],
                    help="T_frontleft_zed_camera_link the bag was recorded with (default: topaz/smaug/hamilton)")
    args = ap.parse_args()
    c, data, versions, rv, tr, t, test = load(args.out, args.zed_extrinsic, args.old_zed_extrinsic)
    f = args.frame if args.frame is not None else pick_frame(data, test, by_key(versions, "lidar").T_cam_lidar)
    print(f"frame {f} (t = {t[f]:.0f} s, speed {c['speed'][f]:.2f} m/s)")
    rgb = rgb_at(args.bag, args.robot, int(c["t_cam"][f])) if args.bag else None
    for ver in versions:
        zz, zl = all_pairs(data, test, ver.T_cam_lidar)
        k = zz < ZED_MAX_DEPTH * 0.98
        lr = np.log(apply_model(ver.model, ver.params, zz[k]) / zl[k])
        print(f"  {ver.name:48s} held-out: median ratio {np.exp(np.median(lr)):.3f}, "
              f"median |err| {np.expm1(np.median(np.abs(lr))):5.1%}, within 5% {np.mean(np.abs(lr) < np.log(1.05)):5.1%}")
    fig1_image(args.out, data, versions, f, rgb)
    fig2_top_down(args.out, c, data, versions, f)
    fig3_error_vs_range(args.out, data, versions, test)
    fig4_shape(args.out, data, versions, test)
    if t[-1] < 200:
        print(f"bag is {t[-1]:.0f} s: too short for fig5 (needs >= 200 s), skipped")
        return
    stats = fig5_online(args.out, c, data, t)
    (args.out / "fig5_online.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: np.round(v, 3).tolist() for k, v in stats.items()}, indent=None))


if __name__ == "__main__":
    main()
