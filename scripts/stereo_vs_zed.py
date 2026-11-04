#!/usr/bin/env python3
"""Offline diagnostic: ZED depth vs classical stereo (SGBM) on the ZED's own rectified pair.

Reads a bag with left/right rectified images, depth and camera_info, runs SGBM, compares
block-median depths and writes a panel video plus aggregate plots and summary.json:

    stereo_vs_zed.py --bag <bag_dir> --out <dir> [--ns /euclid/euclid_zed] [--stride 10]

ratio = z_zed / z_stereo (< 1: ZED reads short). Stereo noise grows as z^2, see
stereo_reference.stereo_sigma; trust the far bins less.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from rosbags.highlevel import AnyReader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from depth_processing_ros.calibration import GroundCalibrator, REPORT_BINS  # noqa: E402
from depth_processing_ros.stereo_reference import (StereoParams, block_compare,  # noqa: E402
                                                   stereo_depth, stereo_sigma)

RANGE_BINS = REPORT_BINS + ((8, 12),)
DEPTH_VIS_MAX = 10.0            # m, colormap range for depth tiles
RATIO_VIS = 0.25                # +-25% on the log-ratio tile
FIT_RANGE = (1.0, 8.0)          # m, stereo range used for the (a, b) fit
MAX_FIT_PAIRS = 200_000
# diverging BGR lookup table (blue: ZED shorter than stereo, red: longer, white: equal)
RATIO_LUT = (255 * plt.get_cmap("coolwarm")(np.linspace(0, 1, 256))[:, 2::-1]).astype(np.uint8)


def to_gray(msg) -> np.ndarray:
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}[msg.encoding]
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)
    if ch == 1:
        return img[..., 0].copy()
    code = {"bgra8": cv2.COLOR_BGRA2GRAY, "rgba8": cv2.COLOR_RGBA2GRAY,
            "bgr8": cv2.COLOR_BGR2GRAY, "rgb8": cv2.COLOR_RGB2GRAY}[msg.encoding]
    return cv2.cvtColor(img, code)


def to_depth(msg) -> np.ndarray:
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == "16UC1":
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def stamp(msg) -> int:
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def colorize_depth(z: np.ndarray) -> np.ndarray:
    v = np.clip(np.nan_to_num(z, nan=0.0) / DEPTH_VIS_MAX, 0, 1)
    out = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~(np.isfinite(z) & (z > 0))] = 0
    return out


def colorize_ratio(stats, shape) -> np.ndarray:
    lr = np.log(stats.z_zed / stats.z_stereo)
    v = np.clip(np.nan_to_num(lr) / np.log(1 + RATIO_VIS), -1, 1)
    img = RATIO_LUT[(127.5 * (1 + v)).astype(np.uint8)]
    img[~stats.valid] = 40
    img = cv2.resize(img, (img.shape[1] * stats.block, img.shape[0] * stats.block),
                     interpolation=cv2.INTER_NEAREST)
    return img[:shape[0], :shape[1]]


def scatter_tile(stats, fxB, shape) -> np.ndarray:
    h, w = shape
    img = np.full((h, w, 3), 255, np.uint8)
    m = 30

    def px(x, y):
        return (int(m + x / DEPTH_VIS_MAX * (w - 2 * m)), int(h - m - y / DEPTH_VIS_MAX * (h - 2 * m)))

    cv2.rectangle(img, px(0, 0), px(DEPTH_VIS_MAX, DEPTH_VIS_MAX), (0, 0, 0), 1)
    zs = np.linspace(0.3, DEPTH_VIS_MAX, 50)
    sig = stereo_sigma(zs, fxB)
    for k, col in ((0, (0, 0, 0)), (1, (160, 160, 160)), (-1, (160, 160, 160))):
        pts = np.array([px(x, x + k * s) for x, s in zip(zs, sig)], np.int32)
        cv2.polylines(img, [pts], False, col, 1)
    for x, y in zip(stats.z_stereo[stats.valid], stats.z_zed[stats.valid]):
        cv2.circle(img, px(x, y), 2, (200, 80, 0), -1)
    cv2.putText(img, "x: stereo  y: ZED (m), 0-10", (m, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ns", default="/euclid/euclid_zed")
    ap.add_argument("--stride", type=int, default=10, help="use every n-th synced frame")
    ap.add_argument("--start", type=float, default=0.0, help="s from bag start")
    ap.add_argument("--end", type=float, default=float("inf"), help="s from bag start")
    ap.add_argument("--block", type=int, default=16, help="px")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = StereoParams(block=args.block)

    topics = {"left": f"{args.ns}/left/image_rect_color", "right": f"{args.ns}/right/image_rect_color",
              "depth": f"{args.ns}/depth/depth_registered", "rinfo": f"{args.ns}/right/camera_info"}
    role = {v: k for k, v in topics.items()}

    rows = []                                   # per-frame stats
    pairs_zed, pairs_st = [], []                # all block pairs
    pos_sum = pos_n = None                      # per-block log-ratio accumulators
    fxB, video, n_synced = None, None, 0
    pending = {}                                # stamp -> {role: msg}

    with AnyReader([args.bag]) as reader:
        conns = [c for c in reader.connections if c.topic in role]
        missing = set(topics.values()) - {c.topic for c in conns}
        if missing:
            raise SystemExit(f"missing topics in bag: {sorted(missing)}")
        t0 = reader.start_time
        for conn, t, raw in reader.messages(connections=conns):
            rel = (t - t0) * 1e-9
            if rel < args.start:
                continue
            if rel > args.end:
                break
            r = role[conn.topic]
            if r == "rinfo":
                if fxB is None:
                    fxB = -reader.deserialize(raw, conn.msgtype).p[3]
                    print(f"fx*B = {fxB:.3f} px*m")
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            entry = pending.setdefault(stamp(msg), {})
            entry[r] = msg
            if len(entry) < 3:
                continue
            del pending[stamp(msg)]
            for s in [s for s in pending if s < stamp(msg)]:     # drop stale partial frames
                del pending[s]
            n_synced += 1
            if fxB is None or (n_synced - 1) % args.stride:
                continue

            left, right = to_gray(entry["left"]), to_gray(entry["right"])
            z_zed = to_depth(entry["depth"])
            z_st, valid, _ = stereo_depth(left, right, fxB, p)
            stats = block_compare(z_zed, z_st, p)
            ok = stats.valid
            if pos_sum is None:
                pos_sum, pos_n = np.zeros(ok.shape), np.zeros(ok.shape)
            lr = np.log(stats.z_zed / stats.z_stereo)
            pos_sum[ok] += lr[ok]
            pos_n[ok] += 1
            pairs_zed.append(stats.z_zed[ok])
            pairs_st.append(stats.z_stereo[ok])
            zed_ok = np.isfinite(z_zed) & (z_zed > 0)
            rows.append(dict(t=rel, brightness=float(left.mean()), stereo_coverage=float(valid.mean()),
                             zed_coverage=float(zed_ok.mean()), n_blocks=int(ok.sum()),
                             median_ratio=float(np.median(np.exp(lr[ok]))) if ok.any() else float("nan")))

            if not args.no_video:
                h, w = left.shape
                tiles = [
                    label(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), f"left  t={rel:.1f}s  mean={left.mean():.0f}"),
                    label(colorize_depth(z_st), f"SGBM depth  cov={valid.mean():.0%}"),
                    label(colorize_depth(z_zed), "ZED depth (0-10 m)"),
                    label(colorize_ratio(stats, (h, w)), f"ZED/stereo per block (+-{RATIO_VIS:.0%}, red=ZED longer)"),
                    label(cv2.cvtColor((valid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), "SGBM valid"),
                    label(scatter_tile(stats, fxB, (h, w)), f"blocks={ok.sum()}  median={rows[-1]['median_ratio']:.3f}"),
                ]
                panel = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
                if video is None:
                    video = cv2.VideoWriter(str(args.out / "panel.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            max(1.0, 15.0 / args.stride), (panel.shape[1], panel.shape[0]))
                video.write(panel)
            if len(rows) % 20 == 0:
                print(f"t={rel:6.1f}s  frames={len(rows)}  blocks={ok.sum():4d}  median ratio={rows[-1]['median_ratio']:.3f}")

    if video is not None:
        video.release()
    if not rows:
        raise SystemExit("no synced frames processed")
    summarize(args, rows, np.concatenate(pairs_zed), np.concatenate(pairs_st), pos_sum, pos_n, fxB)


def summarize(args, rows, zz, zs, pos_sum, pos_n, fxB):
    out = args.out
    ratio = zz / zs
    bins = []
    for lo, hi in RANGE_BINS:
        k = (zs >= lo) & (zs < hi)
        q = np.percentile(ratio[k], [25, 50, 75]) if k.sum() > 30 else [np.nan] * 3
        bins.append(dict(lo=lo, hi=hi, n=int(k.sum()), q25=float(q[0]), median=float(q[1]), q75=float(q[2]),
                         stereo_sigma_frac=float(stereo_sigma((lo + hi) / 2, fxB) / ((lo + hi) / 2))))

    k = (zs >= FIT_RANGE[0]) & (zs <= FIT_RANGE[1])
    idx = np.flatnonzero(k)
    if len(idx) > MAX_FIT_PAIRS:
        idx = np.random.default_rng(0).choice(idx, MAX_FIT_PAIRS, replace=False)
    fit = None
    if len(idx) > 100:
        cal = GroundCalibrator()
        cal.add_pairs(0.0, zz[idx], zs[idx])
        res = cal.fit()
        fit = dict(a=res.a, b=res.b, accepted=res.accepted, reason=res.reason, n_pairs=res.n_pixels,
                   inlier_share=res.inlier_share, ratio_before=res.ratio_before, ratio_after=res.ratio_after)

    summary = dict(bag=str(args.bag), ns=args.ns, fxB=fxB, block=args.block, stride=args.stride,
                   frames=len(rows), block_pairs=int(len(zz)), ratio_by_stereo_range=bins, fit_vs_stereo=fit)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "frames.json").write_text(json.dumps(rows))

    # 1. stereo vs ZED 2D histogram
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hist2d(zs, zz, bins=120, range=[[0, DEPTH_VIS_MAX], [0, DEPTH_VIS_MAX]], cmin=1,
                   norm=matplotlib.colors.LogNorm(), cmap="viridis")
    fig.colorbar(hb[3], ax=ax, label="blocks")
    x = np.linspace(0.3, DEPTH_VIS_MAX, 100)
    ax.plot(x, x, "k-", lw=1, label="y = x")
    ax.fill_between(x, x - stereo_sigma(x, fxB), x + stereo_sigma(x, fxB), color="k", alpha=0.12,
                    label="±1σ stereo (0.25 px)")
    if fit:
        ax.plot(x, fit["a"] / (1 / x - fit["b"]), "r--", lw=1,
                label=f"fit 1/z = a/z_zed + b (a={fit['a']:.3f}, b={fit['b']:+.3f})")
    ax.set(xlabel="stereo block depth (m)", ylabel="ZED block depth (m)", title="ZED vs SGBM (block medians)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=130)
    plt.close(fig)

    # 2. ratio vs range
    fig, ax = plt.subplots(figsize=(6, 4))
    good = [b for b in bins if np.isfinite(b["median"])]
    c = [(b["lo"] + b["hi"]) / 2 for b in good]
    ax.errorbar(c, [b["median"] for b in good],
                yerr=[[b["median"] - b["q25"] for b in good], [b["q75"] - b["median"] for b in good]],
                fmt="o-", capsize=4, label="median, IQR")
    ax.fill_between(c, [1 - b["stereo_sigma_frac"] for b in good], [1 + b["stereo_sigma_frac"] for b in good],
                    color="k", alpha=0.12, label="±1σ stereo noise (per pixel)")
    ax.axhline(1, color="k", lw=1)
    for b, cc in zip(good, c):
        ax.annotate(f"n={b['n']}", (cc, b["q25"]), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7)
    ax.set(xlabel="stereo depth (m)", ylabel="ZED / stereo", title="ZED depth ratio vs range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ratio_vs_range.png", dpi=130)
    plt.close(fig)

    # 3. ratio over image position
    fig, ax = plt.subplots(figsize=(7, 4.2))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.exp(pos_sum / pos_n)
    m[pos_n < 5] = np.nan
    im = ax.imshow(m, cmap="coolwarm", vmin=1 - RATIO_VIS, vmax=1 + RATIO_VIS)
    fig.colorbar(im, ax=ax, label="geo-mean ZED / stereo")
    ax.set(title=f"ratio by image block ({args.block}px)", xlabel="block col", ylabel="block row")
    fig.tight_layout()
    fig.savefig(out / "ratio_by_position.png", dpi=130)
    plt.close(fig)

    # 4. per-frame time series
    t = [r["t"] for r in rows]
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(t, [r["median_ratio"] for r in rows], ".-", ms=3)
    axs[0].axhline(1, color="k", lw=1)
    axs[0].set_ylabel("median ZED/stereo")
    axs[1].plot(t, [r["stereo_coverage"] for r in rows], label="SGBM valid")
    axs[1].plot(t, [r["zed_coverage"] for r in rows], label="ZED valid")
    axs[1].set_ylabel("pixel coverage")
    axs[1].legend(fontsize=8)
    axs[2].plot(t, [r["brightness"] for r in rows], color="tab:orange")
    axs[2].set(ylabel="mean grey (left)", xlabel="t (s)")
    fig.tight_layout()
    fig.savefig(out / "timeseries.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in ("frames", "block_pairs", "ratio_by_stereo_range", "fit_vs_stereo")},
                     indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
