"""Online ZED depth-bias calibration from lidar (numpy only, no ROS dependencies).

Lidar points, moved into the camera frame at the depth image's stamp, are projected into the
depth image; each one that the camera can see and that lands on valid, non-edge depth gives a
(z_zed, z_lidar) pair. Pairs from a window of frames are fitted with the same model as the
ground calibration,

    1 / z_true = a / z_zed + b      ->      corrected z = z_zed / (a + b * z_zed),

robustly on the relative error (corrected / lidar - 1), so near and far points count alike.
Unlike the ground, lidar covers the whole range the ZED reports, which is what pins down b.
Offline evaluation and figures: scripts/lidar_vs_zed.py, scripts/explain_depth_calibration.py.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from depth_processing_ros.calibration import (EPS, HUBER_K, IRLS_ITERATIONS, MAD_TO_STD, MIN_BIN_PIXELS,
                                              OUTER_ITERATIONS, REPORT_BINS, CalibrationResult)


@dataclass
class LidarCalibrationParams:
    min_z: float = 0.5                 # m, lidar depth (camera frame) used for the fit
    max_z: float = 9.5                 # m, keep below the ZED's max depth (10 m by default)
    min_frame_pairs: int = 100         # lidar/ZED pairs a frame needs to be used
    max_frame_pairs: int = 2000        # pairs kept per frame (random subsample)
    visible_cell_px: int = 3           # z-buffer cell for dropping lidar points the camera can't see
    edge_window_px: int = 5            # ZED depth near a jump of more than edge_max_ratio is not used
    edge_max_ratio: float = 1.08
    plausible_ratio: Tuple[float, float] = (0.5, 1.5)   # raw ZED/lidar kept for the first fit
    a_bounds: Tuple[float, float] = (0.7, 1.2)
    b_bounds: Tuple[float, float] = (-0.1, 0.1)
    min_inlier_share: float = 0.5      # share of pairs consistent with the fit
    inlier_tol_frac: float = 0.08      # |corrected / lidar - 1| below this is an inlier


# ---------------------------------------------------------------------------------- geometry

def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def project(p_cam: np.ndarray, K: np.ndarray):
    z = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * p_cam[:, 1] / z + K[1, 2]
    return u, v, z


def visible_mask(u, v, z, shape, cell: int = 3, tol: float = 0.05) -> np.ndarray:
    """In-image points that are the nearest in their cell (within tol): the lidar sees around
    things the camera can't, so points behind a nearer surface are dropped."""
    h, w = shape
    inside = (z > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    keep = np.zeros(len(z), bool)
    idx = np.flatnonzero(inside)
    if not len(idx):
        return keep
    cid = (v[idx] // cell).astype(np.int64) * (w // cell + 1) + (u[idx] // cell).astype(np.int64)
    zmin = np.full(cid.max() + 1, np.inf)
    np.minimum.at(zmin, cid, z[idx])
    keep[idx] = z[idx] <= zmin[cid] * (1 + tol)
    return keep


def mask_depth_edges(depth: np.ndarray, window: int = 5, max_ratio: float = 1.08) -> np.ndarray:
    """Copy of depth with invalid pixels and pixels near depth discontinuities set to NaN, so a
    lidar point that lands a pixel off an edge is not compared with the wrong surface."""
    d = depth.astype(np.float32, copy=True)
    valid = np.isfinite(d) & (d > 0)
    r = window // 2
    win = lambda img, fill: np.lib.stride_tricks.sliding_window_view(
        np.pad(img, r, constant_values=fill), (window, window))
    hi = win(np.where(valid, d, 0.0), 0.0).max(axis=(-2, -1))
    lo = win(np.where(valid, d, np.inf), np.inf).min(axis=(-2, -1))
    all_valid = win(valid, False).all(axis=(-2, -1))
    d[~(all_valid & (hi <= lo * max_ratio))] = np.nan
    return d


def depth_pairs(depth: np.ndarray, K: np.ndarray, p_cam: np.ndarray, p: LidarCalibrationParams,
                rng: Optional[np.random.Generator] = None):
    """(z_zed, z_lidar) for lidar points (camera frame) the camera sees on valid, non-edge depth."""
    u, v, z = project(p_cam, K)
    vis = visible_mask(u, v, z, depth.shape, p.visible_cell_px)
    vis &= (z >= p.min_z) & (z <= p.max_z)
    d = mask_depth_edges(depth, p.edge_window_px, p.edge_max_ratio)
    zz = d[np.round(v[vis]).astype(int), np.round(u[vis]).astype(int)]
    zl = z[vis]
    ok = np.isfinite(zz) & (zz > 0)
    zz, zl = zz[ok], zl[ok]
    if len(zz) > p.max_frame_pairs:
        sel = (rng or np.random.default_rng()).choice(len(zz), p.max_frame_pairs, replace=False)
        zz, zl = zz[sel], zl[sel]
    return zz.astype(np.float64), zl.astype(np.float64)


# ---------------------------------------------------------------------------------- calibrator

class LidarCalibrator:
    """Accumulates (z_zed, z_lidar) pairs from frames and fits the depth correction."""

    def __init__(self, params: Optional[LidarCalibrationParams] = None):
        self.params = params or LidarCalibrationParams()
        self.frames: List[Tuple[float, np.ndarray, np.ndarray]] = []   # (stamp, z_zed, z_lidar)

    def add_pairs(self, stamp: float, z_zed: np.ndarray, z_lidar: np.ndarray) -> Tuple[bool, str]:
        if len(z_zed) < self.params.min_frame_pairs:
            return False, f"only {len(z_zed)} lidar/ZED pairs"
        self.frames.append((stamp, np.asarray(z_zed, np.float64), np.asarray(z_lidar, np.float64)))
        return True, "ok"

    def drop_before(self, stamp: float):
        self.frames = [f for f in self.frames if f[0] >= stamp]

    @property
    def span(self) -> float:
        return self.frames[-1][0] - self.frames[0][0] if len(self.frames) > 1 else 0.0

    def fit(self) -> CalibrationResult:
        p = self.params
        zz = np.concatenate([f[1] for f in self.frames])
        zl = np.concatenate([f[2] for f in self.frames])
        r0 = zz / zl
        keep = (r0 > p.plausible_ratio[0]) & (r0 < p.plausible_ratio[1])
        a, b = 1.0, 0.0
        for _ in range(OUTER_ITERATIONS):
            if keep.sum() < p.min_frame_pairs:
                break
            # relative error of the corrected depth, linear in (a, b):
            #   z_lidar / z_corrected - 1 = (a / z_zed + b) * z_lidar - 1
            X = np.column_stack([zl[keep] / zz[keep], zl[keep]])
            y = np.ones(keep.sum())
            wts = np.ones(len(y))
            for _ in range(IRLS_ITERATIONS):
                sw = np.sqrt(wts)
                coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
                res = np.abs(X @ coef - y)
                s = MAD_TO_STD * np.median(res) + EPS
                wts = np.minimum(1.0, HUBER_K * s / np.maximum(res, EPS))
            a, b = float(coef[0]), float(coef[1])
            keep = np.abs(zz / (a + b * zz) / zl - 1) < p.inlier_tol_frac
        inlier_share = float(np.mean(keep)) if len(keep) else 0.0
        corr = zz / (a + b * zz)

        def ratios(vals):
            out = []
            for lo, hi in REPORT_BINS:
                k = keep & (zl >= lo) & (zl < hi)
                out.append((lo, hi, float(np.median(vals[k] / zl[k])) if k.sum() > MIN_BIN_PIXELS
                            else float("nan")))
            return out

        accepted, reason = True, "ok"
        if not (p.a_bounds[0] <= a <= p.a_bounds[1]):
            accepted, reason = False, f"a = {a:.3f} outside {p.a_bounds}"
        elif not (p.b_bounds[0] <= b <= p.b_bounds[1]):
            accepted, reason = False, f"b = {b:+.3f} outside {p.b_bounds}"
        elif inlier_share < p.min_inlier_share:
            accepted, reason = False, f"only {inlier_share:.0%} of lidar pairs fit"
        return CalibrationResult(a, b, accepted, reason, len(self.frames), int(len(zz)), inlier_share,
                                 ratios(zz), ratios(corr))
