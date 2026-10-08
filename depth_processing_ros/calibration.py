"""Online ZED depth-bias calibration from the ground plane (no ROS dependencies).

Stereo depth from the ZED reads progressively short with range (a small disparity offset,
which changes between sessions). With a known ground plane in the camera frame (Spot's
ground-plane estimate plus the camera mount), the true depth of every pixel whose ray
hits the ground follows from geometry:

    z_true = (n . p) / (n . d),  n: ground normal, p: a ground point (camera frame),
                                 d = ((u - cx) / fx, (v - cy) / fy, 1)

Frames are accumulated and the model

    1 / z_true = a / z_zed + b      ->      corrected z = z_zed / (a + b * z_zed)

is fitted robustly. Frames whose ground pixels mostly don't look like ground (the ground
in view isn't flat, e.g. it slopes away) are rejected, and a fit is only accepted if it
passes sanity checks, so a bad start doesn't produce a bad correction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# GroundCalibrator.fit
MIN_FIT_PIXELS = 100            # stop refitting below this many inlier pixels
OUTER_ITERATIONS = 6            # fit -> re-select inliers -> refit
IRLS_ITERATIONS = 8             # reweighting passes per fit
MAD_TO_STD = 1.4826             # median absolute deviation -> standard deviation (Gaussian)
HUBER_K = 1.5                   # residuals beyond HUBER_K robust sigmas are down-weighted
EPS = 1e-9                      # avoids division by zero
REPORT_BINS = ((1, 2), (2, 3), (3, 5), (5, 8))   # true-depth bins (m) for the before/after report
MIN_BIN_PIXELS = 30             # a bin needs this many inliers to be reported


@dataclass
class CalibrationParams:
    min_z: float = 1.0                 # m, ground pixels used for the fit
    max_z: float = 8.0
    pixel_step: int = 4                # use every n-th pixel
    min_frame_ground_share: float = 0.5   # share of ground-ray pixels whose raw ZED/true ratio is
    plausible_ratio: Tuple[float, float] = (0.6, 1.15)   # plausible, for a frame to be used
    min_frame_pixels: int = 500        # ground-ray pixels a frame needs
    a_bounds: Tuple[float, float] = (0.7, 1.2)
    b_bounds: Tuple[float, float] = (-0.1, 0.1)
    min_inlier_share: float = 0.6      # share of the window's ground pixels consistent with the fit
    inlier_tol_m: float = 0.15         # |corrected - true| < max(tol_m, tol_frac * true)
    inlier_tol_frac: float = 0.08


@dataclass
class CalibrationResult:
    a: float
    b: float
    accepted: bool
    reason: str
    n_frames: int
    n_pixels: int
    inlier_share: float
    ratio_before: List[Tuple[float, float, float]]   # (lo, hi, median ZED/true)
    ratio_after: List[Tuple[float, float, float]]


def correct_depth(depth: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply z = z_zed / (a + b z_zed) to a float depth image (m); invalid pixels stay invalid."""
    out = depth.astype(np.float32, copy=True)
    valid = np.isfinite(out) & (out > 0)
    den = a + b * out[valid]
    good = den > 1e-3
    vals = np.full(den.shape, np.nan, dtype=np.float32)
    vals[good] = out[valid][good] / den[good]
    out[valid] = vals
    return out


def ground_ray_depths(depth: np.ndarray, K: np.ndarray, normal: np.ndarray, point: np.ndarray,
                      params: CalibrationParams):
    """(z_zed, z_true) for sampled pixels whose ray hits the ground within [min_z, max_z].

    normal, point: ground plane in the camera (optical) frame."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:params.pixel_step, 0:w:params.pixel_step]
    z = depth[v, u].astype(np.float64)
    dx = (u - K[0, 2]) / K[0, 0]
    dy = (v - K[1, 2]) / K[1, 1]
    nd = normal[0] * dx + normal[1] * dy + normal[2]
    num = float(normal @ point)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_true = num / nd
    ok = (np.isfinite(z) & (z > 0) & np.isfinite(z_true)
          & (z_true >= params.min_z) & (z_true <= params.max_z))
    return z[ok], z_true[ok]


class GroundCalibrator:
    """Accumulates ground pixels from frames and fits the depth correction."""

    def __init__(self, params: Optional[CalibrationParams] = None):
        self.params = params or CalibrationParams()
        self.frames: List[Tuple[float, np.ndarray, np.ndarray]] = []   # (stamp, z_zed, z_true)

    def add_frame(self, stamp: float, depth: np.ndarray, K: np.ndarray, normal: np.ndarray,
                  point: np.ndarray) -> Tuple[bool, str]:
        """Add a frame if its ground looks usable. Returns (used, reason)."""
        p = self.params
        zz, zt = ground_ray_depths(depth, K, normal, point, p)
        if len(zz) < p.min_frame_pixels:
            return False, f"only {len(zz)} ground-ray pixels"
        r = zz / zt
        share = np.mean((r > p.plausible_ratio[0]) & (r < p.plausible_ratio[1]))
        if share < p.min_frame_ground_share:
            return False, f"ground not flat or not visible ({share:.0%} plausible ground pixels)"
        self.frames.append((stamp, zz, zt))
        return True, "ok"

    def drop_before(self, stamp: float):
        self.frames = [f for f in self.frames if f[0] >= stamp]

    @property
    def span(self) -> float:
        return self.frames[-1][0] - self.frames[0][0] if len(self.frames) > 1 else 0.0

    def fit(self) -> CalibrationResult:
        p = self.params
        zz = np.concatenate([f[1] for f in self.frames])
        zt = np.concatenate([f[2] for f in self.frames])
        r0 = zz / zt
        keep = (r0 > p.plausible_ratio[0]) & (r0 < p.plausible_ratio[1])
        a, b = 1.0, 0.0
        for _ in range(OUTER_ITERATIONS):
            if keep.sum() < MIN_FIT_PIXELS:
                break
            X = np.column_stack([1.0 / zz[keep], np.ones(keep.sum())])
            y = 1.0 / zt[keep]
            wts = np.ones(len(y))
            for _ in range(IRLS_ITERATIONS):    # IRLS, Huber-like weights on the 1/z residual
                sw = np.sqrt(wts)
                coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
                res = np.abs(X @ coef - y)
                s = MAD_TO_STD * np.median(res) + EPS
                wts = np.minimum(1.0, HUBER_K * s / np.maximum(res, EPS))
            a, b = float(coef[0]), float(coef[1])
            corr = zz / (a + b * zz)
            keep = np.abs(corr - zt) < np.maximum(p.inlier_tol_m, p.inlier_tol_frac * zt)
        inlier_share = float(np.mean(keep)) if len(keep) else 0.0
        corr = zz / (a + b * zz)
        def ratios(vals):
            out = []
            for lo, hi in REPORT_BINS:
                k = keep & (zt >= lo) & (zt < hi)
                out.append((lo, hi, float(np.median(vals[k] / zt[k])) if k.sum() > MIN_BIN_PIXELS
                            else float("nan")))
            return out

        accepted, reason = True, "ok"
        if not (p.a_bounds[0] <= a <= p.a_bounds[1]):
            accepted, reason = False, f"a = {a:.3f} outside {p.a_bounds}"
        elif not (p.b_bounds[0] <= b <= p.b_bounds[1]):
            accepted, reason = False, f"b = {b:+.3f} outside {p.b_bounds}"
        elif inlier_share < p.min_inlier_share:
            accepted, reason = False, f"only {inlier_share:.0%} of ground pixels fit"
        return CalibrationResult(a, b, accepted, reason, len(self.frames), int(len(zz)), inlier_share,
                                 ratios(zz), ratios(corr))
