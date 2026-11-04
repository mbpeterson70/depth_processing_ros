"""ZED depth quality against lidar, for the status monitor (numpy only, no ROS dependencies).

Keeps a sliding window of (z_zed raw, z_lidar) pairs (from lidar_calibration.depth_pairs) and
reports one number, the median relative error of the corrected depth z_zed / (a + b z_zed)
against the lidar, plus a flag when the far range (where a bad correction shows first) is biased.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

NOMINAL, WARNING, ERROR = "nominal", "warning", "error"


@dataclass
class DepthQualityParams:
    window_s: float = 60.0             # pairs older than this are dropped
    min_pairs: int = 500               # fewer in the window: not enough to judge
    far_m: float = 4.0                 # far range: lidar depth from here up to the calibration max_z
    # lidar calibration gives 1-3% median error, ground plane 3-9%, raw ZED 13-23% (Spot bags)
    warn_err: float = 0.08             # median error above this: WARNING
    error_err: float = 0.15            # median error above this: ERROR
    far_bias_tol: float = 0.12         # far median ZED/lidar off by more than this: WARNING


class DepthQualityMonitor:
    def __init__(self, params: Optional[DepthQualityParams] = None):
        self.params = params or DepthQualityParams()
        self.frames: List[Tuple[float, np.ndarray, np.ndarray]] = []   # (stamp, z_zed, z_lidar)

    def add_pairs(self, stamp: float, z_zed: np.ndarray, z_lidar: np.ndarray):
        if len(z_zed):
            self.frames.append((stamp, np.asarray(z_zed, np.float64), np.asarray(z_lidar, np.float64)))
        self.frames = [f for f in self.frames if f[0] >= stamp - self.params.window_s]

    def report(self, a: float, b: float) -> Tuple[Optional[str], str]:
        """(status, note) for the correction (a, b); status is None with too few pairs."""
        p = self.params
        n = sum(len(f[1]) for f in self.frames)
        if n < p.min_pairs:
            return None, "measuring against lidar"
        zz = np.concatenate([f[1] for f in self.frames])
        zl = np.concatenate([f[2] for f in self.frames])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = zz / (a + b * zz) / zl
        ok = np.isfinite(ratio) & (ratio > 0)
        ratio, zl = ratio[ok], zl[ok]
        err = float(np.median(np.abs(ratio - 1)))
        note = f"depth error {err:.1%} (vs lidar)"
        far = ratio[zl >= p.far_m]
        far_bias = float(np.median(far)) - 1 if len(far) >= p.min_pairs // 5 else 0.0
        if abs(far_bias) > p.far_bias_tol:
            note += f"; far range reads {abs(far_bias):.0%} {'short' if far_bias < 0 else 'long'}"
        if err > p.error_err:
            return ERROR, note
        return (WARNING if err > p.warn_err or abs(far_bias) > p.far_bias_tol else NOMINAL), note
