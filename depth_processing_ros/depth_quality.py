"""ZED depth quality against lidar, for the status monitor (numpy only, no ROS dependencies).

One number: the median relative error of the corrected depth against lidar, from (z_zed raw,
z_lidar) pairs (lidar_calibration.depth_pairs), plus a flag when the far range, where a bad
correction shows first, is biased.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from depth_processing_ros.calibration import correct_depth

NOMINAL, WARNING, ERROR = "nominal", "warning", "error"


@dataclass
class DepthQualityParams:
    window_s: float = 60.0             # pairs from this many seconds are judged together
    min_pairs: int = 500               # fewer: not enough to judge
    far_m: float = 4.0                 # far range: lidar depth from here up to the calibration max_z
    # lidar calibration gives 1-3% median error, ground plane 3-9%, raw ZED 13-23% (Spot bags)
    warn_err: float = 0.08             # median error above this: WARNING
    error_err: float = 0.15            # median error above this: ERROR
    far_bias_tol: float = 0.12         # far median ZED/lidar off by more than this: WARNING


def depth_quality(z_zed: np.ndarray, z_lidar: np.ndarray, a: float, b: float,
                  p: DepthQualityParams = DepthQualityParams()) -> Tuple[Optional[str], str]:
    """(status, note) of the correction (a, b) on these pairs; status is None with too few pairs."""
    if len(z_zed) < p.min_pairs:
        return None, "measuring against lidar"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = correct_depth(z_zed, a, b) / z_lidar
    ok = np.isfinite(ratio) & (ratio > 0)
    ratio, z_lidar = ratio[ok], z_lidar[ok]
    err = float(np.median(np.abs(ratio - 1)))
    note = f"depth error {err:.1%} (vs lidar)"
    far = ratio[z_lidar >= p.far_m]
    far_bias = float(np.median(far)) - 1 if len(far) >= p.min_pairs // 5 else 0.0
    if abs(far_bias) > p.far_bias_tol:
        note += f"; far range reads {abs(far_bias):.0%} {'short' if far_bias < 0 else 'long'}"
    if err > p.error_err:
        return ERROR, note
    return (WARNING if err > p.warn_err or abs(far_bias) > p.far_bias_tol else NOMINAL), note
