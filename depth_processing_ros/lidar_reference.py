"""Lidar as an absolute reference for the ZED depth (no ROS dependencies).

Lidar points (already deskewed to the depth image's stamp, in the lidar frame) are projected
into the ZED depth image through T_cam_lidar and compared with the ZED depth at that pixel.
The depth correction is fitted robustly in log-depth, optionally together with a small
correction to T_cam_lidar: the mount is hand-measured, so its error would otherwise leak
into the depth fit. A prior keeps that correction small; the depth model stays the thing
we are after.

Depth models (all keep invalid pixels invalid):
    identity:  z = z_zed
    scale:     z = s * z_zed
    inverse:   1/z = a/z_zed + b      (same model as calibration.GroundCalibrator)
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from depth_processing_ros.lidar_calibration import (make_T, mask_depth_edges, project,  # noqa: F401
                                                    transform, visible_mask)

INVALID_RESIDUAL = 0.3          # log-depth residual for points that leave valid depth (outlier level)


@dataclass
class LidarFitParams:
    min_z: float = 0.5                  # m, lidar depth in camera frame
    max_z: float = 20.0
    loss_scale: float = 0.03            # soft-L1 transition, log-depth (~3%)
    prior_rot_deg: float = 1.0          # extrinsic-correction prior (1 sigma)
    prior_trans_m: float = 0.02
    max_points: int = 300_000           # subsample for the fit


@dataclass
class FitResult:
    model: str
    depth_params: Dict[str, float]
    delta_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_trans: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cost: float = float("nan")
    n_points: int = 0


# ---------------------------------------------------------------------------------- geometry

def perturb(T: np.ndarray, rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Left-multiply T by a small correction (applied in the camera frame)."""
    return make_T(Rotation.from_rotvec(rotvec).as_matrix(), trans) @ T


def sample_bilinear(depths: np.ndarray, f: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """depths: (F, H, W); NaN where any neighbour is invalid or outside the image."""
    _, h, w = depths.shape
    out = np.full(len(u), np.nan, np.float32)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    u0, v0 = np.floor(u[ok]).astype(np.int64), np.floor(v[ok]).astype(np.int64)
    du, dv = (u[ok] - u0).astype(np.float32), (v[ok] - v0).astype(np.float32)
    ff = f[ok]
    out[ok] = ((1 - du) * (1 - dv) * depths[ff, v0, u0] + du * (1 - dv) * depths[ff, v0, u0 + 1]
               + (1 - du) * dv * depths[ff, v0 + 1, u0] + du * dv * depths[ff, v0 + 1, u0 + 1])
    return out


# ---------------------------------------------------------------------------------- depth models

DEPTH_MODELS = {"identity": [], "scale": ["s"], "inverse": ["a", "b"]}
DEPTH_INIT = {"s": 1.0, "a": 1.0, "b": 0.0}


def apply_model(model: str, params: Dict[str, float], z_zed: np.ndarray) -> np.ndarray:
    if model == "identity":
        return z_zed
    if model == "scale":
        return params["s"] * z_zed
    if model == "inverse":
        with np.errstate(divide="ignore", invalid="ignore"):
            return z_zed / (params["a"] + params["b"] * z_zed)
    raise ValueError(model)


# ---------------------------------------------------------------------------------- data + fit

@dataclass
class LidarZedData:
    """Frames of masked ZED depth plus lidar points (lidar frame, deskewed to the depth stamp)."""
    depths: np.ndarray                  # (F, H, W) float32, NaN = invalid / edge
    points: np.ndarray                  # (N, 3) lidar frame
    frame: np.ndarray                   # (N,) frame index of each point
    K: np.ndarray
    T_cam_lidar: np.ndarray

    def subset(self, frames: np.ndarray) -> "LidarZedData":
        sel = np.isin(self.frame, frames)
        remap = np.full(len(self.depths), -1)
        remap[frames] = np.arange(len(frames))
        return LidarZedData(self.depths[frames], self.points[sel], remap[self.frame[sel]], self.K,
                            self.T_cam_lidar)

    def pairs(self, rotvec=np.zeros(3), trans=np.zeros(3)):
        """(z_zed, z_lidar, frame, valid) with T_cam_lidar corrected by (rotvec, trans)."""
        p = transform(perturb(self.T_cam_lidar, rotvec, trans), self.points)
        u, v, z = project(p, self.K)
        zz = sample_bilinear(self.depths, self.frame, u, v)
        return zz, z, self.frame, np.isfinite(zz) & (zz > 0) & np.isfinite(z) & (z > 0)


def log_residuals(model, params, zz, zl, valid, p: LidarFitParams):
    r = np.full(len(zz), INVALID_RESIDUAL)
    rng = valid & (zl >= p.min_z) & (zl <= p.max_z)
    zc = apply_model(model, params, zz[rng])
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.log(zc / zl[rng])
    rr[~np.isfinite(rr)] = INVALID_RESIDUAL
    r[rng] = rr
    return r


def fit(data: LidarZedData, model: str, refine_extrinsic: bool = False,
        p: LidarFitParams = LidarFitParams(), seed: int = 0) -> FitResult:
    names = DEPTH_MODELS[model]
    zz, zl, fr, valid = data.pairs()
    keep = np.flatnonzero(valid & (zl >= p.min_z) & (zl <= p.max_z))
    if len(keep) > p.max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, p.max_points, replace=False))
    sub = LidarZedData(data.depths, data.points[keep], data.frame[keep], data.K, data.T_cam_lidar)
    n_frames = len(np.unique(sub.frame))
    # residuals are in units of loss_scale; the per-point residuals within a frame are strongly
    # correlated (same mount error, same surfaces), so the data is counted as n_frames
    # observations rather than len(keep): the prior is scaled up by sqrt(len(keep) / n_frames)
    prior_w = p.loss_scale * np.sqrt(len(keep) / max(n_frames, 1))
    n_ext = 6 if refine_extrinsic else 0

    def unpack(x):
        params = dict(zip(names, x[:len(names)]))
        if refine_extrinsic:
            return params, x[len(names):len(names) + 3], x[len(names) + 3:]
        return params, np.zeros(3), np.zeros(3)

    if refine_extrinsic:
        def residuals(x):
            params, rv, t = unpack(x)
            zz_, zl_, _, ok = sub.pairs(rv, t)
            prior = np.concatenate([rv / np.radians(p.prior_rot_deg), t / p.prior_trans_m]) * prior_w
            return np.concatenate([log_residuals(model, params, zz_, zl_, ok, p), prior])
    else:
        zz0, zl0, _, ok0 = sub.pairs()

        def residuals(x):
            return log_residuals(model, unpack(x)[0], zz0, zl0, ok0, p)

    x0 = np.array([DEPTH_INIT[n] for n in names] + [0.0] * n_ext)
    if len(x0) == 0:
        r = residuals(x0)
        cost = float(np.sum(p.loss_scale ** 2 * 2 * (np.sqrt(1 + (r / p.loss_scale) ** 2) - 1)))
        return FitResult(model, {}, cost=cost, n_points=len(keep))
    res = least_squares(residuals, x0, loss="soft_l1", f_scale=p.loss_scale,
                        diff_step=np.full(len(x0), 1e-4), x_scale="jac")
    params, rv, t = unpack(res.x)
    return FitResult(model, {k: float(v) for k, v in params.items()}, rv, t, float(res.cost), len(keep))


def evaluate(data: LidarZedData, result: FitResult, p: LidarFitParams = LidarFitParams(),
             use_extrinsic: bool = True, bins=((0.5, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20))):
    """Relative-error metrics of the corrected ZED depth against lidar."""
    rv, t = (result.delta_rotvec, result.delta_trans) if use_extrinsic else (np.zeros(3), np.zeros(3))
    zz, zl, _, valid = data.pairs(rv, t)
    k = valid & (zl >= p.min_z) & (zl <= p.max_z)
    zc = apply_model(result.model, result.depth_params, zz[k])
    zl = zl[k]
    lr = np.log(zc / zl)
    alr = np.abs(lr)
    out = dict(n=int(k.sum()), median_abs_log=float(np.median(alr)),
               within_5pct=float(np.mean(alr < np.log(1.05))), within_10pct=float(np.mean(alr < np.log(1.10))),
               median_ratio=float(np.exp(np.median(lr))), bins=[])
    for lo, hi in bins:
        b = (zl >= lo) & (zl < hi)
        if b.sum() < 50:
            out["bins"].append(dict(lo=lo, hi=hi, n=int(b.sum())))
            continue
        q = np.exp(np.percentile(lr[b], [25, 50, 75]))
        out["bins"].append(dict(lo=lo, hi=hi, n=int(b.sum()), q25=float(q[0]), median_ratio=float(q[1]),
                                q75=float(q[2]), median_abs_log=float(np.median(alr[b]))))
    return out
