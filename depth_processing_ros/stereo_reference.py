"""Classical stereo (SGBM) depth as an independent reference for the ZED depth (no ROS dependencies).

The ZED publishes its own rectified left/right pair, so block matching on it gives a purely
geometric depth, z = fx * B / d, with no learned prior. Its error grows as z^2 (a fixed
sub-pixel disparity error), so it is a trustworthy reference only at short to medium range;
`stereo_sigma` gives the expected noise. Block-averaging both depths removes most of the
noise but not a bias, which is what we want to compare.

The disparity scale/offset that stereo exposes is exactly the model the ground calibration
fits (1/z_true = a/z_zed + b), so the block pairs plug straight into GroundCalibrator.fit.
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

SGBM_DISPARITY_SCALE = 16.0     # OpenCV SGBM returns fixed-point disparity * 16


@dataclass
class StereoParams:
    num_disparities: int = 64          # px, multiple of 16; z_min = fxB / num_disparities
    block_size: int = 5
    uniqueness: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    lr_max_diff_px: float = 1.0        # left-right consistency tolerance
    min_disparity_px: float = 2.0      # reject far, poorly-constrained matches
    min_texture: float = 4.0           # mean |d/dx| (grey levels) over the match window
    block: int = 16                    # px, block size for block_compare
    min_block_share: float = 0.3       # share of a block's pixels valid in both depths


@dataclass
class BlockStats:
    z_zed: np.ndarray                  # (by, bx) median ZED depth per block, NaN if dropped
    z_stereo: np.ndarray               # (by, bx) median stereo depth per block, NaN if dropped
    share: np.ndarray                  # (by, bx) share of pixels valid in both
    block: int

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z_zed) & np.isfinite(self.z_stereo)


def _sgbm(p: StereoParams) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=p.num_disparities, blockSize=p.block_size,
        P1=8 * p.block_size ** 2, P2=32 * p.block_size ** 2, disp12MaxDiff=p.disp12_max_diff,
        uniquenessRatio=p.uniqueness, speckleWindowSize=p.speckle_window,
        speckleRange=p.speckle_range, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def texture(gray: np.ndarray, window: int) -> np.ndarray:
    """Mean absolute horizontal gradient over a window: low in dark or featureless regions."""
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) / 4.0
    return cv2.blur(gx, (window, window))


def stereo_depth(left_gray: np.ndarray, right_gray: np.ndarray, fxB: float,
                 p: StereoParams = StereoParams()):
    """(z_stereo, valid, disparity) from a rectified 8-bit grey pair; fxB in px*m.

    Invalid pixels are NaN in z_stereo."""
    matcher = _sgbm(p)
    disp = matcher.compute(left_gray, right_gray).astype(np.float32) / SGBM_DISPARITY_SCALE
    # right-view disparity: match the mirrored pair, then mirror back
    disp_r = matcher.compute(np.ascontiguousarray(right_gray[:, ::-1]),
                             np.ascontiguousarray(left_gray[:, ::-1]))
    disp_r = disp_r[:, ::-1].astype(np.float32) / SGBM_DISPARITY_SCALE

    h, w = disp.shape
    u = np.arange(w)[None, :].repeat(h, axis=0)
    ur = np.clip(np.round(u - disp).astype(int), 0, w - 1)
    lr_ok = np.abs(disp - np.take_along_axis(disp_r, ur, axis=1)) <= p.lr_max_diff_px

    valid = (disp >= p.min_disparity_px) & lr_ok & (texture(left_gray, p.block_size) >= p.min_texture)
    z = np.full(disp.shape, np.nan, dtype=np.float32)
    z[valid] = fxB / disp[valid]
    return z, valid, disp


def stereo_sigma(z, fxB: float, sigma_d: float = 0.25):
    """Expected stereo depth noise (m) for a disparity noise of sigma_d px."""
    return np.asarray(z) ** 2 / fxB * sigma_d


def _blocks(img: np.ndarray, block: int) -> np.ndarray:
    """(by, bx, block*block) view of img, NaN-padded to whole blocks."""
    h, w = img.shape
    by, bx = -(-h // block), -(-w // block)
    pad = np.full((by * block, bx * block), np.nan, dtype=np.float32)
    pad[:h, :w] = img
    return pad.reshape(by, block, bx, block).transpose(0, 2, 1, 3).reshape(by, bx, -1)


def block_compare(z_zed: np.ndarray, z_stereo: np.ndarray, p: StereoParams = StereoParams()) -> BlockStats:
    """Per-block medians of both depths over pixels valid in both."""
    both = np.isfinite(z_zed) & (z_zed > 0) & np.isfinite(z_stereo)
    zz = _blocks(np.where(both, z_zed, np.nan), p.block)
    zs = _blocks(np.where(both, z_stereo, np.nan), p.block)
    share = np.mean(np.isfinite(zz), axis=-1)
    keep = share >= p.min_block_share
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks
        mz, ms = np.nanmedian(zz, axis=-1), np.nanmedian(zs, axis=-1)
    mz[~keep] = np.nan
    ms[~keep] = np.nan
    return BlockStats(mz, ms, share, p.block)
