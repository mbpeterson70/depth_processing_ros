import cv2
import numpy as np

from depth_processing_ros.calibration import GroundCalibrator
from depth_processing_ros.stereo_reference import (StereoParams, block_compare, stereo_depth,
                                                   stereo_sigma)

FXB = 31.34          # px*m, ZED 2i at 640x360 (fx 261.3, B 0.12 m)
H, W = 360, 640


def textured(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (H // 4, W // 4 + 32), dtype=np.uint8)
    return cv2.resize(img, (W + 128, H), interpolation=cv2.INTER_CUBIC)


def pair(disparity_px: int):
    """Rectified pair of a fronto-parallel textured plane at disparity_px."""
    big = textured()
    left = big[:, 64:64 + W].copy()
    right = big[:, 64 + disparity_px:64 + disparity_px + W].copy()
    return left, right


def test_stereo_recovers_plane_depth():
    d = 8
    left, right = pair(d)
    z, valid, _ = stereo_depth(left, right, FXB)
    assert valid.mean() > 0.7
    assert abs(np.nanmedian(z) - FXB / d) < 0.02 * FXB / d


def test_flat_image_is_invalid():
    flat = np.full((H, W), 40, dtype=np.uint8)
    _, valid, _ = stereo_depth(flat, flat, FXB)
    assert valid.mean() < 0.01


def test_block_compare_ratio():
    rng = np.random.default_rng(1)
    zs = rng.uniform(1, 8, (H, W)).astype(np.float32)
    zs[:50] = np.nan                                   # no stereo at the top
    stats = block_compare(0.85 * zs, zs, StereoParams(block=16))
    r = stats.z_zed[stats.valid] / stats.z_stereo[stats.valid]
    assert np.allclose(r, 0.85, atol=0.02)
    assert not stats.valid[0].any()


def test_block_pairs_fit_recovers_bias():
    zt = np.random.default_rng(2).uniform(1, 8, 2000)
    a, b = 0.92, -0.025
    cal = GroundCalibrator()
    cal.add_pairs(0.0, a / (1.0 / zt - b), zt)
    res = cal.fit()
    assert res.accepted, res.reason
    assert abs(res.a - a) < 1e-3 and abs(res.b - b) < 1e-3


def test_sigma_grows_quadratically():
    assert np.isclose(stereo_sigma(4.0, FXB) / stereo_sigma(2.0, FXB), 4.0)
