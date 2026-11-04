import numpy as np

from depth_processing_ros.lidar_calibration import LidarCalibrator, depth_pairs, mask_depth_edges

K = np.array([[261.5, 0.0, 312.4], [0.0, 261.5, 179.9], [0.0, 0.0, 1.0]])
H, W = 360, 640


def biased(z, a, b):                                    # inverse of 1/z = a/z_zed + b
    return a / (1.0 / z - b)


def scene():
    """True depth: a wall at 8 m with a box at 3 m in front of it."""
    z = np.full((H, W), 8.0, np.float32)
    z[100:250, 200:400] = 3.0
    return z


def lidar_points(z_true, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, W - 1, n), rng.uniform(0, H - 1, n)
    zt = z_true[v.astype(int), u.astype(int)]
    return np.stack([(u - K[0, 2]) / K[0, 0] * zt, (v - K[1, 2]) / K[1, 1] * zt, zt], axis=1)


def test_mask_depth_edges_drops_edges_and_invalid():
    d = scene()
    d[0, 0] = np.nan
    m = mask_depth_edges(d)
    assert np.isnan(m[175, 199]) and np.isnan(m[175, 401])      # at the box edges
    assert m[175, 300] == 3.0 and m[50, 50] == 8.0              # well inside
    assert np.isnan(m[1, 1])                                    # next to the invalid pixel


def test_pairs_and_fit_recover_bias():
    z = scene()
    cal = LidarCalibrator()
    for i in range(10):
        zz, zl = depth_pairs(biased(z, 0.86, -0.015), K, lidar_points(z, seed=i), cal.params)
        used, reason = cal.add_pairs(float(i), zz, zl)
        assert used, reason
    res = cal.fit()
    assert res.accepted, res.reason
    assert abs(res.a - 0.86) < 1e-3 and abs(res.b + 0.015) < 1e-3


def test_fit_is_robust_to_wrong_matches():
    rng = np.random.default_rng(1)
    zl = rng.uniform(1, 9, 5000)
    zz = biased(zl, 0.9, -0.01)
    bad = rng.random(5000) < 0.2                                 # 20% matched to another surface
    zz[bad] = rng.uniform(0.5, 10, bad.sum())
    cal = LidarCalibrator()
    cal.add_pairs(0.0, zz, zl)
    res = cal.fit()
    assert res.accepted, res.reason
    assert abs(res.a - 0.9) < 0.01 and abs(res.b + 0.01) < 0.003


def test_misaligned_lidar_is_rejected():
    rng = np.random.default_rng(2)
    zl = rng.uniform(1, 9, 5000)
    zz = rng.uniform(0.5, 10, 5000)                              # no relation: e.g. wrong mount
    cal = LidarCalibrator()
    cal.add_pairs(0.0, zz, zl)
    assert not cal.fit().accepted


def test_frame_with_few_pairs_is_skipped():
    used, _ = LidarCalibrator().add_pairs(0.0, np.ones(10), np.ones(10))
    assert not used
