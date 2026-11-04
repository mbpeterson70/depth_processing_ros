import numpy as np
from scipy.spatial.transform import Rotation

from depth_processing_ros.lidar_reference import (LidarZedData, evaluate, fit, make_T, mask_depth_edges,
                                                  perturb, project, transform, visible_mask)

K = np.array([[261.5, 0.0, 312.4], [0.0, 261.5, 179.9], [0.0, 0.0, 1.0]])
H, W = 360, 640
# lidar 0.6 m behind and 0.3 m above the camera, axes as on Spot (x fwd, y left, z up)
R_CAM_LIDAR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_CAM_LIDAR = make_T(R_CAM_LIDAR, np.array([0.0, 0.3, -0.6]))


def scene(seed):
    """True depth of a pitched-down camera over flat ground with a few boxes and a back wall."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:H, 0:W]
    dx, dy = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    p = np.radians(10)
    n = np.array([0.0, -np.cos(p), -np.sin(p)])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (n @ (-0.6 * n)) / (n[0] * dx + n[1] * dy + n[2])
    z[(z <= 0) | (z > 15)] = 15.0                                     # back wall
    for _ in range(4):                                                # boxes facing the camera
        zb = rng.uniform(1.5, 10)
        cu, cv, s = rng.uniform(80, 560), rng.uniform(60, 220), rng.uniform(20, 70)
        box = (abs(u - cu) < s) & (abs(v - cv) < s) & (z > zb)
        z[box] = zb
    return z.astype(np.float32)


def biased(z, a, b):
    return a / (1.0 / z - b)


def make_data(a, b, delta_rv, delta_t, n_frames=12, pts_per_frame=1500):
    """Lidar truth through the TRUE extrinsic; the data carries a nominal extrinsic that is off
    by (delta_rv, delta_t)^-1, so the fit should recover (delta_rv, delta_t)."""
    T_true = T_CAM_LIDAR
    T_nominal = np.linalg.inv(make_T(Rotation.from_rotvec(delta_rv).as_matrix(), delta_t)) @ T_true
    depths, pts, frame = [], [], []
    rng = np.random.default_rng(0)
    for f in range(n_frames):
        z = scene(f)
        u = rng.uniform(0, W - 1, pts_per_frame)
        v = rng.uniform(0, H - 1, pts_per_frame)
        zt = z[v.astype(int), u.astype(int)]
        p_cam = np.stack([(u - K[0, 2]) / K[0, 0] * zt, (v - K[1, 2]) / K[1, 1] * zt, zt], axis=1)
        pts.append(transform(np.linalg.inv(T_true), p_cam))
        depths.append(mask_depth_edges(biased(z, a, b)))
        frame.append(np.full(pts_per_frame, f))
    return LidarZedData(np.stack(depths), np.concatenate(pts), np.concatenate(frame), K, T_nominal)


def test_inverse_fit_recovers_bias():
    data = make_data(0.93, -0.02, np.zeros(3), np.zeros(3))
    res = fit(data, "inverse")
    assert abs(res.depth_params["a"] - 0.93) < 0.01 and abs(res.depth_params["b"] + 0.02) < 0.005
    assert evaluate(data, res)["median_abs_log"] < 0.005


def test_joint_fit_recovers_bias_and_extrinsic():
    rv, t = np.radians([0.8, -0.6, 0.3]), np.array([0.01, -0.015, 0.02])
    data = make_data(0.93, -0.02, rv, t)
    plain = fit(data, "inverse")
    joint = fit(data, "inverse", refine_extrinsic=True)
    assert abs(joint.depth_params["a"] - 0.93) < 0.02 and abs(joint.depth_params["b"] + 0.02) < 0.006
    # depth-only residuals constrain pitch (rows sweep through depth); yaw and lateral offsets
    # mostly slide points along surfaces of similar depth and stay near the prior
    assert abs(np.degrees(joint.delta_rotvec[0] - rv[0])) < 0.2
    assert evaluate(data, joint)["median_abs_log"] < evaluate(data, plain)["median_abs_log"]


def test_visible_mask_drops_hidden_point():
    p_cam = np.array([[0.0, 0.0, 2.0], [0.001, 0.0, 6.0], [1.0, 0.0, 4.0]])
    u, v, z = project(p_cam, K)
    assert visible_mask(u, v, z, (H, W)).tolist() == [True, False, True]


def test_perturb_identity():
    assert np.allclose(perturb(T_CAM_LIDAR, np.zeros(3), np.zeros(3)), T_CAM_LIDAR)
