import numpy as np

from depth_processing_ros.calibration import GroundCalibrator, correct_depth

K = np.array([[261.5, 0.0, 312.4], [0.0, 261.5, 179.9], [0.0, 0.0, 1.0]])
H, W = 360, 640


def ground_in_camera(height=0.648, pitch_deg=9.0):
    """Ground plane (normal, point) in the optical frame (x right, y down, z forward) of a
    camera `height` m above flat ground, pitched down by pitch_deg."""
    p = np.radians(pitch_deg)
    up = np.array([0.0, -np.cos(p), -np.sin(p)])        # world up in the camera frame
    return up, -height * up


def render(normal, point, wall_z=None):
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (normal @ point) / (d @ normal)
    z[(z <= 0) | (z > 20)] = np.nan
    if wall_z is not None:                             # an object covering the image centre
        z[(abs(u - W / 2) < 60) & (abs(v - H / 2) < 60)] = wall_z
    return z


def biased(z, a, b):                                    # inverse of 1/z = a/z_zed + b
    return a / (1.0 / z - b)


def test_correct_depth_inverts_bias():
    z = np.linspace(0.5, 10, 50).reshape(5, 10)
    out = correct_depth(biased(z, 0.95, -0.03), 0.95, -0.03)
    assert np.allclose(out, z, rtol=1e-5)


def test_fit_recovers_bias():
    n, p = ground_in_camera()
    z = render(n, p, wall_z=3.0)
    cal = GroundCalibrator()
    for i in range(10):
        used, reason = cal.add_frame(float(i), biased(z, 0.9, -0.02), K, n, p)
        assert used, reason
    res = cal.fit()
    assert res.accepted, res.reason
    assert abs(res.a - 0.9) < 1e-3 and abs(res.b + 0.02) < 1e-3


def test_sloped_ground_is_rejected():
    n, p = ground_in_camera()
    # the real ground falls away from the plane the robot stands on: ZED depth is much longer
    z = render(n, p) * 1.6
    used, reason = GroundCalibrator().add_frame(0.0, z, K, n, p)
    assert not used
