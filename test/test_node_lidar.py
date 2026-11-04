"""Calibration source selection: one start-up calibration from lidar or the ground."""

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2, PointField

from depth_processing_ros import depth_scale_correction_node as dsc


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int((t % 1) * 1e9))


def depth_msg(t):
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp(t), "cam"
    msg.height, msg.width, msg.encoding, msg.step = 4, 6, "32FC1", 24
    msg.data = np.full((4, 6), 2.0, np.float32).tobytes()
    return msg


def cloud_msg(t, points=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])):
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp(t), "lidar"
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
    ts = (t + 0.05) * 1e9 * np.ones(len(points))
    msg.fields.append(PointField(name="timestamp", offset=12, datatype=PointField.FLOAT64, count=1))
    rec = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("timestamp", "<f8")])
    rec["x"], rec["y"], rec["z"], rec["timestamp"] = *points.T, ts
    msg.point_step, msg.width, msg.height = 20, len(points), 1
    msg.data = rec.tobytes()
    return msg


def node_with(**params):
    params.setdefault("ground_frame", "gpe")
    node = dsc.DepthScaleCorrectionNode(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    node.K = np.eye(3)
    calls = []
    node.calibrate_step = lambda msg, t, d: calls.append(("ground", t))
    node.lidar_step = lambda msg, t, d: calls.append(("lidar", t))
    node.quality_step = lambda msg, t, d: calls.append(("quality", t))
    return node, calls


def test_cloud_to_xyz_drops_zero_points_and_reads_point_time():
    xyz, t_mean = dsc.cloud_to_xyz(cloud_msg(10.0))
    np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0]])
    assert abs(t_mean * 1e-9 - 10.05) < 1e-6


def test_ground_only_without_lidar_topic():
    node, calls = node_with()
    node.depth_cb(depth_msg(1.0))
    assert calls == [("ground", 1.0)]
    node.destroy_node()


def test_lidar_not_plugged_in_falls_back_to_ground():
    node, calls = node_with(lidar_topic="/lidar", calibration_source="lidar", lidar_wait_s=1.5)
    for t in (1.0, 2.0, 3.0):
        node.depth_cb(depth_msg(t))                # no scans: ground only after lidar_wait_s
    assert calls == [("ground", 3.0)] and node.ground_instead
    node.destroy_node()


def test_lidar_source_calibrates_from_lidar_only_then_measures_quality():
    node, calls = node_with(lidar_topic="/lidar", calibration_source="lidar", lidar_timeout_s=2.0)
    node.lidar_cb(cloud_msg(1.0))
    node.depth_cb(depth_msg(1.1))                  # calibrating: lidar only
    assert calls == [("lidar", 1.1)]
    node.ab, node.source = (0.86, -0.015), "lidar"
    calls.clear()
    node.depth_cb(depth_msg(2.0))                  # calibrated: no more calibration, quality only
    assert calls == [("quality", 2.0)]
    calls.clear()
    node.depth_cb(depth_msg(10.0))                 # scans stopped: keeps the correction
    assert calls == []
    node.destroy_node()


def test_fixed_correction_ignores_lidar():
    node, calls = node_with(lidar_topic="/lidar", fixed_a=0.9, fixed_b=0.0)
    node.lidar_cb(cloud_msg(1.0))
    node.depth_cb(depth_msg(1.1))                  # no calibration; quality is still measured
    assert calls == [("quality", 1.1)]
    node.destroy_node()


def test_lidar_only_config_is_allowed():
    node, _ = node_with(lidar_topic="/lidar", ground_frame="", calibration_source="lidar")
    assert node.lidar_enabled
    node.destroy_node()
