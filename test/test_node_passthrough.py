"""The node must never drop depth: failures pass the raw image through."""

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from depth_processing_ros import depth_scale_correction_node as dsc


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def depth_msg(encoding="32FC1", value=2.0):
    msg = Image()
    msg.height, msg.width, msg.encoding = 4, 6, encoding
    if encoding == "32FC1":
        msg.step = 6 * 4
        msg.data = np.full((4, 6), value, np.float32).tobytes()
    else:
        msg.step = 6 * 3
        msg.data = bytes(4 * 6 * 3)
    return msg


def node_with(**params):
    node = dsc.DepthScaleCorrectionNode(parameter_overrides=[
        Parameter(k, value=v) for k, v in params.items()])
    published = []
    node.depth_pub.publish = published.append
    return node, published


def test_corrected_when_calibration_fixed():
    node, published = node_with(fixed_a=0.9, fixed_b=0.0)
    node.depth_cb(depth_msg(value=0.9))
    assert len(published) == 1
    np.testing.assert_allclose(np.frombuffer(published[0].data, np.float32), 1.0, rtol=1e-6)
    node.destroy_node()


def test_unsupported_encoding_passes_raw_through_and_reports_error():
    node, published = node_with(fixed_a=0.9, fixed_b=0.0)
    msg = depth_msg(encoding="rgb8")
    node.depth_cb(msg)
    assert published == [msg]
    assert node.last_error is not None and "unsupported depth encoding" in node.last_error
    node.destroy_node()


def test_correction_failure_passes_raw_through(monkeypatch):
    node, published = node_with(fixed_a=0.9, fixed_b=0.0)
    monkeypatch.setattr(dsc, "correct_depth", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    msg = depth_msg()
    node.depth_cb(msg)
    assert published == [msg]
    assert "boom" in node.last_error
    node.destroy_node()


def test_calibration_failure_passes_raw_through_even_when_holding_back(monkeypatch):
    node, published = node_with(ground_frame="robot/gpe", publish_before_calibrated=False)
    node.K = np.eye(3)
    monkeypatch.setattr(node, "calibrate_step", lambda *a: (_ for _ in ()).throw(ValueError("bad tf")))
    msg = depth_msg()
    node.depth_cb(msg)
    assert published == [msg]
    node.destroy_node()


def test_passes_raw_through_while_calibrating_by_default():
    node, published = node_with(ground_frame="robot/gpe")
    msg = depth_msg()
    node.depth_cb(msg)        # no camera_info yet, so not calibrating, but depth still flows
    assert published == [msg]
    assert node.last_error is None
    node.destroy_node()
