#!/usr/bin/env python3
"""Correct the ZED depth scale online, calibrated from the ground (see calibration.py).

Subscribes to a depth image and its camera_info. While calibrating, every
`frame_period_s` it looks up the ground plane in the camera frame from TF (Spot's
ground-plane estimate frame, `ground_frame`, e.g. <robot>/gpe) at the image stamp and
adds the frame's ground pixels to the calibrator, skipping frames where the camera isn't
at standing height or the ground in view isn't flat. Once `calibration_duration_s` s of
usable frames are collected, the correction 1/z = a/z_zed + b is fitted; if the fit fails
its checks, the oldest half of the window is dropped and collection continues. After
`max_calibration_time_s` without an accepted fit it falls back to `fallback_a/b`.

Publishes the corrected depth (same encoding as the input) and a copy of camera_info
on `depth_corrected/...`, and the calibration status (latched) on `depth_corrected/calibration`.
Before calibration finishes, depth is held back unless `publish_before_calibrated` is set
(then it is passed through uncorrected). Setting `fixed_a` and `fixed_b` skips calibration.
"""

import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from depth_processing_ros.calibration import CalibrationParams, GroundCalibrator, correct_depth


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def image_to_depth_m(msg: Image) -> np.ndarray:
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1", "mono16"):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return d.astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def depth_m_to_image(depth: np.ndarray, template: Image) -> Image:
    out = Image()
    out.header = template.header
    out.height, out.width = template.height, template.width
    out.encoding = template.encoding
    out.is_bigendian = template.is_bigendian
    if template.encoding == "32FC1":
        out.step = template.width * 4
        out.data = depth.astype(np.float32).tobytes()
    else:
        mm = np.nan_to_num(depth * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        out.step = template.width * 2
        out.data = np.clip(np.round(mm), 0, 65535).astype(np.uint16).tobytes()
    return out


class DepthScaleCorrectionNode(Node):

    def __init__(self):
        super().__init__("depth_scale_correction")
        self.declare_parameters("", [
            ("depth_topic", "depth/depth_registered"),
            ("camera_info_topic", "depth/camera_info"),
            ("output_prefix", "depth_corrected"),
            ("ground_frame", ""),                 # e.g. hamilton/gpe (Spot ground-plane estimate)
            ("calibration_duration_s", 30.0),     # s of usable frames per calibration window
            ("frame_period_s", 0.5),              # one calibration frame per this many s
            ("max_calibration_time_s", 300.0),    # then fall back to fallback_a/b
            ("min_camera_height_m", 0.4),         # below this the robot is taken to be sitting
            ("publish_before_calibrated", False),
            ("fixed_a", float("nan")),            # set both to skip calibration
            ("fixed_b", float("nan")),
            ("fallback_a", 1.0),
            ("fallback_b", 0.0),
            ("min_z", 1.0),
            ("max_z", 8.0),
            ("pixel_step", 4),
            ("tf_timeout_s", 0.1),
            ("reliable_input", True),             # false: best-effort (sensor data) subscriptions
        ])
        gp = lambda name: self.get_parameter(name).value
        self.ground_frame = gp("ground_frame")
        self.duration = gp("calibration_duration_s")
        self.frame_period = gp("frame_period_s")
        self.max_cal_time = gp("max_calibration_time_s")
        self.min_cam_height = gp("min_camera_height_m")
        self.publish_before = gp("publish_before_calibrated")
        self.tf_timeout = Duration(seconds=gp("tf_timeout_s"))
        self.fallback = (gp("fallback_a"), gp("fallback_b"))
        self.calibrator = GroundCalibrator(CalibrationParams(min_z=gp("min_z"), max_z=gp("max_z"),
                                                             pixel_step=gp("pixel_step")))
        self.ab = None
        fa, fb = gp("fixed_a"), gp("fixed_b")
        if not (math.isnan(fa) or math.isnan(fb)):
            self.ab = (fa, fb)
        elif not self.ground_frame:
            raise RuntimeError("set ground_frame (e.g. <robot>/gpe), or fixed_a and fixed_b")

        self.K = None
        self.info = None
        self.first_stamp = None
        self.last_cal_stamp = -math.inf
        self.skip_reasons = {}
        self.n_in = self.n_out = 0

        prefix = gp("output_prefix")
        self.depth_pub = self.create_publisher(Image, f"{prefix}/depth_registered", 10)
        self.info_pub = self.create_publisher(CameraInfo, f"{prefix}/camera_info", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, f"{prefix}/calibration", latched)
        sub_qos = QoSProfile(depth=10) if gp("reliable_input") else qos_profile_sensor_data
        self.create_subscription(CameraInfo, gp("camera_info_topic"), self.info_cb, sub_qos)
        self.create_subscription(Image, gp("depth_topic"), self.depth_cb, sub_qos)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(30.0, self.log_rates)

        if self.ab is not None:
            self.publish_status(f"fixed correction a={self.ab[0]:.4f} b={self.ab[1]:+.4f}")
        else:
            self.publish_status("calibrating")

    def publish_status(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def log_rates(self):
        state = "corrected" if self.ab is not None else (
            "passing through (calibrating)" if self.publish_before else "held back (calibrating)")
        self.get_logger().info(f"last 30 s: {self.n_in} depth images in, {self.n_out} out ({state})")
        self.n_in = self.n_out = 0

    def info_cb(self, msg: CameraInfo):
        self.info = msg
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def depth_cb(self, msg: Image):
        self.n_in += 1
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.first_stamp is None:
            self.first_stamp = stamp
        try:
            depth = image_to_depth_m(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=10.0)
            return
        if self.ab is None and self.K is not None:
            self.calibrate_step(msg, stamp, depth)
        if self.ab is not None:
            out = depth_m_to_image(correct_depth(depth, *self.ab), msg)
        elif self.publish_before:
            out = msg
        else:
            return
        self.depth_pub.publish(out)
        self.n_out += 1
        if self.info is not None:
            info = self.info
            info.header.stamp = msg.header.stamp
            self.info_pub.publish(info)

    def calibrate_step(self, msg: Image, stamp: float, depth: np.ndarray):
        if stamp - self.first_stamp > self.max_cal_time:
            self.ab = self.fallback
            self.publish_status(f"no accepted calibration within {self.max_cal_time:.0f} s "
                                f"(skips: {self.skip_reasons}); falling back to a={self.ab[0]} b={self.ab[1]}")
            return
        if stamp - self.last_cal_stamp < self.frame_period:
            return
        self.last_cal_stamp = stamp
        try:
            tf = self.tf_buffer.lookup_transform(msg.header.frame_id, self.ground_frame,
                                                 msg.header.stamp, self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.skip_reasons["no tf"] = self.skip_reasons.get("no tf", 0) + 1
            return
        q, t = tf.transform.rotation, tf.transform.translation
        R = quat_to_matrix(q.x, q.y, q.z, q.w)
        normal, point = R[:, 2], np.array([t.x, t.y, t.z])       # ground z axis and origin, camera frame
        height = abs(float(normal @ point))
        if height < self.min_cam_height:
            self.skip_reasons["not standing"] = self.skip_reasons.get("not standing", 0) + 1
            return
        used, reason = self.calibrator.add_frame(stamp, depth, self.K, normal, point)
        if not used:
            key = reason.split(" (")[0]
            self.skip_reasons[key] = self.skip_reasons.get(key, 0) + 1
            return
        if self.calibrator.span < self.duration:
            return
        res = self.calibrator.fit()
        fmt = lambda rs: ", ".join(f"{lo}-{hi} m {r:.3f}" for lo, hi, r in rs)
        if res.accepted:
            self.ab = (res.a, res.b)
            self.publish_status(
                f"calibrated a={res.a:.4f} b={res.b:+.4f} from {res.n_frames} frames, {res.n_pixels} ground pixels "
                f"({res.inlier_share:.0%} consistent) at camera height {height:.3f} m; ground ZED/true before: "
                f"{fmt(res.ratio_before)}; after: {fmt(res.ratio_after)}")
        else:
            self.get_logger().warn(f"calibration rejected ({res.reason}); sliding the window")
            self.calibrator.drop_before(self.calibrator.frames[-1][0] - self.duration / 2)


def main():
    rclpy.init()
    node = DepthScaleCorrectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
