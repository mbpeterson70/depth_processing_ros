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
For ros_system_monitor it publishes a NodeInfoMsg on `~/node_status` every second: STARTUP
while calibrating (or before any depth arrives), NOMINAL once depth is being corrected, and
WARNING if calibration timed out and the fallback correction is in use.
Before calibration finishes, raw depth is passed through uncorrected (or held back if
`publish_before_calibrated` is false). Setting `fixed_a` and `fixed_b` skips calibration.
If processing a depth image fails for any reason, that image is passed through uncorrected
and the node reports ERROR, so downstream depth never disappears because of this node.

Lidar (optional, `lidar_topic`): while lidar scans arrive, the correction is also calibrated
from lidar (lidar_calibration.py): every `frame_period_s` the scan nearest the
depth image is moved into the camera frame through TF (via `lidar_odom_frame` to compensate
motion between scan and image, and `lidar_mount_rpy_deg` for a lidar whose TF frame doesn't
match how it is mounted), and its visible points give (z_zed, z_lidar) pairs. After
`lidar_calibration_duration_s` s of frames the same model is fitted; while the lidar keeps
publishing it is re-fitted from the latest window every `lidar_refit_period_s`. A lidar fit
replaces a ground or fallback correction. The ground calibration runs as before alongside it
until there is a correction, so with no lidar (not plugged in, scans stopped for
`lidar_timeout_s`, or lidar fits rejected) the node behaves exactly as without `lidar_topic`.
"""

import math
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from ros_system_monitor_msgs.msg import NodeInfoMsg
from std_msgs.msg import String

from depth_processing_ros.calibration import CalibrationParams, GroundCalibrator, correct_depth
from depth_processing_ros.lidar_calibration import (LidarCalibrationParams, LidarCalibrator, depth_pairs,
                                                    make_T, transform)


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_matrix(roll, pitch, yaw):
    """Fixed-axis x-y-z (roll, pitch, yaw) in radians."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def cloud_to_xyz(msg: PointCloud2):
    """(N, 3) float64 points, and the mean per-point time (ns) if the cloud has a `timestamp`
    field (e.g. Livox), else None."""
    fields = {f.name: f for f in msg.fields}
    fmt = {7: "<f4", 8: "<f8"}             # PointField FLOAT32, FLOAT64
    arr = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
    col = lambda f: arr[:, f.offset:f.offset + np.dtype(fmt[f.datatype]).itemsize].copy().view(
        fmt[f.datatype]).ravel()
    xyz = np.stack([col(fields[c]) for c in "xyz"], axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    t_mean = None
    ts = fields.get("timestamp")
    if ts is not None and ts.datatype in fmt and ok.any():
        t_mean = float(np.mean(col(ts)[ok]))
    return xyz[ok], t_mean


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

    def __init__(self, **kwargs):
        super().__init__("depth_scale_correction", **kwargs)
        self.declare_parameters("", [
            ("depth_topic", "depth/depth_registered"),
            ("camera_info_topic", "depth/camera_info"),
            ("output_prefix", "depth_corrected"),
            ("ground_frame", ""),                 # e.g. hamilton/gpe (Spot ground-plane estimate)
            ("calibration_duration_s", 30.0),     # s of usable frames per calibration window
            ("frame_period_s", 0.5),              # one calibration frame per this many s
            ("max_calibration_time_s", 300.0),    # then fall back to fallback_a/b
            ("min_camera_height_m", 0.4),         # below this the robot is taken to be sitting
            ("publish_before_calibrated", True),  # false: hold depth back until calibrated
            ("fixed_a", float("nan")),            # set both to skip calibration
            ("fixed_b", float("nan")),
            ("fallback_a", 1.0),
            ("fallback_b", 0.0),
            ("min_z", 1.0),
            ("max_z", 8.0),
            ("pixel_step", 4),
            ("tf_timeout_s", 0.1),
            ("reliable_input", True),             # false: best-effort (sensor data) subscriptions
            ("lidar_topic", ""),                  # PointCloud2; empty: ground calibration only
            ("lidar_odom_frame", ""),             # fixed frame for scan->image motion, e.g. hamilton/odom
            ("lidar_mount_rpy_deg", [0.0, 0.0, 0.0]),   # extra rotation of the points in their TF frame
            ("lidar_timeout_s", 2.0),             # no scan for this long: lidar counts as gone
            ("lidar_calibration_duration_s", 10.0),
            ("lidar_refit_period_s", 60.0),       # re-fit from the latest window; <= 0: fit once
            ("lidar_min_z", 0.5),
            ("lidar_max_z", 9.5),                 # stay below the ZED max depth
            ("nickname", "depth_correction"),     # name for ros_system_monitor
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
        self.nickname = gp("nickname")
        self.ab = None
        self.fell_back = False
        self.source = None                    # where self.ab came from: fixed, ground, lidar, fallback
        fa, fb = gp("fixed_a"), gp("fixed_b")
        lidar_topic = gp("lidar_topic")
        if not (math.isnan(fa) or math.isnan(fb)):
            self.ab, self.source = (fa, fb), "fixed"
        elif not self.ground_frame and not lidar_topic:
            raise RuntimeError("set ground_frame (e.g. <robot>/gpe) and/or lidar_topic, or fixed_a and fixed_b")

        self.lidar_enabled = bool(lidar_topic) and self.source != "fixed"
        self.lidar_odom_frame = gp("lidar_odom_frame")
        self.lidar_mount = make_T(rpy_to_matrix(*np.radians(gp("lidar_mount_rpy_deg"))), np.zeros(3))
        self.lidar_timeout = gp("lidar_timeout_s")
        self.lidar_duration = gp("lidar_calibration_duration_s")
        self.lidar_refit_period = gp("lidar_refit_period_s")
        self.lidar_calibrator = LidarCalibrator(LidarCalibrationParams(min_z=gp("lidar_min_z"),
                                                                       max_z=gp("lidar_max_z")))
        self.scans = deque(maxlen=4)          # (stamp s, frame_id, points, point-time stamp s)
        self.last_lidar_frame_stamp = -math.inf
        self.last_lidar_fit_stamp = -math.inf
        self.lidar_skip_reasons = {}

        self.K = None
        self.info = None
        self.first_stamp = None
        self.last_cal_stamp = -math.inf
        self.skip_reasons = {}
        self.n_in = self.n_out = 0
        self.last_error = None                # message of a failure since the last heartbeat

        prefix = gp("output_prefix")
        self.depth_pub = self.create_publisher(Image, f"{prefix}/depth_registered", 10)
        self.info_pub = self.create_publisher(CameraInfo, f"{prefix}/camera_info", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, f"{prefix}/calibration", latched)
        sub_qos = QoSProfile(depth=10) if gp("reliable_input") else qos_profile_sensor_data
        self.create_subscription(CameraInfo, gp("camera_info_topic"), self.info_cb, sub_qos)
        self.create_subscription(Image, gp("depth_topic"), self.depth_cb, sub_qos)
        if self.lidar_enabled:
            self.create_subscription(PointCloud2, lidar_topic, self.lidar_cb, qos_profile_sensor_data)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(30.0, self.log_rates)
        self.monitor_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        self.create_timer(1.0, self.send_node_status)

        if self.ab is not None:
            self.publish_status(f"fixed correction a={self.ab[0]:.4f} b={self.ab[1]:+.4f}")
        else:
            self.publish_status("calibrating")

    def publish_status(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def send_node_status(self):
        """Heartbeat + state for ros_system_monitor."""
        msg = NodeInfoMsg()
        msg.nickname = self.nickname
        msg.node_name = self.get_fully_qualified_name()
        if self.last_error is not None:
            msg.status = NodeInfoMsg.ERROR
            msg.notes = f"passing raw depth through after error: {self.last_error}"
            self.last_error = None
        elif self.fell_back:
            msg.status = NodeInfoMsg.WARNING
            msg.notes = (f"no calibration accepted within {self.max_cal_time:.0f} s; "
                         f"fallback a={self.ab[0]:.4f} b={self.ab[1]:+.4f}")
        elif self.ab is not None:
            msg.status = NodeInfoMsg.NOMINAL
            msg.notes = f"correcting depth ({self.source}): a={self.ab[0]:.4f} b={self.ab[1]:+.4f}"
        else:
            msg.status = NodeInfoMsg.STARTUP
            if self.first_stamp is None:
                msg.notes = "waiting for depth images"
            elif self.K is None:
                msg.notes = "waiting for camera_info"
            else:
                passing = "raw depth passed through" if self.publish_before else "depth held back"
                if self.lidar_calibrator.frames:
                    progress = f"lidar {self.lidar_calibrator.span:.0f}/{self.lidar_duration:.0f} s"
                else:
                    progress = f"ground {self.calibrator.span:.0f}/{self.duration:.0f} s"
                msg.notes = f"calibrating: {progress} of usable frames ({passing})"
        self.monitor_pub.publish(msg)

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
        try:
            out = self.process(msg)
        except Exception as e:  # never drop depth because of this node: pass it through raw
            self.last_error = f"{type(e).__name__}: {e}"
            self.get_logger().error(f"depth correction failed, passing raw depth through: {self.last_error}",
                                    throttle_duration_sec=10.0)
            out = msg
        if out is None:
            return
        self.depth_pub.publish(out)
        self.n_out += 1
        if self.info is not None:
            info = self.info
            info.header.stamp = msg.header.stamp
            self.info_pub.publish(info)

    def process(self, msg: Image):
        """The depth image to publish (corrected, or raw while calibrating), or None to hold it back."""
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.first_stamp is None:
            self.first_stamp = stamp
        depth = image_to_depth_m(msg)
        if self.K is not None and self.source != "fixed":
            if self.lidar_active(stamp):
                self.lidar_step(msg, stamp, depth)
            if self.ab is None:
                self.calibrate_step(msg, stamp, depth)
        if self.ab is not None:
            return depth_m_to_image(correct_depth(depth, *self.ab), msg)
        return msg if self.publish_before else None

    def calibrate_step(self, msg: Image, stamp: float, depth: np.ndarray):
        if stamp - self.first_stamp > self.max_cal_time:
            self.ab, self.source = self.fallback, "fallback"
            self.fell_back = True
            self.publish_status(f"no accepted calibration within {self.max_cal_time:.0f} s "
                                f"(skips: {self.skip_reasons}); falling back to a={self.ab[0]} b={self.ab[1]}")
            return
        if stamp - self.last_cal_stamp < self.frame_period or not self.ground_frame:
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
            self.ab, self.source = (res.a, res.b), "ground"
            self.publish_status(
                f"calibrated from the ground: a={res.a:.4f} b={res.b:+.4f} from {res.n_frames} frames, {res.n_pixels} ground pixels "
                f"({res.inlier_share:.0%} consistent) at camera height {height:.3f} m; ground ZED/true before: "
                f"{fmt(res.ratio_before)}; after: {fmt(res.ratio_after)}")
        else:
            self.get_logger().warn(f"calibration rejected ({res.reason}); sliding the window")
            self.calibrator.drop_before(self.calibrator.frames[-1][0] - self.duration / 2)

    # ------------------------------------------------------------------------------ lidar

    def lidar_cb(self, msg: PointCloud2):
        xyz, t_mean = cloud_to_xyz(msg)
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        # transform the scan at its mean point time: the scan spans ~0.1 s after its header stamp
        t_points = t_mean * 1e-9 if t_mean is not None and abs(t_mean * 1e-9 - stamp) < 1.0 else stamp
        self.scans.append((stamp, msg.header.frame_id, xyz, t_points))

    def lidar_active(self, stamp: float) -> bool:
        return self.lidar_enabled and bool(self.scans) and stamp - self.scans[-1][0] < self.lidar_timeout

    def lidar_skip(self, reason: str):
        self.lidar_skip_reasons[reason] = self.lidar_skip_reasons.get(reason, 0) + 1

    def lidar_step(self, msg: Image, stamp: float, depth: np.ndarray):
        if stamp - self.last_lidar_frame_stamp < self.frame_period:
            return
        self.last_lidar_frame_stamp = stamp
        scan_stamp, lidar_frame, xyz, t_points = min(self.scans, key=lambda s: abs(s[3] - stamp))
        if abs(t_points - stamp) > 0.2:
            return self.lidar_skip("no scan near the image")
        try:
            if self.lidar_odom_frame:
                tf = self.tf_buffer.lookup_transform_full(
                    msg.header.frame_id, msg.header.stamp, lidar_frame,
                    Time(nanoseconds=int(t_points * 1e9)).to_msg(), self.lidar_odom_frame, self.tf_timeout)
            else:
                tf = self.tf_buffer.lookup_transform(msg.header.frame_id, lidar_frame, msg.header.stamp,
                                                     self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return self.lidar_skip("no tf")
        q, t = tf.transform.rotation, tf.transform.translation
        T_cam_lidar = make_T(quat_to_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z]) @ self.lidar_mount
        zz, zl = depth_pairs(depth, self.K, transform(T_cam_lidar, xyz), self.lidar_calibrator.params)
        used, reason = self.lidar_calibrator.add_pairs(stamp, zz, zl)
        if not used:
            return self.lidar_skip(reason.split(" ")[0] + " pairs")
        cal = self.lidar_calibrator
        if cal.span < self.lidar_duration:
            return
        cal.drop_before(stamp - self.lidar_duration)          # keep a sliding window
        refit_due = self.lidar_refit_period > 0 and stamp - self.last_lidar_fit_stamp >= self.lidar_refit_period
        if self.source == "lidar" and not refit_due:
            return
        self.last_lidar_fit_stamp = stamp
        res = cal.fit()
        fmt = lambda rs: ", ".join(f"{lo}-{hi} m {r:.3f}" for lo, hi, r in rs)
        if res.accepted:
            previous = f" (was {self.source}: a={self.ab[0]:.4f} b={self.ab[1]:+.4f})" if self.ab else ""
            self.ab, self.source, self.fell_back = (res.a, res.b), "lidar", False
            self.publish_status(
                f"calibrated from lidar: a={res.a:.4f} b={res.b:+.4f}{previous} from {res.n_frames} frames, "
                f"{res.n_pixels} pairs ({res.inlier_share:.0%} consistent); ZED/lidar before: "
                f"{fmt(res.ratio_before)}; after: {fmt(res.ratio_after)}")
        else:
            self.get_logger().warn(f"lidar calibration rejected ({res.reason}; skips: {self.lidar_skip_reasons})",
                                   throttle_duration_sec=30.0)


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
