# depth_processing_ros

Online correction of the ZED stereo depth bias. On Spot, the ZED depth reads progressively
short with range (about 12-16% at 5 m, 20-28% at 10 m on the West Point 2026 data, measured
against the Livox), and the amount changes between sessions. That distorts maps built from
depth differently on different days, which hurts cross-session registration (ROMAN, ICP).

The node calibrates the bias once at start-up, from lidar or from the ground (`calibration_source`),
fitting `1/z = a/z_zed + b`, and then keeps that correction for the session. From the ground,
Spot's ground-plane estimate (`<robot>/gpe` in TF) plus the camera mount give the true depth of
every pixel whose ray hits the ground; it fits over the first `calibration_duration_s` (30 s) of
usable frames (Spot standing, flat ground in view; frames and fits that fail checks are skipped).
It publishes corrected depth:

    /<robot>/<robot>_zed/depth_corrected/depth_registered   (same encoding as the input)
    /<robot>/<robot>_zed/depth_corrected/camera_info
    /<robot>/<robot>_zed/depth_corrected/calibration        (std_msgs/String, latched)

Until calibration finishes, raw depth is passed through uncorrected (`publish_before_calibrated:
false` holds it back instead). If correcting an image fails for any reason, that image is passed
through uncorrected and the node reports ERROR to ros_system_monitor. `fixed_a` / `fixed_b` skip calibration. If nothing is accepted within
`max_calibration_time_s`, it falls back to `fallback_a` / `fallback_b` (identity by default).

    ros2 launch depth_processing_ros depth_scale_correction.launch.yaml robot_name:=hamilton

### Lidar calibration and depth quality

`calibration_source` picks where the one start-up calibration comes from: `ground` (above) or
`lidar` (`lidar_calibration.py`, needs `lidar_topic`, e.g. `/hamilton/livox/lidar`). For each frame,
the scan nearest the depth image is moved into the camera frame through TF. `lidar_odom_frame`
compensates the motion between scan and image, and `lidar_mount_rpy_deg` covers a lidar whose TF
frame doesn't match how it is mounted (hamilton's Mid-360 is upside down: `[180.0, 0.0, 0.0]`).
Visible points on valid, non-edge depth then give `(z_zed, z_lidar)` pairs. Pairs accumulate from
the start until the window spans `lidar_calibration_duration_s` (10 s) **and** each range bin
(0.5-3, 3-5, 5-9.5 m) has `lidar_min_bin_pairs`; a short window that sees only nearby surfaces gets
the far range wrong. After `lidar_max_calibration_duration_s` (60 s) it fits with what it has and
reports WARNING. If no scan arrives within `lidar_wait_s`, it calibrates from the ground (WARNING).
Either way the correction is then fixed for the session.

Status rows are kept short. `depth_correction` shows the calibration progress, then the source
and the multiplier it applies at 5 m (e.g. `lidar: x1.16 at 5 m`); it is yellow only if there
was no lidar, the far range was never covered, or nothing was calibrated. The full fit (a, b,
per-range ratios) is in the log and on `depth_corrected/calibration`.

With `lidar_topic` set and `quality_enabled`, the node also compares the corrected depth with lidar
every `quality_period_s` over the last 60 s (`depth_quality.py`) and publishes it on
`~/depth_quality_status` as its own ros_system_monitor row (`zed_depth_quality`): the median
depth error, e.g. `depth error 2.1% (vs lidar)`, plus a note if the far range (>= 4 m) is off by
more than 12%. Green up to 8% median error, yellow up to 15%, red above (lidar calibration gives
1-3%, the ground plane 3-9%, raw ZED 13-23% on the Spot bags). Without lidar it shows "not
measured (no lidar)" in green. This runs whichever source calibrated.

    ros2 launch depth_processing_ros depth_scale_correction.launch.yaml robot_name:=hamilton \
        lidar_topic:=/hamilton/livox/lidar calibration_source:=lidar

On held-out data from the West Point and Killian Court bags, the start-up lidar calibration gives
2-3% median depth error against lidar and stays within about 5% out to 10 m, against 3-9% for the
start-up ground calibration and 13-23% for raw ZED depth.
