# depth_processing_ros

Online correction of the ZED stereo depth bias. On Spot, the ZED depth reads progressively
short with range (about 12-16% at 5 m, 20-28% at 10 m on the West Point 2026 data, measured
against the Livox), and the amount changes between sessions. That distorts maps built from
depth differently on different days, which hurts cross-session registration (ROMAN, ICP).

The node calibrates the bias from the ground, with no lidar: Spot's ground-plane estimate
(`<robot>/gpe` in TF) plus the camera mount give the true depth of every pixel whose ray hits
the ground. It fits `1/z = a/z_zed + b` over the first `calibration_duration_s` (30 s) of usable
frames (Spot standing, flat ground in view; frames and fits that fail checks are skipped)
and then publishes corrected depth:

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

On the alpha_dcist bag, the lidar fit lowers the median depth error against lidar from 23% (raw) and
7% (ground calibration) to about 1.5% on held-out data, and converges within about 5 s of data
(see "Lidar reference" below).

## Stereo reference diagnostic (offline)

`scripts/stereo_vs_zed.py` runs classical stereo (OpenCV SGBM, left-right checked, texture-gated;
`depth_processing_ros/stereo_reference.py`) on the ZED's own rectified pair and compares
block-median depths with the ZED depth. It needs `left/image_rect_color`, `right/image_rect_color`,
`depth/depth_registered` and `right/camera_info` in the bag (the standard topic lists do not
record the right image).

    PYTHONPATH=. ~/mit/ros_ws/envs/dcist/roman/bin/python scripts/stereo_vs_zed.py \
        --bag <bag_dir> --out <dir> --ns /euclid/euclid_zed --stride 3

Writes `panel.mp4` (left, SGBM depth, ZED depth, per-block ZED/stereo ratio, SGBM valid mask,
scatter), `scatter.png`, `ratio_vs_range.png`, `ratio_by_position.png`, `timeseries.png` and
`summary.json` (ratio per range bin and a `1/z = a/z_zed + b` fit using `GroundCalibrator.fit`).

Caveats: at 640x360, fx*B is about 31 px*m, so 5 m is about 6 px of disparity and stereo noise is
about 4% at 5 m and 8% at 10 m. SGBM uses the same rectification and baseline as the ZED, so it
finds where the neural depth departs from geometry. It cannot see a calibration (disparity offset)
error that both share; that needs an external reference (ground plane, lidar, tape measure).

## Lidar reference (offline)

`scripts/lidar_vs_zed.py` fits and evaluates the depth correction against lidar
(`depth_processing_ros/lidar_reference.py`). Lidar points are deskewed to the depth stamp with
odom, occluded points are dropped, and depth edges are masked. Depth models (scale, `1/z = a/z_zed + b`)
are fitted robustly in log-depth, optionally together with a small, prior-bounded correction of the
lidar->camera extrinsic. Fitting uses alternating 60 s chunks of the bag; the other chunks are held
out for evaluation.

    PYTHONPATH=. ~/mit/ros_ws/envs/dcist/roman/bin/python scripts/lidar_vs_zed.py extract \
        --bag <bag> --out <dir> --robot hamilton --lidar-rpy-deg 180 0 0     # slow, one pass
    PYTHONPATH=. ~/mit/ros_ws/envs/dcist/roman/bin/python scripts/lidar_vs_zed.py fit --out <dir>

On hamilton the Mid-360 is mounted upside down, but `body -> livox` in TF has identity rotation;
hence `--lidar-rpy-deg 180 0 0`. Depth-only residuals constrain the extrinsic pitch well, but not
yaw or lateral offsets. ZED depth stops at the driver's max depth (10 m here), so ranges beyond it
show that cutoff, not a bias.
