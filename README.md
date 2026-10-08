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

Depth is held back until calibration finishes (`publish_before_calibrated: true` passes raw
depth through instead). `fixed_a` / `fixed_b` skip calibration. If nothing is accepted within
`max_calibration_time_s`, it falls back to `fallback_a` / `fallback_b` (identity by default).

    ros2 launch depth_processing_ros depth_scale_correction.launch.yaml robot_name:=hamilton
