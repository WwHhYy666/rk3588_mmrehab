# Pose estimation

`pose_estimation/` 负责摄像头帧接入和 RKNN 姿态估计。

- `gstreamer_gi_capture.py`：低延迟 GStreamer appsink。
- `backend_selector.py`：共享应用的姿态后端选择。
- `rknn_pose/`：YOLOv5n raw + RTMPose、关键点适配、诊断和测试。

模型统一放在 `models/vision/`。
