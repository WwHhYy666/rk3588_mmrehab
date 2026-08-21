# RKNN YOLOv8-Pose 支线

这个目录是并行的 RKNN/NPU 姿态识别支线，目标是在不破坏现有 MediaPipe CPU Demo 的前提下，为后续 RK3588 NPU 姿态推理预留工程结构。

## 当前阶段

- 默认姿态后端仍然是 MediaPipe。
- 不下载模型，不接真实 RKNN 推理。
- `smoke_test_placeholder.py` 只验证支线脚本可以独立运行。
- `coco17_to_rehab.py` 负责把 COCO17 关键点映射到项目内部 rehab keypoints。
- `pose_result.py` 提供未来 MediaPipe/RKNN 统一返回结构。

## 未来安装与转换

1. 在开发机准备 YOLOv8-Pose 原始模型。
2. 使用 Rockchip 官方 RKNN-Toolkit2 转换为 RKNN 模型。
3. 在 RK3588 板端安装匹配版本的 RKNN runtime。
4. 先运行官方 RKNN demo，确认 NPU load、FPS 和输出关键点正常。
5. 再把输出接入本目录的 `PoseResult` 和 COCO17 映射。

## 板端运行目标

未来真实后端应提供独立 smoke test，至少输出：

- backend: rknn
- FPS
- COCO17 原始关键点
- rehab keypoints
- 可选 annotated frame

## 验收标准

- 官方 RKNN demo 运行时 `/sys/kernel/debug/rknpu/load` 非 0。
- 项目 smoke test 能显示 backend 和 FPS。
- 页面可显示当前 pose backend。
- RKNN 后端失败时 fallback 到 MediaPipe，不影响主 Demo。
