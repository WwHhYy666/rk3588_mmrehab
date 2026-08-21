# RKNN/NPU 姿态识别部署计划

## 为什么保留 MediaPipe CPU

当前 MediaPipe CPU 路线已经完成摄像头预览、姿态识别、医生模板录入和患者实时训练验证，是主 Demo 的稳定路径。RKNN/NPU 接入初期存在模型转换、runtime 版本、后处理和关键点格式差异等风险，因此不能替换现有 CPU 路线。

## 为什么新增 RKNN/NPU 支线

RK3588 有 NPU 资源，后续可用 RKNN YOLOv8-Pose 降低 CPU 占用、提升姿态推理 FPS，并让资源监控中的 NPU load 具备真实意义。支线先独立验证，再封装统一接口，最后才考虑接入训练流程。

## 三阶段

### 1. 官方 demo smoke test

- 使用 Rockchip 官方或已验证的 RKNN YOLOv8-Pose demo。
- 先确认模型、runtime、摄像头输入、NPU load 和 FPS 都能工作。
- 验收：`/sys/kernel/debug/rknpu/load` 在推理时出现非 0，终端可看到关键点或检测结果。

### 2. 项目后端封装

- 在 `vision/rknn_pose/` 中实现真实 RKNN pose backend。
- 输出统一 `PoseResult`。
- 用 `coco17_to_rehab.py` 转换为项目内部 rehab keypoints。
- 验收：独立 smoke test 显示 backend、FPS、关键点数量和 rehab keypoints。

### 3. 接入训练流程

- 通过 `POSE_BACKEND=auto|mediapipe|rknn` 选择后端。
- 默认仍为 `mediapipe`。
- `auto` 模式下 RKNN 初始化失败必须 fallback 到 MediaPipe。
- 页面显示 pose backend、Pose FPS 和 NPU load。
- 验收：RKNN 可用时 NPU load 非 0；RKNN 不可用时主 Demo 不崩溃。

## 当前第一阶段边界

- 不下载模型。
- 不接真实 RKNN 推理。
- 不改训练计数逻辑。
- 不删除或替换现有 MediaPipe 文件。
- 只新增支线骨架、后端选择器、部署文档，并修复 NPU load 显示。
