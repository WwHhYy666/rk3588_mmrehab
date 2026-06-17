# RKNN/NPU 姿态识别接入计划

目标是在当前 8082 三动作康复 Demo 上新增 RKNN YOLOv8-Pose 后端，并尽量复刻现有 MediaPipe CPU 效果。

本计划只描述后续工程实现，不要求立即修改主流程。真正实施时必须遵守：

- 不删除、不重写 MediaPipe。
- 默认仍使用 MediaPipe CPU。
- RKNN/NPU 作为可选后端接入。
- RKNN 失败时必须可回退到 CPU，避免影响当前演示。
- 三动作训练、TTS、有效计数、纠错、attempt 保存、evaluate report 的表现应和当前基本一致。

## 1. 当前状态

已经完成：

- 官方 RKNN YOLOv8-Pose C demo 已跑通。
- 输出图可看到人体框和关键点。
- `/sys/kernel/debug/rknpu/load` 有非 0 波动。
- 板端 `rknn-toolkit-lite2==2.3.2` 可用。
- `from rknnlite.api import RKNNLite` 已验证成功。

当前项目已有：

- `vision/pose_backend_selector.py`：后端选择器骨架，默认 MediaPipe。
- `vision/rknn_pose/pose_result.py`：统一结果容器骨架。
- `vision/rknn_pose/coco17_to_rehab.py`：COCO17 到 rehab keypoints 映射。
- `prescription/banzi/record_prescription_http.py`：当前 8082 主入口，MediaPipe 姿态识别、医生模板录入、患者训练都在这里串起来。
- `realtime/training_session.py`：实时训练会话、有效计数、TTS、attempt 保存、evaluate report。

## 2. 总体路线

两条姿态后端最终统一输出同一种训练帧：

```text
MediaPipe 33 landmarks
  -> MediaPipe adapter
  -> rehab_keypoints
  -> current_frame_data
  -> realtime / evaluate

RKNN YOLOv8-Pose COCO17 keypoints
  -> RKNN adapter
  -> rehab_keypoints
  -> current_frame_data
  -> realtime / evaluate
```

关键原则：

- 不复制三套 NPU 动作 YAML。
- 不让动作 YAML 直接承担不同模型的关键点编号映射。
- MediaPipe 和 RKNN 都先转成统一 `rehab_keypoints`。
- `realtime/` 和 `evaluate/` 尽量只读统一后的名字，不关心原始后端编号。

## 3. Backend Rules

后端选择规则固定如下。

### 3.1 `POSE_BACKEND=mediapipe`

只使用 MediaPipe。

要求：

- 不加载 RKNN。
- 不检查 `.rknn` 模型是否存在。
- 行为必须和当前主 Demo 一致。

### 3.2 `POSE_BACKEND=rknn`

强制使用 RKNN。

要求：

- RKNN 初始化失败时报错。
- RKNN 模型路径错误时报错。
- RKNN 推理失败时报错或标记后端错误。
- 不自动 fallback 到 MediaPipe，方便定位 NPU 问题。

### 3.3 `POSE_BACKEND=auto`

优先使用 RKNN。

要求：

- RKNN 可用时使用 RKNN。
- RKNN 初始化失败、模型不存在、依赖缺失时 fallback 到 MediaPipe。
- 页面必须显示 `fallback_used=true` 和具体错误信息。

### 3.4 默认值

默认：

```bash
POSE_BACKEND=mediapipe
```

新增运行环境变量：

```bash
RKNN_POSE_MODEL=/home/elf/models/yolov8n-pose.rknn
RKNN_POSE_INPUT_SIZE=640
RKNN_POSE_CONF_THRES=0.35
RKNN_POSE_NMS_THRES=0.45
RKNN_POSE_KEYPOINT_THRES=0.12
ALLOW_POSE_BACKEND_MISMATCH=0
```

## 4. 坐标契约

RKNN YOLOv8-Pose 后处理必须处理 letterbox 坐标还原。

流程：

```text
original frame
  -> letterbox resize/pad to model input
  -> RKNN inference
  -> decode keypoints in input/letterbox coordinates
  -> remove pad and divide by scale
  -> restore to original frame pixel coordinates
  -> normalize to 0~1 by original frame width/height
```

保存到 `rehab_keypoints` 的坐标必须是基于原始 frame 的 0~1 归一化坐标：

```json
{
  "left_knee": {
    "x": 0.42,
    "y": 0.68,
    "z": 0.0,
    "visibility": 0.91
  }
}
```

角度计算允许使用：

- 原始 frame 像素坐标。
- 或等价的原始 frame 归一化坐标。

禁止：

- 用未还原的 640x640 letterbox/input 坐标计算角度。
- 用含 padding 的输入坐标直接保存 `rehab_keypoints`。

否则边缘 padding 会导致角度和真实人体位置错位。

## 5. 数据保存契约

运行时对象 `PoseResult` 可以包含：

```text
raw outputs
annotated_frame
numpy arrays
backend internals
```

但是 `current_frame_data` 必须 JSON 可序列化，只能包含：

```text
number
string
boolean
list
dict
null
```

`state.frames.append()` 只能保存：

```python
current_frame_data
```

不能保存：

```text
numpy frame
raw RKNN outputs
annotated_frame
RKNNLite object
OpenCV image
```

医生模板和患者 attempt 里的 `template_frames` 必须继续是当前 `evaluate/` 可读取的轻量 JSON 结构。

## 6. 关键点映射

项目内部统一保留这些 `rehab_keypoints` 名字：

```text
left_shoulder
right_shoulder
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
```

MediaPipe 33 点映射：

```text
left_shoulder  11
right_shoulder 12
left_hip       23
right_hip      24
left_knee      25
right_knee     26
left_ankle     27
right_ankle    28
```

RKNN YOLOv8-Pose / COCO17 映射：

```text
left_shoulder  5
right_shoulder 6
left_hip       11
right_hip      12
left_knee      13
right_knee     14
left_ankle     15
right_ankle    16
```

现有动作 YAML 中的：

```yaml
keypoint_rule:
  hip_index: 23
  knee_index: 25
  ankle_index: 27
```

属于 MediaPipe 历史编号。后续不要让 RKNN 直接复用这些编号。

## 7. 新增模块规划

新增文件：

```text
vision/rknn_pose/rknn_backend.py
vision/rknn_pose/yolov8_pose_postprocess.py
vision/rknn_pose/pose_frame_adapter.py
vision/rknn_pose/smoke_test_rknn_pose.py
vision/rknn_pose/replay_compare.py
realtime/configs/pose_backend_calibration.yaml
```

### 7.1 `rknn_backend.py`

职责：

- 加载 `RKNNLite`。
- 加载 `.rknn` 模型。
- 初始化 `NPU_CORE_0_1_2`。
- 复用同一个模型实例。
- 输入 OpenCV BGR frame。
- 输出 raw inference result 和推理耗时。
- 提供 `release()` 释放资源。

错误规则：

- `POSE_BACKEND=rknn`：初始化失败直接报错。
- `POSE_BACKEND=auto`：初始化失败把错误交给 selector，允许 fallback。

### 7.2 `yolov8_pose_postprocess.py`

职责：

- 尽量从官方 `rknn_model_zoo/examples/yolov8_pose/python/yolov8_pose.py` 迁移：
  - letterbox
  - decode
  - NMS
  - keypoint 解析
  - 标注绘制
- 不自行重写 decode/NMS/keypoint 解析算法。
- 保留官方输出语义，减少后处理错误。

输出结构建议：

```python
{
    "boxes": [...],
    "scores": [...],
    "keypoints": [...],  # COCO17, restored to original frame pixel coordinates
}
```

### 7.3 `pose_frame_adapter.py`

职责：

- 把 RKNN COCO17 转成 `rehab_keypoints`。
- 完成坐标还原和 0~1 归一化。
- 统计左右侧关键点可见度。
- 根据 `side_mode=auto|left|right` 选择侧别。
- 计算髋-膝-踝角度。
- 输出和当前 MediaPipe 路径等价的 `current_frame_data`。

RKNN 第一版只使用 2D 角度：

```text
selected_source = rknn_2d_image
included_angle_3d = null
target_angle_3d = null
```

### 7.4 `smoke_test_rknn_pose.py`

职责：

- 独立验证 RKNN 后端。
- 不启动 8082。
- 支持静态图片和摄像头。

示例命令：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/models/yolov8n-pose.rknn \
  --image /home/elf/yolov8_pose/model/bus.jpg \
  --out outputs/rknn_pose_smoke.jpg
```

输出：

```text
backend
fps
person_count
selected_person_reason
selected_side
selected_angle
rehab_keypoints
annotated image path
```

### 7.5 `replay_compare.py`

职责：

- 使用同一段视频分别跑 MediaPipe 和 RKNN。
- 对比两条后端是否会破坏三动作训练逻辑。

示例命令：

```bash
python3 vision/rknn_pose/replay_compare.py \
  --video samples/rehab_demo.mp4 \
  --rknn-model /home/elf/models/yolov8n-pose.rknn \
  --action seated_knee_extension \
  --out outputs/replay_compare
```

输出：

```text
角度曲线 CSV
角度曲线 PNG
峰值帧对比图
计数结果对比 JSON
FPS 对比
CPU load 对比
NPU load 对比
关键点可见度对比
```

## 8. 多人选择策略

不要只选最高 confidence。

默认训练者选择公式：

```text
score =
  0.45 * bbox_area_score
+ 0.35 * keypoint_visibility_score
+ 0.20 * center_score
```

定义：

- `bbox_area_score`：人体框面积 / frame 面积，归一化后 clamp 到 0~1。
- `keypoint_visibility_score`：肩、髋、膝、踝关键点平均置信度。
- `center_score`：人体框中心距离画面中心越近越高。

状态输出：

```json
{
  "person_count": 3,
  "selected_person_reason": "selected largest centered visible person",
  "selected_person_score": 0.82
}
```

多人干扰严重条件：

```text
person_count > 1
且前两名 person score 差距 < 0.10
```

此时页面提示：

```text
请保持训练者单独入镜
```

训练策略：

- 可继续显示画面。
- 低置信或选择不稳定时不计数。
- 不要因为多人干扰误触发有效 rep。

## 9. 8082 主流程接入要求

`prescription/banzi/record_prescription_http.py` 只做最小接入。

要求：

- MediaPipe 路径保持现状。
- RKNN 路径调用 `rknn_backend + pose_frame_adapter`。
- 两条路径最终都生成同形态 `current_frame_data`。
- `realtime_session.process_frame(current_frame_data, selected_rule)` 不改。
- `state.frames.append(current_frame_data)` 不改。
- `/stream.mjpg` 输出当前后端标注图。

保存医生模板时增加元信息：

```json
{
  "pose_backend": "rknn",
  "pose_backend_version": "rknn_yolov8_pose",
  "pose_keypoint_schema": "coco17_to_rehab_v1",
  "rknn_model_path": "/home/elf/models/yolov8n-pose.rknn"
}
```

训练开始时检查 active template：

- 模板后端和当前后端一致：允许开始。
- 不一致且 `ALLOW_POSE_BACKEND_MISMATCH=0`：阻止开始，提示重新录模板。
- 不一致且 `ALLOW_POSE_BACKEND_MISMATCH=1`：允许开始，但页面显示警告。

原因：

MediaPipe 和 RKNN 的关键点、可见度、抖动特性不同。医生模板和患者训练混用后端可能导致计数和评估不稳定。

## 10. 异步 Worker 约束

摄像头采集、姿态推理、MJPEG 输出、训练状态更新尽量解耦。

建议线程结构：

```text
camera_capture_worker
  -> frame_queue maxsize=1

pose_inference_worker
  -> latest_pose_result
  -> latest_current_frame_data
  -> latest_annotated_frame

mjpeg_stream_handler
  -> 只读取 latest_annotated_frame

training_update_loop
  -> 只读取 latest_current_frame_data
```

队列规则：

```python
Queue(maxsize=1)
```

当队列满时：

```text
丢弃旧帧，保留最新帧。
实时性优先，禁止延迟堆积。
```

要求：

- TTS 不阻塞姿态推理。
- MJPEG 输出不阻塞姿态推理。
- RKNN 单帧推理偶发变慢时，训练状态宁可少处理一帧，也不能积压几秒延迟。
- `pose_fps` 统计实际推理 FPS，不等同摄像头 FPS。

## 11. Backend Calibration 预留

不复制三套动作 YAML。

但允许为不同 pose backend 配置轻量校准参数：

```text
visibility_min
smoothing_window
angle_offset
```

新增可选配置：

```text
realtime/configs/pose_backend_calibration.yaml
```

示例：

```yaml
mediapipe:
  visibility_min: 0.55
  smoothing_window: 5
  angle_offset:
    seated_knee_extension: 0.0
    standing_hamstring_curl: 0.0
    sit_to_stand: 0.0

rknn:
  visibility_min: 0.30
  smoothing_window: 5
  angle_offset:
    seated_knee_extension: 0.0
    standing_hamstring_curl: 0.0
    sit_to_stand: 0.0
```

第一版：

- 可以先读取配置。
- `angle_offset` 默认全部 `0.0`。
- 不急着调阈值。

后续：

- 用真实录制样本和 replay compare 结果微调。

## 12. 页面和系统状态要求

页面系统状态需要显示：

```text
pose_backend
requested_backend
fallback_used
selected_source
pose_fps
person_count
npu_core_loads
rknn_model_path
backend_error_message
selected_person_reason
```

建议 `/status` 或 `/api/system/status` 返回：

```json
{
  "pose_backend": "rknn",
  "requested_backend": "auto",
  "fallback_used": false,
  "selected_source": "rknn_2d_image",
  "pose_fps": 18.6,
  "person_count": 1,
  "selected_person_reason": "selected largest centered visible person",
  "npu_core_loads": {
    "Core0": "12%",
    "Core1": "8%",
    "Core2": "10%"
  },
  "rknn_model_path": "/home/elf/models/yolov8n-pose.rknn",
  "backend_error_message": null
}
```

fallback 时：

```json
{
  "pose_backend": "mediapipe",
  "requested_backend": "auto",
  "fallback_used": true,
  "backend_error_message": "RKNN model not found: /bad/path/yolov8n-pose.rknn"
}
```

## 13. 测试计划

### 13.1 默认 MediaPipe 回归

命令：

```bash
python3 prescription/banzi/record_prescription_http.py
```

成功标准：

- 不需要 RKNN 模型也能启动。
- 当前 CPU Demo 行为不变。
- `/doctor` 正常。
- `/train` 正常。
- 三动作 playlist 正常。

### 13.2 强制 RKNN

命令：

```bash
POSE_BACKEND=rknn \
RKNN_POSE_MODEL=/home/elf/models/yolov8n-pose.rknn \
python3 prescription/banzi/record_prescription_http.py
```

成功标准：

- 页面显示 `pose_backend=rknn`。
- 实时画面有骨架。
- `person_count` 正常。
- `pose_fps` 正常。
- `npu_core_loads` 有非 0 波动。
- 可以录医生模板。
- 可以跑三动作训练。
- 可以保存 attempt。
- 可以生成 evaluate report。

### 13.3 Auto Fallback

命令：

```bash
POSE_BACKEND=auto \
RKNN_POSE_MODEL=/bad/path/yolov8n-pose.rknn \
python3 prescription/banzi/record_prescription_http.py
```

成功标准：

- 页面显示 `requested_backend=auto`。
- 页面显示 `fallback_used=true`。
- 页面显示 RKNN 错误信息。
- 实际后端为 MediaPipe。
- 主 Demo 不崩溃。

### 13.4 坐标还原测试

构造一张带 letterbox padding 的测试图。

成功标准：

- RKNN keypoints 先还原到原始 frame。
- `rehab_keypoints.x/y` 在 0~1。
- 角度使用还原坐标。
- 不使用 input 640 坐标。

### 13.5 多人干扰测试

使用多人图片或视频。

成功标准：

- 页面显示 `person_count`。
- 页面显示 `selected_person_reason`。
- 训练者在画面中心时能稳定选中。
- 多人 score 接近时提示“请保持训练者单独入镜”。

### 13.6 Replay Compare

同一段三动作视频分别跑 MediaPipe 和 RKNN。

成功标准：

- 三动作计数结果一致或差异可解释。
- 角度峰值帧接近。
- RKNN 不产生明显延迟堆积。
- RKNN 后端不会破坏 attempt 保存和 evaluate report。

## 14. 实施顺序建议

建议新对话按这个顺序做：

1. 只实现 `vision/rknn_pose/` 里的独立 RKNN 后端和 smoke test。
2. 验证静态图和摄像头帧输出 `rehab_keypoints`。
3. 验证坐标还原、多人选择、person_count。
4. 再接入 `pose_backend_selector.py`。
5. 再接入 `record_prescription_http.py` 的 RKNN 分支。
6. 先只跑 `/doctor` 录一个模板。
7. 再跑 `/train` 单动作。
8. 最后跑三动作 playlist。
9. 再做 replay compare。
10. 最后根据样本微调 calibration。

## 15. Assumptions

- 官方 RKNN YOLOv8-Pose C demo 已通过。
- NPU load 已确认有非 0 波动。
- 第一版 RKNN 只使用 2D 姿态，不提供 MediaPipe world landmarks。
- 演示时尽量保证训练者单独入镜。
- 动作阈值 YAML 继续复用。
- 后端差异通过轻量 calibration 处理。
- 当前目标是稳定复刻现有 CPU Demo 效果，再逐步优化 NPU 速度和精度。
