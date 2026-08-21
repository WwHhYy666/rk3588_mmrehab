# RKNN/NPU 姿态识别第一版使用说明

> 新说明：当前实时训练主推 NPU 路线是 `YOLOv8n-Pose` 单模型管线，
> 使用 `/home/elf/models/yolov8n-pose.rknn`。`RTMDet + RTMPose`
> 双模型管线已跑通，但 FP16 RTMDet 太慢，只保留为对照和轻量检测器候选验证。
> YOLO 细节请优先阅读 `docs/yolov8_pose_npu_usage.md`。

本文说明当前 RKNN/NPU Pose 接入后，每个文件的作用、推荐使用流程、测试命令，以及哪些文件需要传到 RK3588 板子上。

## 1. 当前原则

- 默认后端仍是 `mediapipe`，不影响原来的 8082 Demo。
- `rknn` 是可选后端，通过环境变量启用。
- `auto` 会优先尝试 RKNN，失败后回退到 MediaPipe。
- RKNN 第一版只使用 2D 图像角度，不提供 3D。
- RKNN 的 `rehab_keypoints.*.z` 为 `null`，并带 `z_valid=false`。
- RKNN 当前要求侧身固定机位，默认单人模式，只从检测结果里选择一个训练者。
- 当前已适配 RKNN YOLOv8-Pose split 输出和 `[1,17,3,8400]` keypoint shape，正常时诊断应显示 `rknn_decoder=model_zoo_split`。
- 当前已加入目标腿锁定和关键点稳定化，但 NPU 骨架仍可能比 MediaPipe CPU 抖；稳定演示优先走 CPU，NPU 作为并行调优路线。
- 关键点质量不足、缺肩/髋/膝/踝或后处理诊断异常时，页面可以显示画面，但不会计数。

## 2. 文件含义

### `vision/pose_backend_selector.py`

负责选择姿态后端。

- `POSE_BACKEND=mediapipe`：只走 MediaPipe，不检查 RKNN 模型。
- `POSE_BACKEND=rknn`：强制 RKNN，失败直接报错，方便排查 NPU。
- `POSE_BACKEND=auto`：优先 RKNN，失败回退 MediaPipe。

返回字段里区分：

- `requested_backend`：用户请求的后端。
- `actual_backend`：实际运行的后端。
- `fallback_used`：auto 是否发生回退。
- `backend_error_message`：回退或失败原因。

模板后端一致性检查看 `actual_backend`。active template 已按后端分组保存，`mediapipe` 和 `rknn` 各自保留一套医生模板，切换后端不会互相覆盖。

### `vision/rknn_pose/rknn_backend.py`

封装 RKNNLite 后端。

职责：

- 加载 `.rknn` 模型。
- 初始化 RK3588 NPU。
- 对 OpenCV BGR frame 做推理。
- 调用 YOLOv8-Pose 后处理。
- 返回标注图、detections、推理耗时和 FPS。

常用环境变量：

```bash
RKNN_POSE_MODEL=/home/elf/models/yolov8n-pose.rknn
RKNN_POSE_INPUT_SIZE=640
RKNN_POSE_CONF_THRES=0.35
RKNN_POSE_NMS_THRES=0.45
RKNN_POSE_KEYPOINT_THRES=0.12
RKNN_KEYPOINT_DECODE_MODE=auto
RKNN_KEYPOINT_ANCHOR_ORDER=auto
RKNN_POSE_TOPK=50
RKNN_POSE_MAX_DET=1
```

排查关键点乱飞时可临时强制：

```bash
RKNN_KEYPOINT_DECODE_MODE=absolute      # 或 anchor_stride / auto
RKNN_KEYPOINT_ANCHOR_ORDER=grid_desc    # 或 output / grid_asc / output_reverse / auto
```

`auto` 会同时尝试常见 anchor 顺序和 decode 模式，并用人体几何分数选择最合理的一组。

### `vision/rknn_pose/yolov8_pose_postprocess.py`

负责 YOLOv8-Pose 的预处理和后处理。

包括：

- letterbox resize/pad。
- 推理输出解析。
- NMS。
- COCO17 keypoints 解析。
- letterbox 坐标还原到原始 frame。
- 标注图绘制。

注意：角度计算和保存都使用还原后的原始 frame 坐标，不能直接使用 640x640 input 坐标。

### `vision/rknn_pose/pose_frame_adapter.py`

负责把 RKNN 输出接成当前康复训练系统能读的帧结构。

包括：

- COCO17 -> `rehab_keypoints`。
- `z=null`、`z_valid=false`。
- 关键点质量判断。
- 缺失关键点提示。
- 单人模式下默认选择训练者。
- 左右侧选择和目标腿锁定。
- RKNN 关键点时间滤波、短暂丢点保持、大跳点拒绝和防串腿。
- 2D 图像角度计算。
- 输出和 MediaPipe 路径同形态的 `current_frame_data`。

关键输出：

- `selected_source="rknn_2d_image"`
- `included_angle_3d=null`
- `target_angle_3d=null`
- `quality_ok`
- `missing_keypoints`
- `quality_message`
- `person_count`
- `selected_person_reason`
- `locked_side`
- `pose_stabilized`
- `held_keypoints`
- `jump_rejected`
- `side_switch_blocked`

### `vision/rknn_pose/smoke_test_rknn_pose.py`

独立 RKNN 测试脚本，不启动 8082。

静态图测试：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/models/yolov8n-pose.rknn \
  --image /home/elf/yolov8_pose/model/bus.jpg \
  --out outputs/rknn_pose_smoke.jpg
```

摄像头测试：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/models/yolov8n-pose.rknn \
  --camera /dev/video21 \
  --out outputs/rknn_pose_camera.jpg
```

输出会打印：

- `actual_backend`
- FPS
- `person_count`
- 关键点质量
- 缺失关键点
- 角度
- 标注图路径

### `vision/rknn_pose/replay_compare.py`

用同一段视频比较 MediaPipe 和 RKNN。

```bash
python3 vision/rknn_pose/replay_compare.py \
  --video samples/rehab_demo.mp4 \
  --rknn-model /home/elf/models/yolov8n-pose.rknn \
  --action seated_knee_extension \
  --out outputs/replay_compare
```

输出：

- `angle_curves.csv`
- `summary.json`
- 如果安装了 matplotlib，会额外输出 `angle_curves.png`

用途：

- 对比两条后端的角度趋势。
- 查看 RKNN 关键点缺失帧。
- 查看 RKNN 关键点抖动、目标腿切换和角度趋势差异。
- 评估 RKNN 是否会破坏当前三动作训练逻辑。

### `realtime/configs/pose_backend_calibration.yaml`

姿态后端轻量校准配置。

当前用于区分 MediaPipe 和 RKNN 的关键点阈值、平滑窗口、动作 offset 和 RKNN 稳定器参数：

- MediaPipe visibility: `0.55`
- RKNN visibility: 默认 `0.12`，可用 `RKNN_POSE_KEYPOINT_THRES` 覆盖
- 三动作 angle offset: `0.0`
- RKNN stabilizer: `max_hold_frames`、`lock_confirm_frames`、`jump_scale` 等

后续可以根据 replay compare 结果微调。

### `prescription/banzi/record_prescription_http.py`

8082 统一训练台主入口。

新增能力：

- 接入 `POSE_BACKEND`。
- MediaPipe 和 RKNN 生成同形态 `current_frame_data`。
- 医生模板和患者 attempt 保存 pose 后端 metadata。
- 训练开始前按当前后端检查 active template；CPU/NPU 模板互不覆盖。
- `/status` 和 `/api/system/status` 返回当前后端、关键点质量、NPU 稳定器诊断和 fallback 信息。

### `prescription/banzi/static/common.js`

系统状态卡片新增：

- Pose Backend
- Keypoint Quality
- Person Count
- fallback error

### `prescription/banzi/static/doctor.js`

医生录模板页面新增显示：

- 当前 `actual_backend`
- requested backend
- fallback 状态
- 关键点质量
- 缺失关键点
- RKNN 2D 侧身固定机位提示

### `prescription/banzi/static/train.js`

患者训练页面新增显示：

- 当前 `actual_backend`
- 关键点质量
- 缺失关键点
- NPU 锁定侧和稳定器状态
- RKNN 2D 侧身固定机位提示

## 3. 推荐使用流程

### 3.1 摄像头基线检查

先确认摄像头硬件和 V4L2 读取没有问题：

```bash
cd /home/elf/project/project_system

RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=360 \
python3 prescription/banzi/camera_preflight.py --device auto
```

成功标准：

```text
opened: True
read_frame: True
```

如果这里失败，先处理摄像头设备号、权限或旧进程占用，不要先排查 RKNN。

### 3.2 默认 MediaPipe 回归

不设置任何 RKNN 环境变量：

```bash
cd /home/elf/project/project_system

RK_CAMERA_DEVICE=auto \
python3 prescription/banzi/record_prescription_http.py
```

浏览器访问：

```text
http://板子IP:8082
```

预期：

- 不需要 `.rknn` 模型。
- `/doctor` 可录模板。
- `/train` 可训练。
- 页面显示 `actual_backend=mediapipe`。

### 3.3 RKNN 独立 smoke test

先确认模型能跑：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/models/yolov8n-pose.rknn \
  --image /home/elf/yolov8_pose/model/bus.jpg \
  --out outputs/rknn_pose_smoke.jpg
```

再测摄像头：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/models/yolov8n-pose.rknn \
  --camera /dev/video21 \
  --out outputs/rknn_pose_camera.jpg
```

如果模型在官方 C demo 目录，使用实际路径：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --model /home/elf/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn \
  --camera /dev/v4l/by-id/usb-iSpring_iSpring_camera-video-index0 \
  --out outputs/rknn_pose_smoke.jpg
```

当前项目版 smoke test 期望看到：

- `rknn_decoder=model_zoo_split`
- `keypoint_decode_mode=anchor_stride` 或 `absolute`
- `keypoint_anchor_order=grid_desc/output/grid_asc/output_reverse`
- `keypoint_geometry_score_range` 不应明显为大负数
- `keypoint_conf_range` 在 `0~1`
- `person_count` 大于 `0`
- `postprocess_error=null`
- 输出图片能看到检测框、关键点和骨架线

### 3.4 强制 RKNN 启动 8082

```bash
cd /home/elf/project/project_system

POSE_BACKEND=rknn \
RKNN_POSE_MODEL=/home/elf/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn \
RK_CAMERA_DEVICE=auto \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=360 \
RKNN_STREAM_WIDTH=640 \
RKNN_STREAM_HEIGHT=360 \
RKNN_POSE_MAX_DET=1 \
python3 prescription/banzi/record_prescription_http.py
```

预期：

- 8082 会先打开摄像头，收到首帧后再懒加载 RKNNLite/NPU runtime。
- 页面显示 `actual_backend=rknn`。
- `/doctor` 重新录三个动作模板。
- `/train` 使用同一个 RKNN 后端训练。
- NPU load 应有非 0 波动。
- 预览默认显示稳定后的训练骨架；如需看原始 RKNN 调试骨架，可临时加 `RKNN_DRAW_RAW_DEBUG_SKELETON=1`。
- 画面 debug 行应能看到 `kpt_mode`、`order`、`geom`；`order` 是最终选择的 keypoint anchor 顺序。
- 诊断卡片应能看到 `NPU 锁定`、`NPU 稳定`、`locked_side`、`pose_stabilized`、`held_keypoints`、`jump_rejected` 等状态。

### 3.5 Auto fallback

```bash
cd /home/elf/project/project_system

POSE_BACKEND=auto \
RKNN_POSE_MODEL=/bad/path/yolov8n-pose.rknn \
python3 prescription/banzi/record_prescription_http.py
```

预期：

- 页面显示 `requested_backend=auto`。
- 页面显示 `actual_backend=mediapipe`。
- 页面显示 `fallback_used=true` 和错误原因。
- 主 Demo 不崩溃。

### 3.6 模板后端一致性

医生模板保存时会记录 `actual_backend`。

训练开始时：

- 系统按当前 `actual_backend` 读取对应后端的 active template。
- `mediapipe` 和 `rknn` 模板互不覆盖；两套都录过后，切换后端不需要重复录制。
- 当前后端缺少某个动作模板时，训练会阻止开始，并提示用当前后端补录对应动作。

强制允许混用：

```bash
ALLOW_POSE_BACKEND_MISMATCH=1 POSE_BACKEND=rknn \
RKNN_POSE_MODEL=/home/elf/models/yolov8n-pose.rknn \
python3 prescription/banzi/record_prescription_http.py
```

不建议演示时混用。

`runtime/active_templates.json` 新格式示例：

```json
{
  "schema_version": 2,
  "by_backend": {
    "mediapipe": {
      "seated_knee_extension": {
        "template_file": "prescription/docs/results/cpu_template.json",
        "actual_backend": "mediapipe"
      }
    },
    "rknn": {
      "seated_knee_extension": {
        "template_file": "prescription/docs/results/npu_template.json",
        "actual_backend": "rknn"
      }
    }
  }
}
```

旧版平铺结构会被新代码自动按模板 metadata 迁移；没有 metadata 的旧模板默认归入 `mediapipe`。

## 4. NPU load 权限持久化

UI 的 NPU 卡片读取：

```text
/sys/kernel/debug/rknpu/load
```

这个 debugfs 节点重启后权限会恢复，手动执行一次 `sudo chmod a+r /sys/kernel/debug/rknpu/load` 不能永久生效。

注意：只给 `load` 文件 `a+r` 还不够，普通用户还需要能穿过父目录 `/sys/kernel/debug` 和 `/sys/kernel/debug/rknpu`，所以 service 里要同时给父目录 `a+rx`。

先查看当前权限：

```bash
sudo ls -ld /sys/kernel/debug /sys/kernel/debug/rknpu /sys/kernel/debug/rknpu/load
```

推荐创建 systemd oneshot 服务：

```bash
sudo tee /etc/systemd/system/rknpu-load-permission.service >/dev/null <<'EOF'
[Unit]
Description=Allow normal user to read RKNN NPU load
After=multi-user.target

[Service]
Type=oneshot
ExecStartPre=/bin/sh -c 'mountpoint -q /sys/kernel/debug || mount -t debugfs debugfs /sys/kernel/debug'
ExecStart=/bin/sh -c 'test -d /sys/kernel/debug && chmod a+rx /sys/kernel/debug; test -d /sys/kernel/debug/rknpu && chmod a+rx /sys/kernel/debug/rknpu; test -e /sys/kernel/debug/rknpu/load && chmod a+r /sys/kernel/debug/rknpu/load'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now rknpu-load-permission.service
```

如果之前已经创建过旧版本 service，改完后运行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart rknpu-load-permission.service
```

立即验证：

```bash
ls -lh /sys/kernel/debug/rknpu/load
cat /sys/kernel/debug/rknpu/load
systemctl status rknpu-load-permission.service --no-pager
```

重启后再次验证：

```bash
cat /sys/kernel/debug/rknpu/load
```

成功标准：普通 `elf` 用户无需再手动 chmod，也能读取 NPU load，UI 不再显示权限不足。

## 5. 传到板子上的文件

必须同步这些项目文件：

```text
vision/pose_backend_selector.py
vision/rknn_pose/yolov8_pose_postprocess.py
vision/rknn_pose/rknn_backend.py
vision/rknn_pose/smoke_test_rknn_pose.py
vision/rknn_pose/replay_compare.py
vision/rknn_pose/pose_frame_adapter.py
prescription/common/active_templates.py
prescription/banzi/record_prescription_http.py
prescription/banzi/static/app.css
prescription/banzi/static/common.js
prescription/banzi/static/doctor.js
prescription/banzi/static/home.js
prescription/banzi/static/train.js
realtime/configs/pose_backend_calibration.yaml
realtime/training_session.py
docs/rknn_pose_npu_usage.md
```

不要从 Windows 侧覆盖板端已有的 `runtime/active_templates.json`。这份文件是运行态模板索引，板端已有模板应由新代码自动兼容；CPU/NPU 两套模板建议直接在板端各录一套。

板端还必须有：

```text
/home/elf/models/yolov8n-pose.rknn
rknn-toolkit-lite2==2.3.2
OpenCV
numpy
PyYAML
MediaPipe
```

官方 demo 可以保留在：

```text
~/yolov8_pose
~/rknn_yolov8_pose_demo
```

它们不属于本项目必传文件，但适合排错时对照。

## 6. 演示建议

1. 先用默认 MediaPipe 确认 8082 仍可用。
2. 用 smoke test 确认 RKNN 图片和摄像头都能输出关键点。
3. 用 `POSE_BACKEND=rknn` 启动 8082。
4. 在 `/doctor` 用侧身固定机位重新录三个模板，NPU 演示建议手动选 `左腿` 或 `右腿`，不要依赖 `auto` 逐帧切换。
5. 在 `/train` 跑完整三动作训练。
6. 如果页面出现关键点缺失、目标腿串腿或角度大跳，先看 `locked_side`、`held_keypoints`、`jump_rejected`、`side_switch_blocked`，再调整机位和选腿。
7. 跑通后再用 `replay_compare.py` 对比 MediaPipe/RKNN 的角度趋势。

## 7. RKNN 低 FPS 诊断

RKNN 模式如果只有 0.x FPS，先不要直接判断是摄像头问题。当前 8082 的系统监控卡片会显示分段耗时：

- `Infer ms`：RKNNLite/NPU 推理耗时。
- `Post ms`：YOLOv8-Pose Python 后处理耗时。
- `JPEG ms`：MJPEG 页面流编码耗时。
- `Pose ms`：预处理、推理、后处理、画框的总耗时。

RKNN 第一版默认按康复单人训练链路配置：采集 `640x360`、页面流 `640x360`、最终只保留 `1` 个 person detection。若关键点不稳，可把采集和页面流调到 `960x540`；`1280x720` 只建议远距离或必须高清预览时使用。

常见判断：

- `Infer ms` 很大：优先检查 RKNN runtime、模型版本、NPU 是否真的跑起来。
- `Post ms` 很大：说明 Python 后处理是瓶颈，可以继续降低 `RKNN_POSE_TOPK`。
- `JPEG ms` 很大：说明页面流编码压力大，可以降低 `RKNN_STREAM_WIDTH` 和 `RKNN_STREAM_HEIGHT`。

本版默认优化：

```bash
RKNN_POSE_TOPK=50
RKNN_POSE_MAX_DET=1
RK_CAMERA_WIDTH=640
RK_CAMERA_HEIGHT=360
RKNN_STREAM_WIDTH=640
RKNN_STREAM_HEIGHT=360
RKNN_PREVIEW_FPS=6
```

如果仍然卡，可以临时降一点：

```bash
POSE_BACKEND=rknn \
RKNN_POSE_MODEL=/home/elf/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn \
RK_CAMERA_DEVICE=auto \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=360 \
RKNN_POSE_TOPK=50 \
RKNN_POSE_MAX_DET=1 \
RKNN_STREAM_WIDTH=640 \
RKNN_STREAM_HEIGHT=360 \
RKNN_PREVIEW_FPS=6 \
python3 prescription/banzi/record_prescription_http.py
```

同时检查 RKNNLite wheel 和 runtime 路径：

```bash
python3 -c "from rknnlite.api import RKNNLite; import rknnlite; print(rknnlite.__file__)"
```

如果终端出现类似 `model version 2.3.2 not match runtime 2.1.0`，说明模型转换版本、Python wheel 或板端 `librknnrt` 版本不一致。官方 C demo 虽然能跑，但 Python RKNNLite 这边仍可能被版本不一致拖慢或异常，建议把模型转换版本、`rknn-toolkit-lite2` wheel、板端 runtime 库统一到同一版本，例如 2.3.2。
