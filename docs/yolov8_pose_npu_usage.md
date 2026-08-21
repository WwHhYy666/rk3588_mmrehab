# YOLOv8n-Pose RKNN 使用说明

这条路线用于替代当前很慢的 RTMDet + RTMPose 双模型。`rtmdet_fp16.rknn`
单次检测约 343ms，不适合实时训练；YOLOv8n-Pose 是单模型，同时输出人体框和
COCO17 关键点，更适合先把 `/train` 页面跑流畅。

本方案不改 UI，不覆盖医生模板、训练结果和报告。

## 1. 上传文件

Windows PowerShell:

```powershell
scp "D:\rk3588\project\scripts\prepare_yolov8_pose_model.sh" "D:\rk3588\project\scripts\start_8082_rknn_yolov8_pose.sh" elf@192.168.137.232:/home/elf/project/project_system/scripts/
scp "D:\rk3588\project\docs\yolov8_pose_npu_usage.md" elf@192.168.137.232:/home/elf/project/project_system/docs/
```

需要上传到板端项目：

```text
scripts/prepare_yolov8_pose_model.sh
scripts/start_8082_rknn_yolov8_pose.sh
docs/yolov8_pose_npu_usage.md
```

如果板端还没有最新 YOLOv8-pose 后端代码，也一起上传：

```text
vision/rknn_pose/rknn_backend.py
vision/rknn_pose/yolov8_pose_postprocess.py
vision/rknn_pose/smoke_test_rknn_pose.py
vision/rknn_pose/pose_frame_adapter.py
prescription/banzi/record_prescription_http.py
realtime/configs/pose_backend_calibration.yaml
```

不要覆盖这些运行态数据：

```text
runtime/active_templates.json
prescription/docs/results/
evaluate/reports/
prescription/banzi/static/
```

## 2. 准备模型

先在板端找已有 YOLOv8-pose 模型：

```bash
find /home/elf -name "*yolov8*pose*.rknn" 2>/dev/null
```

如果找到了官方 demo 的 `yolov8n-pose.rknn`，执行：

```bash
cd /home/elf/project/project_system
chmod +x scripts/prepare_yolov8_pose_model.sh
bash -n scripts/prepare_yolov8_pose_model.sh scripts/start_8082_rknn_yolov8_pose.sh
scripts/prepare_yolov8_pose_model.sh
```

脚本会把第一个非 `_fp` 的 YOLOv8-pose RKNN 模型复制到：

```text
/home/elf/models/yolov8n-pose.rknn
```

如果找不到模型，需要先用 Rockchip Model Zoo 的 `examples/yolov8_pose`
转换 INT8 版 `yolov8n-pose.rknn`，不要继续用 FP16 大模型。

## 3. Smoke Test

```bash
cd /home/elf/project/project_system

python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline yolov8_pose \
  --model /home/elf/models/yolov8n-pose.rknn \
  --camera /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  --out outputs/yolov8_pose_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person
```

成功标准：

- JSON 里 `pipeline` 是 `yolov8_pose`。
- `person_count >= 1`。
- `postprocess_error` 是 `null`。
- `outputs/yolov8_pose_smoke.jpg` 能看到人体框和骨架。

## 4. 启动 8082

```bash
cd /home/elf/project/project_system
chmod +x scripts/start_8082_rknn_yolov8_pose.sh
scripts/start_8082_rknn_yolov8_pose.sh
```

等价完整命令：

```bash
POSE_BACKEND=rknn \
RKNN_POSE_PIPELINE=yolov8_pose \
RKNN_POSE_MODEL=/home/elf/models/yolov8n-pose.rknn \
RK_CAMERA_DEVICE=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=480 \
RKNN_STREAM_WIDTH=640 \
RKNN_STREAM_HEIGHT=360 \
RKNN_POSE_INPUT_SIZE=640 \
RKNN_POSE_CONF_THRES=0.35 \
RKNN_POSE_NMS_THRES=0.45 \
RKNN_POSE_KEYPOINT_THRES=0.12 \
RKNN_POSE_TOPK=50 \
RKNN_POSE_MAX_DET=1 \
python3 prescription/banzi/record_prescription_http.py
```

浏览器仍访问：

```text
http://板子IP:8082/train
```

## 5. 验证性能

另开终端：

```bash
curl -s http://127.0.0.1:8082/status \
  | python3 -m json.tool \
  | grep -E "rknn_pipeline|model_path|inference_ms|postprocess_ms|total_pose_ms|pose_fps|frame_queue_drops|postprocess_error"
```

期望：

- `rknn_pipeline` 是 `yolov8_pose`。
- `model_path` 是 `/home/elf/models/yolov8n-pose.rknn`。
- 不再加载 `rtmdet_fp16.rknn` 或 `rtmpose_fp16.rknn`。
- `total_pose_ms` 明显低于 RTMDet + RTMPose 的约 350ms。
- `active_template_backend` 是 `rknn`。NPU 模板和 MediaPipe 模板分开保存，切到 NPU 后必须在 `/doctor` 重新录三个动作模板。

实时看 NPU：

```bash
watch -n 0.2 cat /sys/kernel/debug/rknpu/load
```

## 6. 识别效果调参

人必须完整入镜，尤其髋、膝、踝。只露上半身时，康复动作会被标成 invalid，
这是正确行为。

如果乱框或多人干扰：

```bash
RKNN_POSE_CONF_THRES=0.35 RKNN_POSE_MAX_DET=1 scripts/start_8082_rknn_yolov8_pose.sh
```

如果骨架点少：

```bash
RKNN_POSE_KEYPOINT_THRES=0.10 scripts/start_8082_rknn_yolov8_pose.sh
```

当前默认值已经按康复单人训练调成：

```bash
RKNN_POSE_CONF_THRES=0.35
RKNN_POSE_KEYPOINT_THRES=0.12
RKNN_POSE_MAX_DET=1
```

如果速度仍不够：

```bash
RKNN_POSE_TOPK=20 RKNN_POSE_MAX_DET=1 scripts/start_8082_rknn_yolov8_pose.sh
```
