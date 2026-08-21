# RTMDet + RTMPose NPU 使用说明

本文说明如何在 RK3588 上使用 `rknn/rtmdet_fp16.rknn` 和
`rknn/rtmpose_fp16.rknn` 运行 NPU 姿态识别，并接入现有 8082 训练台。

本版不改 UI，不改 `/doctor`、`/train`、`/ai` 的操作流程。NPU 只替换姿态
识别来源，后面的录模板、实时计数、纠错、保存 attempt、生成 report 和
`/ai` 复盘仍走原来的 CPU 版流程。

## 1. 文件作用

- `vision/rknn_pose/rtmdet_rtmpose_backend.py`
  - 新增双模型 NPU 后端。
  - RTMDet 检测人体框，RTMPose 输出 COCO17 关键点。
  - 输出格式对齐现有 `adapt_rknn_pose_frame()`。
- `vision/rknn_pose/rknn_backend.py`
  - 统一 RKNN 入口。
  - `RKNN_POSE_PIPELINE=rtmdet_rtmpose` 时走双模型。
  - `RKNN_POSE_PIPELINE=rtmpose_fixed` 时跳过 RTMDet，直接使用固定 ROI 跑 RTMPose。
  - 不设置该变量时仍保留旧 `yolov8_pose` 支线。
- `vision/rknn_pose/smoke_test_rknn_pose.py`
  - 独立 smoke test，不启动 8082。
- `realtime/configs/pose_backend_calibration.yaml`
  - NPU 关键点阈值、平滑窗口和目标腿稳定参数。
- `prescription/banzi/record_prescription_http.py`
  - 8082 主入口，现有 `POSE_BACKEND=rknn` 可使用新双模型后端。

## 2. 上板文件

需要传到 RK3588 的项目文件：

```text
vision/rknn_pose/rknn_backend.py
vision/rknn_pose/rtmdet_rtmpose_backend.py
vision/rknn_pose/smoke_test_rknn_pose.py
vision/rknn_pose/pose_frame_adapter.py
prescription/banzi/record_prescription_http.py
prescription/banzi/camera_preflight.py
vision/rknn_pose/check_rknn_runtime.py
scripts/start_8082_rknn_rtmdet_rtmpose.sh
scripts/start_8082_rknn_rtmpose_fixed.sh
scripts/check_rtmdet_light_model.sh
scripts/install_rknnrt_system.sh
scripts/restore_rknnrt_system.sh
realtime/configs/pose_backend_calibration.yaml
docs/rtmdet_rtmpose_npu_usage.md
docs/rknn_pose_npu_usage.md
```

需要传到 RK3588 的模型文件：

```text
rknn/rtmdet_fp16.rknn
rknn/rtmpose_fp16.rknn
```

不要覆盖这些板端运行态数据：

```text
runtime/active_templates.json
prescription/docs/results/
evaluate/reports/
prescription/banzi/static/
```

## 3. 依赖检查

板端 Python 需要能导入：

```bash
python3 -c "import cv2, numpy, yaml; from rknnlite.api import RKNNLite; print('ok')"
```

如果 `rknnlite` 导入失败，先安装和板端 runtime 匹配的
`rknn-toolkit-lite2`。

## 4. Smoke Test

先确认摄像头可打开：

```bash
cd /home/elf/project/project_system

RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=480 \
python3 prescription/banzi/camera_preflight.py --device auto --frames 300
```

如果 `auto` 找到了可用设备，正式启动时建议用预检输出里的稳定设备路径。
USB 摄像头优先用 `/dev/v4l/by-id/...index0`，例如：

```bash
RK_CAMERA_DEVICE=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
```

不要直接套用 `RK_CAMERA_DEVICE=0`，因为 RK3588 上 `/dev/video0` 往往是板载
ISP/MIPI 节点，不一定是 USB 摄像头。

再跑 NPU 姿态 smoke test：

```bash
cd /home/elf/project/project_system

python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline rtmdet_rtmpose \
  --det-model rknn/rtmdet_fp16.rknn \
  --pose-model rknn/rtmpose_fp16.rknn \
  --camera auto \
  --out outputs/rtmdet_rtmpose_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person
```

成功标准：

- 输出图片能看到人体框和 COCO17 骨架。
- JSON 里 `pipeline` 为 `rtmdet_rtmpose`。
- `person_count >= 1`。
- `postprocess_error` 为 `null`。
- `rtmdet_compatible` 为 `true`，说明检测头仍是当前代码支持的 cls/bbox 输出。
- 推理时 NPU load 有波动。

## 4. 轻量 RTMDet 候选模型验证

如果拿到 `RTMDet-nano` 或 `RTMDet-tiny` 的 RKNN 文件，先只替换检测模型，
继续复用当前 `rtmpose_fp16.rknn`。不要直接改 8082 启动脚本上线。

```bash
cd /home/elf/project/project_system
chmod +x scripts/check_rtmdet_light_model.sh

RKNN_DET_MODEL=rknn/rtmdet_tiny.rknn \
RKNN_RTMPOSE_MODEL=rknn/rtmpose_fp16.rknn \
RK_CAMERA_DEVICE=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
scripts/check_rtmdet_light_model.sh
```

通过标准：

- `ok` 为 `true`。
- `rtmdet_compatible` 为 `true`。
- `det_inference_ms <= 60`。
- `total_pose_ms <= 100`。
- `person_count >= 1`，输出图 `outputs/rtmdet_light_smoke.jpg` 有人体框和骨架。

如果 `rtmdet_compatible=false` 或 `postprocess_error` 提示找不到 cls/bbox heads，
说明模型输出头和当前 `pair_rtmdet_outputs()` / `postprocess_rtmdet()` 不兼容。
这种情况不要强行跑 8082，需要先改检测后处理。

## 5. RTMPose fixed ROI 单模型路线

比赛训练位固定、画面里只有一个患者时，可以跳过 RTMDet，每帧直接裁固定人体区域
给 RTMPose。默认固定框按 `640x480` 画面设置：

```text
RKNN_RTMPOSE_FIXED_BBOX=80,20,560,470
```

先跑 smoke test：

```bash
cd /home/elf/project/project_system

python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline rtmpose_fixed \
  --pose-model rknn/rtmpose_fp16.rknn \
  --fixed-bbox 80,20,560,470 \
  --camera /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  --out outputs/rtmpose_fixed_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person \
  --max-total-ms 100
```

再启动 8082：

```bash
cd /home/elf/project/project_system
chmod +x scripts/start_8082_rknn_rtmpose_fixed.sh
scripts/start_8082_rknn_rtmpose_fixed.sh
```

等价完整环境变量：

```bash
POSE_BACKEND=rknn \
RKNN_POSE_PIPELINE=rtmpose_fixed \
RKNN_RTMPOSE_MODEL=rknn/rtmpose_fp16.rknn \
RKNN_RTMPOSE_FIXED_BBOX=80,20,560,470 \
RK_CAMERA_OPEN_MODE=opencv \
RK_CAMERA_GST_FORMAT=MJPG \
RK_CAMERA_FPS=30 \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=480 \
RKNN_STREAM_WIDTH=640 \
RKNN_STREAM_HEIGHT=360 \
RKNN_PREVIEW_FPS=3 \
REHAB_KEYFRAME_EVERY_N=3 \
RKNN_FAST_PREVIEW=1 \
RKNN_FAST_FRAME_DATA=1 \
RKNN_RTMPOSE_DRAW=0 \
RKNN_DRAW_FIXED_BBOX=1 \
RKNN_FIXED_STRICT_LEG_VISIBILITY=1 \
RKNN_FIXED_LEG_VISIBILITY_THRESHOLD=0.30 \
RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD=0.35 \
RK_CAMERA_FAIL_DISPLAY_THRESHOLD=30 \
python3 prescription/banzi/record_prescription_http.py
```

RTMPose fixed 预览默认显示固定 bbox；如果目标侧髋/膝/踝不完整，`quality_ok=false` 且提示 `请让髋、膝、踝完整入镜`。

验收标准：

- `/status` 中 `rknn_pipeline` 为 `rtmpose_fixed`。
- `model_path` 只包含 `rtmpose_fp16.rknn`，不加载 `rtmdet_fp16.rknn`。
- `outputs/rtmpose_fixed_smoke.jpg` 有骨架，髋、膝、踝在固定框内。
- 切到这条 NPU 路线后，必须进 `/doctor` 重新录三个动作模板。

查看外层耗时：

```bash
curl -s http://127.0.0.1:8082/status \
  | python3 -m json.tool \
  | grep -E "camera_status|camera_display_failures|camera_consecutive_read_failures|camera_capture_fps|camera_read_ms|pose_fps|pose_worker_idle_ms|inference_ms|total_pose_ms|pose_loop_ms|pose_process_ms|rknn_infer_call_ms|rknn_action_context_ms|rknn_adapt_ms|rknn_keypoint_copy_ms|rknn_fixed_visibility_guard_ms|rknn_frame_data_ms|rknn_frame_prelude_ms|rknn_angle_smooth_ms|rknn_threshold_ms|rknn_current_frame_data_ms|rknn_realtime_frame_data_ms|rknn_side_view_ms|rknn_draw_ms|realtime_process_ms|keyframe_encode_ms|stream_resize_ms|jpeg_encode_ms|state_update_ms|fixed_bbox|target.*visibility|visibility_min|visibility_avg|person_box|quality_message|missing_keypoints|frame_queue_drops|pose_worker_error|postprocess_error"
```

如果患者出框，优先调整 `RKNN_RTMPOSE_FIXED_BBOX`，格式为 `x1,y1,x2,y2`。

## 5.1 推荐：YOLO 人体框 + RTMPose 关键点路线

如果你的目标不是“固定机位调试”，而是要像 MediaPipe 那样稳定给后续比对提供骨架点，优先用 `yolov8_rtmpose`：

```bash
scripts/start_8082_rknn_yolov8_rtmpose.sh
```

这个路线先用 YOLOv8 找人体框，再把人体框送入 RTMPose。它比大 fixed bbox 更适合做训练比对，因为骨架点通常会更贴近人体，`target_leg_visibility` 也更容易过。

验收时同样看：

- `/status` 中 `rknn_pipeline` 为 `yolov8_rtmpose`。
- `person_box_quality_ok` 为 `true`。
- `quality_ok` 为 `true`。
- `rehab_keypoints` 里有完整的 `hip / knee / ankle / shoulder` 点。
- `/train` 里的角度和 `target_angle_smoothed` 会随着动作变化。

## 6. 启动 8082 NPU 版

流畅优先启动命令：

```bash
cd /home/elf/project/project_system

POSE_BACKEND=rknn \
RKNN_POSE_PIPELINE=rtmdet_rtmpose \
RKNN_DET_MODEL=rknn/rtmdet_fp16.rknn \
RKNN_RTMPOSE_MODEL=rknn/rtmpose_fp16.rknn \
RK_CAMERA_DEVICE=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
RK_CAMERA_OPEN_MODE=opencv \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=480 \
RKNN_STREAM_WIDTH=640 \
RKNN_STREAM_HEIGHT=360 \
RKNN_MAX_POSE_PERSONS=1 \
RKNN_DET_SCORE_THRES=0.50 \
RKNN_POSE_KEYPOINT_THRES=0.20 \
RKNN_DET_INTERVAL=8 \
RKNN_DET_CACHE_SECONDS=1.5 \
RKNN_DET_NMS_PRE=100 \
RKNN_PERSON_SELECT=largest_center \
python3 prescription/banzi/record_prescription_http.py
```

## 10. RKNN runtime / smooth preview quick fix

If the board log says the RKNN model version does not match the runtime version, do not only check
`python3 -m pip show rknn-toolkit-lite2`. That shows the Python package version, while the actual
runtime may still be an older `librknnrt.so` loaded from the system.

Run this on the board:

```bash
cd /home/elf/project/project_system
python3 vision/rknn_pose/check_rknn_runtime.py
```

If `ldd` or the 8082 startup log still shows an old system `librknnrt.so`, replace the
system runtime. Do not rely on a project-local `LD_LIBRARY_PATH` for the final deployment.

This project also includes a helper startup script with these defaults:

```bash
cd /home/elf/project/project_system
chmod +x scripts/start_8082_rknn_rtmdet_rtmpose.sh
scripts/start_8082_rknn_rtmdet_rtmpose.sh
```

For smoother preview with the large FP16 RTMDet + RTMPose models, use:

```bash
RKNN_DET_INTERVAL=8
RKNN_DET_CACHE_SECONDS=1.5
RKNN_POSE_INTERVAL=2
RKNN_POSE_CACHE_SECONDS=1.0
RKNN_MIN_PERSON_BOX_HEIGHT_RATIO=0.42
RKNN_MIN_PERSON_BOX_AREA_RATIO=0.08
```

`RKNN_POSE_INTERVAL` reused frames are preview-only. They draw the last skeleton but do not enter
doctor template recording, attempt saving, or realtime rep counting.

## 11. Install system RKNN runtime 2.3.2

If the board still prints `librknnrt version: 2.1.0`, replace the system runtime after
backing it up.

On Windows PowerShell, copy only the aarch64 runtime to the board:

```powershell
scp "E:\王义龙大学\网站上下载的\rknn-toolkit2-2.3.2\rknpu2\runtime\Linux\librknn_api\aarch64\librknnrt.so" elf@192.168.137.232:/home/elf/librknnrt.so
```

On the board:

```bash
cd /home/elf/project/project_system
chmod +x scripts/install_rknnrt_system.sh scripts/restore_rknnrt_system.sh scripts/start_8082_rknn_rtmdet_rtmpose.sh

strings /home/elf/librknnrt.so | grep -i "librknnrt version"
scripts/install_rknnrt_system.sh /home/elf/librknnrt.so
```

Then restart 8082:

```bash
scripts/start_8082_rknn_rtmdet_rtmpose.sh
```

The startup log must show `librknnrt version: 2.3.2`, and must not show runtime
`2.1.0`. Check speed in another terminal:

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool | grep -E "det_inference_ms|pose_inference_ms|total_pose_ms|rknn_pose_reused"
```

If the system runtime replacement breaks RKNN, restore the latest backup:

```bash
cd /home/elf/project/project_system
scripts/restore_rknnrt_system.sh
```

浏览器仍访问：

```text
http://板子IP:8082
```

页面流程和 CPU 版一致：

1. 进入 `/doctor`，用 NPU 后端重新录入三个医生模板。
2. 进入 `/train`，开始完整训练。
3. 检查三动作连续切换、实时计数、纠错、TTS 和 report 生成。
4. 进入 `/ai`，确认训练后复盘仍可用。

注意：CPU 模板和 NPU 模板按后端分开保存，NPU 演示必须重新录模板。

## 6. 常用参数

```bash
RKNN_DET_SCORE_THRES=0.50      # 检人阈值，弱光误检多时继续提高
RKNN_DET_NMS_THRES=0.45        # 检人 NMS
RKNN_POSE_KEYPOINT_THRES=0.20  # 关键点质量阈值，骨架丢点时可降到 0.15
RKNN_MAX_POSE_PERSONS=1        # 流畅优先，只对一个训练者做人姿态
RKNN_DET_INTERVAL=8            # 每 8 帧跑一次 RTMDet，中间帧复用检测框
RKNN_DET_CACHE_SECONDS=1.5     # 检测框最多复用 1.5 秒
RKNN_DET_NMS_PRE=100           # 降低检测后处理候选数量
RK_CAMERA_OPEN_MODE=opencv     # 默认 OpenCV；卡顿时可改成 gstreamer 试低延迟采集
RK_CAMERA_GST_FORMAT=MJPG      # GStreamer 模式默认按 MJPEG 摄像头打开
RKNN_PERSON_SELECT=largest_center  # 优先选面积大、靠中心、像人体的框
RKNN_INPUT_LAYOUT=nchw         # 默认与当前模型一致
RKNN_RTMDET_HEAD_LAYOUT=auto   # 自动识别 RTMDet 输出布局
RKNN_RTMDET_BBOX_DECODE_MODE=auto
RKNN_RTMPOSE_SIMCC_SCORE_MODE=sqrt
```

如果骨架偶尔丢点，可以先降一点关键点阈值：

```bash
RKNN_POSE_KEYPOINT_THRES=0.15
```

如果误检的人框太多或卡顿，先提高检人阈值：

```bash
RKNN_DET_SCORE_THRES=0.60
```

如果画面仍然卡，先增大检测间隔：

```bash
RKNN_DET_INTERVAL=8
```

## 7. 卡顿排查

`/status` 和系统状态里会保留分段耗时：

- `det_inference_ms`：RTMDet NPU 推理耗时。
- `pose_inference_ms`：RTMPose NPU 推理耗时。
- `det_postprocess_ms`：检测后处理耗时。
- `pose_postprocess_ms`：SimCC 解码耗时。
- `jpeg_encode_ms`：页面 MJPEG 编码耗时。
- `total_pose_ms`：整帧姿态耗时。

判断方法：

- `pose_inference_ms` 高：确认 `RKNN_MAX_POSE_PERSONS=1`，避免多人体重复跑 RTMPose。
- `jpeg_encode_ms` 高：保持 `RKNN_STREAM_WIDTH=640`、`RKNN_STREAM_HEIGHT=360`。
- `det_inference_ms` 高：增大 `RKNN_DET_INTERVAL`，让 RTMDet 少跑几次。
- `det_postprocess_ms` 高：提高 `RKNN_DET_SCORE_THRES` 或降低 `RKNN_DET_NMS_PRE`。
- `camera_read_ms` 高或 `camera_capture_fps` 低：先查摄像头/USB 链路，或尝试
  `RK_CAMERA_OPEN_MODE=gstreamer`。
- NPU load 没波动：检查 RKNNLite/runtime 是否匹配，模型路径是否正确。

## 8. 回退 CPU

默认不设置 `POSE_BACKEND` 就是 CPU/MediaPipe：

```bash
cd /home/elf/project/project_system

RK_CAMERA_DEVICE=auto \
RK_CAMERA_WIDTH=640 \
RK_CAMERA_HEIGHT=360 \
python3 prescription/banzi/record_prescription_http.py
```

如果想让 NPU 失败时自动回退 CPU：

```bash
POSE_BACKEND=auto \
RKNN_POSE_PIPELINE=rtmdet_rtmpose \
RKNN_DET_MODEL=rknn/rtmdet_fp16.rknn \
RKNN_RTMPOSE_MODEL=rknn/rtmpose_fp16.rknn \
python3 prescription/banzi/record_prescription_http.py
```
