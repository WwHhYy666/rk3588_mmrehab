# 新手快速上手路线：8082 与 RTMPose fixed

这份文档给第一次接手本项目的同学使用。目标不是一次看懂所有代码，而是先知道当前系统跑到哪里、先看哪些文件、怎么验证 RTMPose 版本，以及后面改 UI 和语音助手时应该从哪里下手。

## 1. 当前项目进度

- 主入口是 `readme.md`，建议先看第 1、3、4、7、8 节，了解项目全貌、当前进度、关键文件和下一步。
- 当前 Demo 是 `8082` 统一训练台：
  - `/doctor`：医生录标准动作模板。
  - `/train`：患者实时训练。
  - `/ai`：训练后 AI 复盘、报告问答和可选朗读。
- RTMPose 已接入两种路线：
  - 推荐路线：`yolov8_rtmpose`，YOLOv8 先找人体框，RTMPose 再提取 COCO17 骨架点，适合做成参考作品那种“人体框 + 骨架”效果。
  - 调试路线：`rtmpose_fixed`，固定训练位时手动指定人体框，只适合机位固定且 bbox 贴近人体的场景。
  - 启动脚本：`scripts/start_8082_rknn_rtmpose_fixed.sh`
  - 推荐启动脚本：`scripts/start_8082_rknn_yolov8_rtmpose.sh`
  - 8082 主程序：`prescription/banzi/record_prescription_http.py`
  - RKNN 后端：`vision/rknn_pose/rtmdet_rtmpose_backend.py`
  - 模型文件：`rknn/rtmpose_fp16.rknn`
- 现在已有的是 TTS 播报，不等于已有“小爱式麦克风助手”。麦克风 ASR、唤醒词和连续语音交互仍属于后续功能。

## 2. 新手先看哪些文件

第一优先级：

- `readme.md`：项目地图。
- `docs/unified_training_station.md`：8082 怎么用。
- `docs/rk3588_hdmi_8082_browser_guide.md`：板子接显示屏、打开浏览器流程。
- `docs/rtmdet_rtmpose_npu_usage.md`：RTMPose smoke test、启动和 `/status` 验证命令。

第二优先级：

- `prescription/banzi/record_prescription_http.py`：8082 后端、摄像头、姿态、路由都在这里。
- `prescription/banzi/static/train.js`：训练页面逻辑。
- `prescription/banzi/static/app.css`：页面样式。
- `realtime/training_session.py`：连续训练、计数、TTS 调度。
- `realtime/knee_flexion.py`：动作状态机和每次 rep 是否有效。

暂时别优先看：

- 旧备份文件。
- `__pycache__`。
- 旧 `feedback/tts` 入口。
- 双模型 RTMDet 路线，除非是在排查历史问题。

## 3. 推荐 RTMPose 测试顺序

如果目标是像参考作品一样提取人体骨架点，优先跑 `yolov8_rtmpose`，不要先用大范围 fixed bbox。

先确认模型存在：

```bash
ls -lh /home/elf/models/yolov8n-pose.rknn
ls -lh rknn/rtmpose_fp16.rknn
```

如果 YOLOv8 模型还没有整理，先执行：

```bash
scripts/prepare_yolov8_pose_model.sh
```

先跑 YOLO + RTMPose smoke test，不要一上来进 8082：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline yolov8_rtmpose \
  --model /home/elf/models/yolov8n-pose.rknn \
  --pose-model rknn/rtmpose_fp16.rknn \
  --camera auto \
  --out outputs/yolov8_rtmpose_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person
```

smoke test 出图后，再启动 8082：

```bash
scripts/start_8082_rknn_yolov8_rtmpose.sh
```

只有固定训练位和手动框已经贴近人体时，才跑 fixed ROI：

```bash
python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline rtmpose_fixed \
  --pose-model rknn/rtmpose_fp16.rknn \
  --fixed-bbox 80,20,560,470 \
  --camera auto \
  --out outputs/rtmpose_fixed_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person \
  --max-total-ms 100
```

fixed smoke test 出图后，再启动 8082：

```bash
scripts/start_8082_rknn_rtmpose_fixed.sh
```

浏览器进入：

```text
http://127.0.0.1:8082
```

重点看 `/status`：

```bash
curl -s http://127.0.0.1:8082/status \
  | python3 -m json.tool \
  | grep -E "actual_backend|rknn_pipeline|fixed_bbox|quality_ok|target.*visibility|total_pose_ms|pose_fps|state_update_ms|rknn_fixed_visibility_guard_ms|pose_worker_error|postprocess_error"
```

验收时优先确认：

- `actual_backend` 是 `rknn`。
- 推荐路线的 `rknn_pipeline` 是 `yolov8_rtmpose`；fixed 调试路线的 `rknn_pipeline` 是 `rtmpose_fixed`。
- `yolov8_rtmpose` 应该有贴近人体的检测框；`rtmpose_fixed` 的 `fixed_bbox` 必须与实际人体边缘匹配。
- `quality_ok` 为 `true`，或者不为 true 时能看到明确的 `quality_message`。
- `target_leg_visibility` 中目标侧髋、膝、踝完整可见。
- `total_pose_ms`、`pose_fps`、`state_update_ms` 没有明显异常。

如果患者出框，先调 `RKNN_RTMPOSE_FIXED_BBOX`，格式是：

```bash
RKNN_RTMPOSE_FIXED_BBOX=x1,y1,x2,y2 scripts/start_8082_rknn_rtmpose_fixed.sh
```

## 4. 之后功能怎么排

- 先把 `yolov8_rtmpose` 跑通，再改 UI。否则 UI 改完也不知道问题来自界面还是姿态链路。
- UI 改版主要动：
  - `prescription/banzi/static/app.css`
  - `prescription/banzi/static/train.js`
  - `prescription/banzi/static/doctor.js`
  - `prescription/banzi/static/ai.js`
- 小爱式助手建议分两步做：
  - 第一步：页面文字输入问答 + TTS 朗读，用现有 `/ai` 和 LLM/TTS 链路验证业务逻辑。
  - 第二步：再加麦克风 ASR、唤醒词和连续对话。
- `realtime/configs/rehab_demo_plan.yaml` 是连续训练流程配置，但当前中文文案存在编码显示问题。新手先理解它负责“练哪几个动作、播什么提示、每组休息多久”，暂时不要随手改中文文案。

## 5. 最小成功标准

一次合格的 RTMPose 上手验证应做到：

- 推荐 smoke test 生成 `outputs/yolov8_rtmpose_smoke.jpg`，图里有人体框和关键点。
- `scripts/start_8082_rknn_yolov8_rtmpose.sh` 能启动 8082。
- 板端浏览器能打开 `http://127.0.0.1:8082`。
- `/status` 确认 `actual_backend=rknn` 且 `rknn_pipeline=yolov8_rtmpose`。
- `/doctor` 在 RTMPose 后端下重新录三个 active template。
- `/train` 能开始完整训练，并能看到实时指标、关键点质量、计数和 TTS 状态。
