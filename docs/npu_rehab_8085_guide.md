# NPU 8085 完整康复训练链路说明

## 1. 这次新增了什么

这次没有修改稳定的 `8082 + MediaPipe` 展示入口，而是在同一个项目中新增一套独立的 `8085` NPU 康复服务。

8085 保留现有完整业务流程：

```text
医生录入 NPU 标准模板
-> 患者依次完成三动作训练
-> YOLOv5n 检测人体框
-> RTMPose 输出 COCO-17 骨架
-> 实时计数、ROM/TUT 纠错和 TTS
-> 保存 NPU attempt
-> 生成 NPU 报告
-> ONNX CPU 异步完成度评分
-> 释放姿态 RKNN Runtime.destop
-> GLM / 本地 Qwen / 小爱解释报告
```

本地复制进来的 `project_system/` 只用于读取板端现状，本次没有修改其中任何文件。不要把整个 `project_system/` 子目录再次上传到板子。

本次更新把默认检测器从旧的 1.28 GB `yolov5n_nonms_fp.rknn` 切换为约 4.9 MB `yolov5n_raw_fp.rknn`，并新增 `/npu-debug` 页面用于在板端显示屏上先单独验证人体框和17点骨架，再进行医生模板与患者训练验收。

完整 8085 明确独占 `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`，不读取 8082 的 MJPEG。摄像头使用 `1280x720@30`。板端 OpenCV 即使显示 `GStreamer: NO`，8085 也会通过 Python GI 直接创建 GStreamer appsink，优先尝试 `mppjpegdec` 硬件 JPEG 解码，再尝试 `jpegdec`，最后回退 OpenCV/V4L2。姿态线程保留完整 `1280x720` 源图供人体 ROI 取样，网页视频输出为 `960x540`、JPEG 质量 72。YOLOv5n输入保持 `640x640`，RTMPose输入保持 `192x256`，无需重新转换权重。

开机后默认不启动8082、8085或浏览器显示管理器，避免任一路线提前占用摄像头。显示屏桌面保留“康复训练 CPU 8082”和“康复训练 NPU 8085”两个快捷方式；双击时会停止两条旧服务、等待摄像头释放、启动目标服务并打开对应页面。重复双击同一快捷方式会重新拉起该路线，可用于恢复摄像头无画面状态。

8085网页在 `stream_available=true` 时立即连接 `/stream.mjpg`，不再等待第一张JPEG后才建立连接，避免服务端等待客户端、客户端等待首帧的启动死锁。静态JS/CSS使用文件版本号并禁止直接缓存，上传后重启服务即可加载新页面逻辑。

8085默认设置 `RK_CAMERA_FIXED_FPS=1`。摄像头支持 `exposure_auto_priority` 时，启动脚本将其设为0，防止室内自动曝光稳定后从30 FPS降到15 FPS。若画面因此明显变暗，可在systemd环境中设置 `RK_CAMERA_FIXED_FPS=0` 恢复摄像头默认曝光策略，并优先增加现场照明。

8085 默认启用 `RKNN_FAST_PREVIEW=1`、`RKNN_FAST_FRAME_DATA=1` 和 `RKNN_ASYNC_PIPELINE=1`。摄像头采集、顺序姿态推理、网页 JPEG 渲染和训练关键帧编码分别运行；采集与渲染队列只保留最新一帧，关键帧任务按动作 generation 有序完成，慢渲染和关键帧写盘不会反向阻塞姿态线程。训练状态机、角度、计数、纠错、模板、attempt 和报告数据保持不变。

第二阶段把人体框、COCO-17 骨架和 JPEG 编码全部放到渲染线程。没有 `/stream.mjpg` 客户端时直接跳过缩放、绘制和 JPEG；网页默认最多 `20 FPS`，限流窗口内持续合并为最新任务，避免摄像头轻微抖动时错误退化成约一半帧率。关键帧只在动作指标刷新最佳值时复制低分辨率 BGR，rep 结束后异步编码一次。若板端 `capture_to_stream_age_ms` P95超过250ms，可设置 `RKNN_STREAM_FPS=15`回退。

单人康复低延迟调度默认使用：

```text
RKNN_ADAPTIVE_DETECTOR=1       YOLO 按首次捕获、750ms校准和姿态质量下降触发
RKNN_DET_SCORE_THRES=0.80      现场杂乱背景下提高人体框门槛；1.5秒缓存缓冲短时漏检
RKNN_DET_RETRY_SECONDS=0.25    漏检或异常后限制重试频率，避免每帧连续跑YOLO
RKNN_DET_CACHE_SECONDS=1.5    短暂漏检时复用最近人体框
RTMPose                        每个姿态帧仍然运行
RKNN_YOLOV5_PERSON_ONLY_FAST=1 只计算康复需要的 person 类
RKNN_YOLOV5_BACKEND_DRAW=0    关闭原始骨架，改画稳定后的康复骨架
RKNN_STABILIZER_MAX_HOLD_FRAMES=8 短暂姿态漏检不立即清空骨架
RKNN_DISPLAY_MAX_HOLD_FRAMES=4 脸部、肘和腕短暂低置信度时保持显示
RKNN_DISPLAY_BBOX_HOLD_FRAMES=6 人体框短暂漏检时保持并平滑
RKNN_RTMPOSE_DEBUG_CROP_EVERY=0 关闭生产环境调试图片落盘
RK_JPEG_QUALITY=72            提高 960x540 清晰度且控制编码延迟
```

如果现场真实人体开始出现持续漏检，可先只把 `RKNN_DET_SCORE_THRES` 回调到 `0.76`，不要同时改关键点阈值、NMS和动作判定参数。修改后重启8085并用 `./scripts/check_npu_rehab_8085.sh` 确认 `det.score_threshold` 已加载，再在同一站位分别测试空背景、人体静止和完整动作。RTMPose跟踪只允许在最后一次真实YOLO检出后的1.5秒内调整缓存框，不得刷新缓存时间；完全遮挡后 `det.cache_valid` 必须在1.5秒后变为 `False`，人体框和骨架同时清空。

训练判定继续使用肩、髋、膝、踝等可靠康复关节点和原有阈值；显示层单独绘制完整 COCO-17，包括脸部五点、双肘和双腕。低显示阈值不会进入计数、ROM 或 TUT 判定。检测间隔只影响人体框刷新频率，不降低 RTMPose 关键点和状态机处理频率。

## 2. CPU 与 NPU 一键切换

板子只有一个 USB 摄像头，因此 8082 和 8085 由 systemd 托管并严格二选一运行。切换成功后会记录当前模式，下一次开机自动恢复；首次安装默认进入 NPU 8085。

### 2.1 第一次安装开机自启动

把本指南第 8 节列出的文件上传后，在板端执行一次：

```bash
cd /home/elf/project/project_system
chmod +x scripts/install_rehab_station_autostart.sh
./scripts/install_rehab_station_autostart.sh
```

安装器会创建：

```text
rehab-station-qwen.service       CPU 8082 + MediaPipe
rehab-station-npu-8085.service  NPU 8085 + YOLOv5n raw + RTMPose
rehab-station-mode.service      读取上次模式并启动其中一路
runtime/selected_rehab_mode     cpu_8082 或 npu_8085
```

同时会配置桌面自动登录、关闭息屏、启动显示管理器，并在桌面生成“康复训练 CPU 8082”和“康复训练 NPU 8085”两个快捷方式。快捷方式不保存 sudo 密码，安装器只授权启停上述两个服务。

安装完成后立即检查：

```bash
sudo systemctl status rehab-station-mode.service --no-pager
cat runtime/selected_rehab_mode
curl -s http://127.0.0.1:8085/status | python3 -m json.tool
```

首次应看到 `npu_8085`。以后切到 CPU 后重启会恢复 CPU，切回 NPU 后重启会恢复 NPU。

### 2.2 日常操作速查

所有命令先进入板端项目目录：

```bash
cd /home/elf/project/project_system
```

第一次上传脚本后设置执行权限：

```bash
chmod +x scripts/switch_to_cpu_8082.sh \
  scripts/switch_to_npu_8085.sh \
  scripts/switch_rehab_mode.sh \
  scripts/start_selected_rehab_mode.sh \
  scripts/rehab_display_manager.sh \
  scripts/switch_rehab_mode_desktop.sh \
  scripts/start_npu_rehab_8085.sh \
  scripts/stop_npu_rehab_8085.sh \
  scripts/check_npu_rehab_8085.sh \
  scripts/open_rehab_station_kiosk.sh \
  scripts/open_npu_debug_8085_kiosk.sh
```

启动或切换到 CPU 8082：

```bash
./scripts/switch_to_cpu_8082.sh
```

这个命令既可以用于从 NPU 切回 CPU，也可以用于重新启动 CPU。它会停止 8085、启动 `rehab-station-qwen.service`、检查 MediaPipe 和摄像头状态，成功后才把开机模式写成 `cpu_8082`。

启动或切换到 NPU 8085：

```bash
./scripts/switch_to_npu_8085.sh
```

这个命令会停止 CPU 8082、释放摄像头、启动 `rehab-station-npu-8085.service`，并检查 `service_mode=npu_rehab`、`actual_backend=rknn`、`rknn_pipeline=yolov5n_rtmpose`。验证失败会自动恢复切换前的服务和开机模式。

训练结束后不需要手动停止姿态模型。系统会自动释放 YOLOv5n 和 RTMPose 的 RKNN Runtime，让 NPU 可以交给本地 Qwen；8085 Web 服务和摄像头预览会继续运行。

只停止 NPU 8085（仅用于维护，不会修改已记住的开机模式）：

```bash
./scripts/stop_npu_rehab_8085.sh
```

这个命令会关闭 8085 和摄像头，但不会停止 Qwen 8080/18080，也不会启动 CPU 8082。

重新启动 NPU 8085：

```bash
sudo systemctl restart rehab-station-npu-8085.service
```

完全停止 CPU 8082 和 Qwen：

```bash
sudo systemctl stop rehab-station-qwen.service
./scripts/stop_rehab_station_qwen.sh
```

完全停止 NPU、CPU、摄像头和 Qwen：

```bash
./scripts/stop_npu_rehab_8085.sh
sudo systemctl stop rehab-station-qwen.service
./scripts/stop_rehab_station_qwen.sh
```

电脑浏览器地址：

```text
CPU 医生录入：http://板子IP:8082/doctor
CPU 患者训练：http://板子IP:8082/train
NPU 医生录入：http://板子IP:8085/doctor
NPU 患者训练：http://板子IP:8085/train
NPU 独立检测：http://板子IP:8085/npu-debug
```

当前板子 IP 为 `192.168.137.232` 时：

```text
CPU：http://192.168.137.232:8082/train
NPU：http://192.168.137.232:8085/train
NPU 检测：http://192.168.137.232:8085/npu-debug
```

板子显示屏全屏打开：

```bash
# CPU 8082
REHAB_STATION_URL="http://127.0.0.1:8082/train?display=1" ./scripts/open_rehab_station_kiosk.sh

# NPU 8085
REHAB_STATION_URL="http://127.0.0.1:8085/train?display=1" ./scripts/open_rehab_station_kiosk.sh

# NPU 独立检测调试页
./scripts/open_npu_debug_8085_kiosk.sh
```

`/npu-debug` 页面打开后不会自动占用 NPU。点击“开始 NPU 检测”才加载 YOLOv5n raw 和 RTMPose；点击“停止并释放 NPU”、关闭页面或超过 15 秒没有页面心跳后，姿态模型会释放给 Qwen。

检查 CPU 8082：

```bash
systemctl status rehab-station-qwen.service --no-pager
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

CPU 正常时应看到 `actual_backend=mediapipe`、`camera_live_ok=true`、`vision_boot_error=null`。

检查 NPU 8085：

```bash
./scripts/check_npu_rehab_8085.sh
curl -s http://127.0.0.1:8085/status | python3 -m json.tool
```

检查端口占用：

```bash
sudo ss -ltnp | grep -E ':8080 |:8082 |:8085 |:18080 '
```

查看日志：

```bash
# CPU 8082
tail -f logs/rehab_8082.log
sudo journalctl -u rehab-station-qwen.service -f

# NPU 8085
tail -f runtime/npu/logs/npu_switch_launcher.log
tail -f runtime/npu/logs/npu_rehab_8085.log
sudo journalctl -u rehab-station-npu-8085.service -f
```

日常操作只使用 `switch_to_cpu_8082.sh` 和 `switch_to_npu_8085.sh`。不要在一键切换命令前后重复调用原来的 stop/start 脚本。

显示屏桌面的两个快捷方式调用相同的切换器。通过 SSH 执行切换命令后，桌面显示管理器会监听 `runtime/selected_rehab_mode` 和 `runtime/display_refresh`，只关闭康复专用 Chromium profile，并自动打开对应的全屏训练页，不影响其他浏览器。

### 2.3 切换过程说明

上传并设置脚本权限后，每次只需要记住下面两个命令。它们本身已经包含停止旧版本、释放摄像头和启动新版本，不需要再额外执行 `stop_rehab_station_qwen.sh` 或 `start_rehab_station_qwen.sh`。

从 CPU 8082 切换到 NPU 8085：

```bash
cd /home/elf/project/project_system
./scripts/switch_to_npu_8085.sh
```

这个脚本会依次执行：读取旧模式、停止 `rehab-station-qwen.service`、启动 `rehab-station-npu-8085.service`、等待并校验 `/status`、原子写入 `npu_8085`、通知显示屏切换浏览器。它不会修改或删除 CPU 配置。

新版脚本成功后会输出 `[OK] Active mode: npu_8085` 并把终端还给你。此时不需要再执行 `start_npu_rehab_8085.sh`。

打开：

```text
医生录入：http://板子IP:8085/doctor
患者训练：http://板子IP:8085/train
NPU 检测：http://板子IP:8085/npu-debug
状态接口：http://板子IP:8085/status
```

例如当前板子 IP 是 `192.168.137.232`，电脑浏览器打开：

```text
http://192.168.137.232:8085/train
```

从 NPU 8085 切回稳定 CPU 8082：

```bash
cd /home/elf/project/project_system
./scripts/switch_to_cpu_8082.sh
```

这个脚本会严格按顺序执行：

```text
停止 8085
-> 停止 systemd 中的旧 8082
-> 调用原 stop_rehab_station_qwen.sh 清理残留进程
-> 确认 8082 端口已经释放
-> 写入 CPU 专用 systemd drop-in
-> 重新启动全新的 8082
```

CPU 专用 drop-in 内容为：

```text
REHAB_PORT=8082
POSE_BACKEND=mediapipe
RK_CAMERA_ENABLED=1
```

然后复位 systemd 的 `failed` 状态并启动 CPU 服务。这样即使板端旧启动脚本曾被改成“摄像头留给 8085”，切回 CPU 后也会强制重新启用摄像头。如果板子没有安装 systemd 服务，就带着相同环境变量直接调用原 CPU 启动脚本。

脚本输出 `[OK]` 后，电脑浏览器打开：

```text
http://板子IP:8082/train
```

例如：

```text
http://192.168.137.232:8082/train
```

浏览器不会因为后端切换而自动改端口。切到 NPU 时把地址中的 `8082` 改成 `8085`；切回 CPU 时改回 `8082`，然后刷新页面。

板子连接的显示屏要全屏打开时，另开一个终端执行：

```bash
# CPU 8082
REHAB_STATION_URL="http://127.0.0.1:8082/train?display=1" ./scripts/open_rehab_station_kiosk.sh

# NPU 8085
REHAB_STATION_URL="http://127.0.0.1:8085/train?display=1" ./scripts/open_rehab_station_kiosk.sh
```

下面是故障排查时才使用的手动完整命令。日常不要和一键切换脚本重复执行。CPU -> NPU 的完整命令是：

```bash
sudo systemctl stop rehab-station-qwen.service
sudo systemctl reset-failed rehab-station-qwen.service
./scripts/stop_rehab_station_qwen.sh
./scripts/start_npu_rehab_8085.sh
```

NPU -> CPU 的完整命令是：

```bash
./scripts/stop_npu_rehab_8085.sh
sudo systemctl stop rehab-station-qwen.service
RK_CAMERA_ENABLED=1 POSE_BACKEND=mediapipe REHAB_PORT=8082 \
  ./scripts/start_rehab_station_qwen.sh
```

8085 启动脚本发现 8082 仍在运行时只会报错退出，不会擅自停止 8082。

## 3. NPU 与 Qwen 如何轮流使用资源

8085 的摄像头线程一直保留，但姿态模型不是从启动到退出一直占用 NPU。

```text
空闲：姿态模型未加载，NPU 可供 Qwen 使用
录模板/训练开始：加载 YOLOv5n 和 RTMPose
三动作训练及动作间休息：保持姿态模型加载
训练报告生成完成：释放两个 RKNNLite Runtime
训练结束问答：允许本地 Qwen 使用 NPU
```

Qwen 的 `8080` 服务和 `18080` proxy 可以保持运行。训练期间系统禁止提交 Qwen 问答，因此 Qwen 不会和姿态检测同时执行 NPU 推理。训练结束后，系统先确认姿态模型已经释放，再提交本地 Qwen 请求。

如果小爱正在生成回答，此时点击开始训练，8085 会提示等待当前回答结束，不会强行中断 Qwen。

页面顶部会显示：

- `NPU 模型加载中`
- `NPU 姿态检测中`
- `NPU 正在释放`
- `NPU 已释放，可问小爱`
- `NPU 资源异常`

状态接口中的关键字段：

```text
npu_resource.state
npu_resource.owner
npu_resource.models_loaded
npu_resource.det_model_loaded
npu_resource.pose_model_loaded
npu_resource.det_model_path
npu_resource.pose_model_path
npu_resource.core_mask
npu_resource.last_loaded_at
npu_resource.last_released_at
npu_resource.last_error
npu_debug.active
npu_debug.lease_expires_at
npu_pose_debug.det_decoder
npu_pose_debug.det_output_contract
npu_pose_debug.detector_contract_error
npu_pose_debug.selected_yolo_bbox
```

## 4. 姿态模型权重分别有什么用

### 4.1 `rknn/yolov5n_raw_fp.rknn`

这是当前完整 8085 默认使用的第一阶段人体检测模型，板端文件约 4.9 MB。它输入整张摄像头画面，输出 YOLOv5n 三个尺度的 raw head；8085 只解码 COCO 类别 `0`，也就是 `person`。

源 ONNX 输出 9 个张量：stride `8/16/32` 每层各有 `cls`、`bbox`、`objectness`。RKNN Runtime 也可能把每层合并为一个 255 通道输出，程序同时支持 9 输出和3输出，并按通道数与网格尺寸识别顺序。

后处理使用固定 YOLOv5 anchors 和 stride，计算 `sigmoid(objectness) * sigmoid(person class)`，再执行 letterbox 坐标恢复和 CPU NMS。raw 权重已经在板端，本次不重新转换、不重复上传。

模型输入：

```text
尺寸：640 x 640
布局：NHWC
颜色：RGB
默认预处理：0-1 浮点归一化
```

### 4.2 `rknn/yolov5n_nonms_fp.rknn`（旧版回滚）

这是以前使用的约 1.28 GB TopK/Gather 检测产物。板端日志显示它可能输出负宽高框，现有状态会给出 `Legacy TopK/Gather detector output is deprecated` 警告。

新 8085 不再默认加载它，只在显式设置下面环境变量时用于故障回滚：

```bash
RKNN_DET_MODEL=rknn/yolov5n_nonms_fp.rknn ./scripts/switch_to_npu_8085.sh
```

旧模型回滚路径只按已记录的 `xyxy + 640 输入像素坐标`解释，不再猜测 `cxcywh` 或归一化坐标，避免无效框被错误修复成人体框。

### 4.3 `rknn/rtmpose_m_256x192_fp.rknn`

这是第二阶段姿态模型。程序先扩展 YOLOv5n 的人体框，再把人体区域整理成 `192 x 256` 输入。

模型输出 COCO-17 骨架点的 SimCC X/Y 分布。程序分别寻找 X 和 Y 峰值，再映射回原摄像头画面。

模型输入：

```text
宽度：192
高度：256
布局：NHWC
输出：17 个关键点的 SimCC X/Y
```

### 4.4 `/home/elf/models/qwen/qwen1_5b.rkllm`

这是本地 Qwen2.5-1.5B 问答模型，由板端 `8080/rkllm_chat` 服务加载。它不负责姿态检测，只负责训练结束后的报告解释和康复问答。

训练期间问答被阻止；姿态模型释放以后，Qwen 才开始实际生成。

### 4.5 `quality_model/models/{action_id}/model.onnx`

这是动作完成度评分模型。输入一次动作的固定 30 帧骨架序列，输出 0-100 完成度。

当前在线评分使用 ONNX CPU，并通过后台队列异步运行，不占用 NPU Core，也不会阻塞摄像头和训练状态机。8085 的最终显示分数采用规则优先校准：正确动作由规则分占 75%、ONNX 原始分占 25%，最高 96 分；错误动作按 `ROM_LOW`、`TUT_LOW`、`SHAPE_BAD`、`TOO_FAST`、`VISIBILITY_LOW` 分别封顶。ONNX 原始分只保留在报告和 `/status` 中用于诊断。

### 4.6 `quality_model/models/{action_id}/best.pt`

这是质量评分模型训练后的 PyTorch 检查点，用于继续训练或导出 ONNX。板端在线训练页面不会直接加载 `best.pt`。

### 4.7 当前 8085 不使用的旧姿态权重

下面这些属于以前的调试或对比路线，不能和新 8085 混为一谈：

```text
rknn/rtmpose_fp16.rknn
rknn/rtmdet_fp16.rknn
/home/elf/models/yolov8n-pose.rknn
```

8085 正确状态应显示：

```text
rknn_pipeline = yolov5n_rtmpose
det_model_path = rknn/yolov5n_raw_fp.rknn
pose_model_path = rknn/rtmpose_m_256x192_fp.rknn
det_output_contract = 9 split raw tensors 或 3 combined raw tensors
detector_contract_error = null
```

## 5. 为什么只有 17 点仍然能完成三个动作

三个动作实际依赖的是肩、髋、膝、踝，不需要 MediaPipe 的全部 33 点。

COCO-17 映射：

| 关键点 | 左侧索引 | 右侧索引 |
| --- | ---: | ---: |
| 肩 | 5 | 6 |
| 髋 | 11 | 12 |
| 膝 | 13 | 14 |
| 踝 | 15 | 16 |

动作计算方式：

- 坐站训练：髋部相对坐姿 baseline 的上升高度，再除以肩髋距离。
- 站姿后勾腿：髋、膝、踝形成的二维膝屈曲角。
- 坐姿抬膝：膝相对髋的上升高度，再除以肩髋距离。

NPU 只有二维图像坐标，没有 MediaPipe world landmark 的三维深度，因此 NPU 必须重新录制三个医生模板。系统把 NPU 模板和 NPU attempt 放在独立注册表中，不允许拿 CPU 模板直接比较。

## 6. NPU 专用阈值

公共参数：

```text
关键点置信度：0.18
目标腿有效阈值：0.20
平滑窗口：7 帧
动作确认：4 帧
缺失关键点最多保持：8 帧
```

初始动作阈值：

| 动作 | 启动增量 | 有效动作增量 | 返回增量 | 稳定返回 |
| --- | ---: | ---: | ---: | ---: |
| 坐站 | 0.08 | 0.18 | 0.10 | 0.30 秒 |
| 后勾腿 | 6 度 | 8 度 | 7 度 | 0.25 秒 |
| 坐姿抬膝 | 0.05 | 0.10 | 0.10 | 0.20 秒 |

后续只调整：

```text
evaluate/configs/npu/*.yaml
realtime/configs/knee_flexion_realtime_npu.yaml
realtime/configs/rehab_demo_plan_npu.yaml
```

不要为了调 NPU 修改 CPU 的三个动作 YAML。

## 7. NPU 数据保存在哪里

```text
runtime/npu/active_templates.json
prescription/docs/npu/doctor_templates/
prescription/docs/npu/patient_attempts/
prescription/docs/npu/summaries/
evaluate/reports/npu/
evaluate/reports/npu/keyframes/
```

第一次使用 8085 时，必须进入 `/doctor` 依次录入：

```text
sit_to_stand
standing_hamstring_curl
seated_knee_raise
```

录完三个 NPU 模板后再进入 `/train` 启动完整训练。

## 8. 需要上传哪些文件

不上传姿态检测权重。`yolov5n_raw_fp.rknn` 和 `rtmpose_m_256x192_fp.rknn` 已经在板端 `rknn/` 目录中。

如果此前已经完成 8085 双模式安装，本次 `npu_training_v8_stage2_pipeline` 稳定低延迟修复需要重新上传：

```text
prescription/banzi/record_prescription_http.py
prescription/banzi/npu_rehab_8085.py
prescription/banzi/static/common.js
prescription/banzi/static/train.js
vision/rknn_pose/pose_frame_adapter.py
vision/rknn_pose/yolov5n_rtmpose_backend.py
realtime/training_session.py
scripts/start_npu_rehab_8085.sh
scripts/check_npu_rehab_8085.sh
scripts/set_npu_8085_pipeline.sh
scripts/benchmark_npu_rehab_8085.py
vision/gstreamer_gi_capture.py
scripts/install_rehab_station_autostart.sh
scripts/switch_rehab_mode.sh
docs/npu_rehab_8085_guide.md
```

上传后必须在板端执行语法检查和服务重启，VSCode 上传本身不会让已经运行的 Python 进程重新加载代码：

```bash
cd /home/elf/project/project_system
python3 -m py_compile \
  prescription/banzi/record_prescription_http.py \
  prescription/banzi/npu_rehab_8085.py \
  vision/rknn_pose/yolov5n_rtmpose_backend.py \
  vision/rknn_pose/pose_frame_adapter.py \
  realtime/training_session.py
sudo systemctl restart rehab-station-npu-8085.service
./scripts/check_npu_rehab_8085.sh
```

板端基线必须按场景分别采样 60 秒。先打开对应页面并进入待测状态，再执行：

```bash
python3 scripts/benchmark_npu_rehab_8085.py --scenario idle
python3 scripts/benchmark_npu_rehab_8085.py --scenario npu-debug
python3 scripts/benchmark_npu_rehab_8085.py --scenario doctor
python3 scripts/benchmark_npu_rehab_8085.py --scenario train
```

结果保存在 `runtime/npu/benchmarks/`。摄像头真实采集约 15 FPS 时，姿态验收目标是稳定 `12-15 FPS`；只有 `camera_capture_fps >= 25` 时才要求姿态达到 `20-25 FPS`。先保持 `960x540@15`，如果该档 `stream_fps < 12` 或 `capture_to_stream_age_ms` P95 超过 `250ms`，再用 systemd 环境变量将 `RKNN_STREAM_WIDTH=640`、`RKNN_STREAM_HEIGHT=360` 做同场景 A/B，不能用重复旧帧伪造高 FPS。

需要立即回退旧8085同步显示路径时，不改8082，执行：

```bash
chmod +x scripts/set_npu_8085_pipeline.sh
./scripts/set_npu_8085_pipeline.sh sync
```

恢复新版异步路径：

```bash
./scripts/set_npu_8085_pipeline.sh async
```

不需要重新上传两个 RKNN 模型，也不改变 COCO-17 数据结构。现有 NPU 模板只要 `/status` 中对应动作的 `template.ok=true` 就不需要重录；模板本身无效时，再到 `8085/doctor` 单独重录该动作。

修复后的 8085 只在姿态输入层保留 COCO-17 适配，动作切分、ROM/TUT 判定、回位条件和反馈优先级与 8082 对齐。它不再在每次尝试后重建 baseline，也不再使用 NPU 专用动态目标和额外 ROM 下限。达到站起幅度但保持不足时应判定 `TUT_LOW`，不会再因为第二次 baseline 漂移而提示“再站起来一点”。

高清显示稳定器只作用于画面绘制，不修改动作模板、训练关键点置信度、动作切分或 ROM/TUT 阈值。未站够高、未勾够、未抬够时仍只更新屏幕提示，患者真正返回起始姿势以后才结算错误，不会在动作中途打断。

8085 训练语音启用 `training_fixed_audio_only: true`：只允许播放 `prescription/banzi/static/assets/tts/` 中已经录制的豆包 WAV。没有固定 WAV 的恢复话术只显示在屏幕上，绝不初始化或调用自然 TTS、pyttsx3 或 espeak；8082 继续保持原来的语音兜底行为。

Windows PowerShell 示例：

```powershell
$BOARD_USER = "elf"
$BOARD_IP = "192.168.137.232"
$LOCAL = "D:\rk3588\project"
$REMOTE = "/home/elf/project/project_system"

scp $LOCAL\vision\rknn_pose\pose_frame_adapter.py ${BOARD_USER}@${BOARD_IP}:$REMOTE/vision/rknn_pose/
scp $LOCAL\vision\rknn_pose\yolov5n_rtmpose_backend.py ${BOARD_USER}@${BOARD_IP}:$REMOTE/vision/rknn_pose/
scp $LOCAL\prescription\banzi\record_prescription_http.py ${BOARD_USER}@${BOARD_IP}:$REMOTE/prescription/banzi/
scp $LOCAL\prescription\banzi\npu_rehab_8085.py ${BOARD_USER}@${BOARD_IP}:$REMOTE/prescription/banzi/
scp $LOCAL\scripts\start_npu_rehab_8085.sh ${BOARD_USER}@${BOARD_IP}:$REMOTE/scripts/
scp $LOCAL\scripts\check_npu_rehab_8085.sh ${BOARD_USER}@${BOARD_IP}:$REMOTE/scripts/
scp $LOCAL\docs\npu_rehab_8085_guide.md ${BOARD_USER}@${BOARD_IP}:$REMOTE/docs/
```

板端设置执行权限：

```bash
cd /home/elf/project/project_system
chmod +x \
  scripts/start_npu_rehab_8085.sh \
  scripts/stop_npu_rehab_8085.sh \
  scripts/check_npu_rehab_8085.sh \
  scripts/switch_rehab_mode.sh \
  scripts/switch_to_npu_8085.sh \
  scripts/switch_to_cpu_8082.sh \
  scripts/start_selected_rehab_mode.sh \
  scripts/rehab_display_manager.sh \
  scripts/switch_rehab_mode_desktop.sh \
  scripts/install_rehab_station_autostart.sh \
  scripts/open_rehab_station_kiosk.sh \
  scripts/open_npu_debug_8085_kiosk.sh

./scripts/install_rehab_station_autostart.sh
```

## 9. 状态检查和日志

```bash
./scripts/check_npu_rehab_8085.sh
sudo systemctl status rehab-station-mode.service --no-pager
sudo systemctl status rehab-station-npu-8085.service --no-pager
tail -n 160 runtime/npu/logs/npu_rehab_8085.log
tail -n 100 runtime/npu/logs/qwen_flask.log
tail -n 100 runtime/npu/logs/qwen_proxy.log
```

空闲时预期：

```text
npu.state                 qwen_available
npu.models_loaded         False
training.status           idle 或 completed
```

训练时预期：

```text
pose.backend              rknn
pose.pipeline             yolov5n_rtmpose
npu.state                 pose_active
npu.owner                 pose
npu.models_loaded         True
```

独立检测页验收：

```bash
./scripts/open_npu_debug_8085_kiosk.sh
```

在显示屏点击“开始 NPU 检测”，然后检查：

```text
debug.active              True
det_model_path            rknn/yolov5n_raw_fp.rknn
det.decoder               yolov5_raw_head 或 yolov5_combined_raw_head
det.output_contract       9 split raw tensors 或 3 combined raw tensors
det.contract_error        None
pose.keypoint_range       非空
pose_fps                   训练时目标 12-15
camera.source              direct_device
camera.device              /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
camera.open_mode           gstreamer_mppjpegdec 或 gstreamer_jpegdec
camera.requested           [1280, 720]
camera.capture_fps         当前摄像头实测约 15
pose.inference_fps         训练时目标 12-15
pose.stream_fps            训练时目标 12-15
pose.capture_to_stream_ms  P95 小于 250
camera.uses_8082_stream    False
perf.fast_preview          True
perf.fast_frame_data       True
perf.adaptive_detector     True
perf.det_refresh_seconds   0.75
perf.det_cache_seconds     1.5
perf.backend_draw          False
perf.person_only_fast      True
perf.debug_crop_every      0
perf.jpeg_quality          72
perf.process_resolution    [1280, 720]
perf.stream_resolution     [960, 540]
perf.loop_ms               非YOLO帧应接近RTMPose与状态机总耗时；以P95和结果年龄验收
display.keypoint_count     正常完整入画时接近 17
display.held_keypoints     仅短暂低置信度时非空
```

点击“停止并释放 NPU”后，`debug.active=False` 且 `npu.models_loaded=False`。直接关闭页面时，15 秒租约到期后也会自动释放。

患者训练开始后，初始方向流程应依次出现：

```text
orientation.phase        awaiting_front
orientation.front_count  0 -> 1 -> 2
orientation.front_need   2
orientation.mode         torso_ratio

播放“请侧身对准镜头”后：
orientation.phase        awaiting_side
orientation.side_count   0 -> 1 -> 2 -> 3 -> 4
orientation.side_need    4
```

上传并重启后先确认版本，避免只覆盖 YAML、Python 仍是旧版：

```text
training.logic_version   npu_training_v8_stage2_pipeline
runtime.build_id         非空且重启后与新文件一致
deployment.hashes_ok     True
template.ok              True
performance_profile.async_pipeline        True
performance_profile.adaptive_detector     True
performance_profile.det_cache_seconds     1.5
performance_profile.backend_draw_enabled  False
performance_profile.process_resolution    [1280, 720]
performance_profile.stream_resolution     [960, 540]
```

如果 `runtime.build_id` 没变化、`deployment.hashes_ok` 不是 `True`，或 `training.logic_version` 仍是旧值，说明板端进程没有加载刚上传的代码，不能开始验收。

正面识别使用左右肩、左右髋的横向宽度与躯干高度比值。`orientation.ratio >= 0.55` 为正面，`<= 0.32` 为侧面；中间角度不会通过。双侧躯干点不足时，仅允许完整近侧三点链以 `side_chain_fallback` 模式通过侧面确认，不允许其跳过初始正面确认。

“请先回到起始姿势站稳”来自8082和8085共用的动作起始校准状态。正常情况下约2秒后变为“可以开始动作”。如果长期不消失，检查 `start.guard_remaining`、`start.baseline`、`start.current_metric`、`start.ready` 和 `start.motion_delta`；坐站训练必须先坐稳，并保持肩、髋、膝、踝完整可见。

如果画面提示长期停在“慢慢放下”，执行检查脚本并观察：

```text
return_pose.count       回到起始姿势后的连续确认数
return_pose.seconds     起始姿势稳定时间
rebaseline.state        当前是否正在为下一次重建 baseline
watchdog.reason         最近一次自动恢复原因
watchdog.recoveries     自动恢复累计次数
```

正常坐回或放腿后，`return_pose.count` 应连续增长，动作随即结算；即使状态还显示 `HOLDING`，稳定起始姿势也会触发收尾，主指标有少量漂移时不会卡死。若未真正回到起始姿势，最多经过对应状态看门狗时间后明确结算无效尝试或回到起始姿势确认，不会连续播放 ROM/TUT 纠错。

NPU 动作过程中允许目标腿关键点短暂不稳定最多约 `1.5 秒`，期间暂停当前动作判断但不清空已经完成的站起、保持或放腿阶段。连续超过容忍时间后，已经形成有效幅度的动作会记录为 `VISIBILITY_LOW` 无效尝试，而不是直接丢失；没有形成有效动作的噪声才回到起始姿势确认。“请先回到起始姿势站稳”只作为屏幕恢复提示。正常动作介绍、计数、ROM/TUT 反馈只播放已有豆包 WAV，不做 TTS 合成。

训练结束后再次检查，`models_loaded` 必须回到 `False`，再验证小爱本地 Qwen 问答。

问答录音改为单按钮控制：第一次点击 `唤醒监听` 后，板端 `arecord` 持续录制且按钮变成 `结束监听`；用户完整说完问题后再次点击，系统才停止同一 `session_id` 的录音、把整段WAV提交ASR并覆盖文字框旧问题，之后由用户手动点击 `提交问答`。页面刷新可通过 `/api/voice/status.voice.manual_capture` 恢复录音状态，旧会话不能停止新会话；监听期间训练启动和助手TTS均被阻止。

如果完整训练帧率低于纯姿态测试，执行 `./scripts/check_npu_rehab_8085.sh`，重点比较：

```text
perf.infer_ms          YOLOv5n + RTMPose 本身耗时
perf.pose_process_ms   推理、关键点适配和康复骨架绘制总耗时
perf.realtime_ms       训练状态机单帧耗时
perf.jpeg_ms           网页 MJPEG 编码耗时
perf.loop_ms           完整姿态线程单帧耗时
```

`camera.source` 必须为 `direct_device`，`camera.uses_8082_stream` 必须为 `False`。如果 `perf.infer_ms` 接近纯测试而 `perf.loop_ms` 明显更高，才说明瓶颈仍在康复终端逐帧处理；如果两者同时升高，则继续检查 NPU 频率、温度和模型是否误用了旧权重。

## 10. 完整验收顺序

1. 运行自启动安装器，确认首次模式为 `npu_8085`，显示屏自动打开 `8085/train?display=1`。
2. 检查空闲时姿态模型未加载。
3. 显示屏打开 `/npu-debug`，点击开始，验证 raw detector 人体框、17 点骨架、FPS 和输出契约。
4. 点击停止或关闭页面，确认姿态 Runtime 自动释放。
5. 在 8085 `/doctor` 只重新录入坐站和站姿后勾腿模板；保留第三组坐姿抬膝模板。
6. 在 8085 `/train` 完成三个动作。
7. 检查实时骨架、计数、ROM/TUT 纠错、固定豆包 WAV、无 TTS 合成和离屏恢复。
8. 检查 NPU attempt、报告、关键帧、规则优先完成度和保留的 ONNX 原始诊断分。
9. 检查训练结束后姿态 RKNN Runtime 已释放，再提交小爱问题验证 Qwen。
10. 用桌面快捷方式或 SSH 切到 CPU 8082，确认 CPU 摄像头、骨架、三动作、CPU 模板和历史报告完全不变。
11. 在 CPU 模式重启一次、NPU 模式再重启一次，确认都能恢复最后一次成功模式并自动打开正确全屏页面。

## 11. 常见启动问题

### `ModuleNotFoundError: No module named 'prescription'`

这是旧版 8085 入口没有把项目根目录加入 Python 搜索路径导致的，不是 NPU 模型错误。重新上传下面两个更新后的文件：

```text
prescription/banzi/npu_rehab_8085.py
scripts/start_npu_rehab_8085.sh
```

新版入口会主动加入项目根目录，启动脚本也会设置 `PYTHONPATH=/home/elf/project/project_system`。

### systemd 显示 `failed (status=143)`

`143` 表示原 CPU 服务被停止信号结束，不代表代码损坏。切换脚本会自动执行：

```bash
sudo systemctl reset-failed rehab-station-qwen.service
```

因此日常切换直接使用 `switch_to_npu_8085.sh` 和 `switch_to_cpu_8082.sh` 即可。

### 切回 8082 后显示 `RK_CAMERA_ENABLED=0`

这表示旧 systemd 单元没有明确启用 CPU 摄像头。重新运行安装器并切回 CPU：

```bash
./scripts/install_rehab_station_autostart.sh
./scripts/switch_to_cpu_8082.sh
```

CPU systemd 单元固定包含：

```text
POSE_BACKEND=mediapipe
RK_CAMERA_ENABLED=1
```

新版还会先杀掉旧的 8082 进程。只执行 `systemctl start` 不够，因为如果旧 8082 已经在运行，systemd 不会用新环境重新创建进程。

检查：

```bash
systemctl show rehab-station-qwen.service -p Environment
curl -s http://127.0.0.1:8082/status | python3 -m json.tool | grep -E 'camera enabled|actual_backend|camera_live_ok'
```

## 12. 报告图片、训练后 TTS 与小爱验收

8085 的训练语音继续只播放固定 WAV。完整训练和最后一条 `finished.wav` 播放结束前，问答、自然女声和唤醒录音都必须保持关闭。训练后自然女声与固定 WAV 统一走板载音频口：

```bash
grep -q '^REHAB_AUDIO_OUTPUT_DEVICE=' runtime/llm.env \
  && sed -i 's/^REHAB_AUDIO_OUTPUT_DEVICE=.*/REHAB_AUDIO_OUTPUT_DEVICE=plughw:CARD=rockchipnau8822,DEV=0/' runtime/llm.env \
  || echo 'REHAB_AUDIO_OUTPUT_DEVICE=plughw:CARD=rockchipnau8822,DEV=0' >> runtime/llm.env
aplay -D plughw:CARD=rockchipnau8822,DEV=0 prescription/banzi/static/assets/tts/count_1.wav
```

更新代码并重启 8085 后，用已有 attempt 重算最新三个 NPU 报告，不需要重录模板：

```bash
python3 scripts/regenerate_latest_npu_reports.py
./scripts/switch_to_npu_8085.sh
```

检查报告和音频所有权：

```bash
curl -s http://127.0.0.1:8085/api/reports/latest_by_action | python3 -m json.tool | grep -E 'keyframes|image_path'
curl -s http://127.0.0.1:8085/api/voice/status | python3 -m json.tool | grep -E 'audio_owner|assistant_tts_blocked_reason|audio_output_device|last_audio_returncode'
```

成功标准：

```text
训练中和组间休息：audio_owner = training_fixed_wav，assistant TTS 不初始化、不排队
完成提示播放结束后：audio_owner = idle，允许开启小爱
问答朗读时：audio_owner = assistant_tts，audio_output_device = plughw:CARD=rockchipnau8822,DEV=0（失败时回退 plughw:1,0）
NPU 最新报告：keyframes 非空，图文建议能打开骨架大图
```
