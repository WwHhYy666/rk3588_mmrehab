# NPU 8085 康复训练部署与验收

本文是 `latest` 分支的板端操作手册。该分支只维护 `8085 + YOLOv5n raw + RTMPose` 路线。

## 1. 运行链路

```text
USB 摄像头
-> GStreamer GI / OpenCV 低延迟采集
-> YOLOv5n raw RKNN 人体框
-> RTMPose RKNN COCO-17 关键点
-> 康复关键点适配
-> 三动作状态机与纠错
-> 固定提示音 / 可选 TTS
-> attempt、关键帧和评估报告
-> 可选完成度评分与 GLM/Qwen 问答
```

默认页面：

```text
医生录入  http://板子IP:8085/doctor
患者训练  http://板子IP:8085/train
NPU 调试  http://板子IP:8085/npu-debug
运行状态  http://板子IP:8085/status
```

## 2. 板端前提

- RK3588 Linux，Python 3.10 或兼容版本。
- Rockchip RKNNLite2 板端 wheel 与匹配的 `librknnrt.so`。
- Python 模块：`cv2`、`numpy`、`yaml`、`rknnlite`。
- `curl`、`v4l2-ctl`、ALSA；浏览器展示时需要 Chromium。
- 摄像头默认设备：
  `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`。

桌面测试依赖可用 `requirements.txt` 安装。RKNNLite2 必须使用 Rockchip 提供的板端包。

## 3. 模型文件

默认需要：

```text
models/vision/yolov5n_raw_fp.rknn
models/vision/rtmpose_m_256x192_fp.rknn
```

启动脚本会在启动任何服务前检查两个文件。模型被 `.gitignore` 排除，应从项目受控存储单独复制：

```bash
ssh elf@板子IP 'mkdir -p /home/elf/project/models/vision'
scp yolov5n_raw_fp.rknn elf@板子IP:/home/elf/project/models/vision/
scp rtmpose_m_256x192_fp.rknn elf@板子IP:/home/elf/project/models/vision/
```

不要使用 `yolov5n_nonms_fp.rknn`。检查脚本会把超过 100 MiB 或带该旧文件名的检测模型标记为过时模型。

可通过环境变量改路径：

```bash
export RKNN_DET_MODEL=/home/elf/project/models/vision/yolov5n_raw_fp.rknn
export RKNN_RTMPOSE_MODEL=/home/elf/project/models/vision/rtmpose_m_256x192_fp.rknn
```

## 4. 首次配置

```bash
cd /home/elf/project
cp .env.example runtime/llm.env
chmod +x scripts/*.sh
```

若使用 GLM，在 `runtime/llm.env` 填写 `GLM_API_KEY`。若使用本地 Qwen，确认：

```text
QWEN_RKLLM_MODEL=models/language/qwen/qwen1_5b.rkllm
QWEN_SERVER_DIR=/home/elf/qwen_server/rkllm_server
```

密钥和机器路径不得提交到 Git。

## 5. 启动、检查与停止

前台启动：

```bash
./scripts/start_npu_rehab_8085.sh
```

该脚本会：

1. 检查 8082 和 8085 端口冲突；
2. 检查两个 RKNN 模型与 Python 依赖；
3. 按需启动外部 Qwen RKLLM server 和本仓库的 HTTP proxy；
4. 固定 `POSE_BACKEND=rknn` 和 `RKNN_POSE_PIPELINE=yolov5n_rtmpose`；
5. 启动 `rehab_app/server/npu_rehab_server.py`；
6. 把日志与 PID 写入 `runtime/npu/`。

另开终端检查：

```bash
./scripts/check_npu_rehab_8085.sh
```

关键结果应满足：

```text
service_mode             npu_rehab
actual_backend           rknn
rknn_pipeline            yolov5n_rtmpose
camera.source            direct_device
camera.uses_8082_stream  False
deployment.hashes_ok     True
```

停止：

```bash
./scripts/stop_npu_rehab_8085.sh
```

停止脚本校验 PID 对应的命令行，只会终止 8085 入口进程。

## 6. systemd 与全屏页面

安装并启动服务：

```bash
./scripts/install_npu_rehab_8085_autostart.sh
```

默认行为：

- 安装并启用 `rehab-station-npu-8085.service`；
- 当前用户登录桌面后自动打开 8085 患者页；
- 使用独立浏览器 profile，不影响用户的其他 Chromium 窗口。

只安装、不立即启用：

```bash
REHAB_ENABLE_SERVICE=0 ./scripts/install_npu_rehab_8085_autostart.sh
```

不安装浏览器自启动：

```bash
REHAB_INSTALL_KIOSK=0 ./scripts/install_npu_rehab_8085_autostart.sh
```

手动打开页面：

```bash
./scripts/open_npu_rehab_8085_kiosk.sh
./scripts/open_npu_debug_8085_kiosk.sh
```

## 7. 模板、训练数据与报告

8085 使用隔离数据目录：

```text
runtime/npu/active_templates.json
data/npu/
data/reports/npu/
```

默认训练计划与参数：

```text
training/configs/rehab_demo_plan_npu.yaml
training/configs/training_defaults_npu.yaml
evaluation/configs/npu/
```

三个动作必须用当前 NPU 姿态后端重新录制医生模板。不要把其他关键点体系生成的模板直接复制到 NPU 注册表。

生成数据、报告、关键帧和日志均被 Git 忽略。

## 8. 性能基准

在不同页面状态运行：

```bash
python3 scripts/benchmark_npu_rehab_8085.py --scenario idle
python3 scripts/benchmark_npu_rehab_8085.py --scenario npu-debug
python3 scripts/benchmark_npu_rehab_8085.py --scenario doctor
python3 scripts/benchmark_npu_rehab_8085.py --scenario train
```

重点比较 `p50`、`p95`、推理、渲染、JPEG、队列等待和 capture-to-stream 延迟。出现抖动时先判断瓶颈属于摄像头、检测、姿态、渲染还是关键帧落盘，不要只看总 FPS。

异步姿态流水线默认开启。需要诊断同步路径时：

```bash
./scripts/set_npu_pose_execution_mode.sh sync
./scripts/set_npu_pose_execution_mode.sh async
```

## 9. 摄像头故障

列出设备：

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
```

覆盖设备路径：

```bash
RK_CAMERA_DEVICE=/dev/video0 ./scripts/start_npu_rehab_8085.sh
```

若启动脚本提示 8082 占用端口或摄像头，先停止对应的外部旧服务，再启动 8085。`latest` 分支不提供 8082 控制脚本。

查看日志：

```bash
tail -f runtime/npu/logs/npu_rehab_8085.log
journalctl -u rehab-station-npu-8085.service -f
```

## 10. NPU 与运行库故障

检查：

```bash
python3 pose_estimation/rknn_pose/check_rknn_runtime.py
./scripts/check_npu_rehab_8085.sh
```

仓库提供运行库安装和回滚辅助：

```bash
./scripts/install_rknn_runtime.sh
./scripts/restore_rknn_runtime.sh /home/elf/rknnrt_backups/<timestamp>
```

运行库必须与模型和 RKNNLite2 wheel 版本匹配。升级前保留备份，升级后重启 8085 服务并重新执行完整检查。

## 11. 最小验收清单

1. `check_npu_rehab_8085.sh` 无模型、摄像头或部署哈希错误。
2. `/npu-debug` 骨架稳定，关键点与人体方向一致。
3. `/doctor` 可为三个动作录入并启用模板。
4. `/train` 可完成三动作、计数、ROM/TUT 纠错、休息和完成提示。
5. 训练结束生成 attempt、报告和关键帧。
6. 固定 WAV 不阻塞训练；可选 GLM/Qwen 故障不影响主闭环。
7. 重启 systemd 服务后页面恢复，模型和摄像头仍可加载。
8. 运行 `python -m pytest` 的非硬件测试全部通过。
