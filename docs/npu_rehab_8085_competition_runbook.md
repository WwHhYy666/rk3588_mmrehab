# 8085 国赛现场运行手册

本手册只针对国赛使用的 `8085 + YOLOv5n raw + RTMPose` 路线，`8082 + MediaPipe` 不在本次修改范围内。

## 1. 启动与切换

本项目默认不开机启动。Ubuntu 登录后，通过桌面上的“康复训练 NPU 8085”图标启动；图标会调用 `switch_rehab_mode_desktop.sh npu_8085`，启动服务并打开专用显示浏览器。

如果板子以前配置过开机自启动，先执行一次：

```bash
cd /home/elf/project/project_system
REHAB_AUTO_START_MODE=0 ./scripts/install_rehab_station_autostart.sh
```

该命令会关闭 8082、8085 和模式控制器的开机启动，但保留桌面快捷方式。

```bash
cd /home/elf/project/project_system
./scripts/switch_to_npu_8085.sh
curl -s http://127.0.0.1:8085/status | python3 -m json.tool
```

确认状态中包含：

```text
actual_backend = rknn
rknn_pipeline  = yolov5n_rtmpose
det_score_thres = 0.68
process_resolution = [1280, 720]
stream_resolution  = [854, 480]
stream_fps_limit   = 24
jpeg_quality       = 68
```

显示屏全屏页面由专用 Chromium profile 打开：

```bash
REHAB_STATION_URL="http://127.0.0.1:8085/train?display=1" \
  ./scripts/open_rehab_station_kiosk.sh
```

旧版板端 Chromium 默认启用 GPU 合成兼容参数和 `/dev/shm` 回退，不影响电脑浏览器访问。

## 2. 音箱测试

本项目默认使用板载 `rockchipnau8822` 的 3.5mm 音频口，不使用 HDMI 显示器扬声器。

```bash
aplay -l
aplay -D plughw:CARD=rockchipnau8822,DEV=0 \
  prescription/banzi/static/assets/tts/count_1.wav
```

如果声卡名在当前系统不可用，播放程序会自动回退到：

```text
plughw:1,0
```

现场确认训练开始音、计数音、组间休息音和结束音均能从同一个音箱听到。

## 3. 120 秒显示稳定性测试

启动后保持 `/train?display=1` 页面静置至少 120 秒，同时观察摄像头流和状态文字。

```bash
curl -s http://127.0.0.1:8085/status | python3 -m json.tool \
  | grep -E 'stream_frame_age_ms|stream_fps|camera_live_ok|capture_to_stream_age_ms'
```

正常条件：

- 页面不出现整页白屏。
- `stream_frame_age_ms` 持续较小，通常不超过 1000 ms。
- 摄像头画面停止时，前端只重连 `/stream.mjpg`，不会刷新整个训练页面。
- 如果浏览器进程确实崩溃，显示管理器只重启康复专用 profile。

## 4. 三动作完整验收

依次验证坐站、站姿屈膝后勾腿、坐姿抬膝：

1. 医生模板和站位/方向确认正常。
2. 训练中检测框稳定，空背景不触发有效关键点和计数。
3. ROM、TUT、纠错、组间休息和 TTS 顺序不变。
4. 离开画面再返回后能恢复当前动作，不重新破坏完整训练流程。
5. 训练结束后 attempt、报告、关键帧和 Qwen/GLM 问答入口正常。

## 5. 帧率与延迟检查

```bash
./scripts/check_npu_rehab_8085.sh
```

重点查看：

```text
perf.infer_ms
perf.pose_process_ms
perf.render_total_ms
perf.jpeg_ms
perf.loop_ms
stream_fps
capture_to_stream_age_ms
stale_inference_drops
```

优先目标是稳定 20 FPS 以上、`capture_to_stream_age_ms` P95 约 250 ms 以内，同时不牺牲姿态推理和训练更新频率。如果板端负载过高，可临时设置 `RKNN_STREAM_FPS=20` 回退显示上限，不修改模型和状态机。

## 6. 白屏恢复

先确认 8085 服务仍然正常：

```bash
curl -fsS http://127.0.0.1:8085/status >/dev/null
```

只重启康复显示浏览器：

```bash
pkill -f -- "--user-data-dir=${HOME}/.cache/rehab-station-browser" || true
REHAB_STATION_URL="http://127.0.0.1:8085/train?display=1" \
  ./scripts/open_rehab_station_kiosk.sh
```

不要在训练中执行整页刷新或重启 8085；那会破坏当前动作状态。只有服务本身不可达时，才执行：

```bash
./scripts/switch_to_npu_8085.sh
```
