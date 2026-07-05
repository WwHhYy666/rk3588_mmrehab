# Sherpa-ONNX Paraformer FP32 ASR 与小爱助手验收指南

本文档用于把小爱助手 ASR 从较容易误识别的 `model.int8.onnx` 切到更高精度的 `model.onnx`，并按当前板端真实情况先用 `rockchipnau8822 / plughw:1,0` 录音链路跑通“小爱康复助手”。

## 当前推荐路线

当前实测结论：

```text
USB 麦克风：lsusb 能看到，但当前系统缺 snd-usb-audio，暂时不能作为初赛阻塞项。
nau8822：arecord -D plughw:1,0 能录音，aplay test.wav 能听到人声，可以先用于 ASR。
```

所以当前路线是：

```text
初赛线上视频：
板子本地 nau8822/plughw:1,0 录音 -> Sherpa-ONNX FP32 ASR -> 小爱助手 -> GLM/Qwen 回答

最终脱机展示：
板子接显示屏，本机浏览器打开 /train；只要 plughw:1,0 仍能录音，就可以继续脱机演示。

后续增强：
再找厂家补 snd-usb-audio，或换支持当前系统的 USB 麦克风 / USB 声卡。
```

注意：GLM 需要手机热点联网，严格说不是“无网”，但仍然是板子独立运行；Qwen 本地 NPU 推理可以断网展示。

当前项目 ASR 入口：

```text
voice/asr_worker.py
POST /api/voice/asr
POST /api/voice/asr_capture
GET  /api/voice/asr_result
GET  /api/voice/status
```

代码现在会优先找 `model.onnx`，找不到才退回 `model.int8.onnx`。只要模型目录里有 `model.onnx`，并且 `REHAB_ASR_PREFER_INT8` 不是 `1`，就会优先使用高精度模型。

## 1. 推荐下载哪个模型

推荐下载官方 Sherpa-ONNX 模型：

```text
sherpa-onnx-paraformer-zh-2024-03-09
```

官方文档：

```text
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html
```

官方下载包：

```text
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
```

这个包里通常包含：

```text
model.onnx        约 785M，精度更高，当前推荐优先用
model.int8.onnx   约 217M，速度更快，但短句和噪声下更容易误识别
tokens.txt
```

当前小爱助手问答更看重识别准确率，尤其是短句、康复问题和唤醒词，所以初赛视频和正式演示都先用 `model.onnx`。只有 FP32 太慢或 CPU 压力太大时，才临时切回 `model.int8.onnx`。

## 2. 在电脑上下载到 D 盘

不要下载到 C 盘。Windows 电脑统一放到：

```text
D:\rk3588_models\sherpa_asr\
```

如果命令行下载太慢，推荐直接手动下载：

1. 在电脑浏览器打开这个链接：

```text
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
```

2. 浏览器弹出保存位置时，保存到：

```text
D:\rk3588_models\sherpa_asr\
```

3. 下载完成后，文件应该是：

```text
D:\rk3588_models\sherpa_asr\sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
```

4. 用 7-Zip 解压。右键这个 `.tar.bz2` 文件，先解压出 `.tar`，再右键 `.tar` 解压出目录：

```text
D:\rk3588_models\sherpa_asr\sherpa-onnx-paraformer-zh-2024-03-09\
```

5. 解压完成后进入这个目录，确认能看到：

```text
model.onnx
model.int8.onnx
tokens.txt
```

如果你想继续用命令行下载，也可以用 Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path D:\rk3588_models\sherpa_asr | Out-Null
cd D:\rk3588_models\sherpa_asr
curl.exe -L -o sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
tar -xvf sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
```

如果你用 WSL / Git Bash，也建议直接操作 D 盘目录：

```bash
mkdir -p /mnt/d/rk3588_models/sherpa_asr
cd /mnt/d/rk3588_models/sherpa_asr
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
```

## 3. 在电脑上先检查文件

进入解压后的目录：

```bash
cd sherpa-onnx-paraformer-zh-2024-03-09
ls -lh
```

必须看到：

```text
model.onnx
model.int8.onnx
tokens.txt
```

重点确认 `model.onnx` 存在。如果没有 `model.onnx`，说明下载或解压不完整。

## 4. 上传到 RK3588 板子

假设板子用户名是 `elf`，板子 IP 是 `192.168.x.x`。

Windows PowerShell：

```powershell
cd D:\rk3588_models\sherpa_asr
$BOARD_IP="192.168.x.x"
scp -r .\sherpa-onnx-paraformer-zh-2024-03-09 elf@${BOARD_IP}:/home/elf/models/
```

Linux / WSL / Git Bash：

```bash
cd /mnt/d/rk3588_models/sherpa_asr
BOARD_IP=192.168.x.x
scp -r sherpa-onnx-paraformer-zh-2024-03-09 elf@${BOARD_IP}:/home/elf/models/
```

上传完成后，SSH 到板子检查：

```bash
ssh elf@192.168.x.x
ls -lh /home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09
```

成功标准：

```text
model.onnx 存在，约 785M
model.int8.onnx 存在，约 217M
tokens.txt 存在
```

## 5. 配置项目使用 FP32 模型和 nau8822 录音

项目启动脚本会读取：

```text
/home/elf/project/project_system/runtime/llm.env
/home/elf/project/project_system/.env.llm
```

推荐写到 `runtime/llm.env`：

```bash
cd /home/elf/project/project_system
mkdir -p runtime
nano runtime/llm.env
```

加入或修改这些行：

```bash
REHAB_ASR_MODEL_DIR=/home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09
REHAB_ASR_PREFER_INT8=0
REHAB_ASR_CAPTURE_DEVICE=plughw:1,0
REHAB_ASR_MIN_RMS=0.0005
REHAB_ASR_MIN_PEAK=0.004
```

含义：

```text
REHAB_ASR_MODEL_DIR        指向新模型目录
REHAB_ASR_PREFER_INT8      0 表示优先 model.onnx；1 表示优先 model.int8.onnx
REHAB_ASR_CAPTURE_DEVICE   当前实测可用的 nau8822 录音设备 plughw:1,0
REHAB_ASR_MIN_RMS          声音能量门槛，太大可能完全识别不到
REHAB_ASR_MIN_PEAK         峰值门槛，太大可能把小声说话挡掉
```

`plughw:1,0` 来自当前 `arecord -l` 里的 `rockchipnau8822`。如果后面换系统、换板子、换外接声卡，设备号必须重新用 `arecord -l` 确认。

保存后检查：

```bash
grep -n "REHAB_ASR" runtime/llm.env
```

## 6. 先确认 nau8822 录音链路

先不用网页，直接在板子上测：

```bash
arecord -l
```

当前成功标准是能看到类似：

```text
card 1: rockchipnau8822 [rockchip-nau8822], device 0: dailink-multicodecs nau8822-hifi-0
```

录 3 秒：

```bash
arecord -D plughw:1,0 -d 3 -r 16000 -c 1 -f S16_LE test.wav
```

回放：

```bash
aplay test.wav
```

如果能听到人声，就判定：

```text
板端录音链路可用，可以继续测试 ASR 和小爱助手。
```

如果听到的是静音，先不要测 ASR。需要检查说话距离、板载麦克风位置、`alsamixer -c 1` 里的 Capture/Mic/ADC 输入和增益。

## 7. USB 麦克风作为后续增强项

当前 USB 麦克风的状态是：`lsusb` 能看到设备，但 `snd-usb-audio` 模块不存在，`arecord -l` 不出现 USB capture 设备。这说明 USB 层识别到了，但当前系统不能把它作为 ALSA 录音设备。

可检查：

```bash
lsusb
arecord -l
cat /proc/asound/cards
cat /proc/modules | grep snd_usb_audio || true
sudo modprobe snd-usb-audio
```

如果出现：

```text
modprobe: FATAL: Module snd-usb-audio not found in directory /lib/modules/5.10.209
```

说明当前镜像缺 USB Audio 驱动模块。此时项目配置改不了，`REHAB_ASR_CAPTURE_DEVICE` 也不能凭空选出一个 USB 麦克风。

后续可以问厂家要：

```text
启用 CONFIG_SND_USB_AUDIO 的系统镜像，或和当前内核 5.10.209 完全匹配的 snd-usb-audio.ko 及依赖模块。
```

但初赛线上视频不再把 USB 麦克风作为阻塞项，先用已验证的 `plughw:1,0`。

## 8. 单独测试 ASR 是否用了 model.onnx

在项目根目录执行：

```bash
cd /home/elf/project/project_system
python3 -m voice.asr_worker test.wav --model-dir /home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09
```

成功时会输出类似：

```json
{
  "ok": true,
  "text": "屈膝训练要注意什么",
  "latency_ms": 1234,
  "model_dir": "/home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09",
  "model_path": "/home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09/model.onnx",
  "tokens_path": "/home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt",
  "recognizer_api": "OfflineRecognizer.from_paraformer",
  "audio_stats": {
    "duration_seconds": 3.0,
    "rms": 0.01,
    "peak": 0.2
  },
  "suppressed_text": null
}
```

重点看：

```text
model_path 必须以 model.onnx 结尾
text 应该接近你说的话
audio_stats.rms / peak 不能太低
suppressed_text 应该是 null
```

如果 `suppressed_text = audio_gate`，说明录音音量太小，或门槛太高。可以临时把门槛调低：

```bash
export REHAB_ASR_MIN_RMS=0.0002
export REHAB_ASR_MIN_PEAK=0.002
python3 -m voice.asr_worker test.wav --model-dir /home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09
```

如果调低后能识别，说明不是模型坏，而是声音太小或门槛太严。

## 9. 重启 8082 并测试网页麦克风

重启：

```bash
cd /home/elf/project/project_system
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

打开：

```text
http://板子IP:8082/train
```

初赛录视频如果要按脱机展示口径，建议板子接显示屏，在板子本机浏览器打开 `/train`。如果临时用电脑浏览器访问板子页面，现在前端也会优先调用板端 `/api/voice/asr_capture`，让后端走 `plughw:1,0` 录音，避免误用电脑麦克风。

网页测试：

1. 先不要训练，保持空闲。
2. 点击“麦克风测试”。
3. 说一句：“屈膝训练要注意什么”。
4. 看文字框是否填入接近的识别文本。
5. 再点击“提交问答”。

成功标准：

```text
不再固定识别成“没有没有有没有没有异议”
文字框里能出现接近真实问题的文字
Qwen/GLM 能基于报告或当前动作回答
```

## 10. 检查网页后端状态

在板子上执行：

```bash
curl -s http://127.0.0.1:8082/api/voice/status | python3 -m json.tool
```

重点字段：

```text
voice.asr.model_dir
voice.asr.model_path
voice.asr.prefer_int8
voice.asr.audio_gate
voice.asr.last_audio_stats
voice.asr.last_suppressed_text
voice.asr.last_error
```

判断：

```text
model_path 以 model.onnx 结尾      说明使用高精度模型
prefer_int8 = false                说明没有强制 int8
last_audio_stats.rms 很低          说明录音太小或没录到人声
last_suppressed_text = audio_gate  说明被音频门槛挡住
last_suppressed_text 是一串文字     说明被当成固定幻听过滤掉
last_error 非空                    说明模型加载或 ASR 调用报错
```

## 11. 测小爱唤醒词

在 `/train` 空闲、休息或训练结束后：

1. 点击“开启唤醒监听”。
2. 说：“小爱，屈膝训练要注意什么？”
3. 看是否自动把“小爱”后面的内容提交成问题。

当前前端支持这些唤醒词：

```text
你好小爱
小爱康复
小爱助手
小爱同学
小爱
```

训练进行中会自动阻止唤醒和问答，避免影响摄像头、骨架、计数和训练语音。

## 12. 能不能脱机演示

可以脱机演示的条件：

```text
板子接显示屏
板子本机浏览器能打开 http://127.0.0.1:8082/train 或 http://板子IP:8082/train
arecord -D plughw:1,0 能录到人声
Qwen 本地 NPU 推理可用
训练结束后小爱助手能识别并提交问题
```

GLM 路线需要手机热点联网，所以是“板子独立运行 + 手机热点联网”；Qwen 路线可以作为“本地无网部署”展示。

如果初赛只是线上拍视频，可以先拍：

```text
板子 8082 页面训练
板子 nau8822 录音测试
麦克风测试识别文字
小爱唤醒提交问题
Qwen 本地回答
GLM 联网回答
```

## 13. 常见问题

### 13.1 还是完全识别不到

先看 `aplay test.wav` 是否能听到声音。如果 `test.wav` 都听不到人声，问题是麦克风、ALSA 设备号或音量，不是 ASR 模型。

### 13.2 `suppressed_text = audio_gate`

说明录音能量太低。先试：

```bash
export REHAB_ASR_MIN_RMS=0.0002
export REHAB_ASR_MIN_PEAK=0.002
```

如果有效，再把这两行写入 `runtime/llm.env`。

### 13.3 `model_path` 还是 `model.int8.onnx`

说明新目录没有 `model.onnx`，或者环境变量没有生效。

检查：

```bash
ls -lh /home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09/model.onnx
grep -n "REHAB_ASR" /home/elf/project/project_system/runtime/llm.env
```

然后重启：

```bash
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

### 13.4 识别变准但速度慢

`model.onnx` 比 `model.int8.onnx` 大，CPU 推理会慢一些。当前设计是训练中不跑唤醒问答，训练结束后再识别和问答，所以第一版优先保证准确率。

如果后面觉得太慢，可以临时切回 int8：

```bash
export REHAB_ASR_PREFER_INT8=1
```

但初赛视频和正式演示都建议优先准确率，用 `model.onnx`。

### 13.5 USB 麦克风不可用

如果 USB 麦克风 `lsusb` 能看到，但 `arecord -l` 看不到 USB capture，而且 `sudo modprobe snd-usb-audio` 提示模块不存在，说明当前系统缺 USB Audio 驱动。

这不影响当前用 `plughw:1,0` 做初赛视频。后续再处理：

```text
1. 问厂家要启用 CONFIG_SND_USB_AUDIO 的系统镜像。
2. 问厂家要和当前内核完全匹配的 snd-usb-audio.ko 及依赖模块。
3. 换当前系统已支持的 USB 声卡或音频小板。
4. 如果继续用 nau8822，优化拾音距离和环境噪声。
```

## 14. 最终验收清单

必选项：

```text
[ ] 板子上存在 /home/elf/models/sherpa-onnx-paraformer-zh-2024-03-09/model.onnx
[ ] runtime/llm.env 指向新模型目录
[ ] REHAB_ASR_PREFER_INT8=0
[ ] REHAB_ASR_CAPTURE_DEVICE=plughw:1,0
[ ] arecord -l 能看到 rockchipnau8822
[ ] arecord -D plughw:1,0 -d 3 -r 16000 -c 1 -f S16_LE test.wav 能录音
[ ] aplay test.wav 能听到人声
[ ] python3 -m voice.asr_worker test.wav 能识别出接近文本
[ ] 输出里的 model_path 以 model.onnx 结尾
[ ] /api/voice/status 里 last_error 为空
[ ] /train 的“麦克风测试”能把问题填入文字框
[ ] “小爱，屈膝训练要注意什么”能触发唤醒问答
```

可选增强项：

```text
[ ] USB 麦克风或 USB 声卡能出现在 arecord -l
[ ] snd-usb-audio 已加载，或系统镜像已内置 USB Audio 支持
[ ] 外接麦克风录音质量明显好于 nau8822
```

完成必选项后，就可以先认为 ASR FP32 和当前板端录音链路已经能支撑小爱助手演示。

