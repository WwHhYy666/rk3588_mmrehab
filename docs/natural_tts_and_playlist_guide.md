# 自然女声 TTS 与三动作连续训练说明

这份文档说明如何使用新的自然女声 TTS，以及如何完成“医生录入 3 个个性化模板 -> 患者一键连续训练 3 个动作”的完整流程。

## 1. 当前主入口

启动 8082 统一训练台：

```bash
python3 prescription/banzi/record_prescription_http.py
```

浏览器访问：

```text
http://板子IP:8082
```

页面：

| 路由 | 作用 |
| --- | --- |
| `/` | 训练台首页 |
| `/doctor` | 医生录制标准模板 |
| `/train` | 患者实时训练 |

## 2. 新自然女声 TTS

当前实时训练优先使用：

```text
tts/tts_model_pack/vits-aishell3.onnx
```

对应代码封装在：

```text
realtime/natural_tts.py
```

实时训练不会直接运行 `tts/test.py`。`tts/test.py` 只作为单独测试脚本保留。

实时训练使用：

```text
realtime/tts_worker.py
```

它会按顺序尝试：

1. `natural_tts`：sherpa-onnx VITS 自然女声
2. `pyttsx3/espeak`：旧兜底播报
3. `mock`：只打印文本

在 `/train` 页面可以看到 TTS backend 状态。

## 3. 板端依赖检查

在板子上检查：

```bash
python3 -c "import sherpa_onnx, soundfile"
which aplay
```

如果缺 Python 依赖：

```bash
pip install sherpa-onnx soundfile
```

如果缺 `aplay`，需要安装 ALSA 工具，或者把 `NaturalTTS` 里的播放命令替换成板子上可用的播放器。

## 4. 医生录入三个动作模板

进入：

```text
http://板子IP:8082/doctor
```

### 动作 1：坐姿伸膝

动作名称填写：

```text
seated_knee_extension
```

建议姿势：

```text
坐在椅子上，侧对镜头，保持髋、膝、踝可见。
```

操作：

```text
录入标准动作 -> 做一遍坐姿伸膝 -> 保存为 active template
```

### 动作 2：站姿屈膝后勾腿

动作名称填写：

```text
standing_hamstring_curl
```

建议姿势：

```text
站稳，侧对镜头，手扶椅背或墙面，保持髋、膝、踝可见。
```

操作：

```text
录入标准动作 -> 做一遍站姿屈膝后勾腿 -> 保存为 active template
```

### 动作 3：坐站训练

动作名称填写：

```text
sit_to_stand
```

建议姿势：

```text
坐在稳定椅子上，侧对或斜侧对镜头，缓慢站起再坐下。
```

操作：

```text
录入标准动作 -> 做一遍坐站训练 -> 保存为 active template
```

三个动作都保存后，`runtime/active_templates.json` 中应包含：

```text
seated_knee_extension
standing_hamstring_curl
sit_to_stand
```

患者训练时会读取这些最新医生模板。

## 5. 患者完整连续训练

进入：

```text
http://板子IP:8082/train
```

填写：

```text
患者编号：patient_001
目标次数：建议拍视频时填 3
侧别模式：auto，或与医生录制时一致
```

点击：

```text
开始完整训练
```

系统会按 `realtime/configs/rehab_demo_plan.yaml` 的顺序执行：

```text
坐姿伸膝
-> 休息 10 秒
-> 站姿屈膝后勾腿
-> 休息 10 秒
-> 坐站训练
-> 全部完成
```

## 6. 语音流程

自然女声会播报：

```text
康复训练即将开始，请坐稳并面向镜头。
现在开始坐姿伸膝，请侧对镜头。
一
二
三
做得很好，请休息一下。
现在开始站姿屈膝后勾腿，请侧对镜头。
一
二
三
做得很好，请休息一下。
现在开始坐站训练，请侧对镜头。
一
二
三
今天的训练完成得很好，请注意休息。
```

纠错语音优先级低于计数语音。计数只播：

```text
一、二、三
```

只有动作达到当前医生模板的目标区间并返回后才计数。动作不到位时只播纠错，例如“再伸直一点”“小腿再往后弯一点”“再站起来一点”，不会增加完成次数。

## 7. 训练结果输出

每个动作完成后会保存一个 patient attempt，并调用 evaluate 生成报告。

三个动作使用不同的 evaluate 指标：

```text
坐姿伸膝：knee_extension_angle，髋-膝-踝膝关节伸展角。
站姿屈膝后勾腿：hamstring_curl_flexion_angle，髋-膝-踝膝关节屈曲角。
坐站训练：hip_rise_height_ratio，髋部相对坐姿 baseline 的上升高度 / 肩髋距离；辅助检查 knee_extension_angle。
```

报告里的 `metric.metric_name` 可以直接看到本动作使用的指标。三个动作仍然和医生最新 active template 比，不使用统一固定角度。

输出位置：

```text
prescription/docs/results/
evaluate/reports/
```

训练页会显示三个动作的 report 路径。


## 8. 常见问题

### 8.1 自然女声没有声音

看 `/train` 页面里的 TTS backend。

如果显示：

```text
pyttsx3/espeak
```

说明自然女声初始化失败，系统降级到了旧播报。

如果显示：

```text
mock
```

说明真实播报都失败了，只打印文本。

### 8.2 完整训练无法开始

通常是三个 active template 没录齐。

请回到 `/doctor`，分别录入：

```text
seated_knee_extension
standing_hamstring_curl
sit_to_stand
```

### 8.3 动作 2 为什么侧对镜头

站姿屈膝后勾腿主要看小腿向后弯曲，侧对镜头更容易看到髋、膝、踝三点，也更适合 MediaPipe 识别和拍视频演示。
