# 8085 TTS 与三动作连续训练

8085 训练语音按优先级分为：

1. `rehab_app/server/static/assets/tts/` 中的固定 WAV；
2. 可选 Sherpa-ONNX VITS 自然女声；
3. 可选 pyttsx3/espeak 兜底；
4. 仅显示文字。

计数、纠错、休息和完成提示优先于训练后 AI 问答。

## 固定提示音

固定 WAV 随代码仓库提交，适合现场演示。文件名与 `training/configs/tts_phrases.yaml` 的 phrase key 对应。

可先检查 ALSA：

```bash
aplay -l
aplay rehab_app/server/static/assets/tts/welcome.wav
```

通过 `REHAB_AUDIO_OUTPUT_DEVICE` 指定设备：

```bash
REHAB_AUDIO_OUTPUT_DEVICE=plughw:1,0
```

## 自然女声模型

模型包放在：

```text
models/audio/tts/sherpa_vits/
```

通常包含 VITS ONNX、tokens、lexicon 和 FST 资源。实际模型资源被 Git 忽略。

安装可选依赖：

```bash
python3 -m pip install -r requirements-optional.txt
```

若模型不完整，`training/natural_tts.py` 会报告不可用，主训练仍继续。

## 三动作计划

8085 使用：

```text
training/configs/rehab_demo_plan_npu.yaml
```

默认动作：

```text
sit_to_stand
standing_hamstring_curl
seated_knee_raise
```

每个动作包含介绍、站位/方向提示、目标次数、组间休息和固定音频 key。状态机位于 `training/training_session.py`。

## 验收

```bash
./scripts/start_npu_rehab_8085.sh
```

在 `http://板子IP:8085/train` 完成一轮训练，确认：

1. 欢迎、动作开始、计数、纠错、休息和完成提示顺序正确；
2. 固定 WAV 不重复叠播；
3. 离开画面与恢复训练时提示正确；
4. 问答播报不会打断计数提示；
5. 音频设备缺失时页面和状态机仍可继续；
6. 三个动作按计划生成独立报告。
