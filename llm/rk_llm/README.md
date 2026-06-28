# RK LLM 语音反馈管线

这是一个面向 RK3588 骨科居家数字康复终端的语音和端侧 LLM 验证模块。它把麦克风录音、ASR 语音识别、Qwen 推理、Windows SAPI TTS 语音合成串成可单独调试的脚本，也可以跑完整闭环。

当前模块适合作为主仓库的 `llm/rk_llm/` 子模块维护。虚拟环境、本地模型权重和运行输出不进入 Git。

## 功能

- 麦克风录音并保存为 WAV
- WAV 音频转文本，支持 `google`、`sphinx`、`whisper`、`faster-whisper`
- Qwen2-VL 文本推理，也支持 `echo` 后端做快速 smoke test
- Windows SAPI 文本转语音
- 单轮实时语音交互
- 固定间隔流式语音交互
- 完整测试记录保存到 `pipeline/runs/`

## 目录

```text
.
├── asr_audio.py              # 音频转文字
├── llm_infer.py              # 文本输入 -> Qwen/echo 输出
├── play_wav.py               # 播放 WAV
├── realtime_voice_qwen.py    # 单轮或多轮实时语音交互
├── record_audio.py           # 麦克风录音
├── run_pipeline.py           # 文件驱动的完整管线
├── stream_voice_qwen.py      # 定时循环语音管线
├── tts_text.py               # 文本转 WAV
├── voice_qwen_core.py        # 公共核心逻辑
├── requirements*.txt
└── pipeline/
    ├── test_text/            # 可提交的测试文本
    ├── asr/                  # 本地运行输出，Git 忽略
    ├── llm/                  # 本地运行输出，Git 忽略
    ├── record_speech/        # 本地录音输出，Git 忽略
    ├── runs/                 # 完整管线运行记录，Git 忽略
    └── tts/                  # 本地 TTS 输出，Git 忽略
```

## 环境

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-whisper.txt
```

如果需要 Google Cloud TTS，再安装：

```powershell
pip install -r requirements-google-tts.txt
```

## 模型

默认模型优先级：

1. 环境变量 `QWEN_MODEL_PATH`
2. 当前目录下的 `qwen2_vl_2b_instruct/`
3. Hugging Face model id `Qwen/Qwen2-VL-2B-Instruct`

本地模型权重很大，不提交到 Git。离线运行时可以把模型放到模块目录下：

```text
qwen2_vl_2b_instruct/
```

或者通过环境变量指定：

```powershell
$env:QWEN_MODEL_PATH="D:\models\qwen2_vl_2b_instruct"
```

## 常用命令

列出录音和播放设备：

```powershell
python .\record_audio.py --list-devices
python .\play_wav.py --list-devices
```

录音：

```powershell
python .\record_audio.py --duration 5 --output .\pipeline\record_speech\input.wav
```

ASR：

```powershell
python .\asr_audio.py --audio-file .\pipeline\record_speech\input.wav --backend faster-whisper --language zh-CN --whisper-model tiny --output .\pipeline\asr\asr_text.txt
```

LLM 推理：

```powershell
python .\llm_infer.py --input-text .\pipeline\asr\asr_text.txt --output .\pipeline\llm\llm_output.txt --device-map none --max-new-tokens 64
```

快速 smoke test 可以不加载 Qwen：

```powershell
python .\llm_infer.py --backend echo --input-text .\pipeline\test_text\zh.txt --output .\pipeline\llm\echo_output.txt
```

TTS：

```powershell
python .\tts_text.py --input-text .\pipeline\llm\llm_output.txt --language zh-CN --voice Huihui --output-audio .\pipeline\tts\tts_output.wav --play
```

如果系统播放器打不开 WAV，可以用当前 Python 音频设备播放：

```powershell
python .\play_wav.py --audio-file .\pipeline\tts\tts_output.wav
```

## 完整管线

使用 `pipeline/test_text/zh.txt` 生成输入音频，再依次执行 ASR、LLM、TTS：

```powershell
python .\run_pipeline.py --source test-text --language zh-CN --asr-backend faster-whisper --whisper-model tiny --llm-backend qwen --device-map none --max-new-tokens 32 --play-final
```

输出目录示例：

```text
pipeline/runs/<timestamp>_pipeline/
  00_seed_text.txt
  01_input_audio.wav
  02_asr_text.txt
  03_llm_output.txt
  04_tts_output.wav
  manifest.json
```

英文测试：

```powershell
python .\run_pipeline.py --source test-text --language en-US --test-text .\pipeline\test_text\en.txt --asr-backend faster-whisper --whisper-model tiny --tts-voice Zira
```

## 实时和流式

单轮实时语音交互：

```powershell
python .\realtime_voice_qwen.py --language zh-CN --duration 5 --asr-backend faster-whisper --whisper-model tiny --llm-backend qwen --device-map none --tts-voice Huihui --once
```

固定间隔循环：

```powershell
python .\stream_voice_qwen.py --language zh-CN --interval 20 --duration 5 --asr-backend faster-whisper --whisper-model tiny --llm-backend qwen --device-map none --tts-voice Huihui
```

不加载 Qwen、不播放 TTS 的快速检查：

```powershell
python .\stream_voice_qwen.py --language zh-CN --interval 3 --duration 1 --max-cycles 1 --asr-backend faster-whisper --whisper-model tiny --llm-backend echo --skip-tts
```

## 备注

- `faster-whisper tiny` 速度快，中文可用，但准确率一般；需要更高准确率可改为 `base` 或更大模型。
- `sphinx` 更适合基础英文离线测试，不建议用于中文。
- Windows SAPI 中文语音通常使用 `Huihui`，英文可用 `Zira`。
- CPU 也能跑 Qwen2-VL，但速度较慢；有 NVIDIA GPU 时建议使用 CUDA 版 PyTorch。
