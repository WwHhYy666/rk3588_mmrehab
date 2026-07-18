# Speech interaction

`speech/` 维护训练后语音交互 worker：

- `asr_worker.py`：Sherpa-ONNX Paraformer ASR；
- `llm_worker.py`：异步 GLM/Qwen 请求、job 状态与 TTS 回调；
- `tests/`：ASR 波形和并发 job 测试。

ASR 模型统一放在 `models/audio/asr/paraformer/`。
