# Paraformer ASR 部署与验收

语音识别是 8085 训练后问答的可选输入方式，不属于计数和纠错的硬依赖。缺少麦克风、模型或 `sherpa_onnx` 时，患者仍可使用页面文字输入。

## 模型目录

默认目录：

```text
models/audio/asr/paraformer/
```

至少包含模型和 tokens。实际文件名由所用 Sherpa-ONNX 模型包决定，可通过 `REHAB_ASR_MODEL_DIR` 覆盖。

模型文件不进入 Git。

## 环境

```bash
python3 -m pip install sherpa-onnx
export REHAB_ASR_PROVIDER=sherpa_paraformer
export REHAB_ASR_MODEL_DIR=models/audio/asr/paraformer
```

把长期配置写入 `runtime/llm.env`。

## 单独测试

```bash
python3 -m speech.asr_worker /path/to/16k-mono.wav
```

建议先用 16 kHz、单声道 PCM WAV。若音频来自浏览器，先确认系统能正确录音和转码，再排查识别器。

## 8085 页面测试

```bash
./scripts/start_npu_rehab_8085.sh
curl -s http://127.0.0.1:8085/api/voice/status | python3 -m json.tool
```

状态应显示：

- provider 与模型目录正确；
- `sherpa_available=true`；
- 模型文件完整；
- 没有初始化错误。

训练中语音问答会受保护，推荐在训练完成后测试。

## FP32 与 INT8

- FP32 通常准确率更稳，但占用更多内存、推理更慢。
- INT8 更省资源，但必须用实际康复问句和唤醒短句重新验收。
- ASR 不应与姿态主线程争抢到导致视频、计数或提示音卡顿。

## 验收

1. 安静环境下短句可稳定识别。
2. “小爱”、动作名称、幅度、保持时间等关键词识别正确。
3. 连续录音不会残留旧结果。
4. 模型缺失时页面显示可理解的错误，并保留文字问答入口。
5. ASR 失败不影响 `/train`、报告和固定 TTS。
