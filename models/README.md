# Model registry

本目录统一存放项目运行所需的深度学习模型及其配套资源。Git 只跟踪各级 `README.md`，实际权重、词表和运行时资源均由 `.gitignore` 排除。

```text
models/
├─ vision/                 人体检测与姿态估计
├─ audio/
│  ├─ asr/paraformer/      语音识别
│  └─ tts/sherpa_vits/     自然语音合成
├─ language/qwen/          本地 Qwen RKLLM
└─ action_quality/         轻量级动作完成度时序模型
```

| 功能 | 默认位置 | 是否为 8085 必需 | 路径覆盖变量 |
| --- | --- | --- | --- |
| YOLOv5n 人体检测 | `models/vision/yolov5n_raw_fp.rknn` | 是 | `RKNN_DET_MODEL` |
| RTMPose 关键点 | `models/vision/rtmpose_m_256x192_fp.rknn` | 是 | `RKNN_RTMPOSE_MODEL` |
| Paraformer ASR | `models/audio/asr/paraformer/` | 否 | `REHAB_ASR_MODEL_DIR` |
| Sherpa VITS TTS | `models/audio/tts/sherpa_vits/` | 否 | `REHAB_TTS_MODEL_DIR` |
| Qwen RKLLM | `models/language/qwen/qwen1_5b.rkllm` | 否 | `QWEN_RKLLM_MODEL` |
| 动作完成度评分 | `models/action_quality/<action_id>/model.onnx` | 否 | 各动作配置文件 |

模型应从项目受控存储或对应官方渠道获取，并遵守各自许可证。不要把患者数据、私有模型或密钥提交到公开仓库。
