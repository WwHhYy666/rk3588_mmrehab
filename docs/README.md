# 文档索引

本目录只保留与 NPU 8085 康复训练闭环、板端部署及当前可选能力直接相关的文档。运行入口和最短操作步骤以仓库根目录的 [README](../README.md) 为准。

## 部署与验收

| 文档 | 内容 |
| --- | --- |
| [npu_rehab_8085_guide.md](npu_rehab_8085_guide.md) | 8085 服务、摄像头、双 RKNN 模型、性能基准、systemd 和故障排查 |
| [rk3588_qwen_rkllm_rknn_conversion_guide.md](rk3588_qwen_rkllm_rknn_conversion_guide.md) | Qwen RKLLM、GLM 路由和模型转换 |
| [asr_paraformer_fp32_upgrade_guide.md](asr_paraformer_fp32_upgrade_guide.md) | Paraformer ASR 模型部署与验收 |
| [natural_tts_and_playlist_guide.md](natural_tts_and_playlist_guide.md) | 自然女声 TTS、固定提示音和连续训练 |
| [npu_quality_scoring_for_beginners.md](npu_quality_scoring_for_beginners.md) | 完成度评分模型的训练、ONNX/RKNN 导出与部署 |

## 设计与说明

| 文档 | 内容 |
| --- | --- |
| [medical_rehab_actions.md](medical_rehab_actions.md) | 三个康复动作及医学依据 |
| [architecture.md](architecture.md) | 主闭环数据流、模块职责和扩展约束 |
| [quality_scoring_voice_assistant_zh.md](quality_scoring_voice_assistant_zh.md) | 完成度评分与语音助手设计 |
| [technical_overview_zh.md](technical_overview_zh.md) | 系统技术说明 |
| [rk3588_npu_defense_notes_zh.md](rk3588_npu_defense_notes_zh.md) | NPU 方案答辩与工程细节 |
| [technical_document_zh.pdf](technical_document_zh.pdf) | 排版版技术文档 |

## 模块内文档

- [models/README.md](../models/README.md)：所有模型的功能分类、默认路径与提交规则。
- [rehab_app/README.md](../rehab_app/README.md)：8085 应用入口与服务编排。
- [pose_estimation/README.md](../pose_estimation/README.md)：摄像头和姿态估计。
- [training/README.md](../training/README.md)：动作状态机、TTS 与设备状态。
- [evaluation/README.md](../evaluation/README.md)：评估输出和命令行入口。
- [action_feedback/README.md](../action_feedback/README.md)：可解释纠错规则。
- [action_scoring/README.md](../action_scoring/README.md)：完成度模型代码。
- [speech/README.md](../speech/README.md)：ASR 与异步问答 worker。
- [llm/README.md](../llm/README.md)：本地 RKLLM 代理。
- [scripts/README.md](../scripts/README.md)：部署、检查和运维脚本。
- [rehab_data_storage_guide.md](rehab_data_storage_guide.md)：模板、attempt、报告与隐私边界。

历史 8082、YOLOv8、RTMDet、固定 ROI、Windows 和双模式文档已从 `latest` 分支移除，避免与当前唯一维护路线混淆。
