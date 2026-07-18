# 架构与模块边界

`latest` 分支只维护 RK3588 NPU 8085 康复训练闭环。目录按照“应用编排、感知、训练、评估、反馈、可选智能能力、数据与模型”划分，业务模块通过稳定的数据结构连接，不通过复制脚本连接。

## 主闭环

```text
rehab_app/server/npu_rehab_server.py
  ├─ pose_estimation/       摄像头帧 -> COCO-17/康复关键点
  ├─ training/              关键点序列 -> 动作阶段、计数、实时错误
  ├─ action_feedback/       错误码 -> 页面文案与 TTS
  ├─ evaluation/            attempt -> 指标、错误和报告
  ├─ action_scoring/        关键点时序 -> 可选完成度分数
  ├─ speech/ + llm/         可选 ASR、GLM/Qwen 问答
  ├─ models/                按功能注册本地模型，不进入 Git
  └─ data/ + runtime/       患者数据和进程状态，不进入 Git
```

## 模块职责

| 模块 | 输入 | 输出 | 不负责 |
| --- | --- | --- | --- |
| `rehab_app/` | HTTP 请求、摄像头状态、各模块结果 | 8085 页面与 API | 训练模型、保存权重 |
| `pose_estimation/` | 摄像头帧、RKNN 模型 | 人体框、COCO-17、诊断指标 | 动作计数和医学结论 |
| `training/` | 标准化关键点、训练计划 | 状态、计数、纠错事件、attempt | 离线模型转换 |
| `evaluation/` | 完整 attempt 与动作配置 | 指标、主错误、JSON 报告 | 实时摄像头采集 |
| `action_feedback/` | 稳定错误码和规则 | 页面/TTS 反馈结构 | 姿态推理 |
| `action_scoring/` | 定长关键点序列 | `0-1` 完成度分数 | 替代 ROM/TUT 规则 |
| `speech/`、`llm/` | 音频、问题和报告摘要 | 转写与回答 | 控制训练状态机 |

## 关键接口

- 姿态层输出统一的关键点和可见度结构；训练层不直接依赖 RKNN API。
- 训练层保存 attempt 后调用评估入口，评估层不读取摄像头。
- 实时纠错和报告使用同一组错误码，例如 `ROM_LOW`、`TUT_LOW`、`SHAPE_BAD`。
- 可选动作评分失败时只降级评分，不阻断规则计数、报告和 TTS。
- 所有模型路径从 `models/` 或环境变量解析；所有运行数据写入 `data/`、`runtime/`。

## 扩展约束

新增能力时优先扩展已有模块和注册表。只有当输入、输出和生命周期都形成独立边界时才新增根目录；实验脚本进入仓库前必须接入 8085 闭环、补充文档和测试。
