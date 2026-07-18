# 动作完成度评分：训练、导出与 8085 接入

完成度评分是 8085 规则状态机之后的增强层。实时计数仍由 ROM、TUT、回位和可见性规则决定；模型只对已切分的动作片段异步评分，不能阻塞摄像头和训练线程。

## 数据流

```text
训练状态机
-> quality_attempt_segments
-> 固定 30 帧关键点张量
-> ONNX CPU 或后续 RKNN 推理
-> 0-100 分与 grade
-> /status 和训练报告
```

当前稳定在线后端是 ONNX Runtime CPU。RKNN 完成度模型属于可选部署目标，不应写成默认已跑通能力。

## 目录

```text
action_scoring/features.py
action_scoring/dataset.py
action_scoring/model.py
action_scoring/train.py
action_scoring/export_onnx.py
action_scoring/export_rknn.py
action_scoring/service.py
models/action_quality/<action_id>/
```

模型权重和真实训练数据被 Git 忽略。

## 数据检查

将匿名化数据放到本地数据集目录后运行：

```bash
python3 -m action_scoring.check_dataset
```

重点检查动作 ID、片段长度、关键点完整性、标签分布和异常样本。

## 训练与导出

```bash
python3 -m action_scoring.train --action-id sit_to_stand
python3 -m action_scoring.export_onnx --action-id sit_to_stand
```

其他动作：

```text
standing_hamstring_curl
seated_knee_raise
```

ONNX 验证：

```bash
python3 -m action_scoring.infer_onnx --action-id sit_to_stand --input <attempt.json>
```

RKNN 导出需要单独的 x86 转换环境和匹配的 RKNN Toolkit2：

```bash
python3 -m action_scoring.export_rknn --action-id sit_to_stand
```

转换机和板端运行库版本必须匹配。

## 8085 接入

`training/training_session.py` 在动作片段结束后把评分任务放入后台队列。服务状态可从：

```bash
curl -s http://127.0.0.1:8085/status | python3 -m json.tool
```

查看 `quality_model`、backend、model path、最近错误和最近分数。

## 验收

1. 没有模型时计数、纠错和报告仍能完成。
2. 有 ONNX 模型时后台分数能进入 `/status` 和报告。
3. 慢推理不会降低姿态采集和页面视频帧率。
4. 正确、幅度不足、保持不足样本的分数排序合理。
5. 模型原始分与规则结果冲突时，页面最终建议仍以可解释规则为主。
6. 真实患者数据不进入公开仓库。
