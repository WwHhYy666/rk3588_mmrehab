# Quality Model

每个动作一个轻量质量评分模型，输入固定为 `30` 帧关键点序列，输出 `0-1` 分数，线上乘以 `100` 变成 `0-100`。

默认优先级：

1. `model.rknn` on NPU Core 2
2. `model.onnx` on CPU
3. 两者都不存在时静默跳过

训练入口：

```bash
python quality_model/train.py --action-id knee_flexion
python quality_model/export_onnx.py --action-id knee_flexion
python quality_model/export_rknn.py --action-id knee_flexion
```

当前数据集优先从 `prescription/docs/results/*.json` 中的 `runtime_meta.rep_segments` 读取单 rep 序列；历史模板若没有 `rep_segments`，会退回到整段动作作为单样本。
