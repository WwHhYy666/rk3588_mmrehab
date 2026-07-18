# Lightweight action scoring

`action_scoring/` 维护动作完成度时序模型的代码；权重统一放在根目录 `models/action_quality/`。

```text
action_scoring/
├─ features.py             关键点序列预处理
├─ dataset.py              本地数据集加载
├─ model.py                轻量时序网络
├─ train.py                PyTorch 训练
├─ export_onnx.py          ONNX 导出
├─ export_rknn.py          RKNN 导出
├─ infer_onnx.py           ONNX 推理
├─ infer_rknn.py           RKNN 推理
├─ service.py              在线异步评分接口
├─ registry.py             动作与模型路径注册
├─ configs/                各动作训练参数
└─ tests/                  特征和校准测试
```

输入固定为 30 帧 COCO-17/康复关键点序列，输出 `0-1`，页面展示时映射为 `0-100`。

当前稳定在线后端为 `model.onnx` + ONNX Runtime CPU。`model.rknn` 是可选后续目标，不能替代实时 ROM/TUT 规则。

```bash
python -m action_scoring.train --action-id sit_to_stand
python -m action_scoring.export_onnx --action-id sit_to_stand
python -m action_scoring.infer_onnx --action-id sit_to_stand --input <attempt.json>
```
