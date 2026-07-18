# Action-quality models

轻量级动作完成度时序模型按动作 ID 分目录：

```text
models/action_quality/sit_to_stand/
models/action_quality/standing_hamstring_curl/
models/action_quality/seated_knee_raise/
```

每个目录可包含：

```text
best.pt
model.onnx
model.rknn
train_summary.json
```

当前稳定在线后端为 ONNX Runtime CPU；RKNN 是可选的后续部署目标。
