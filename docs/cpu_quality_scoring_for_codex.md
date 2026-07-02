# CPU 动作质量评分模型交接说明

这份文档给后续接手质量评分模型的 Codex 使用。当前决定是：姿态检测主流程继续优先保证摄像头、骨架、计数和 TTS 稳定；动作质量评分先走 CPU ONNX 异步推理，不走 RKNN/NPU。

## 目标

- 每个动作训练一个轻量质量评分模型，输入单次动作的骨架序列，输出 0-100 分。
- 评分只作为报告和建议的补充，不参与实时计数，不影响主训练流程。
- 主训练线程只负责保存 `quality_attempt_segments` 并投递到后台队列；CPU 推理必须在独立 worker 线程中完成。
- NPU 占用率现阶段可以先显示模拟值，后续等 NPU 姿态检测同学修好预处理和解码后再接真实占用。

## 数据流程

推荐流程：

```text
1. 板端采数据，每个动作采 OK / ROM_LOW / TUT_LOW 等情况
2. 拷回 prescription/docs/patient_attempts/*.json 或 prescription/docs/results/*.json
3. 电脑端运行 quality_model/train.py，训练出 best.pt
4. 电脑端运行 quality_model/export_onnx.py，导出 model.onnx
5. 跳过 RKNN 转换，不生成 model.rknn
6. 上传 model.onnx 到板端 quality_model/models/{action_id}/model.onnx
7. 板端 quality_model 后台 worker 使用 ONNXRuntime CPUExecutionProvider 推理
```

目标模型路径：

```text
quality_model/models/sit_to_stand/model.onnx
quality_model/models/standing_hamstring_curl/model.onnx
quality_model/models/seated_knee_raise/model.onnx
```

## 实现要求

- `quality_model/service.py` 默认优先选择 `model.onnx`，backend 名称建议显示为 `onnx_cpu`。
- `quality_model/infer_onnx.py` 使用 `CPUExecutionProvider`，并设置 ONNXRuntime 线程数为 1，避免和摄像头/姿态推理抢 CPU。
- 保留现有异步队列设计：`training_session.py` 只调用 enqueue，不在主训练线程里直接推理。
- 队列满、模型缺失、推理失败都只写入 `quality_model.last_error`，不能打断训练。
- 报告里保留 `quality_attempts[]`、`overall_quality`、`quality_model.backend`、`quality_model.last_score_time_ms` 等字段。

## 数据集要求

优先读取：

```text
runtime_meta.quality_attempt_segments
```

如果旧文件没有这个字段，再兼容：

```text
runtime_meta.rep_segments
```

每个 segment 至少需要：

- `action_id`
- `primary_error`
- `rom`
- `rom_target`
- `rom_diff`
- `tut_seconds`
- `tut_target`
- `tut_ratio`
- `skeleton_sequence`

第一版可跑通流程的数据量：每个动作 30-50 段。

比较稳的数据量：每个动作 80-120 段。

建议每个动作尽量均衡：

```text
OK       40%-50%
ROM_LOW  25%-30%
TUT_LOW  25%-30%
```

如果后面加入 `TOO_FAST`、`EARLY_RETURN`、`SHAPE_BAD`，每类至少先采 10 段以上。

## 旧日志能不能用

能用，而且要优先用 `runtime_meta.quality_attempt_segments`，不要只看 `runtime_meta.rep_results`。

字段含义：

```text
runtime_meta.rep_results              只包含成功计数的 reps，通常看起来都是 OK
runtime_meta.quality_attempt_segments 包含成功和失败的每一次尝试，训练质量模型优先读这里
runtime_meta.invalid_attempts         失败尝试摘要
runtime_meta.rep_segments             成功计数的骨架段，是 quality_attempt_segments 的子集
```

当前发现第一个动作 `sit_to_stand` 和第三个动作 `seated_knee_raise` 之前阈值偏宽，部分明显不到位的动作也被写成 OK。处理建议：

- 新阈值修好以后继续采新数据，新数据优先用于训练。
- 旧日志里的 `ROM_LOW`、`TUT_LOW` 通常可以保留。
- 旧日志里的 OK 要按当前动作配置重新算一次，不要直接相信旧标签。
- 对 `seated_knee_raise`，旧 OK 中 `rom_diff > 0.10` 的样本改成 ROM_LOW。
- 对 `sit_to_stand`，旧 OK 中 `rom_diff > 0.08` 的样本改成 ROM_LOW。
- 如果没有 ROM_LOW，但 `tut_ratio` 低于当前 `realtime.tut_ratio_min`，改成 TUT_LOW。

本仓库已提供非破坏式重标注脚本：

```bash
python quality_model/relabel_attempt_segments.py --input-glob "prescription/docs/patient_attempts/*.json" --out "quality_model/label_reviews/threshold_relabel_manifest.csv"
```

脚本不会修改原始 JSON，只输出 CSV。训练脚本后续应优先使用 CSV 里的 `relabel_error` 作为训练标签，并用 `keep_for_training=true` 过滤有骨架序列的样本。

当前 20260702 这批日志已经筛好：

```text
quality_model/label_reviews/threshold_relabel_20260702.csv
```

筛选结果：110 个 attempt segment，其中 20 个标签按新阈值发生变化。按 `relabel_error` 统计：

```text
seated_knee_raise: OK 7, ROM_LOW 21, TUT_LOW 5
sit_to_stand: OK 15, ROM_LOW 15, TUT_LOW 9
standing_hamstring_curl: OK 18, ROM_LOW 11, TUT_LOW 9
```

## 给 Codex 的实施提示

实现时不要改摄像头采集、姿态检测、实时计数和 TTS 的主链路。只改质量评分模块的 backend 选择、ONNX CPU 推理线程设置、dataset 读取兼容和文档。完成后至少验证：

```bash
python -m py_compile quality_model/service.py quality_model/infer_onnx.py quality_model/dataset.py realtime/training_session.py
python quality_model/train.py --action-id sit_to_stand
python quality_model/export_onnx.py --action-id sit_to_stand
```

板端验证时打开 `/train`，确认摄像头 FPS、骨架绘制、动作计数、TTS 都不受质量评分影响。
