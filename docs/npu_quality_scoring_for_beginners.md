# RK3588 动作质量评分模型新手落地文档

这份文档给明天去实验室调试用。目标不是一上来就折腾 NPU，而是先保证现有 8082 训练主流程稳定，再逐步加入质量评分模型、GLM/Qwen、小爱助手、WiFi 模块、显示屏和 UI。顺序很重要，别反过来。

## 1. 总体目标

最终作品形态应该是：

- `/train` 页面能稳定显示摄像头 + 骨架图。
- CPU 跑 MediaPipe 姿态识别，负责实时训练主流程。
- 三个动作按计划训练，每个动作目标次数可以调整，动作之间休息 10 秒。
- 做得标准时计数通过，故意做错时能实时纠错和 TTS 提示。
- NPU Core 2 后台跑动作质量评分模型，训练主线程不卡。
- 训练结束后显示每次尝试的质量分、原因和平均分。
- 训练结束后再调用 GLM/Qwen/小爱助手生成建议，不影响训练。
- 最后装上 WiFi 模块和显示屏，尽量做到脱机或弱联网可演示。

一句话：训练主流程必须先稳，NPU 评分、大模型、小爱助手、UI 都是锦上添花。

## 2. 明天去实验室先干什么

### 第一步：先验证完整训练流程，不要先搞模型

先把板子启动起来：

```bash
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

打开：

```text
http://板子IP:8082/train
```

先检查这些东西：

- 摄像头画面是否持续刷新。
- 骨架图是否稳定显示。
- FPS 是否接近之前稳定状态，目标是尽量接近 15 FPS。
- 三个动作是否能按顺序进入。
- 每个动作完成后是否休息 10 秒进入下一个动作。
- 做标准动作时是否能计数。
- 故意动作不到位时是否能提示 `ROM_LOW`。
- 故意保持时间不够时是否能提示 `TUT_LOW`。
- TTS 是否能正常纠错。
- 终端不要反复刷 `BrokenPipeError` 或其他 traceback。

这一步的判断标准：不用管质量评分有没有分，先确认训练主流程没有错误。

### 第二步：采集质量评分模型需要的数据

质量评分模型需要数据集。现在系统已经会在训练时记录每次尝试的骨架序列，位置在训练结果 JSON 的：

```text
runtime_meta.quality_attempt_segments
runtime_meta.rep_segments
```

你需要为每个动作采一些样本：

```text
sit_to_stand
standing_hamstring_curl
seated_knee_raise
```

建议每个动作都采：

- 标准动作：多做几组。
- 幅度不到位：故意少抬一点，产生 `ROM_LOW`。
- 保持时间不够：故意很快放下，产生 `TUT_LOW`。
- 速度或轨迹不稳：后面可选。

演示时不一定固定 5 次。比如目标设置为 3 次标准通过，你中间故意做 2 次错的，那么最终报告可能显示 5 次尝试：3 次通过 + 2 次纠错。平均分按实际有分数的尝试计算，不写死 5 次。

### 第三步：把数据从板子拷回电脑

训练结果通常在项目的结果目录里，重点保留包含 `runtime_meta.quality_attempt_segments` 的患者训练结果 JSON。你可以把相关结果目录拷回电脑项目里，用来训练模型。

不要只拷 report。训练模型更需要患者尝试 JSON 里的骨架序列。

### 第四步：在电脑上训练并导出 ONNX

推荐在电脑上做训练和 ONNX 导出，因为 PyTorch 环境更容易装，速度也更稳。

在项目根目录执行：

```bash
python quality_model/train.py --action-id sit_to_stand
python quality_model/export_onnx.py --action-id sit_to_stand

python quality_model/train.py --action-id standing_hamstring_curl
python quality_model/export_onnx.py --action-id standing_hamstring_curl

python quality_model/train.py --action-id seated_knee_raise
python quality_model/export_onnx.py --action-id seated_knee_raise
```

生成结果：

```text
quality_model/models/sit_to_stand/best.pt
quality_model/models/sit_to_stand/model.onnx
quality_model/models/standing_hamstring_curl/best.pt
quality_model/models/standing_hamstring_curl/model.onnx
quality_model/models/seated_knee_raise/best.pt
quality_model/models/seated_knee_raise/model.onnx
```

如果提示没有样本，说明数据还不够，或者训练结果没有包含 `quality_attempt_segments`。

### 第五步：用 WSL 或 Linux 环境把 ONNX 转 RKNN

你的理解基本对：

1. 板子采数据。
2. 数据拷回电脑。
3. 电脑训练出 `best.pt`。
4. 电脑导出 `model.onnx`。
5. 在 WSL/Linux 的 RKNN Toolkit2 环境里把 `model.onnx` 转成 `model.rknn`。
6. 再把 `model.rknn` 传回板子。
7. 板子运行时优先用 RKNN/NPU。

转换命令：

```bash
python quality_model/export_rknn.py --action-id sit_to_stand
python quality_model/export_rknn.py --action-id standing_hamstring_curl
python quality_model/export_rknn.py --action-id seated_knee_raise
```

生成结果：

```text
quality_model/models/sit_to_stand/model.rknn
quality_model/models/standing_hamstring_curl/model.rknn
quality_model/models/seated_knee_raise/model.rknn
```

注意：`export_rknn.py` 需要 `rknn-toolkit2`。一般在 WSL/Linux 转换环境里做，不建议在板子上临时折腾转换环境。

### 第六步：把 RKNN 模型传回板子

必传模型：

```text
quality_model/models/sit_to_stand/model.rknn
quality_model/models/standing_hamstring_curl/model.rknn
quality_model/models/seated_knee_raise/model.rknn
```

可选兜底模型：

```text
quality_model/models/sit_to_stand/model.onnx
quality_model/models/standing_hamstring_curl/model.onnx
quality_model/models/seated_knee_raise/model.onnx
```

ONNX 是兜底：如果 RKNN 不存在或加载失败，会尝试 ONNX CPU 推理。但演示 NPU 时，重点还是 `.rknn`。

### 第七步：验证 CPU 和 NPU 是否真的异步并行

重启 8082：

```bash
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

打开训练页，再做完整训练。观察：

- 摄像头和骨架不能因为评分而卡顿。
- 计数、纠错、TTS 不能变慢。
- `/status` 里应该能看到质量模型状态。

可以看：

```bash
curl http://127.0.0.1:8082/status
```

重点字段：

```json
"quality_model": {
  "available": true,
  "backend": "rknn",
  "action_id": "sit_to_stand",
  "model_path": ".../quality_model/models/sit_to_stand/model.rknn",
  "last_score_time_ms": 3.2,
  "queue_size": 0,
  "worker_alive": true
}
```

判断方式：

- `backend = rknn`：说明走 NPU 模型。
- `backend = onnx`：说明只走 ONNX CPU 兜底。
- `available = false`：说明当前动作没有可用模型，训练仍应正常。
- `queue_size` 长期很大：说明评分跟不上，需要优化模型或降低评分频率。

### 第八步：训练结束后再看大模型和小爱助手

质量评分、GLM/Qwen、小爱助手应该都在训练结束后检查，不要训练中疯狂点。

训练结束后检查：

- 报告卡片是否显示每次尝试的评分。
- 是否显示平均分。
- 好动作分数是否相对高。
- `ROM_LOW`、`TUT_LOW` 的分数是否明显降低。
- GLM 有 Key 且联网时，能不能生成图文建议。
- 没有 GLM Key 或断网时，Qwen 是否能给文字回答。
- 小爱助手是否能基于报告回答问题。

如果大模型失败，不应该影响训练报告生成。

### 第九步：装 WiFi 模块和显示屏，做脱机演示

等上面都稳定后，再做硬件集成：

- WiFi 模块：保证板子能联网或能稳定连热点。
- 显示屏：训练页能直接在屏幕上打开。
- 如果完全脱机：GLM 不可用是正常的，要走本地 Qwen 或显示明确错误。
- 脱机状态下核心演示仍然应该成立：摄像头骨架 + 三动作训练 + 纠错 + NPU 质量评分。

最后再改 UI，不要一开始就改 UI。UI 是最后包装，主流程稳定才值得包装。

## 3. 评分模型到底怎么算

每个动作最好一个模型：

```text
sit_to_stand 一个模型
standing_hamstring_curl 一个模型
seated_knee_raise 一个模型
```

原因是每个动作的标准姿态、幅度、保持时间和轨迹都不一样。一个模型硬吃所有动作，第一版容易不稳定。

现在第一版质量分是补充指标，不替代原来的规则：

- 原规则负责判断：`OK`、`ROM_LOW`、`TUT_LOW` 等。
- 质量模型负责给 0-100 分。
- 报告里同时显示规则原因和模型分数。

分档：

```text
>= 85：优秀
70 - 84.99：良好
50 - 69.99：一般
< 50：需改进
```

你演示时可以这样讲：

- 标准动作分数更高。
- 幅度不到位会因为 `ROM_LOW` 降分。
- 保持时间不够会因为 `TUT_LOW` 降分。
- 最后平均分反映本组动作整体质量。

## 4. CPU/NPU 并行是不是这样就可以了

是，整体路线是对的：

- CPU/MediaPipe 继续实时跑摄像头和骨架。
- 每次尝试结束后，把那次骨架序列扔进后台队列。
- 后台 worker 调 `quality_model.service.score_rep()`。
- 有 `.rknn` 就优先用 RKNNLite 跑 NPU Core 2。
- 没有 `.rknn` 但有 `.onnx` 就用 ONNX CPU 兜底。
- 都没有就跳过评分，训练继续。

这就形成了 CPU 和 NPU 的并行：CPU 管训练主链路，NPU 管训练后的单次尝试评分。注意它不是“每一帧都让 NPU 评分”，而是“每次尝试结束后评分”，这样更不容易卡。

## 5. 要传到板子上的文件

代码必传：

```text
quality_model/
realtime/training_session.py
evaluate/run_evaluate.py
prescription/banzi/record_prescription_http.py
prescription/common/llm_assistant.py
prescription/banzi/static/common.js
prescription/banzi/static/train.js
docs/npu_quality_scoring_for_beginners.md
```

模型必传：

```text
quality_model/models/sit_to_stand/model.rknn
quality_model/models/standing_hamstring_curl/model.rknn
quality_model/models/seated_knee_raise/model.rknn
```

模型可选兜底：

```text
quality_model/models/sit_to_stand/model.onnx
quality_model/models/standing_hamstring_curl/model.onnx
quality_model/models/seated_knee_raise/model.onnx
```

不要传：

```text
rk3588_mmrehab-main/
__pycache__/
*.pyc
训练中间缓存
无关历史结果
```

## 6. 明天建议时间安排

建议按这个顺序：

1. 先启动 8082，完整跑一遍三动作训练。
2. 故意做错动作，确认纠错和 TTS 正常。
3. 确认报告能生成，质量评分字段即使是空也不能报错。
4. 开始采每个动作的数据。
5. 把数据拷回电脑。
6. 训练每动作模型，导出 ONNX。
7. WSL/Linux 转 RKNN。
8. 把 RKNN 传回板子。
9. 再跑完整训练，确认 `backend = rknn`。
10. 检查训练时不卡、训练后有分数。
11. 再调 GLM/Qwen/小爱助手。
12. 最后接 WiFi 模块、显示屏，调 UI。

## 7. 有没有拿奖概率

有概率，但前提是演示要稳。

这个作品的亮点是完整度：

- 有实时骨架训练。
- 有多动作流程。
- 有错误纠正和语音提示。
- 有 NPU 动作质量评分。
- 有训练报告。
- 有大模型/小爱助手解释。
- 还能上显示屏做端侧演示。

评委通常更看重稳定闭环，而不是单个功能多炫。最容易扣分的是现场卡顿、摄像头没画面、模型调用失败却没有兜底。所以优先级一定是：稳定主流程 > NPU 分数 > 大模型建议 > 脱机硬件 > UI 美化。

如果你明天能把完整流程跑稳，再把 RKNN 评分接上，作品就已经有比较完整的比赛形态了。后面 UI 和脱机体验做好，拿奖概率会明显提高。

## 8. 常见问题

### 没有数据集怎么办

先用板子多跑几次训练采数据。标准动作和故意错误动作都要有。没有数据集，训练脚本就没有东西可学。

### 是不是必须先有 ONNX 才能转 RKNN

是。流程一般是：

```text
best.pt -> model.onnx -> model.rknn
```

### 是不是要传回电脑再用 WSL 转 RKNN

推荐这样做。板子主要负责采数据和演示，电脑/WSL 负责训练和转换，最后把 `.rknn` 传回板子跑。

### 只有 ONNX 可以吗

可以作为临时兜底，但它是 CPU 推理，不算真正使用 NPU。比赛展示 NPU 时最好有 `.rknn`。

### 评分会不会影响摄像头

正常不会。现在评分是后台队列异步跑，摄像头、计数、TTS 不等评分结果。如果它影响了，说明模型太慢、队列堆积、或板子上还有别的重任务。

### 完全做错为什么没有分

如果完全没有形成一次可识别尝试，系统就没有骨架片段可评分。要展示错误评分，动作要“能被识别成一次尝试”，比如幅度不够或保持不够，而不是完全乱动或走出画面。



