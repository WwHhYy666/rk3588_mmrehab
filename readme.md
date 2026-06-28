# RK3588 居家康复训练终端 - 当前进度与交接说明

这份 README 记录当前项目的真实状态、稳定演示路线、板端启动命令、训练数据保存位置，以及下一阶段 NPU 动作质量评分模型的落地流程。

当前主线是：

```text
医生录入标准动作模板
-> 患者进入 /train 做三动作训练
-> CPU/MediaPipe 实时骨架、计数、纠错、TTS
-> 每次可识别尝试保存骨架片段
-> 训练结束生成评估报告
-> NPU Core 2 异步输出动作质量评分
-> 训练结束后再调用 GLM/Qwen/小爱助手解释报告
```

核心原则：训练主链路优先。摄像头、骨架、计数、纠错、TTS 不能被 GLM、Qwen、小爱助手、质量评分模型拖慢。

## 1. 当前稳定演示路线

稳定演示默认仍然是：

```text
姿态识别：CPU / MediaPipe
训练服务：8082
训练页面：http://板子IP:8082/train
医生/模板页面：http://板子IP:8082/doctor
本地大模型：Qwen2.5 RKLLM proxy，可作为 GLM 不可用时的文字兜底
```

当前完整训练动作：

```text
seated_knee_extension       坐姿伸膝
standing_hamstring_curl     站姿屈膝后勾腿
seated_knee_raise           坐姿抬膝
```

当前训练流程支持：

- 三动作按计划进入。
- 每个动作目标次数可调整。
- 动作之间休息 10 秒。
- 标准动作计数通过。
- 幅度不到位触发 `ROM_LOW`。
- 保持时间不足触发 `TUT_LOW`。
- TTS 实时纠错。
- 训练结束后生成每个动作的 report。

## 2. 当前新增进度：异步 NPU 动作质量评分

已新增 `quality_model/`，来源是同学给的 `rk3588_mmrehab-main` 中有用的质量评分模型代码，并迁移到主项目根目录，不整包覆盖原项目。

已接入的能力：

- `realtime/training_session.py`：每次可识别尝试结束后保存骨架序列，并投递到后台评分队列。
- `evaluate/run_evaluate.py`：报告新增 `quality_attempts[]`、`reps[]`、`overall_quality`、`quality_model`。
- `prescription/banzi/record_prescription_http.py`：`/status` 返回轻量 `quality_model` 状态。
- `prescription/common/llm_assistant.py`：GLM/Qwen 摘要带质量评分摘要，但不塞完整骨架序列。
- `prescription/banzi/static/common.js`：报告卡片显示平均分、每次尝试分数、原因、backend。
- `docs/npu_quality_scoring_for_beginners.md`：中文新手落地文档，写了数据、ONNX、RKNN、上传和验证流程。

评分模型不会改变实时计数逻辑。原规则仍然判断 `OK`、`ROM_LOW`、`TUT_LOW` 等，模型只额外给 0-100 分。

推理优先级：

```text
1. quality_model/models/{action_id}/model.rknn  -> RKNNLite / NPU Core 2
2. quality_model/models/{action_id}/model.onnx  -> ONNXRuntime / CPU 兜底
3. 都没有                                      -> 跳过评分，训练正常继续
```

当前仓库里模型目录已有占位，但真正出 NPU 分数还需要后续生成并上传 `.rknn` 文件。

## 3. 每次启动命令

板端进入项目根目录后启动。项目目录以实际上传位置为准；如果沿用之前部署，一般是：

```bash
cd /home/elf/project/project_system
```

如果你明天实际放在另一个目录，比如 `/home/elf/rk3588_mmrehab`，就进入那个目录。关键是目录下要有 `scripts/`、`prescription/`、`realtime/`、`quality_model/`。

启动 8082 + Qwen 相关服务：

```bash
chmod +x scripts/start_rehab_station_qwen.sh scripts/stop_rehab_station_qwen.sh scripts/check_llm_status.sh
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

打开训练页面：

```text
http://板子IP:8082/train
```

打开医生模板页面：

```text
http://板子IP:8082/doctor
```

停止服务：

```bash
./scripts/stop_rehab_station_qwen.sh
```

检查 GLM/Qwen/小爱助手状态：

```bash
./scripts/check_llm_status.sh
```

日志位置：

```text
logs/rehab_8082.log
logs/qwen_flask.log
logs/qwen_proxy.log
```

## 4. 从零开始时删哪些数据

如果明天想完全重新录模板和采数据，可以删除旧模板和旧训练日志。建议先备份再清空。

备份并清空：

```bash
mkdir -p backup_before_lab
mv runtime/active_templates.json backup_before_lab/ 2>/dev/null || true
mv prescription/docs/results/*.json backup_before_lab/ 2>/dev/null || true
mv prescription/docs/summaries/*_summary.md backup_before_lab/ 2>/dev/null || true
mv prescription/docs/results_log.md backup_before_lab/ 2>/dev/null || true
mv evaluate/reports/report_*.json backup_before_lab/ 2>/dev/null || true
mv evaluate/reports/keyframes backup_before_lab/keyframes 2>/dev/null || true
```

如果确定不要备份，才直接删除：

```bash
rm -f runtime/active_templates.json
rm -f prescription/docs/results/*.json
rm -f prescription/docs/summaries/*_summary.md
rm -f prescription/docs/results_log.md
rm -f evaluate/reports/report_*.json
rm -rf evaluate/reports/keyframes
```

不要删除这些代码和配置目录：

```text
quality_model/
evaluate/configs/
feedback/rules/
realtime/configs/
scripts/
prescription/banzi/
prescription/common/
```

清空后必须重新录 3 个医生标准模板，并保存 active template，否则 `/train` 会提示缺少模板。

## 5. JSON 保存在哪里

医生模板 JSON 和患者训练 attempt JSON 都保存到：

```text
prescription/docs/results/
```

摘要保存到：

```text
prescription/docs/summaries/
```

训练结束后的评估报告保存到：

```text
evaluate/reports/report_{action_id}_{时间}.json
```

关键帧图片保存到：

```text
evaluate/reports/keyframes/
```

active template 索引保存到：

```text
runtime/active_templates.json
```

患者训练现在是“一个动作保存一个患者 JSON”。如果跑完整三动作训练，通常会生成 3 个患者 attempt JSON，例如：

```text
prescription/docs/results/patient_001_seated_knee_extension_时间.json
prescription/docs/results/patient_001_standing_hamstring_curl_时间.json
prescription/docs/results/patient_001_seated_knee_raise_时间.json
```

每个患者 JSON 里会包含这个动作的一组训练尝试，包括标准通过、`ROM_LOW`、`TUT_LOW`，以及 `runtime_meta.quality_attempt_segments`。

## 6. 训练质量评分模型需要哪些数据

训练模型最重要的是患者 attempt JSON：

```text
prescription/docs/results/*.json
```

优先使用满足这些条件的 JSON：

```text
runtime_meta.record_role = patient_attempt
runtime_meta.quality_attempt_segments 不为空
```

建议拷回电脑：

```text
prescription/docs/results/*.json
evaluate/reports/report_*.json
```

可选拷回：

```text
evaluate/reports/keyframes/
prescription/docs/summaries/
runtime/active_templates.json
```

模型训练的意义：用多组患者训练数据，让模型学习“这个动作的一段骨架序列应该给多少质量分”。例如标准动作更高分，幅度不足、保持时间不足、轨迹不稳会更低分。

第一版建议每个动作一个模型：

```text
seated_knee_extension       一个模型
standing_hamstring_curl     一个模型
seated_knee_raise           一个模型
```

## 7. ONNX / RKNN 生成流程

推荐流程：

```text
板子采数据
-> 把 prescription/docs/results/*.json 拷回电脑
-> 电脑训练 best.pt
-> 电脑导出 model.onnx
-> WSL/Linux + rknn-toolkit2 转 model.rknn
-> 把 model.rknn 传回板子
-> 板子用 NPU Core 2 异步评分
```

训练和导出 ONNX：

```bash
python quality_model/train.py --action-id seated_knee_extension
python quality_model/export_onnx.py --action-id seated_knee_extension

python quality_model/train.py --action-id standing_hamstring_curl
python quality_model/export_onnx.py --action-id standing_hamstring_curl

python quality_model/train.py --action-id seated_knee_raise
python quality_model/export_onnx.py --action-id seated_knee_raise
```

转 RKNN：

```bash
python quality_model/export_rknn.py --action-id seated_knee_extension
python quality_model/export_rknn.py --action-id standing_hamstring_curl
python quality_model/export_rknn.py --action-id seated_knee_raise
```

最终需要传回板子的模型：

```text
quality_model/models/seated_knee_extension/model.rknn
quality_model/models/standing_hamstring_curl/model.rknn
quality_model/models/seated_knee_raise/model.rknn
```

可选 ONNX 兜底：

```text
quality_model/models/seated_knee_extension/model.onnx
quality_model/models/standing_hamstring_curl/model.onnx
quality_model/models/seated_knee_raise/model.onnx
```

## 8. 板端需要上传的文件

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
quality_model/models/seated_knee_extension/model.rknn
quality_model/models/standing_hamstring_curl/model.rknn
quality_model/models/seated_knee_raise/model.rknn
```

不要上传：

```text
rk3588_mmrehab-main/
__pycache__/
*.pyc
训练缓存
无关历史结果
```

## 9. 质量评分验证

重启服务后：

```bash
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

查看 `/status`：

```bash
curl http://127.0.0.1:8082/status
```

重点看：

```json
"quality_model": {
  "available": true,
  "backend": "rknn",
  "action_id": "seated_knee_extension",
  "model_path": ".../quality_model/models/seated_knee_extension/model.rknn",
  "last_score_time_ms": 3.2,
  "queue_size": 0,
  "worker_alive": true
}
```

判断：

```text
backend = rknn    说明走 NPU 模型
backend = onnx    说明走 ONNX CPU 兜底
available = false 当前动作没有模型，训练仍应正常
queue_size 长期很大 说明评分跟不上，需要优化模型或降低负载
```

验收标准：

- 摄像头和骨架不卡。
- 计数和纠错不卡。
- TTS 不被评分影响。
- 标准动作分数相对高。
- `ROM_LOW`、`TUT_LOW` 分数相对低。
- 报告里有每次尝试分数、原因、平均分。

## 10. GLM / Qwen / 小爱助手当前状态

训练中不应该提交慢 LLM 请求。训练中小爱问答会被快速阻止，避免影响主流程。

训练结束后再检查：

- 有网 + GLM Key：优先 `glm4v_api`。
- 无 GLM Key 或 GLM 不通：降级本地 Qwen 文字回答。
- 两者都失败：显示明确错误，不影响训练报告。

检查命令：

```bash
./scripts/check_llm_status.sh
```

重点字段：

```text
llm.provider
llm.expected_provider_now
llm.api_key_configured
llm.qwen_ready
voice.qa_allowed
current_job.*
last_done.*
```

## 11. 明天实验室建议顺序

```text
1. 清空/备份旧模板和旧训练数据
2. 启动 8082
3. 重新录 3 个医生标准模板
4. 打开 /train 跑完整三动作流程
5. 故意做 ROM_LOW / TUT_LOW，确认纠错和 TTS
6. 采每个动作的数据集
7. 拷 prescription/docs/results/*.json 回电脑
8. 电脑训练 best.pt 并导出 ONNX
9. WSL/Linux 转 RKNN
10. 把 RKNN 传回板子
11. 再跑完整训练，确认 backend = rknn 且摄像头不卡
12. 训练结束后调 GLM/Qwen/小爱助手
13. 最后接 WiFi 模块、显示屏，做脱机演示和 UI 美化
```

## 12. 作品当前成型度

当前已经具备比较完整的比赛作品形态：

- 实时摄像头骨架训练。
- 三动作流程和 10 秒休息。
- 实时计数、纠错、TTS。
- 训练报告。
- 异步质量评分接口和 UI 展示。
- GLM/Qwen/小爱助手训练后解释。
- 后续可接 WiFi 模块、显示屏，做端侧演示。

拿奖概率取决于现场稳定性。优先级是：

```text
稳定主流程 > NPU 评分 > 大模型建议 > 脱机硬件 > UI 美化
```

只要主流程不卡、NPU 评分能跑通、报告和小爱助手能在训练后解释清楚，这个作品就已经有比较完整的展示闭环。
