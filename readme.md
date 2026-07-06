# RK3588 居家康复训练终端

这是一个面向居家康复和康复训练评估的 RK3588 端侧作品。系统用摄像头采集人体姿态，让医生先录入标准动作模板，再让患者在 `/train` 页面完成训练；训练过程中实时显示骨架、计数、纠错和语音提示，训练结束后生成动作报告，并在后台异步计算动作完成度、调用 GLM/Qwen/小爱助手解释结果。

项目的核心目标不是单独做一个姿态 demo，而是做一个尽量完整的康复训练闭环：

```text
医生录入标准动作模板
-> 患者按计划完成三动作训练
-> 摄像头实时姿态识别和骨架展示
-> 状态机计数、纠错、TTS 提醒
-> 保存患者 attempt JSON 和训练报告
-> 后台完成度评分
-> 训练结束后由 GLM/Qwen/小爱助手解释报告
```

当前稳定演示主线仍然优先保证摄像头、骨架、计数、纠错和报告稳定。GLM/Qwen、ASR、质量评分模型、NPU 姿态路线都是围绕主训练链路扩展，不能阻塞训练主线程。

## 当前稳定路线

默认稳定路线：

```text
硬件平台：RK3588
训练服务：8082
稳定姿态后端：CPU / MediaPipe
医生录入页面：http://板子IP:8082/doctor
患者训练页面：http://板子IP:8082/train
本地大模型：Qwen2.5 RKLLM 独立进程，作为 GLM 不可用时的文字兜底
```

当前默认完整训练计划来自 `realtime/configs/rehab_demo_plan.yaml`：

| 动作 ID | 中文名 | 训练要点 |
| --- | --- | --- |
| `sit_to_stand` | 坐站训练 | 坐稳后慢慢站起，站稳后再坐下 |
| `standing_hamstring_curl` | 站姿屈膝后勾腿 | 站稳扶支撑物，目标腿慢慢后勾再放下 |
| `seated_knee_raise` | 坐姿抬膝 | 坐稳侧对镜头，慢慢抬膝并保持 |

系统里还保留了 `seated_knee_extension`、`knee_flexion` 等历史/可扩展动作入口，但当前作品演示主流程以上面三个动作为准。

## 页面与用户流程

### 医生录入 `/doctor`

医生页负责录入标准动作模板。医生或演示者选择动作后，面对摄像头完成一次标准动作，系统保存该动作的骨架序列、关键帧和运行元数据。患者训练时不会和固定角度模板硬编码比较，而是尽量和当前患者/医生录入的 active template 对齐。

关键数据：

```text
runtime/active_templates.json              当前启用的动作模板索引
prescription/docs/results/*.json           医生模板和患者 attempt JSON
prescription/docs/summaries/*_summary.md   模板或训练摘要
```

### 患者训练 `/train`

患者页是作品的主展示页面，包含摄像头预览、骨架图、训练状态、动作计数、实时指标、完成度后台、报告侧栏、康复问答和设备运行状态。

训练时系统按计划依次执行三个动作。每个动作开始前会提示站位和镜头角度；动作过程中状态机判断是否进入动作、是否达到目标幅度、是否保持够时间、是否回到起始姿势。训练中慢任务不会抢主流程资源，AI 问答会在动作进行中被阻止。

关键状态：

```text
running                    正在训练
resting                    组间休息
awaiting_orientation       等待角度/侧身确认
awaiting_return            离开画面后等待回到起始姿势
awaiting_care_response     连续错误后询问是否需要休息
finished                   完整训练结束
```

## 核心功能怎么实现

### 1. 姿态识别与骨架显示

稳定路线使用 MediaPipe Pose 在 CPU 上输出人体关键点。后端把 MediaPipe 的关键点转换成项目内部统一的康复关键点结构，并在视频帧上绘制骨架、关键点、提示文本和调试信息。

相关实现：

```text
prescription/banzi/record_prescription_http.py   摄像头、姿态后端、HTTP 服务总入口
vision/pose_backend_selector.py                  姿态后端选择
vision/rknn_pose/                                RKNN 姿态路线相关实验代码
prescription/banzi/static/train.js               患者训练页面和骨架/状态 UI
prescription/banzi/static/common.js              报告、状态、通用前端工具
```

NPU 姿态路线已经有多条脚本和调试代码，例如 YOLOv8 pose、RTMPose fixed、RTMDet + RTMPose，但它们当前更适合调试和对比。README 默认不把这些路线写成稳定演示默认路径，避免现场演示时牺牲主流程稳定性。

### 2. 实时动作判断、计数和纠错

实时训练逻辑集中在 `realtime/training_session.py`。它负责加载当前动作配置、启动训练状态机、处理每一帧姿态数据、判断动作阶段、生成计数结果，并把提示和报告状态暴露给 `/status`。

每个动作的评估配置在 `evaluate/configs/` 中。系统主要关心两类常见错误：

| 错误码 | 含义 | 示例 |
| --- | --- | --- |
| `ROM_LOW` | 动作幅度不足 | 没有站够高、抬膝不够、后勾腿幅度不够 |
| `TUT_LOW` | 保持时间不足 | 到位后没有稳定保持到目标时长 |

动作被识别为一次尝试后，会进入 `quality_attempt_segments`。其中可计数的尝试进入 `reps[]`，不可计数但有意义的尝试仍可用于纠错和后续质量评分。

相关实现：

```text
realtime/training_session.py                     实时训练状态机和异步评分投递
realtime/configs/rehab_demo_plan.yaml            三动作训练计划
realtime/configs/knee_flexion_realtime.yaml      实时阈值和状态机基础配置
evaluate/configs/*.yaml                          各动作评估目标、幅度和保持时间
feedback/rules/*_feedback.yaml                   错误反馈和纠错文案
```
### 3. TTS 语音提示

训练提示音走实时训练自己的播报链路，优先级高于 AI 问答。动作开始、计数、纠错、休息和训练结束都有对应提示。当前训练提示支持固定 wav 优先，缺少音频时再走 TTS 兜底。

设计原则：

```text
训练计数/纠错播报优先
AI 问答播报低优先级
动作进行中不让慢 LLM 请求抢摄像头和训练状态机
```

相关实现：

```text
realtime/tts_worker.py
realtime/configs/tts_phrases.yaml
prescription/banzi/static/assets/tts/
realtime/natural_tts.py
```

### 4. 训练报告与图文建议

训练结束后，系统把患者 attempt JSON 送入评估流程，生成结构化 report。报告记录动作名、完成次数、错误类型、关键帧、质量评分摘要和 AI 可读的总结字段。患者页右侧报告侧栏按动作显示最近一次报告，不再只显示“报告 1/2/3”这种不清楚的列表。

报告主要保存到：

```text
evaluate/reports/report_{action_id}_{时间}.json
evaluate/reports/keyframes/
```

相关实现：

```text
evaluate/run_evaluate.py                         离线/训练后评估和报告生成
prescription/common/report_visuals.py            报告图片、关键帧和指标卡片
prescription/common/result_storage.py            模板/attempt 保存
prescription/banzi/static/common.js              前端报告卡片和数据展示
```

### 5. 动作完成度评分模型

完成度评分模型是锦上添花层，不改变实时计数逻辑。系统先用规则状态机判断 `OK`、`ROM_LOW`、`TUT_LOW`，再把每次可识别动作尝试的骨架片段交给后台模型，输出 0-100 的完成度分数。

当前实现要点：

```text
输入：固定 30 帧关键点序列
输出：0-100 完成度分数和 grade
投递方式：训练 session 后台队列
当前在线 backend：onnx_cpu
未来目标：model.rknn + RKNNLite / NPU Core 2
```

也就是说，当前代码里的在线评分服务实际检测和加载的是 `model.onnx`，backend 名称是 `onnx_cpu`。仓库中保留了 `export_rknn.py`、`infer_rknn.py` 和 `model.rknn` 路径设计，但 RKNN/NPU 评分应作为下一阶段验证目标，不能在演示说明里写成已经稳定跑通的默认能力。

相关实现：

```text
quality_model/features.py         把骨架序列整理成模型输入
quality_model/model.py            轻量评分模型
quality_model/dataset.py          从患者 JSON 中读取训练样本
quality_model/train.py            训练 best.pt
quality_model/export_onnx.py      导出 ONNX
quality_model/export_rknn.py      导出 RKNN，后续 NPU 化使用
quality_model/service.py          在线评分服务，目前加载 onnx_cpu
```

### 6. GLM / Qwen / 小爱助手

训练结束后，患者可以在右侧康复问答里询问“我刚才哪里没做好”“这个动作要注意什么”等问题。系统会根据当前选中的动作报告构造上下文，有网和 GLM Key 可用时优先走 GLM；无网或强制本地时走 Qwen2.5 RKLLM；两者都不可用时显示明确错误，不用假回答冒充。

Qwen 不加载在 8082 主进程里，而是拆成独立进程：

```text
Qwen RKLLM Flask server   127.0.0.1:8080/rkllm_chat
RKLLM proxy wrapper       127.0.0.1:18080/health 和 /generate
8082 主服务               /train 调用异步 voice worker
```

为了避免旧回答覆盖新问题，前端用 `voicePollToken` 和 `displayedVoiceAnswerJobId` 守住当前 job；后端也通过 voice worker 和 speech generation 状态管理异步结果。

相关实现：

```text
voice/llm_worker.py                         异步问答 worker
llm/rkllm_proxy_server.py                   本地 Qwen proxy
prescription/common/llm_assistant.py        GLM/Qwen 报告总结和问答
prescription/banzi/static/train.js          患者页 AI 报告和康复问答
scripts/check_llm_status.sh                 快速检查 GLM/Qwen 当前状态
```

### 7. ASR 语音识别

项目保留了 Paraformer ASR 接口，用于后续麦克风接入后把语音问题转成文字，再提交给小爱问答。ASR 不属于当前训练主链路，模型缺失或麦克风未接入时不应影响 `/train` 的摄像头、骨架和计数。

相关实现：

```text
voice/asr_worker.py
docs/asr_paraformer_fp32_upgrade_guide.md
```

### 8. 设备运行状态

患者页右侧设备状态侧栏用于查看板端运行情况，例如 CPU、内存、NPU、摄像头和服务状态。这个侧栏是现场调试的重要入口，不能因为添加 AI 面板或报告面板被删掉。

相关实现：

```text
realtime/system_monitor.py
prescription/banzi/static/train.js
```
## 最小运行入口

板端进入项目目录后：

```bash
chmod +x scripts/start_rehab_station_qwen.sh scripts/stop_rehab_station_qwen.sh scripts/check_llm_status.sh
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

打开页面：

```text
医生录入：http://板子IP:8082/doctor
患者训练：http://板子IP:8082/train
展示模式：http://板子IP:8082/train?display=1
```

检查 GLM/Qwen/小爱助手：

```bash
./scripts/check_llm_status.sh
```

可选的自启动和大屏展示脚本：

```text
scripts/install_rehab_station_autostart.sh
scripts/open_rehab_station_kiosk.sh
```

## 关键 API 和数据入口

常用状态/功能接口：

```text
/status                         训练、姿态、报告、质量评分等综合状态
/api/system/status              板端 CPU / NPU / 内存等设备状态
/api/reports/latest_by_action   每个动作最近一次报告
/api/voice/status               语音、ASR、LLM worker 状态
/api/voice/ask                  提交小爱康复问答
/api/voice/ask_result           轮询异步问答结果
```

常用数据目录：

```text
runtime/active_templates.json
prescription/docs/results/*.json
prescription/docs/summaries/*_summary.md
evaluate/reports/report_*.json
evaluate/reports/keyframes/
quality_model/models/{action_id}/model.onnx
quality_model/models/{action_id}/model.rknn
```

## 项目文件索引

| 路径 | 作用 |
| --- | --- |
| `prescription/banzi/record_prescription_http.py` | 8082 后端总入口，管理摄像头、姿态后端、页面路由、实时训练、报告、语音和 LLM worker |
| `prescription/banzi/static/train.js` | 患者训练页，包含摄像头预览、训练 HUD、完成度、报告侧栏、康复问答和设备状态 |
| `prescription/banzi/static/doctor.js` | 医生录入页，负责模板录制和保存 |
| `realtime/training_session.py` | 实时训练状态机，负责动作判断、计数、纠错、休息、离屏恢复和质量评分投递 |
| `realtime/configs/rehab_demo_plan.yaml` | 当前三动作训练计划 |
| `evaluate/run_evaluate.py` | 训练后评估和 report 生成 |
| `feedback/rules/` | 各动作错误反馈规则 |
| `quality_model/` | 完成度评分模型训练、导出、推理和在线服务 |
| `voice/llm_worker.py` | 异步 GLM/Qwen 问答 worker |
| `llm/rkllm_proxy_server.py` | 本地 Qwen RKLLM proxy |
| `voice/asr_worker.py` | Paraformer ASR 语音识别 worker |
| `scripts/` | 板端启动、停止、状态检查、自启动和 NPU 姿态路线脚本 |
| `docs/` | 详细部署、模型转换、ASR、HDMI 展示、NPU 路线说明 |

## 详细文档入口

| 文档 | 用途 |
| --- | --- |
| `docs/rk3588_qwen_rkllm_rknn_conversion_guide.md` | Qwen2.5 RKLLM、GLM/Qwen 路由、8080/18080/8082 架构和验收 |
| `docs/npu_quality_scoring_for_beginners.md` | 质量评分模型数据采集、ONNX/RKNN、上传和验证 |
| `docs/rk3588_hdmi_8082_browser_guide.md` | HDMI、大屏、展示模式和浏览器自启动 |
| `docs/asr_paraformer_fp32_upgrade_guide.md` | Paraformer ASR 模型下载、上传、启用和验收 |
| `docs/newcomer_rtmpose_fixed_quickstart.md` | RTMPose fixed 路线快速调试 |
| `docs/rtmdet_rtmpose_npu_usage.md` | RTMDet + RTMPose NPU 路线 |
| `docs/yolov8_pose_npu_usage.md` | YOLOv8 pose NPU 路线 |
| `docs/rknn_yolov8_pose_smoke_test.md` | YOLOv8 pose smoke test |
| `docs/动作质量评分及语音交互助手.md` | 质量评分和语音交互助手说明 |
## 当前完成度

已经成型的部分：

- 医生模板录入和 active template 管理。
- 患者 `/train` 三动作训练流程。
- CPU/MediaPipe 稳定骨架识别和画面叠加。
- 实时动作状态机、计数、`ROM_LOW` / `TUT_LOW` 纠错。
- 训练 TTS、休息提示和训练结束提示。
- 患者 attempt JSON、summary、report 和关键帧保存。
- 按动作显示最近报告、AI 图文建议和康复问答。
- GLM 优先、本地 Qwen 兜底的问答路线。
- 完成度评分模型的数据结构、训练/导出脚本和 ONNX CPU 在线评分入口。
- 设备运行状态侧栏和状态检查脚本。

需要继续验证或完善的部分：

- RKNN/NPU 质量评分需要真实 `model.rknn` 和板端 smoke test 后才能作为稳定能力展示。
- 多条 NPU 姿态路线仍属于调试/对比路线，现场默认不替代 CPU/MediaPipe。
- ASR 需要麦克风和 Paraformer 模型完整到位后再验收。
- 质量评分数据集还需要更多患者样本，尤其是每个动作的 `OK`、`ROM_LOW`、`TUT_LOW` 平衡样本。
- 展示模式、自启动、HDMI 和 UI 细节需要按比赛现场设备做最终压测。

## 后续改进方向

1. 扩大质量评分数据集

   每个动作采集更多患者 attempt JSON，覆盖标准动作、幅度不足、保持不足、速度过快、动作轨迹不稳等情况。先保证数据可训练，再追求模型分数稳定。

2. 把完成度评分从 ONNX CPU 推到 RKNN/NPU

   当前在线服务实际走 `onnx_cpu`。下一步应完成 `model.rknn` 转换、板端加载、NPU Core 2 运行和 `/status.quality_model.backend` 验证，再写成稳定展示能力。

3. 继续压测实时训练状态机

   重点验证离开画面后返回、回到 baseline、连续错误、休息后继续、TTS 播报期间动作开始等场景，保证不会卡在“请保持静止”“请慢慢坐下”等提示状态。

4. 优化离线语音闭环

   麦克风到位后，把 ASR -> 小爱问答 -> TTS 串起来，同时继续保证训练进行中不触发慢 LLM 推理。

5. 收敛 NPU 姿态路线

   YOLOv8 pose、RTMPose fixed、RTMDet + RTMPose 中选择一条最稳定路线，明确输入尺寸、关键点映射、可见性阈值和现场启动脚本，再考虑替换默认演示路线。

6. 打磨比赛展示体验

   优先保证主流程不卡，再优化大屏展示、状态文案、报告可读性、AI 建议的简洁度和现场一键启动恢复能力。

## 作品展示优先级

现场演示时按这个顺序保稳定：

```text
稳定摄像头和骨架
> 三动作训练流程
> 实时计数和纠错
> 报告生成
> 完成度评分
> GLM/Qwen/小爱助手解释
> ASR、NPU 姿态路线和 UI 美化
```

只要摄像头不卡、三动作流程完整、报告能生成、完成度和 AI 能在训练后解释清楚，这个作品就已经具备完整闭环。后续 NPU 和语音能力要服务这个闭环，而不是反过来拖慢它。
