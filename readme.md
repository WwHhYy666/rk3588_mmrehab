# RK3588 骨科居家数字康复终端

> 基于 RK3588 的端侧视觉康复训练、动作评估、语音反馈与智能问答一体化成品工程。

本项目面向骨科术后康复、老年下肢训练、社区康复和居家运动监督场景，将 RK3588 开发板、RGB 摄像头、HDMI 显示、麦克风、扬声器和可选震动马达组合为一套可现场演示的康复训练终端。

项目已经从单点摄像头或姿态识别 Demo 发展为完整产品闭环：医生录入个体化动作模板，患者按训练计划完成动作，系统实时识别人体姿态并完成计数、纠错和语音提示，训练结束后生成图文报告和动作完成度，用户还可以围绕报告进行 GLM/Qwen 康复问答。

本项目用于康复训练辅助评估和交互展示，不进行疾病诊断，不替代医生制定处方。疼痛、头晕、肿胀、麻木或站立不稳时应立即停止训练并联系医生或康复师。

## 1. 成品概览

### 1.1 完整训练闭环

```text
医生选择动作并录入标准模板
        |
        v
系统保存模板、关键帧、动作元数据并更新 active template
        |
        v
患者进入训练台，完成站位/正面/侧面方向确认
        |
        v
摄像头采集 -> MediaPipe 或 RKNN 姿态推理 -> 统一关键点结构
        |
        v
实时状态机判断动作阶段、ROM、TUT、速度、回位和代偿
        |
        +--> 计数、纠错、固定 WAV/TTS 提示、必要时震动反馈
        |
        v
保存 attempt JSON、关键帧和报告
        |
        +--> 规则指标与 ONNX CPU 完成度评分
        +--> GLM / 本地 Qwen / 规则兜底生成解释
        +--> 患者版、医生版总结和康复问答
```

### 1.2 已交付能力

- 医生标准动作模板录入、健康检查、启用模板和模板持久化。
- 患者训练台 `/train`，支持训练计划、目标次数、侧别、站位确认和离屏恢复。
- 坐站训练、站姿屈膝后勾腿、坐姿抬膝三动作主流程。
- 8082 CPU/MediaPipe 稳定模式与 8085 RKNN/NPU 独立模式。
- YOLOv5n raw + RTMPose NPU 主路线，摄像头直接接入，不复用 8082 视频流。
- 实时骨架、动作阶段、有效次数、无效尝试、ROM_LOW、TUT_LOW、TOO_FAST、SHAPE_BAD 等反馈。
- 训练固定 WAV、休息音乐、训练后自然 TTS 和音频所有权管理。
- attempt JSON、模板 JSON、Markdown 摘要、结构化报告、关键帧图片和图文指标卡片。
- 每个动作独立的轻量时序质量评分模型，当前在线服务使用 ONNX CPU 后端。
- GLM 在线模型、本地 Qwen RKLLM、规则问答三层路由和安全兜底。
- Paraformer ASR worker、录音会话管理、问答文本覆盖和问答朗读。
- CPU、内存、温度、NPU、摄像头、姿态后端和服务状态监控。
- 扩展动作目录：上肢训练组和全身训练组各 3 个动作，并支持动作模板、训练、休息音乐和报告。
- RK3588 现场启动、停止、自启动、浏览器大屏、双模式切换和运行状态检查脚本。
- 23 个项目测试文件，覆盖双模式契约、问答路由、报告图片、显示稳定性、扩展训练和语音唤醒等关键合同。

### 1.3 当前进度状态

| 交付模块 | 当前状态 | 成品说明 |
| --- | --- | --- |
| 产品主流程 | 已完成 | 医生模板 -> 患者训练 -> 实时反馈 -> 报告 -> AI 解释完整闭环 |
| 8082 训练台 | 已完成 | CPU/MediaPipe 稳定运行，适合作为默认演示和回退模式 |
| 8085 NPU 训练台 | 已完成 | YOLOv5n raw + RTMPose 独立服务、直连摄像头和 NPU 调试页 |
| 三动作处方 | 已完成 | 坐站、站姿屈膝后勾腿、坐姿抬膝均有配置、反馈和报告入口 |
| 模板与数据 | 已完成 | active template、医生模板、患者 attempt、摘要、报告和关键帧均有落盘路径 |
| 实时评估 | 已完成 | 状态机、起始姿势、方向确认、ROM/TUT、计数、回位和离屏恢复已集成 |
| 语音链路 | 已完成 | 固定训练音频、休息音乐、训练后 TTS、ASR worker 和问答朗读已集成 |
| 智能问答 | 已完成 | GLM、Qwen RKLLM、规则兜底和医疗风险拦截已集成 |
| 完成度评分 | 已完成 | 三个主动作均有训练模型和 ONNX CPU 在线评分入口 |
| 设备监控 | 已完成 | CPU、内存、温度、NPU、摄像头、姿态 FPS 和后端状态均可查看 |
| 扩展动作 | 已完成 | 上肢组、全身组各 3 个动作，支持模板、训练、休息和报告 |
| 部署运维 | 已完成 | 启停、自启动、浏览器大屏、双模式切换、诊断和恢复脚本齐全 |
| 工程测试 | 已完成 | 已建立项目级契约测试和模块测试；板端实机验收通过脚本执行 |

成品口径不是“某个模型文件已经存在”，而是主训练链路在模型、页面、数据、服务和设备资源之间能够协同工作。当前仓库中的可选模型转换路线、历史动作和实验脚本均作为扩展能力保留，不会改变上述成品主流程。

## 2. 产品运行模式

两个服务共用同一个 USB 摄像头，因此通过双模式服务严格二选一。切换脚本会先停止另一模式、等待摄像头释放、启动目标服务、检查 `/status`，确认模式和姿态后端正确后才写入最后成功模式。

| 模式 | 服务端口 | 姿态后端 | 典型用途 | 页面 |
| --- | ---: | --- | --- | --- |
| CPU 稳定模式 | `8082` | MediaPipe Pose | 日常演示、开发机联调、可靠回退 | `/`, `/doctor`, `/train` |
| NPU 成品模式 | `8085` | YOLOv5n raw + RTMPose | RK3588 板端端侧推理、竞赛展示 | `/doctor`, `/train`, `/npu-debug` |

### 2.1 CPU/MediaPipe 模式

8082 由 `prescription/banzi/record_prescription_http.py` 提供统一训练服务，摄像头、医生录入、患者训练、报告、设备状态和问答都在同一个 Web 服务中完成。默认摄像头输入为 640x360，适合先确认主流程和页面交互。

启动脚本会同时尝试管理本地 Qwen Flask 服务和 `18080` RKLLM proxy；没有 Qwen 模型或 GLM Key 时，训练主流程仍可独立运行，问答会显示明确的 provider 状态或使用规则回答。

### 2.2 NPU/RKNN 模式

8085 由 `prescription/banzi/npu_rehab_8085.py` 提供，与 8082 的运行目录、报告目录、姿态初始化和资源释放逻辑隔离。默认模型为：

```text
检测：rknn/yolov5n_raw_fp.rknn
姿态：rknn/rtmpose_m_256x192_fp.rknn
摄像头：/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
处理分辨率：1280x720
网页流：960x540
默认推理流水线：yolov5n_rtmpose
```

NPU 服务支持空闲时释放姿态 Runtime，进入 `/npu-debug` 后按租约加载模型；训练结束或调试页面关闭后释放 NPU 资源，从而把 NPU 计算和本地 Qwen 资源隔离开。

### 2.3 模式切换与自启动

常用脚本：

```bash
./scripts/switch_to_cpu_8082.sh
./scripts/switch_to_npu_8085.sh
./scripts/start_selected_rehab_mode.sh
./scripts/install_rehab_station_autostart.sh
./scripts/open_rehab_station_kiosk.sh
./scripts/check_npu_rehab_8085.sh
./scripts/check_llm_status.sh
```

`runtime/selected_rehab_mode` 保存最后一次成功模式。安装自启动后，设备重启会恢复该模式，并由显示管理脚本打开对应的训练页面或大屏展示页面。

## 3. 用户流程与页面

### 3.1 首页 `/`

统一训练台首页提供医生录入、患者训练和当前服务状态入口。展示设备时可以直接从首页进入患者训练，也可以先进入医生页录制新模板。

### 3.2 医生录入 `/doctor`

医生页用于为指定患者录制动作模板。录制时系统持续采集关键点、可见度、方向信息和时间戳，并在保存前进行模板健康检查。

主要流程：

1. 确认摄像头和人体骨架正常。
2. 选择动作、患者编号和侧别模式（`auto`、`left`、`right`）。
3. 点击录入，完成一遍完整标准动作。
4. 检查有效帧、无效帧、动作方向和模板健康状态。
5. 保存为 active template。

保存后会更新：

```text
runtime/active_templates.json
prescription/docs/doctor_templates/*.json
prescription/docs/results/*.json
prescription/docs/summaries/*_summary.md
```

训练时优先读取当前 active template。模板不是固定角度常量，而是包含患者/医生实际录制的动作序列、目标幅度、动作方向和运行元数据。

### 3.3 患者训练 `/train`

患者页是成品的主展示界面，包含：

- 摄像头实时画面和人体骨架。
- 当前训练动作、目标次数、已完成次数和无效尝试。
- ROM、TUT、速度、动作阶段和完成度。
- 正面/侧面确认、站位提示、回到起始姿势提示。
- 动作纠错、休息倒计时、固定 WAV 和训练结束播报。
- 最近一次报告、关键帧、图文建议和患者/医生总结。
- 康复问答、模型状态、ASR 状态和问答朗读。
- CPU、内存、温度、NPU、摄像头及姿态后端运行状态。

主训练状态包括：

```text
idle
running
resting
awaiting_orientation
awaiting_return
awaiting_care_response
awaiting_action_audio
awaiting_rep_feedback
finished / completed
```

训练期间，慢速评分、LLM、ASR 和问答不会抢占摄像头与动作状态机。问答 worker 会根据训练状态阻止训练中的请求，训练结束后才允许提交或朗读康复建议。

### 3.4 NPU 调试 `/npu-debug`

该页面用于独立验收 RKNN 姿态路线，不需要开始完整康复训练。页面可查看人体框、COCO-17 关键点、姿态 FPS、推理延迟、关键点稳定器、当前 NPU 后端和模型资源状态。

## 4. 默认动作与扩展动作

### 4.1 三动作主训练计划

默认计划位于 `realtime/configs/rehab_demo_plan.yaml`，训练顺序如下：

| 动作 ID | 中文名称 | 主要评价内容 |
| --- | --- | --- |
| `sit_to_stand` | 坐站训练 | 坐姿到站姿的髋膝协同、站起幅度、站稳保持和缓慢坐回 |
| `standing_hamstring_curl` | 站姿屈膝后勾腿 | 膝关节屈曲幅度、支撑侧稳定、髋部代偿和控制速度 |
| `seated_knee_raise` | 坐姿抬膝 | 抬膝高度、髋屈曲、保持时间、骨盆和躯干稳定 |

仓库同时保留 `knee_flexion`、`seated_knee_extension` 等兼容动作配置，可用于历史数据、单动作测试和后续处方扩展。

### 4.2 扩展动作训练

`extension_rehab/` 提供独立的动作目录、模板存储、动作计算和训练会话。扩展动作通过运行时开关启用，默认不改变三动作主训练台。

上肢训练组：

- `seated_biceps_curl`：坐姿屈肘训练。
- `seated_shoulder_flexion`：坐姿前平举。
- `standing_shoulder_abduction`：站姿侧平举。

全身训练组：

- `lateral_step_touch`：站姿侧向迈步。
- `low_impact_step_jack`：低冲击开合步。
- `mini_squat`：半蹲训练。

扩展训练同样支持模板健康检查、目标次数、动作计数、无效尝试、ROM、TUT、速度、代偿错误、组间休息音乐和 JSON 报告。相关配置位于 `extension_rehab/configs/` 与 `extension_rehab/plans/`。

## 5. 核心技术实现

### 5.1 姿态识别与关键点统一

系统通过 `vision/pose_backend_selector.py` 选择姿态后端，并将不同后端输出转换为统一的康复关键点结构。

CPU 路线使用 MediaPipe Pose；NPU 路线使用 YOLOv5n 人体检测和 RTMPose 关键点估计，输出 COCO-17 关键点，再通过 `vision/rknn_pose/pose_frame_adapter.py` 转换、筛选、稳定和映射。

相关模块：

```text
vision/camera_http.py                         摄像头 HTTP 预览
vision/pose_http.py                           姿态 HTTP 预览
vision/gstreamer_gi_capture.py                GStreamer/直连摄像头采集
vision/pose_backend_selector.py               后端选择与状态描述
vision/rknn_pose/rknn_backend.py              RKNN 推理封装
vision/rknn_pose/pose_frame_adapter.py        COCO-17 适配、稳定器和方向指标
vision/rknn_pose/yolov5n_rtmpose_backend.py   YOLOv5n + RTMPose 后处理
```

### 5.2 动作状态机与实时评估

`realtime/training_session.py` 是主流程的实时状态机，负责：

- 读取动作配置和医生模板。
- 建立起始姿势 baseline。
- 判断动作开始、峰值、保持、回位和动作完成。
- 过滤低可见度、跳变、离屏和错误方向。
- 生成有效/无效 attempt 和计数结果。
- 将动作片段投递给后台质量评分。
- 管理休息、恢复、连续错误和训练结束。

每个动作的规则位于 `evaluate/configs/`，反馈文案位于 `feedback/rules/`。常见错误包括：

| 错误码 | 中文含义 | 典型反馈 |
| --- | --- | --- |
| `ROM_LOW` | 动作幅度不足 | 增加抬膝、站起或后勾腿幅度 |
| `TUT_LOW` | 目标位置保持不足 | 到位后先停稳再返回 |
| `TOO_FAST` | 动作过快 | 放慢速度、避免惯性 |
| `SHAPE_BAD` | 轨迹或姿态不规范 | 减少晃动和代偿 |
| `VISIBILITY_LOW` | 关键点可见度不足 | 回到镜头范围并完整入画 |
| `OK` | 动作通过 | 保持当前节奏 |

规则评价先于异步模型评价，确保即使评分模型或 LLM 暂时不可用，计数、纠错和报告主流程仍能完成。

### 5.3 指标、报告与关键帧

训练结果按动作和 attempt 保存。报告包含动作名称、完成次数、无效尝试、错误码、ROM、TUT、速度、模板信息、完成度、关键帧、患者总结、医生总结、风险提示和下一步建议。

主要实现：

```text
evaluate/run_evaluate.py                    规则评估和报告生成
evaluate/core/                              模板健康、指标和报告核心逻辑
prescription/common/result_storage.py       模板与 attempt 存储
prescription/common/report_visuals.py       关键帧、指标卡片和图文报告
scripts/regenerate_latest_npu_reports.py    NPU 历史 attempt 报告重算
```

常用产物：

```text
evaluate/reports/report_*.json
evaluate/reports/keyframes/*
evaluate/reports/extensions/*.json
prescription/docs/patient_attempts/*.json
prescription/docs/doctor_templates/*.json
prescription/docs/summaries/*_summary.md
```

### 5.4 完成度评分模型

质量模型对单次动作片段进行时序评分：将关键点序列整理为固定 30 帧输入，模型输出 0 到 1 的质量值，在线显示时转换为 0 到 100 分并映射等级。

```text
quality_model/features.py       关键点归一化、重采样和输入构造
quality_model/dataset.py        从 attempt/rep_segments 构建训练样本
quality_model/train.py          训练每个动作的 best.pt
quality_model/export_onnx.py    导出 model.onnx
quality_model/export_rknn.py    RKNN 导出工具
quality_model/service.py        在线评分服务
quality_model/models/            三个主动作的模型与训练摘要
```

当前仓库中三个主动作均包含 `best.pt`、`model.onnx` 和 `train_summary.json`。在线 `quality_model/service.py` 实际使用 `onnx_cpu` 后端；RKNN 导出脚本和模型路径为 NPU 化保留的工程接口，不影响默认训练主链路。

### 5.5 GLM、Qwen 与康复问答

问答模块将报告中的结构化指标、动作名称、错误统计和用户问题组合为受约束上下文，提供三层路线：

1. 有网络且配置 Key 时使用 GLM API。
2. 离线或强制本地时使用独立 Qwen RKLLM 服务。
3. provider 不可用、超时或回答不符合结构化事实时使用本地规则回答。

本地 Qwen 不加载到 8082/8085 主进程，而是独立运行：

```text
Qwen RKLLM Flask       127.0.0.1:8080/rkllm_chat
RKLLM proxy             127.0.0.1:18080/health、/generate
康复训练服务             127.0.0.1:8082 或 127.0.0.1:8085
```

问答实现会锁定当前动作和报告，防止旧报告覆盖新问题，也会校验模型回答中的次数、合格数、平均完成度和动作名称。医疗风险问题不调用 Qwen，直接返回停止训练和联系医生的安全建议。

### 5.6 ASR、TTS 与音频优先级

语音模块由 `voice/asr_worker.py`、`voice/llm_worker.py`、`realtime/tts_worker.py` 和 `realtime/audio_player.py` 组成。

- 训练计数、纠错、站位、休息和结束提示优先使用固定 WAV。
- 训练期间阻止问答、ASR 和助手 TTS 抢占音频设备。
- 训练结束后允许自然 TTS 朗读总结和问答结果。
- 录音采用 session 机制，开始监听和结束监听属于同一会话，旧会话不能停止新会话。
- Paraformer ASR 使用独立 worker，模型缺失时不会影响视觉训练。

固定训练音频位于 `prescription/banzi/static/assets/tts/`；TTS 模型和资源位于 `tts/tts_model_pack/`。

### 5.7 震动反馈

`hardware/motro_control/` 提供可选的 GPIO 震动反馈。当前 RK3588 接线配置为 `/dev/gpiochip3` line `11`，对应 GPIO3_B3；代码支持真实 GPIO 和 mock 模式，并提供短震、长震、快速脉冲、间隔脉冲及安全清理。

该模块可以由上层反馈队列调用，也可以单独进行硬件 bring-up。未接入马达时自动回退 mock，不会阻塞训练台。

## 6. 关键 API

### 6.1 页面和视频

```text
GET /                         统一训练台首页
GET /doctor                   医生模板录入
GET /train                    患者训练
GET /npu-debug                RKNN/NPU 调试页（8085）
GET /stream.mjpg              摄像头 MJPEG 流
GET /status                   训练与姿态综合状态
```

### 6.2 状态、报告和设备

```text
GET  /api/system/status       CPU、内存、温度、NPU 和设备状态
GET  /api/reports/latest_by_action
GET  /api/voice/status
GET  /api/voice/mic_status
GET  /api/voice/asr_result?job_id=...
GET  /api/active_template?action_id=...
```

### 6.3 语音与问答

```text
POST /api/voice/ask           提交文本或录音后的康复问题
GET  /api/voice/ask_result    轮询异步问答结果
POST /api/voice/listen_start  开始一次手动录音会话
POST /api/voice/listen_stop   结束当前录音会话
POST /api/voice/asr_capture   结束录音并提交 ASR 任务
POST /api/llm/speak           朗读报告或问答结果
GET  /api/llm/status          GLM/Qwen/provider 状态
```

8082 和 8085 的训练控制、模板录入和扩展动作接口由各自后端实现；前端统一通过状态接口轮询，不直接依赖某一种姿态后端。

## 7. 数据和配置

### 7.1 运行数据

```text
runtime/active_templates.json          当前启用模板索引
runtime/selected_rehab_mode            最后成功服务模式
runtime/llm.env                         板端 LLM/音频环境变量（不纳入 Git）
runtime/npu/                            8085 日志、PID 和运行快照
evaluate/reports/                       训练报告和关键帧
prescription/docs/                      医生模板、患者 attempt、摘要和历史结果
```

### 7.2 核心配置

```text
realtime/configs/rehab_demo_plan.yaml          三动作默认训练计划
realtime/configs/rehab_demo_plan_npu.yaml      8085 NPU 训练计划
realtime/configs/knee_flexion_realtime*.yaml   实时阈值与 NPU 参数
realtime/configs/tts_phrases.yaml              训练播报文案
evaluate/configs/*.yaml                         ROM/TUT/速度等动作目标
feedback/rules/*.yaml                           错误码与反馈文案
extension_rehab/configs/{upper,full}/*.yaml    扩展动作配置
extension_rehab/plans/{upper,full}.yaml         扩展动作训练计划
```

### 7.3 环境变量原则

服务支持通过环境变量覆盖摄像头设备、分辨率、姿态后端、RKNN 模型、NPU Core、LLM provider、ASR 模型目录和音频输出设备。现场优先使用 `scripts/start_rehab_station_qwen.sh` 或 `scripts/start_npu_rehab_8085.sh`，脚本会设置一组经过验证的默认值。

## 8. 快速启动

### 8.1 板端推荐启动

```bash
cd /home/elf/project/project_system
chmod +x scripts/*.sh
./scripts/start_rehab_station_qwen.sh
```

打开：

```text
http://板子IP:8082/
http://板子IP:8082/doctor
http://板子IP:8082/train?display=1
```

### 8.2 直接启动 8082

```bash
python3 prescription/banzi/record_prescription_http.py
```

### 8.3 启动 8085 NPU 成品模式

```bash
./scripts/start_npu_rehab_8085.sh
```

打开：

```text
http://板子IP:8085/doctor
http://板子IP:8085/train?display=1
http://板子IP:8085/npu-debug
```

8085 启动前会检查 8082 是否已释放、检测模型和姿态模型是否存在、Python 依赖是否可用，并按需启动本地 Qwen 与 RKLLM proxy。

### 8.4 停止服务

```bash
./scripts/stop_rehab_station_qwen.sh
./scripts/stop_npu_rehab_8085.sh
```

开发机或没有 systemd 的环境可以直接结束对应 Python 进程；板端正式使用建议始终通过模式切换脚本管理，避免 USB 摄像头被两个服务同时占用。

## 9. 现场演示流程

1. 上电后确认 HDMI、摄像头、扬声器和网络连接。
2. 运行 `check_llm_status.sh`，确认 GLM、Qwen proxy 和问答 provider 状态。
3. 进入 `/doctor`，检查或重新录制三个主动作模板。
4. 进入 `/train`，填写患者编号和目标次数。
5. 按屏幕提示完成正面/侧面站位确认和起始姿势校准。
6. 展示实时骨架、计数、ROM/TUT 纠错、语音提示和休息音乐。
7. 训练完成后打开报告侧栏，展示关键帧、指标卡片、动作完成度和 AI 建议。
8. 训练结束后提交一个康复问题，展示本地 Qwen 或 GLM 回答和 TTS 朗读。
9. 打开设备状态侧栏，展示 CPU、内存、温度、NPU、摄像头和姿态后端。
10. 如需展示 NPU 能力，切换到 8085 并打开 `/npu-debug`，查看人体框、17 点骨架和推理状态。

现场展示优先级：

```text
摄像头与骨架 -> 三动作流程 -> 计数与纠错 -> 报告与关键帧
-> 完成度评分 -> GLM/Qwen 问答 -> ASR/TTS -> 扩展动作与 NPU 调试
```

## 10. 测试与验收

### 10.1 自动化测试

项目测试主要位于 `tests/`、`voice/tests/`、`realtime/tests/`、`vision/rknn_pose/` 和各模块测试目录。根目录测试覆盖：

```text
tests/test_rehab_dual_mode_contract.py       8082/8085 双模式和摄像头归属
tests/test_rehab_qa_action_routing.py        问答意图、动作锁定和安全兜底
tests/test_report_image_paths.py              报告关键帧和图片路径
tests/test_rehab_display_stability.py        展示屏稳定与保活
tests/test_restore_stable_8085_display.py    8085 显示恢复
tests/test_extension_rehab.py                扩展动作和训练会话
tests/test_voice_wake_contract.py             录音会话与语音唤醒契约
```

运行：

```bash
python -m pytest -q
```

### 10.2 板端验收重点

```text
8082: actual_backend=mediapipe，camera_live_ok 非 False
8085: service_mode=npu_rehab，actual_backend=rknn
8085: rknn_pipeline=yolov5n_rtmpose
摄像头: camera.source=direct_device
训练中: 计数、ROM/TUT、固定 WAV 和回位恢复正常
训练后: 报告、关键帧、问答和 TTS 正常
资源: 训练/调试结束后 NPU models_loaded 回到 False
切换: CPU 与 NPU 互不覆盖模板、报告和运行配置
```

### 10.3 常用诊断

```bash
./scripts/check_npu_rehab_8085.sh
./scripts/check_rtmdet_light_model.sh
./scripts/check_llm_status.sh
python3 scripts/benchmark_npu_rehab_8085.py
```

## 11. 项目目录说明

```text
project1/
├── analyze/                  报告分析与总结生成
├── api_use/                  外部模型/API 使用示例
├── docs/                     部署、NPU、语音、显示和竞赛运行手册
├── evaluate/                 动作规则、指标、报告和质量评估
├── extension_rehab/          上肢/全身扩展动作训练
├── feedback/                 反馈规则和前端反馈资源
├── hardware/motro_control/   GPIO 震动马达控制
├── llm/                      RKLLM proxy、音频和本地模型工具
├── prescription/             训练台后端、模板、报告和 Web 前端
├── quality_model/            完成度模型训练、导出、推理和在线服务
├── realtime/                 实时状态机、音频、TTS 和系统监控
├── rknn/                     RKNN 模型、转换配置和 NPU demo
├── runtime/                  本地运行状态、PID、日志和模型资源
├── scripts/                  启停、切换、自启动、诊断和恢复脚本
├── tests/                    项目级自动化契约测试
├── tts/                      TTS 模型和测试资源
├── vision/                   摄像头、姿态后端和 RKNN 姿态适配
├── voice/                    ASR、LLM worker 和语音测试
└── readme.md                项目总说明
```

### 11.1 主要代码入口

| 文件 | 作用 |
| --- | --- |
| `prescription/banzi/record_prescription_http.py` | 8082 统一训练台后端 |
| `prescription/banzi/npu_rehab_8085.py` | 8085 NPU 独立训练服务 |
| `prescription/banzi/static/train.js` | 患者训练页、报告、问答和设备状态 |
| `prescription/banzi/static/doctor.js` | 医生模板录入页面 |
| `realtime/training_session.py` | 主训练状态机和实时动作评估 |
| `realtime/system_monitor.py` | 设备运行状态采集 |
| `evaluate/run_evaluate.py` | 训练后评估与报告生成 |
| `quality_model/service.py` | 在线完成度评分 |
| `prescription/common/llm_assistant.py` | GLM/Qwen/规则问答与总结 |
| `voice/llm_worker.py` | 异步问答 worker |
| `voice/asr_worker.py` | Paraformer ASR worker |
| `llm/rkllm_proxy_server.py` | 本地 Qwen RKLLM proxy |
| `extension_rehab/session.py` | 扩展动作训练会话 |
| `hardware/motro_control/motor_controller.py` | GPIO 震动反馈 |

## 12. 详细文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/unified_training_station.md` | 8082 统一训练台使用说明 |
| `docs/npu_rehab_8085_guide.md` | 8085 NPU 完整训练与板端验收 |
| `docs/npu_rehab_8085_competition_runbook.md` | 竞赛现场运行手册 |
| `docs/rk3588_hdmi_8082_browser_guide.md` | HDMI、大屏、浏览器和自启动 |
| `docs/rk3588_qwen_rkllm_rknn_conversion_guide.md` | Qwen/RKLLM、GLM 路由与部署 |
| `docs/npu_quality_scoring_for_beginners.md` | 完成度评分模型采集、导出和验证 |
| `docs/cpu_quality_scoring_for_codex.md` | ONNX CPU 评分模型说明 |
| `docs/asr_paraformer_fp32_upgrade_guide.md` | Paraformer ASR 模型和启用方法 |
| `docs/natural_tts_and_playlist_guide.md` | 自然 TTS、训练音频和休息播放 |
| `docs/rehab_extensions_competition_guide.md` | 扩展动作竞赛展示流程 |
| `docs/rehab_extensions_board_handoff_zh.md` | 扩展动作板端交接说明 |
| `docs/rtmdet_rtmpose_npu_usage.md` | RTMDet/RTMPose NPU 路线 |
| `docs/rknn_pose_npu_usage.md` | RKNN 姿态适配与验收 |
| `docs/medical_rehab_actions.md` | 康复动作和评价指标设计 |

仓库中的 PDF 资料属于背景学习和过程归档，本 README 以当前代码、脚本、配置和测试为准，不依赖 PDF 才能部署或理解项目。

## 13. 成品交付边界

当前项目的成品主线是一个可以在 RK3588 上部署和现场演示的康复训练终端，核心能力已经完整串联：

```text
模板录入
-> 实时姿态识别
-> 三动作训练
-> 计数与纠错
-> 音频反馈
-> attempt/报告/关键帧
-> 完成度评分
-> GLM/Qwen/规则问答
-> 设备状态与双模式切换
```

仓库中保留的 YOLOv8、RTMDet、RTMPose fixed、RKNN 质量评分导出、ASR 模型替换和扩展动作配置，是产品的可选工程路线和继续迭代接口。它们不会改变当前三动作主流程，也不会成为 8082 稳定模式的启动前置条件。

从产品形态看，本项目已经具备完整的“采集、识别、评价、反馈、记录、解释、部署”闭环，后续工作主要属于模型数据扩大、外观结构优化、更多动作处方和现场长期运行参数沉淀，而不是补齐基础功能。
