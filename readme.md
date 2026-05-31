# RK3588 骨科居家数字康复终端项目总说明

这份 `readme.md` 是当前项目的总入口文档。

它的用途不是只做展示，而是给下面两类人快速接手：

- 新开的 Codex 对话
- 第一次接手这个项目的新同学

如果你后面在 `vision/`、`hardware/`、`prescription/` 或其他新文件夹里单独开对话，建议先把这个文件发给 Codex 或先让它阅读这个文件，再继续具体任务。

## 1. 项目是做什么的

本项目的目标是做一套：

**基于 RK3588 的骨科居家数字康复终端**

核心思路是把下面这条闭环跑通：

```text
视觉感知 -> 动作评估 -> 数字处方录入/读取 -> 文本转语音反馈 -> 物理反馈
```

更具体一点，就是：

- 用摄像头采集人体动作
- 用姿态识别提取骨骼关键点
- 根据康复动作标准判断动作是否达标
- 如果动作不规范，就通过语音和震动提醒患者

这个项目的整体方向来自 `嵌赛开发流程_v2.pdf`，当前已经明确的五大功能模块是：

- 视觉感知层：摄像头 + MediaPipe 提取骨骼关键点，RKNN/NPU Pose 作为可选后端已接入 8082
- 数字处方层：保存患者的动作模板
- 临床评估层：DTW、ROM、TUT 等动作评估逻辑
- 智能反馈层：端侧大模型 + TTS 语音反馈
- 物理反馈层：GPIO 驱动震动马达

## 2. 当前目录结构

当前项目根目录是：

`D:\rk3588\project`

目前已经看到的结构如下：

```text
project/
├── docs/
│   ├── offline_closed_loop_demo.md
│   ├── medical_rehab_actions.md
│   ├── natural_tts_and_playlist_guide.md
│   ├── unified_training_station.md
│   ├── rknn_pose_deployment_plan.md
│   └── rknn_pose_npu_usage.md
├── evaluate/
│   ├── windows/
│   │   └── run_evaluate.py
│   ├── banzi/
│   │   └── run_evaluate.py
│   ├── core/
│   │   └── action_metrics.py
│   ├── configs/
│   │   ├── knee_flexion.yaml
│   │   ├── seated_knee_extension.yaml
│   │   ├── seated_knee_raise.yaml
│   │   ├── standing_hamstring_curl.yaml
│   │   └── sit_to_stand.yaml        # 旧坐站动作配置，当前演示不使用
│   ├── samples/
│   ├── reports/
│   ├── legacy/
│   ├── run_evaluate.py
│   └── README.md
├── feedback/
│   ├── rules/
│   │   ├── knee_flexion_feedback.yaml
│   │   ├── seated_knee_extension_feedback.yaml
│   │   ├── seated_knee_raise_feedback.yaml
│   │   ├── standing_hamstring_curl_feedback.yaml
│   │   └── sit_to_stand_feedback.yaml        # 旧坐站动作反馈，当前演示不使用
│   ├── feedback_engine.py
│   └── tts/
│       ├── banzi/
│       │   ├── tts_board.py
│       │   └── explain.md
│       ├── windows/
│       │   ├── prescription_voice_demo.py
│       │   └── tts_test.py
│       ├── _current_script.js
│       ├── _legacy_script.js
│       └── _tmp_record_page.js
├── hardware/
│   ├── docs/
│   └── motro_control/
│       ├── motor_controller.py
│       ├── motor_hardware_test.py
│       ├── hardware_bringup.md
│       ├── explain.md
│       └── readme.md
├── prescription/
│   ├── docs/
│   │   ├── results/
│   │   ├── summaries/
│   │   ├── results_log.md
│   │   └── rk3588_prescription_guide.md
│   ├── common/
│   │   └── result_storage.py
│   ├── windows/
│   │   ├── local_result_sink.py
│   │   └── read_prescription_json.py
│   ├── banzi/
│   │   ├── record_prescription_http.py
│   │   ├── camera_preflight.py
│   │   └── static/
│   ├── local_result_sink.py
│   ├── read_prescription_json.py
│   ├── record_prescription_http.py
│   ├── record_prescription_http.py.bak_20260517_legacy_rebuild
│   ├── record_prescription_http_legacy.py
│   └── record_prescription_json_improved.py
├── realtime/
│   ├── configs/
│   │   ├── knee_flexion_realtime.yaml
│   │   └── rehab_demo_plan.yaml
│   ├── feedback_runtime.py
│   ├── knee_flexion.py
│   ├── natural_tts.py
│   ├── state_machine.py
│   ├── system_monitor.py
│   ├── training_session.py
│   └── tts_worker.py
├── tts/
│   ├── test.py
│   └── tts_model_pack/
│       ├── vits-aishell3.onnx
│       ├── lexicon.txt
│       ├── tokens.txt
│       ├── phone.fst
│       ├── date.fst
│       └── number.fst
├── vision/
│   ├── camera_test.py
│   ├── pose_mediapipe_demo.py
│   ├── camera_http.py
│   ├── pose_http.py
│   ├── pose_backend_selector.py
│   ├── rknn_pose/
│   ├── guide.md
│   ├── explain.md
│   ├── x11_windows_preview.md
│   └── apt_update_troubleshooting.md
├── runtime/
│   └── active_templates.json
└── readme.md
```

说明：

- `readme.md`
  - 当前这个文件
  - 是全项目总入口
- `feedback/tts/`
  - 当前板端和 Windows 侧文本转语音模块目录
  - `banzi/` 存放板端播报脚本
  - `windows/` 存放 Windows 侧语音验证脚本
- `feedback/rules/` 和 `feedback/feedback_engine.py`
  - 当前离线评估和实时训练的反馈规则目录
  - 三动作反馈规则已拆分，输出 screen / TTS 文案 / motor mock pattern
  - 当前训练流程中的马达仍严格 mock，不访问真实 GPIO
- `evaluate/`
  - 当前离线评估与动作专属指标目录
  - 已包含 ROM、TUT、DTW、速度检查、错误归类和 `core/action_metrics.py`
  - 当前演示三动作不复用同一指标：坐姿伸膝、站姿屈膝后勾腿、坐姿抬膝各有独立 metric
  - `windows/run_evaluate.py` 是 Windows 本机评估入口，`banzi/run_evaluate.py` 是 RK3588 板端评估入口
  - `run_evaluate.py` 保留为兼容入口，评估报告输出到 `evaluate/reports/`
  - 旧 replay 原型已归档到 `evaluate/legacy/`
- `hardware/motro_control/`
  - 目前是硬件马达控制模块的主要工作目录
  - 已包含真实硬件测试脚本和上板联调文档
- `prescription/`
  - 当前统一训练台 Web 入口所在目录
  - 已具备 RK3588 浏览器版医生模板录制、患者实时训练、板端本地保存、Windows 同步保存、结果读取和评估报告生成
  - `/train` 新 UI 已接入实时训练状态、系统资源卡片、摄像头预览降级提示和视觉链路诊断
  - `banzi/` 放板端录制入口，`windows/` 放本机接收和读取入口，`common/` 放共享保存逻辑
  - 根目录同名脚本保留为兼容 wrapper，旧命令仍可用
  - 当前结果默认保存在 `prescription/docs/results/`，中文摘要保存在 `prescription/docs/summaries/`
- `realtime/`
  - 当前患者实时训练核心模块目录
  - 包含状态机、三动作连续训练会话、实时反馈映射、TTS worker、自然女声 TTS backend 和板端资源监控
  - NPU load 监控已改为只解析 `/sys/kernel/debug/rknpu/load` 中 `CoreX` 后面的百分比，读不到时显示 `N/A`
  - `configs/rehab_demo_plan.yaml` 定义三动作连续训练顺序和 10 秒组间休息
- `runtime/`
  - 运行态管理目录
  - `active_templates.json` 保存当前 active template，路径使用相对 `PROJECT_ROOT` 的写法，方便后续迁移到 RK3588 Linux
- `tts/`
  - 当前自然女声 TTS 模型目录
  - `tts/test.py` 是单独测试脚本，实时训练通过 `realtime/natural_tts.py` 复用 sherpa-onnx VITS 模型
  - `tts_model_pack/` 保存 `vits-aishell3.onnx`、词典、tokens 和 fst 规则文件
- `vision/`
  - 目前是摄像头、骨骼识别和板端预览验证目录
  - 默认姿态后端仍是 MediaPipe CPU；RKNN/NPU Pose 已作为可选后端接入 8082，主 Demo 稳定演示仍建议先走 MediaPipe
  - NPU 支线已适配 RKNN YOLOv8-Pose split 输出和 `[1,17,3,8400]` keypoint shape，并加入目标腿锁定与关键点稳定化；目前仍比 CPU 抖，适合作为并行调优路线

注意：

- 当前目录名写的是 `motro_control`，不是 `motor_control`
- 这是现有事实，后续如果你想统一命名，可以再单独整理

## 3. 当前开发进度总览

为了避免新对话误判，下面统一使用这几种状态：

- 已完成
- 已验证 Demo
- 待联调
- 待硬件
- 待建立

### 3.1 hardware 模块现状

状态总结：

- `motor_controller.py`：已完成并已上板验证
- Mock 模式本地运行：已验证 Demo
- 真实硬件 GPIO 联调：已完成

目前已完成内容：

- 已实现 `MotorController` 类
- 已实现三种震动模式：
  - `short_buzz()`
  - `long_buzz()`
  - `interval_buzz()`
- 已保留 `rapid_buzz()` 作为旧命令兼容写法
- 已实现 `gpio_worker()` 队列工作线程函数
- 已支持：
  - `mock_mode=True` 模拟模式
  - `mock_mode=False` 真实 GPIO 模式
- 已实现 `cleanup()` 资源清理
- 已加 `threading.Lock` 保护 GPIO 操作
- 已写模块说明文档 `explain.md`

当前已经验证过的事情：

- 在普通电脑上直接运行 `motor_controller.py`，Mock 模式可以正常打印：
  - `short_buzz`
  - `long_buzz`
  - `interval_buzz`
  - `cleanup`
- 程序可以正常退出，软件流程没有报错
- 真实硬件使用 `GPIO3_B3` 已测试通过
  - sysfs 手动测试使用 `GPIO107`
  - `python-periphery` 使用 `/dev/gpiochip3` line `11`
- `motor_hardware_test.py` 已可作为真实硬件测试入口：
  - `--real short`
  - `--real long`
  - `--real interval`
  - `--real all`

后续还需要联调的事情：

- 马达模块已经独立验证，后面需要接入主线程
- 主程序里应让 `gpio_worker()` 消费 `Motor_Queue`，再和视觉判断、TTS 反馈一起联调

### 3.2 vision 模块现状

状态总结：

- `camera_test.py`：代码已完成，窗口预览路径待补验证
- `pose_mediapipe_demo.py`：代码已完成，窗口预览路径待补验证
- `camera_http.py`：已验证 Demo
- `pose_http.py`：已验证 Demo
- `pose_backend_selector.py`：已建立后端选择器，支持 `mediapipe`、`rknn`、`auto`
- `vision/rknn_pose/`：已具备 RKNN/NPU Pose 后端、后处理、frame adapter、smoke test 和 replay compare
- 与主程序联动：已接入 8082，当前重点是板端稳定演示和性能调优

目前已完成内容：

- `camera_test.py`
  - 用 OpenCV 打开摄像头
  - 代码层面已支持显示实时画面
  - 代码层面已支持按 `ESC` 或 `q` 退出
  - 但在当前“纯 SSH + 笔记本不能直接当板端显示器”的环境下，这条本地窗口路径还没有真正完成验证
- `pose_mediapipe_demo.py`
  - 用 MediaPipe Pose 识别人体骨架
  - 绘制姿态骨架
  - 打印左膝关键点坐标与可见度
  - 当前同样受限于本地窗口显示路径，尚未完成最终的窗口预览验证
- `camera_http.py`
  - 提供浏览器版摄像头实时预览
  - 用于在 X11 暂时没打通时快速确认画面
- `pose_http.py`
  - 提供浏览器版骨骼识别实时预览
  - 可在浏览器中查看骨架识别结果
- `pose_backend_selector.py`
  - 读取 `POSE_BACKEND`，支持 `mediapipe`、`rknn`、`auto`
  - 默认仍选择 `mediapipe`，`auto` 可优先尝试 RKNN 再回退
- `vision/rknn_pose/`
  - RKNN YOLOv8-Pose / NPU 姿态识别支线已具备可运行后端
  - 已接入后处理、frame adapter、smoke test、replay compare 和 8082 主流程
  - RKNN 第一版只使用 2D 图像角度，不提供 3D，`z` 视为无效

当前说明：

- 视觉链路已经能在板端独立跑起来
- 板端摄像头链路已经能工作
- 板端骨骼识别已经能工作
- 当前主要通过浏览器方案进行实时查看
- 当前 8082 主 Demo 默认仍走 MediaPipe CPU 姿态识别路线
- RKNN/NPU Pose 是并行支线，已能接入 8082 并识别骨架；后处理已对齐官方 YOLOv8-Pose split 输出，关键点已做目标腿锁定和稳定化，但侧身场景仍比 CPU 更容易抖
- NPU 当前目标是验证 RK3588 NPU 和继续降低 CPU 占用，不删除也不替换现有 MediaPipe 稳定路线
- 当前“本地窗口 / X11 弹窗预览”仍未完全打通，因此不能把 `camera_test.py` 和 `pose_mediapipe_demo.py` 的窗口显示路径算作最终已验证
- X11 / 本地窗口仍然是可选优化方向，不再是当前必须先解决的主阻塞项
- 这些脚本当前仍属于“功能验证脚本”，还没有和主线程架构、动作评估逻辑、硬件反馈逻辑做正式集成

截至当前阶段，同学 B 第 1 周的骨骼识别任务仍可以视为已完成，但这个“完成”主要是指浏览器方案已经够用，不是指本地窗口预览路径已经完全打通。

### 3.3 处方模块现状

状态总结：

- `prescription/`：已完成当前阶段核心实现
- 录入、读取、Windows 同步保存、板端本地保存：已完成
- 8082 浏览器统一训练台：已形成当前主 Demo 入口
- `prescription/banzi/record_prescription_http.py`：已支持 `/doctor` 医生录入、`/train` 患者三动作连续实时训练、整组结束自动评估

当前更细的状态判断如下：

- `prescription/`
  - 已完成当前阶段核心实现
  - 已完成处方录入、读取、Windows 同步保存和板端本地保存
  - 已支持板端采集、推理、浏览器录制、实时训练、结果读取与双端保存
  - 当前已显式区分三类 JSON：
    - `doctor_template`：医生/标准动作模板
    - `patient_attempt`：患者训练动作
    - `evaluation_report`：评估输出报告
  - 标准动作保存后会写入 `runtime/active_templates.json`，作为当前 active template
  - 患者完成 `target_reps` 后保存 attempt JSON，并调用 `evaluate/banzi/run_evaluate.py` 生成完整报告
  - 当前完整训练要求三个 active template：`seated_knee_extension`、`standing_hamstring_curl`、`seated_knee_raise`
  - `/train` 新 UI 已区分 `stream_available` 和 `stream_ready`：摄像头对象打开不等于浏览器画面已经可用
  - 摄像头诊断会显示请求设备、实际打开设备、已尝试设备、等待首帧或读取失败原因
  - `prescription/banzi/camera_preflight.py` 可在演示前检查 OpenCV 是否能打开摄像头并读到一帧
  - 后续重点是板端稳定演示、拍摄 Demo 视频、继续调 RKNN/NPU Pose 的性能和机位，再考虑大模型和真实马达扩展

重点说明：

- `prescription/` 不是“未来可能有”的模块
- 它已经完成了当前阶段浏览器录制、active template、实时训练与双端保存主链路
- 当前保存方式是：板端本地先落盘，同时浏览器保存流程继续同步到 Windows 本机
- 当前闭环不再依赖“最新 JSON”隐式判断模板或患者动作，而是通过 `doctor_template`、`patient_attempt`、`evaluation_report` 显式区分角色
- 板端唯一页面入口：`http://板子IP:8082`
- `/doctor` 用于医生录入标准模板，`/train` 用于患者实时训练
- 本机结果接收器：`http://127.0.0.1:8090`
- 标准动作和患者动作 JSON 在 `prescription/docs/results/`
- 评估报告 JSON 在 `evaluate/reports/`，不作为动作输入
- 中文摘要在 `prescription/docs/summaries/`

### 3.4 TTS 模块现状

状态总结：

- `tts/` + `realtime/natural_tts.py`：自然女声 TTS 已接入实时训练优先链路
- `feedback/tts/`：旧板端播报链路保留
- `tts_board.py`：已可读取板端最新摘要并播报
- `sherpa-onnx VITS` 优先，`pyttsx3/espeak/mock` 兜底

当前说明：

- `feedback/tts/banzi/tts_board.py` 负责从板端最新处方摘要里取内容并播报
- 当前实时训练优先通过 `realtime/natural_tts.py` 加载 `tts/tts_model_pack/vits-aishell3.onnx`
- `realtime/tts_worker.py` 会启动时加载一次模型，后续播报复用，减少延迟
- 计数语音只播“一、二、三”
- 当前固定话术已按演示节奏精简，并加入动作开始保护期，避免系统还在播报时患者轻微移动就被记为错误
- 组间休息音乐目标是每组结束后 10 秒休息时播放 WAV 并淡出；板端 `aplay prescription/banzi/static/rest_music.wav` 已可直接出声，若训练页不出声，后续优先排查浏览器/后端播放链路，固定话术录音也应复用同一链路
- 自然女声失败时会降级到 `pyttsx3/espeak/mock`，训练页可查看 TTS backend 状态
- 后续还可以继续优化为更稳定的音频播放、缓存清理和更多语音风格

### 3.5 其他模块现状

从项目总目标来看，后续还应有这些模块：

- `evaluate/`
- `feedback/rules/` 和 `feedback/feedback_engine.py`
- `llm/`
- `main.py` 或等价的主程序集成入口

当前更细的状态判断如下：

- `evaluate/`
  - 已完成离线评估 MVP，并已接入 8082 三动作连续训练 Demo
  - `run_evaluate.py` 支持模板 JSON + 患者尝试 JSON + YAML 配置，输出标准评估报告 JSON
  - `core/action_metrics.py` 已支持动作专属指标，不再把三动作都套同一套角度标准
  - 当前三动作指标：
    - `seated_knee_extension`：`knee_extension_angle`
    - `standing_hamstring_curl`：`hamstring_curl_flexion_angle`
    - `seated_knee_raise`：`knee_raise_height_ratio`
  - `sit_to_stand` 坐站配置仍保留在仓库中，但不属于当前固定演示流程
  - `core/tut.py` 已升级为线性插值 TUT，且 TUT 目标时长来自 active template 的医生模板实测保持时间，不再固定写死 3 秒
  - `core/dtw_compare.py` 已支持平滑和最大时间扭曲限制
- `feedback/rules/` 和 `feedback/feedback_engine.py`
  - 已完成三动作反馈规则
  - 当前输出 screen / TTS 文案 / motor mock pattern，马达仍严格 mock，不访问真实 GPIO
- `realtime/`
  - 已完成三动作连续训练会话、有效 rep 计数、10 秒组间休息、TTS worker 和系统资源监控
  - 已加入演示剧情功能：第一个动作前侧身提示、离画超过 5 秒暂停并提醒回到画面、连续 3 次明显不到位后的关怀弹窗、组间休息流程
  - 只有动作达到医生模板目标区间并返回后才计数；不到位只播纠错，不增加完成次数
  - `/api/system/status` 可展示 CPU、内存、温度、NPU load 可用性和 Pose FPS
  - NPU load 现在只解析 `/sys/kernel/debug/rknpu/load` 的 `Core0/Core1/Core2` 百分比；文件不存在、权限不足或解析失败时显示 `N/A`，不会再假显示 100% 或拖垮其他资源监控
  - 当前 MediaPipe CPU 训练不会让 NPU 动起来，真正看到非 0% 需要另跑真实 RKNN workload
- `llm/`
  - 独立推进，不属于当前三动作核心 Demo 的必要链路
- `main.py`
  - 待建立或待联调

### 3.6 当前闭环 Demo

当前主 Demo 已从单动作屈膝离线闭环升级为 8082 统一训练台三动作连续实时训练。

当前可演示流程如下：

1. 启动 `prescription/banzi/record_prescription_http.py`
2. 浏览器访问 `http://板子IP:8082`
3. 进入 `/doctor`，分别录入并保存三个医生模板：
   - `seated_knee_extension`：坐姿伸膝
   - `standing_hamstring_curl`：站姿屈膝后勾腿
   - `seated_knee_raise`：坐姿抬膝
4. 系统把每个标准动作写入 `runtime/active_templates.json`，设置为对应 action 的 active template
5. 进入 `/train`，点击“开始完整训练”
6. 系统执行：坐姿伸膝 -> 休息 10 秒 -> 站姿屈膝后勾腿 -> 休息 10 秒 -> 坐姿抬膝
7. 每个动作只有达到医生模板目标区间并返回后才计数，不到位只播纠错语音
8. 每个动作完成后保存 patient attempt，并调用 `evaluate/banzi/run_evaluate.py` 生成 report

当前要特别注意：

- `prescription/docs/results/*.json` 是录制结果，可以作为标准模板或患者尝试
- `evaluate/reports/*.json` 是评估输出报告，不作为动作输入
- `runtime/active_templates.json` 保存相对 `PROJECT_ROOT` 的路径，不保存固定 Windows 绝对路径
- 当前患者训练使用真实 TTS 优先链路；马达仍然是 mock，不接真实 GPIO
- 当前 CPU/MediaPipe 路线是稳定演示默认路线；NPU/RKNN 路线建议先单独 smoke test，再进入 8082 验证
- 如果摄像头打不开，优先检查 `/dev/video*`、旧 Python 进程占用，以及 `RK_CAMERA_DEVICE=/dev/videoX`
- 演示前建议先跑 `RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=360 python3 prescription/banzi/camera_preflight.py --device auto`，确认 `opened: True` 且 `read_frame: True`
- CPU 稳定演示建议用自动选摄像头启动，例如 `RK_CAMERA_DEVICE=auto RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=360 python3 prescription/banzi/record_prescription_http.py`
- NPU 单独验证时再加 `POSE_BACKEND=rknn RKNN_POSE_MODEL=/home/elf/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn RKNN_STREAM_WIDTH=640 RKNN_STREAM_HEIGHT=360 RKNN_POSE_MAX_DET=5`
- 当前 MediaPipe 仍是 CPU 路线；NPU load 监控已接入系统监控页面，权限不足时显示 `N/A`
- 如果希望页面显示 `Core0/Core1/Core2 0%`，需要让运行 8082 的用户能读取 `/sys/kernel/debug/rknpu/load`

## 4. 关键文件索引

如果你是新开对话，最推荐先读这些文件。

### 4.1 总入口

- `D:\rk3588\project\readme.md`
  - 全项目总说明

### 4.2 硬件模块

- `D:\rk3588\project\hardware\motro_control\motor_controller.py`
  - 震动马达控制源码
- `D:\rk3588\project\hardware\motro_control\explain.md`
  - 硬件模块详细说明
  - 解释了参数、方法、Mock 模式、硬件到货后要改哪些地方
- `D:\rk3588\project\hardware\motro_control\readme.md`
  - 硬件模块最初背景说明
  - 内容较简略，且有部分旧状态描述

### 4.3 视觉模块

- `D:\rk3588\project\vision\camera_test.py`
  - 摄像头基础验证脚本
- `D:\rk3588\project\vision\pose_mediapipe_demo.py`
  - MediaPipe Pose 骨架识别验证脚本
- `D:\rk3588\project\vision\camera_http.py`
  - 浏览器版摄像头预览脚本
- `D:\rk3588\project\vision\pose_http.py`
  - 浏览器版骨骼识别预览脚本
- `D:\rk3588\project\vision\pose_backend_selector.py`
  - 姿态后端选择器骨架，默认仍为 MediaPipe
- `D:\rk3588\project\vision\rknn_pose\`
  - RKNN/NPU Pose 后端目录，已具备后处理、frame adapter、smoke test、replay compare、目标腿锁定和关键点稳定化
- `D:\rk3588\project\vision\guide.md`
  - 视觉模块上板操作指南
- `D:\rk3588\project\vision\explain.md`
  - 视觉模块代码说明
- `D:\rk3588\project\vision\x11_windows_preview.md`
  - 把板端实时预览映射到 Windows 的说明文档

### 4.4 当前主任务目录

- `D:\rk3588\project\prescription\`
  - 当前统一训练台 Web 入口所在模块目录
- `D:\rk3588\project\prescription\banzi\record_prescription_http.py`
  - RK3588 板端 8082 统一训练台主入口
  - 支持 `/doctor` 医生录制 active template、`/train` 患者三动作连续实时训练、摄像头流、状态展示和 API 转发
- `D:\rk3588\project\prescription\banzi\camera_preflight.py`
  - 板端摄像头预检脚本，用于演示前确认设备能打开并读到一帧，支持 `auto`
- `D:\rk3588\project\prescription\banzi\static\train.js`
  - `/train` 新 UI 逻辑，显示训练状态、资源卡片、摄像头流 ready/waiting/fallback 状态
- `D:\rk3588\project\prescription\banzi\static\common.js`
  - 新 UI 公共工具，当前按 `stream_ready` 决定是否切到 `/stream.mjpg`
- `D:\rk3588\project\prescription\windows\local_result_sink.py`
  - Windows 本机结果保存入口
- `D:\rk3588\project\prescription\windows\read_prescription_json.py`
  - 最新结果读取入口
- `D:\rk3588\project\prescription\common\result_storage.py`
  - 板端和 Windows 共用的 JSON、摘要、日志保存逻辑
- `D:\rk3588\project\prescription\record_prescription_http.py`
  - 兼容旧命令的 wrapper，新流程优先使用 `prescription\banzi\record_prescription_http.py`
- `D:\rk3588\project\prescription\docs\rk3588_prescription_guide.md`
  - 处方模块操作文档入口
- `D:\rk3588\project\docs\natural_tts_and_playlist_guide.md`
  - 当前三动作连续训练和自然女声 TTS 使用说明
- `D:\rk3588\project\docs\medical_rehab_actions.md`
  - 当前三动作医学依据、镜头姿势和检测指标说明
- `D:\rk3588\project\docs\unified_training_station.md`
  - 8082 统一训练台模块说明和使用流程
- `D:\rk3588\project\docs\rknn_pose_deployment_plan.md`
  - RKNN/NPU Pose 支线部署计划，说明三阶段接入和验收标准
- `D:\rk3588\project\docs\rknn_pose_npu_usage.md`
  - RKNN/NPU Pose 使用说明，包含 8082 启动方式、低延迟参数、性能诊断和上板文件列表

### 4.5 评估、实时训练与反馈

- `D:\rk3588\project\evaluate\windows\run_evaluate.py`
  - Windows 本机离线评估入口
- `D:\rk3588\project\evaluate\banzi\run_evaluate.py`
  - RK3588 板端离线评估入口
- `D:\rk3588\project\evaluate\run_evaluate.py`
  - 离线评估共享实现和兼容入口
  - 输入模板 JSON、患者尝试 JSON 和动作 YAML 配置，输出评估报告 JSON
- `D:\rk3588\project\evaluate\core\action_metrics.py`
  - 三动作专属指标提取逻辑
  - 当前支持 `knee_extension_angle`、`hamstring_curl_flexion_angle`、`hip_rise_height_ratio`
- `D:\rk3588\project\evaluate\configs\knee_flexion.yaml`
  - 屈膝动作评估配置
  - 保存关键点编号、角度字段、ROM/TUT/DTW/速度阈值和判定策略
- `D:\rk3588\project\evaluate\configs\seated_knee_extension.yaml`
  - 坐姿伸膝评估配置
- `D:\rk3588\project\evaluate\configs\standing_hamstring_curl.yaml`
  - 站姿屈膝后勾腿评估配置
- `D:\rk3588\project\evaluate\configs\seated_knee_raise.yaml`
  - 坐姿抬膝评估配置，当前第三个演示动作
- `D:\rk3588\project\evaluate\configs\sit_to_stand.yaml`
  - 坐站训练旧配置，当前固定演示流程不使用
- `D:\rk3588\project\evaluate\README.md`
  - 评估模块详细说明，包含目录结构、核心函数和新增动作方法
- `D:\rk3588\project\realtime\training_session.py`
  - 患者训练会话管理，负责三动作 playlist、target reps、attempt 保存和 evaluate 调用
- `D:\rk3588\project\realtime\knee_flexion.py`
  - 通用实时 rep 状态机，当前通过动作专属 primary signal 做 baseline、到位判断和有效计数
- `D:\rk3588\project\realtime\configs\rehab_demo_plan.yaml`
  - 三动作连续训练顺序、中文名、镜头姿势、TTS 话术和组间休息配置
- `D:\rk3588\project\realtime\system_monitor.py`
  - CPU、内存、温度、NPU load 可用性和 Pose FPS 监控
- `D:\rk3588\project\feedback\feedback_engine.py`
  - 离线反馈生成逻辑
  - 根据 `error_code` 输出 screen / TTS mock / motor mock
- `D:\rk3588\project\feedback\rules\knee_flexion_feedback.yaml`
  - 屈膝动作反馈规则
  - 定义 `ROM_LOW`、`TUT_LOW`、`SHAPE_BAD`、`TOO_FAST`、`OK` 对应文案和 mock 马达模式
- `D:\rk3588\project\feedback\rules\standing_hamstring_curl_feedback.yaml`
  - 站姿屈膝后勾腿反馈规则，`ROM_LOW` 会提示“小腿再往后弯一点”
- `D:\rk3588\project\runtime\active_templates.json`
  - 当前 active template 注册表
  - 保存相对项目根目录的模板路径和配置路径
- `D:\rk3588\project\docs\offline_closed_loop_demo.md`
  - 旧第三阶段离线闭环 Demo 使用说明
  - 说明 UI 流程、接口、JSON 角色、反馈规则和常见错误排查

### 4.6 TTS 模块

- `D:\rk3588\project\realtime\natural_tts.py`
  - 实时训练自然女声 TTS backend，复用 `tts/tts_model_pack/vits-aishell3.onnx`
- `D:\rk3588\project\realtime\tts_worker.py`
  - 异步 TTS worker，支持优先级队列和 natural_tts / pyttsx3 / espeak / mock fallback
- `D:\rk3588\project\tts\test.py`
  - sherpa-onnx VITS 女声单独测试脚本
- `D:\rk3588\project\tts\tts_model_pack\`
  - 自然女声模型、lexicon、tokens 和 fst 文件目录
- `D:\rk3588\project\feedback\tts\banzi\tts_board.py`
  - 旧板端文本转语音播报入口
- `D:\rk3588\project\feedback\tts\banzi\explain.md`
  - TTS 模块说明文档

## 5. 团队分工

根据当前 PDF 中的项目规划，三人分工如下：

### 同学 A

定位：

- 核心架构与大模型部署

主责方向：

- 本地 Qwen 模型部署
- 多线程架构
- DTW / 角度计算核心函数

### 同学 B

定位：

- 视觉数据与交互逻辑

主责方向：

- MediaPipe 骨骼点提取
- JSON 数字处方读写
- Prompt 模板
- OpenCV 交互提示
- TTS 语音合成

当前焦点：

- 已完成：
  - MediaPipe / 骨骼识别基础验证
  - 板端浏览器实时预览链路
- 当前主任务：
  - CPU/MediaPipe 版 8082 三动作连续实时训练稳定演示，NPU/RKNN 作为可选后端继续单独 smoke test 和 8082 调优
- 紧接下一任务：
  - 真实 TTS 接入、真实马达接入和多动作扩展

### 同学 C

定位：

- 硬件驱动与实物总装

主责方向：

- RK3588 环境准备
- GPIO + MOS 驱动马达
- 护具缝制与走线
- 实物装配与联调

当前你已经完成的内容，主要落在：

- 同学 C 的硬件驱动方向
- 同时 `vision/` 目录也已经完成了当前阶段够用的骨骼识别验证

## 6. 8 周计划摘要

下面是从 PDF 浓缩出来的项目推进主线，便于新对话理解当前阶段。

### 第 1 周

目标：

- 基础设施点亮

典型任务：

- 跑通 MediaPipe 示例
- 板子装系统
- 阅读 GPIO 例程

### 第 2 周

目标：

- 核心模块跑通

典型任务：

- DTW 脚本验证
- JSON 处方读写
- TTS 初步验证
- 用 `periphery` 跑通震动马达三种模式

### 第 3 周

目标：

- 业务逻辑成型

典型任务：

- 角度计算 API
- 多线程骨架
- 视觉引导 UI
- 把代码往板子上迁移

### 第 4 周

目标：

- 智能助教发声

典型任务：

- LLM 输出康复指导
- TTS 接队列
- 实物护具与马达固定

### 第 5 周

目标：

- 软硬大缝合

典型任务：

- 主线程接管视觉流和硬件流
- 真实摄像头测试
- 马达反馈延迟测试

### 第 6 周

目标：

- 业务逻辑联调

典型任务：

- 线程锁与竞态处理
- 反馈冷却机制
- 故意错误动作联调
- 处理物理接触问题

### 第 7 周

目标：

- 系统优化排雷

典型任务：

- 控帧率和 CPU 占用
- UI 和 Prompt 微调
- 训练报告页
- 外壳和展示包装

### 第 8 周

目标：

- 冲刺答辩

典型任务：

- 停止大改代码
- 写文档与 PPT
- 录 Demo
- 准备实物演示

## 7. 当前阶段判断

结合现有代码，当前项目可以做两层判断。

### 7.1 从项目整体看

- 整体已经从第 3 周“业务逻辑成型”推进到可拍 Demo 的连续训练联调阶段
- 8082 统一训练台已基本成型，当前重点是板端 CPU/MediaPipe 三动作连续实时训练稳定演示，同时 RKNN/NPU Pose 已进入可测试、可调优阶段

原因：

- 硬件马达控制的软件逻辑已经写出来了
- 板端摄像头和骨骼识别的独立验证已经够用
- 处方录入 / 读取 / 保存链路已经完成
- 自然女声 TTS 已接入实时训练优先链路，并有 fallback
- `evaluate/` 已支持动作专属指标和完整报告输出
- 浏览器 UI 已能串起“医生录入 active template -> 患者三动作连续训练 -> 不到位纠错 -> 有效 rep 计数 -> attempt 保存 -> report 生成”
- 当前马达仍是 mock；MediaPipe CPU 路线是稳定演示默认路线；RKNN/NPU Pose 已可选接入但仍在稳定性调优；大模型/真实马达属于后续增强

所以当前最适合的整体状态判断是：

- 基础 demo 已经点亮
- 处方、实时训练、动作专属评估和自然女声 TTS 的三动作闭环已基本成型
- 下一步应先用 CPU/MediaPipe 稳定板端 8082 三动作演示，并把 RKNN/NPU Pose 作为可选后端持续调优，再考虑大模型和真实马达

### 7.2 从同学 B 的任务视角看

- 第 1 周的骨骼识别目标已经完成
- 数字处方、自然女声 TTS、三动作反馈话术和实时训练页面已经跑通
- 当前重点是把医生模板录入、患者连续训练、有效计数、报告生成和拍摄脚本跑稳；RKNN/NPU 可选后端继续单独调优

## 8. 当前主任务

如果你现在要继续推进项目，当前最合理的顺序是：

1. 先确认 `vision` 和 `prescription` 的现有链路都能正常跑
2. 在 `/doctor` 重新录入三个医生模板：`seated_knee_extension`、`standing_hamstring_curl`、`seated_knee_raise`
3. 到 `/train` 点击“开始完整训练”，确认三动作连续切换、侧身提示、离画提醒、关怀弹窗、10 秒休息和自然女声播报正常
4. 若要测试 NPU，先跑 `vision/rknn_pose/smoke_test_rknn_pose.py`，再用 `POSE_BACKEND=rknn` 和 `RK_CAMERA_DEVICE=auto` 启动 8082，确认 `actual_backend=rknn`
5. 故意做不到位动作，确认只播纠错、不计数；做到位并返回后才播“一、二、三”
6. 检查 `prescription/docs/results/` 的 attempt 和 `evaluate/reports/` 的 report
7. 跑通后拍老师要看的 Demo 视频，再继续调 NPU、大模型和真实马达

也就是说：

- 现在新开 `prescription/` 对话，不需要重做视觉验证
- 也不需要再从零开始做 JSON 处方读写
- 应该直接在现有 8082 统一训练台上做 CPU/MediaPipe 三动作连续演示验收，RKNN/NPU Pose 作为可选后端继续打磨

## 9. 当前阶段后的建议

### 9.1 先完成当前主任务

顺序建议：

1. 先在板端用 CPU/MediaPipe 完整验证 8082 三动作连续训练
2. 根据真实录制样本微调三个动作 YAML 的 ROM、TUT、DTW、速度阈值和反馈文案
3. 保持马达 mock，先保证视频演示稳定；真实 GPIO 马达放到后续增强
4. RKNN/NPU Pose 继续作为可选后端做关键点稳定性、目标腿锁定和性能调优
5. 再考虑大模型解释和更完整的系统资源压测

### 9.2 再接入主线程

后续主程序应该做的事情：

- 建立主线程 + 多个工作线程
- 让视觉线程输出判断结果
- 让 GPIO 线程消费 `Motor_Queue`
- 让数字处方模块提供动作模板数据
- 让 TTS 模块提供语音反馈
- 让 `evaluate/` 提供动作专属离线报告，`realtime/` 提供实时有效计数和纠错
- 让 `runtime/active_templates.json` 成为模板选择的运行态入口

### 9.3 最后做软硬联调

最终要联起来的是：

```text
摄像头录制 -> 医生 active template -> 患者三动作实时训练 -> 有效 rep 计数和纠错 -> attempt 保存 -> evaluate 报告 -> 屏幕/自然女声 TTS/马达 mock -> 后续 NPU 或真实 GPIO
```
