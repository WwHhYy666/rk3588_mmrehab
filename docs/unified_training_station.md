# RK3588 统一训练台使用说明

这份文档说明当前 `8082` 统一训练台怎么启动、怎么使用，以及每个模块负责什么。

当前训练台把两件事整合到同一个 Web 服务里：

- 医生录制标准动作
- 患者实时屈膝训练与反馈

也就是说，后续演示和联调只需要打开一个入口：

```text
http://板子IP:8082
```

不需要启动额外端口。

## 1. 快速开始

### 1.1 启动服务

在项目根目录运行：

```bash
python prescription/banzi/record_prescription_http.py
```

如果在 RK3588 板端使用 `python3`，则运行：

```bash
python3 prescription/banzi/record_prescription_http.py
```

服务启动后，终端会提示：

```text
8082 统一训练台已启动: http://板子IP:8082
```

### 1.2 打开页面

在浏览器中访问：

```text
http://板子IP:8082
```

页面入口如下：

| 路由 | 作用 |
| --- | --- |
| `/` | 统一训练台首页 |
| `/doctor` | 医生标准动作录制 |
| `/train` | 患者实时屈膝训练 |
| `/stream.mjpg` | 实时摄像头画面流 |
| `/status` | 当前系统状态 |

## 2. 推荐使用流程

### 第一步：医生录制标准动作

进入：

```text
http://板子IP:8082/doctor
```

操作顺序：

1. 确认摄像头画面正常。
2. 填写或保持默认患者编号和动作名称。
3. 选择侧别模式：
   - `auto`：自动选择可见度更好的腿
   - `left`：固定左腿
   - `right`：固定右腿
4. 点击“录入标准动作”。
5. 医生或示范者完成一遍标准屈膝动作。
6. 点击“保存为 active template”。

保存成功后，系统会：

- 把标准动作 JSON 保存到 `prescription/docs/results/`
- 把中文摘要保存到 `prescription/docs/summaries/`
- 更新 `runtime/active_templates.json`

患者实时训练会读取这个 active template。

### 第二步：患者进入实时训练

进入：

```text
http://板子IP:8082/train
```

如果还没有 active template，训练页面不能正常开始。需要先回到 `/doctor` 录制并保存标准动作。

### 第三步：开始实时训练

在 `/train` 页面中：

1. 填写患者编号。
2. 设置目标次数 `target_reps`，默认是 `10`。
3. 选择侧别模式。
4. 点击“开始训练”。

系统会先进入 baseline 校准阶段，提示患者保持静止。

校准完成后，患者连续完成多次屈膝动作。每一遍动作不需要重新点击开始。

### 第四步：实时提示与每遍小评估

训练过程中，系统会实时显示：

- 当前膝关节角度
- 状态机状态
- 已完成次数 / 目标次数
- 当前提示语
- TTS 文案
- motor mock pattern

实时提示包括：

- 请保持静止，正在校准
- 准备开始下一遍
- 再抬高一点
- 保持住
- 慢慢放下
- 请站到摄像头前

每一遍动作结束后，系统会生成小评估：

| 字段 | 含义 |
| --- | --- |
| `rep_index` | 第几遍 |
| `rom` | 本遍活动度 |
| `tut_seconds` | 本遍目标区间保持时间 |
| `peak_speed` | 本遍峰值角速度 |
| `primary_error` | 本遍主要问题 |
| `screen_prompt` | 屏幕提示 |
| `tts_text` | 语音播报文本 |
| `motor_mock_pattern` | 马达 mock 模式 |

`primary_error` 目前包括：

| 错误码 | 含义 |
| --- | --- |
| `ROM_LOW` | 抬高幅度不够 |
| `TUT_LOW` | 保持时间不够 |
| `TOO_FAST` | 动作过快 |
| `OK` | 本遍达标 |

### 第五步：整组完成并生成完整报告

当患者完成 `target_reps` 次动作后，系统会自动：

1. 保存整组 patient attempt JSON。
2. 调用现有离线评估入口：

```text
evaluate/banzi/run_evaluate.py
```

3. 生成完整 evaluation report。
4. 在页面显示报告摘要和报告路径。

## 3. 模块作用

### 3.1 Web 入口模块

文件：

```text
prescription/banzi/record_prescription_http.py
```

职责：

- 启动 `8082` Web 服务
- 提供首页 `/`
- 提供医生录制页 `/doctor`
- 提供患者训练页 `/train`
- 提供实时摄像头流 `/stream.mjpg`
- 提供统一状态接口 `/status`
- 保留医生录制 API
- 转发患者实时训练 API 到 `realtime/` 模块
- 打开摄像头并运行 MediaPipe 推理

注意：实时屈膝检测、状态机、rep 计数、TTS worker 不放在这个文件里，它们已经下沉到 `realtime/` 模块。

### 3.2 实时状态机基础

文件：

```text
realtime/state_machine.py
```

职责：

- 定义实时训练状态：
  - `BASELINE`
  - `IDLE`
  - `RISING`
  - `HOLDING`
  - `RETURNING`
  - `REP_DONE`
- 提供连续帧确认工具，避免关键点抖动导致误判。

### 3.3 屈膝实时检测

文件：

```text
realtime/knee_flexion.py
```

职责：

- 维护屈膝动作状态机
- 采集 baseline 静止角度
- 识别每一遍动作的开始、保持、放下和完成
- 自动生成 rep 小评估
- 计算本遍：
  - ROM
  - TUT
  - peak speed
  - primary error

它不负责页面展示，也不负责保存文件。

### 3.4 训练会话管理

文件：

```text
realtime/training_session.py
```

职责：

- 管理一次患者实时训练会话
- 读取 `runtime/active_templates.json`
- 读取医生标准动作模板
- 根据模板和评估配置推导实时目标
- 接收摄像头 worker 传入的每帧角度数据
- 统计完成次数
- 达到 `target_reps` 后保存 patient attempt JSON
- 自动调用 `evaluate/banzi/run_evaluate.py`
- 保存完整报告路径和报告内容

这是患者实时训练的主要业务编排模块。

### 3.5 实时反馈映射

文件：

```text
realtime/feedback_runtime.py
```

职责：

- 把实时状态转换成屏幕提示
- 把每遍小评估结果转换成：
  - screen prompt
  - TTS 文案
  - motor mock pattern
- 读取 `feedback/rules/knee_flexion_feedback.yaml` 中的反馈规则

### 3.6 TTS worker

文件：

```text
realtime/tts_worker.py
```

职责：

- 独立线程异步播报 TTS
- 使用队列接收播报文本
- 避免视觉线程和状态机等待 TTS 播放完成
- 支持全局冷却时间
- 支持同一句话冷却时间
- 优先真实播报
- 真实 TTS 失败时降级为 mock 打印

### 3.7 实时配置

文件：

```text
realtime/configs/knee_flexion_realtime.yaml
```

职责：

- 保存实时训练参数

当前主要字段：

| 字段 | 含义 |
| --- | --- |
| `baseline_seconds` | baseline 静止校准时间 |
| `start_delta` | 判断动作开始的角度变化阈值 |
| `return_delta` | 判断动作回落完成的阈值 |
| `confirm_frames` | 状态切换需要连续满足的帧数 |
| `visibility_threshold` | 关键点可见度阈值 |
| `min_rom` | 最小 ROM 兜底值 |
| `min_rep_seconds` | 单遍最短时间 |
| `target_reps` | 默认目标次数 |
| `tts_global_cooldown_seconds` | TTS 全局冷却 |
| `tts_same_text_cooldown_seconds` | 同一句 TTS 冷却 |

### 3.8 离线评估模块

目录：

```text
evaluate/
```

职责：

- 在整组训练完成后生成完整评估报告
- 计算：
  - ROM
  - TUT
  - DTW
  - speed
  - primary error

实时阶段不做每帧 DTW。DTW 仍交给整组结束后的离线 evaluate。

### 3.9 反馈规则

目录：

```text
feedback/rules/
```

当前屈膝反馈规则：

```text
feedback/rules/knee_flexion_feedback.yaml
```

职责：

- 定义不同错误码对应的屏幕文案
- 定义 TTS 文案
- 定义 motor mock pattern

### 3.10 active template 注册表

文件：

```text
runtime/active_templates.json
```

职责：

- 保存当前动作使用哪个医生标准模板
- 保存对应评估配置路径

患者训练开始前必须能在这里找到 `knee_flexion` 的 active template。

## 4. API 一览

### 4.1 页面路由

| 路由 | 说明 |
| --- | --- |
| `/` | 统一训练台首页 |
| `/doctor` | 医生标准动作录制 |
| `/train` | 患者实时训练 |
| `/stream.mjpg` | 摄像头实时画面 |
| `/status` | 系统状态 |

### 4.2 医生录制 API

| API | 方法 | 说明 |
| --- | --- | --- |
| `/api/start` | POST | 开始录制医生模板或患者动作 |
| `/api/save` | POST | 保存当前录制结果 |
| `/api/evaluate` | POST | 对保存的患者动作运行离线评估 |
| `/api/active_template` | GET | 查询当前 active template |

### 4.3 实时训练 API

| API | 方法 | 说明 |
| --- | --- | --- |
| `/api/realtime/start` | POST | 开始患者实时训练 |
| `/api/realtime/pause` | POST | 暂停或继续训练 |
| `/api/realtime/stop` | POST | 停止训练 |
| `/api/realtime/status` | GET | 查询实时训练状态 |

## 5. 数据输出位置

### 5.1 医生模板和患者 attempt

```text
prescription/docs/results/
```

医生标准动作和患者训练结果都会保存到这里。

### 5.2 中文摘要

```text
prescription/docs/summaries/
```

每次保存动作结果时，会生成一份中文摘要。

### 5.3 完整评估报告

```text
evaluate/reports/
```

患者实时训练完成后，系统会调用 evaluate 并把完整报告保存到这里。

### 5.4 active template

```text
runtime/active_templates.json
```

医生模板保存为 active template 后，这个文件会被更新。

## 6. 注意事项

### 6.1 只使用 8082

当前统一训练台只使用：

```text
http://板子IP:8082
```

不要再启动额外的实时训练端口。

### 6.2 患者训练前必须先录标准动作

患者实时训练依赖 active template。

如果 `/train` 页面无法开始训练，优先检查：

```text
runtime/active_templates.json
```

是否已经有 `knee_flexion` 对应的模板。

### 6.3 摄像头只打开一次

摄像头由 `record_prescription_http.py` 中的 worker 统一打开。

医生录制和患者训练共享同一个摄像头 worker，不要再启动其他会抢占摄像头的脚本。

### 6.4 TTS 不阻塞训练

TTS 在独立 worker 线程中播报。

即使 TTS 播放较慢，也不应该阻塞摄像头推理和状态机判断。

### 6.5 马达当前严格 mock

当前患者实时训练阶段不会访问真实 GPIO，也不会打开 `/dev/gpiochip*`。

页面和终端只会显示类似：

```text
[MOTOR MOCK] short_double
```

后续如果要接真实马达，应单独把 pattern 映射到：

```text
hardware/motro_control/motor_controller.py
```

## 7. 常见问题

### 7.1 打不开摄像头

确认摄像头设备路径是否正确。

默认使用：

```text
/dev/video21
```

如果实际设备不同，可以设置环境变量：

```bash
export RK_CAMERA_DEVICE=/dev/videoX
python3 prescription/banzi/record_prescription_http.py
```

### 7.2 患者训练不能开始

优先检查：

1. 是否已经在 `/doctor` 保存标准动作。
2. `runtime/active_templates.json` 是否存在。
3. active template 中的模板路径是否真实存在。
4. `evaluate/configs/knee_flexion.yaml` 是否存在。

### 7.3 页面有角度但不计数

可能原因：

- 患者没有先保持静止完成 baseline。
- 抬腿角度变化小于 `start_delta`。
- 关键点 visibility 太低。
- 状态切换需要连续满足 `confirm_frames` 帧。

可以调整：

```text
realtime/configs/knee_flexion_realtime.yaml
```

### 7.4 TTS 没有声音

系统会优先尝试真实 TTS。

如果真实 TTS 初始化失败，会降级为终端 mock 打印。此时训练逻辑仍然可以继续。

## 8. 当前能力边界

当前已经做：

- 8082 统一入口
- 医生标准动作录制
- active template 保存
- 患者实时屈膝状态机
- rep 自动计数
- 每遍小评估
- TTS worker
- motor mock
- 整组完成后自动 evaluate

当前暂不做：

- 不接 LLM
- 不做多动作扩展
- 不做每帧 DTW
- 不接真实马达
- 不启动额外实时训练端口
