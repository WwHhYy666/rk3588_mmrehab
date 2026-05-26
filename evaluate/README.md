# evaluate 离线评估模块 MVP

`evaluate/` 是 RK3588 骨科居家数字康复终端里的动作评估模块。当前阶段只保证离线评估 MVP 干净可运行：

```text
标准模板 JSON + 患者尝试 JSON + 动作配置 YAML
        -> run_evaluate.py
        -> 结构化评估报告 JSON
```

本阶段不接 UI、不接摄像头、不接 TTS、不接马达、不做实时逐帧反馈。

## 唯一 MVP 主入口

当前 MVP 唯一主入口是：

```text
evaluate/run_evaluate.py
```

它负责读取两份动作 JSON 和一份动作配置 YAML，计算 ROM、TUT、DTW、速度指标，并输出给后续 feedback / LLM / 主程序使用的标准报告 JSON。

旧 replay 原型已经归档到 `evaluate/legacy/`，不再作为主流程入口。

## 算法严谨性说明

当前 MVP 不是简单逐帧打分，而是使用几类可解释指标组合评估：

- ROM：比较标准模板和患者尝试的活动范围。
- TUT：先统计医生模板在目标角度区间内的有效保持时间，再统计患者实际保持时间并计算达成比例。
- DTW：比较标准模板曲线和患者动作曲线的整体形状。
- speed：比较患者动作速度是否明显快于模板。

TUT 已从早期的“相邻两帧都在区间内才累计”升级为线性插值估算。对每两个相邻有效点，程序假设角度随时间线性变化，计算该时间段内落在目标区间的真实比例。这样能更准确处理低帧率采样下进入目标区间、离开目标区间，以及两端都在区间外但中间穿过目标区间的情况。当前 TUT target 来自医生标准模板在该工作区间内的实测保持时间，不再使用 YAML 固定 3 秒。

DTW 当前主要比较动作角度曲线形状。为了减少关键点抖动和过度对齐，`dtw_score` 支持角度序列移动平均平滑和 Sakoe-Chiba 最大时间扭曲窗口。后续如需更严格的节奏评估，可以扩展为 `angle + time + velocity` 的复合距离。

## 目录结构

```text
evaluate/
├── core/
│   ├── __init__.py
│   ├── angle_utils.py
│   ├── rom.py
│   ├── tut.py
│   ├── dtw_compare.py
│   ├── speed_check.py
│   └── error_classifier.py
├── configs/
│   └── knee_flexion.yaml
├── samples/
│   ├── template.json
│   └── attempt.json
├── reports/
│   └── result_sample.json
├── legacy/
│   ├── actions.json
│   ├── engine.py
│   ├── run_local_evaluate.py
│   └── run_replay_compare.py
├── run_evaluate.py
└── README.md
```

## JSON 文件角色

### `prescription/docs/results/*.json`

这是录制流程产生的真实动作结果。每一份都可以作为：

- 标准模板 JSON：医生或标准示范者录入的动作。
- 患者尝试 JSON：患者本次训练录入的动作。

正式比对时不要简单理解为“永远用最新 JSON”。正确逻辑是：明确选择一份作为标准模板，再选择一份作为患者尝试，然后把这两份交给 `run_evaluate.py`。

### `evaluate/samples/*.json`

这是开发测试样例，只用于验证评估模块是否能跑通。

- `samples/template.json`：小型标准模板样例。
- `samples/attempt.json`：小型患者尝试样例。

### `evaluate/reports/*.json`

这是评估输出报告，不作为动作输入。后续 feedback、TTS、LLM 或主程序应该读取这里的报告结果。

## YAML 阈值是否主观

`configs/knee_flexion.yaml` 里的阈值是第一版规则基线，不是随意拍脑袋。专业表述是：

- 第一版阈值来自专家经验、处方目标和演示安全边界。
- 阈值全部外置到 YAML，医生或调试人员可以调整，不需要改核心算法。
- 后续采集更多样本后，可以用真实训练数据校准阈值。

也就是说，YAML 负责的是“容差和判定策略”，不是把所有临床目标写死在代码里。TUT 的目标秒数由标准模板实测得到，YAML 只定义有效保持区间和最低达成比例。

## 个体化目标从哪里来

评估模块不是拿固定角度强行评所有人，而是把标准模板当作个体化目标来源：

- ROM target 来自标准模板 JSON 的实际 ROM。
- TUT 工作区相对标准模板最大角度计算。
- TUT target 来自标准模板在工作区内的实际保持时间。
- DTW 比较的是标准模板曲线和患者动作曲线。
- YAML 主要负责允许偏差、TUT 比例、速度倍率等判定策略。

这样做的好处是：不同患者、不同动作、不同康复阶段都可以先录一份标准模板，再用同一套算法评估患者尝试。

## 快速运行

在项目根目录 `D:\rk3588\project` 下运行：

```powershell
D:\anaconda\python.exe evaluate\run_evaluate.py --template evaluate\samples\template.json --attempt evaluate\samples\attempt.json --config evaluate\configs\knee_flexion.yaml --out evaluate\reports\result_sample.json
```

在 `D:\rk3588\project\evaluate` 下运行：

```powershell
D:\anaconda\python.exe run_evaluate.py --template samples\template.json --attempt samples\attempt.json --config configs\knee_flexion.yaml --out reports\result_sample.json
```

使用真实录制 JSON 的示例：

```powershell
D:\anaconda\python.exe evaluate\run_evaluate.py --template prescription\docs\results\patient_action_20260519_145637.json --attempt prescription\docs\results\patient_action_20260519_154038.json --config evaluate\configs\knee_flexion.yaml --out evaluate\reports\result_real_pair.json
```

## 输出报告结构

输出报告至少包含：

```json
{
  "evaluated_at": "2026-05-25T20:00:00",
  "action_name": "屈膝",
  "template_file": "...",
  "attempt_file": "...",
  "config_file": "...",
  "metrics": {},
  "errors": {},
  "structured_feedback": {}
}
```

### `metrics`

评估指标集合：

- `rom`：模板 ROM、患者 ROM、差值。
- `tut`：模板实测目标保持时间、患者实际保持时间、达成比例、目标角度区间、目标来源和计算方法。
- `dtw`：动作曲线 DTW 距离、归一化距离、路径长度。默认限制最大时间扭曲，避免节奏差异过大的动作被强行对齐。
- `speed`：模板和患者的峰值角速度、平均角速度、速度倍率。

### `errors`

错误分类结果：

- `all_errors`：全部错误码。
- `primary_error`：最高优先级错误码。
- `is_ok`：是否达标。

当前优先级：

```text
ROM_LOW > TUT_LOW > SHAPE_BAD > TOO_FAST > OK
```

### `structured_feedback`

给后续反馈模块读取的结构化结果：

- `error_code`：主错误码。
- `params`：反馈需要的参数，例如 ROM 差值、目标值、实际值。

## core 文件说明

### `core/angle_utils.py`

核心函数：

```python
angle_from_3_points(a, b, c)
```

用余弦公式计算以 `b` 为顶点的三点夹角，作为角度字段缺失时的兜底工具。

### `core/rom.py`

核心函数：

```python
compute_rom(frames, angle_field="flexion_angle")
```

从帧序列中计算最小角、最大角、ROM、最大/最小角所在帧。

### `core/tut.py`

核心函数：

```python
compute_tut(frames, target_range, angle_field="flexion_angle")
```

统计角度落在目标角度区间内的累计时长。当前使用线性插值估算：相邻两帧之间假设角度线性变化，只累计真实落在目标区间内的那一段时间。这样比离散帧累计更适合低帧率数据。

### `core/dtw_compare.py`

核心函数：

```python
dtw_score(template_angles, attempt_angles)
```

比较标准模板曲线和患者动作曲线，输出总距离、归一化距离和路径长度。默认使用带 Sakoe-Chiba 窗口的 DTW，限制最大时间扭曲；当 `smooth_window > 1` 时，会先对角度序列做简单移动平均，降低关键点抖动影响。

### `core/speed_check.py`

核心函数：

```python
check_speed(frames, angle_field="flexion_angle")
```

根据相邻帧 `|Δangle / Δtime|` 计算峰值角速度和平均角速度。

### `core/error_classifier.py`

核心函数：

```python
classify(metrics, config)
```

根据 YAML 阈值和 metrics 输出错误码，保证后续反馈模块只需要消费结构化结果。

## 配置文件说明

`configs/knee_flexion.yaml` 是当前屈膝动作配置，也是多动作扩展模板。

关键字段：

- `action_name`：报告中的动作名称。
- `keypoint_rule`：关键点编号和目标关节。
- `angle_field`：默认角度字段。
- `angle_fields`：兼容多个 JSON 角度字段的优先级列表。
- `thresholds.rom_diff_max`：ROM 差超过该值判定 `ROM_LOW`。
- `thresholds.tut_ratio_min`：TUT 达成比例低于该值判定 `TUT_LOW`。
- `thresholds.dtw_normalized_max`：曲线差异超过该值判定 `SHAPE_BAD`。
- `thresholds.speed_ratio_max`：速度倍率超过该值判定 `TOO_FAST`。
- `tut_work_zone`：相对模板最大角度定义 TUT 工作区。
- TUT 目标时长不再写死在 YAML 中，而是由标准模板在 `tut_work_zone` 内的实测保持时间自动生成。

## 如何新增动作

新增动作时，不改 `core/` 算法，新增一份 YAML 即可。

例如新增屈肘：

1. 新建 `configs/elbow_flexion.yaml`。
2. 修改 `action_name`。
3. 修改 `keypoint_rule` 的三个关键点编号和 `target_joint`。
4. 修改 `angle_field` / `angle_fields`，匹配新动作 JSON。
5. 修改阈值和 TUT 工作区。
6. 用 `run_evaluate.py --config configs/elbow_flexion.yaml` 运行。

## 验收命令

在项目根目录运行：

```powershell
D:\anaconda\python.exe evaluate\run_evaluate.py --template evaluate\samples\template.json --attempt evaluate\samples\attempt.json --config evaluate\configs\knee_flexion.yaml --out evaluate\reports\result_sample.json
```

检查输出报告：

```powershell
Get-Content evaluate\reports\result_sample.json
```

报告中必须包含：

- `metrics`
- `errors`
- `structured_feedback`

## 归档说明

`evaluate/legacy/` 里的旧文件只用于查历史原型：

- `engine.py`
- `actions.json`
- `run_local_evaluate.py`
- `run_replay_compare.py`

它们不参与当前 MVP 主流程。
