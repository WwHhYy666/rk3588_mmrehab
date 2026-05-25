# 离线闭环 Demo 使用说明

本文档说明第三阶段闭环 Demo 的使用方法和各部分作用。

当前闭环是：

```text
录入标准动作
  -> 保存 doctor_template
  -> 设置 active template
  -> 录入患者动作
  -> 保存 patient_attempt
  -> 调用 evaluate/run_evaluate.py
  -> 生成 evaluation_report
  -> 生成屏幕反馈 / TTS mock / 马达 mock
```

第三阶段不做实时逐帧反馈，不接真实 TTS，不接真实马达，不引入 LLM。

## 1. 入口文件

浏览器 UI 入口：

```powershell
D:\anaconda\python.exe prescription\record_prescription_http.py
```

启动后在浏览器打开：

```text
http://板子IP:8082
```

在 Windows 本机调试时，如果服务运行在本机，也可以打开：

```text
http://127.0.0.1:8082
```

## 2. 文件和目录作用

### `prescription/record_prescription_http.py`

浏览器版录制和闭环 Demo 主入口。

它负责：

- 打开摄像头。
- 用 MediaPipe 提取骨骼关键点。
- 录制标准动作或患者动作。
- 保存动作 JSON。
- 维护当前 session 的 patient attempt。
- 提供 `/api/evaluate` 接口。
- 在页面展示评估结果和 mock 反馈。

### `prescription/result_storage.py`

动作 JSON 和摘要文件的保存工具。

它负责把录制结果保存到：

```text
prescription/docs/results/
prescription/docs/summaries/
```

### `runtime/active_templates.json`

当前 active template 管理文件。

它按动作保存当前标准模板。例如：

```json
{
  "knee_flexion": {
    "action_id": "knee_flexion",
    "template_file": "prescription/docs/results/patient_001_knee_flexion_20260525_120000.json",
    "config_file": "evaluate/configs/knee_flexion.yaml",
    "updated_at": "2026-05-25T12:00:00"
  }
}
```

注意：这里保存的是相对 `D:\rk3588\project` 的路径，不是 Windows 绝对路径，方便后续迁移到 RK3588 Linux。

### `evaluate/run_evaluate.py`

离线评估主入口。

后端会自动包装调用：

```powershell
D:\anaconda\python.exe evaluate\run_evaluate.py --template <active_template> --attempt <patient_attempt> --config <config_file> --out <report_file>
```

UI 不直接拼命令行。

### `evaluate/reports/`

评估报告输出目录。

里面的 JSON 是 `evaluation_report`，只作为评估输出，不允许作为标准模板或患者输入。

### `feedback/rules/knee_flexion_feedback.yaml`

屈膝动作的离线反馈规则。

每个错误码都有：

- screen：屏幕提示颜色、标题、文案模板。
- tts：TTS mock 文案模板。
- motor：马达 mock pattern。

### `feedback/feedback_engine.py`

反馈生成器。

输入：

- evaluate report JSON
- feedback rule YAML

输出：

```json
{
  "error_code": "ROM_LOW",
  "raw_params": {
    "rom_diff": 8.0,
    "missing_seconds": 1.2,
    "speed_ratio": 1.1,
    "dtw_normalized_distance": 0.18
  },
  "screen": {
    "color": "red",
    "title": "动作不到位",
    "message": "还差 8.0 度，再抬高一点"
  },
  "tts": {
    "text": "您还差 8 度，再抬高一点"
  },
  "motor": {
    "pattern": "short_double"
  }
}
```

## 3. 三类 JSON

### doctor_template

医生或标准示范者录入的标准动作。

保存后会：

- 写入 `prescription/docs/results/`
- 标记 `runtime_meta.record_role = doctor_template`
- 设置为 active template
- 写入 `runtime/active_templates.json`

### patient_attempt

患者本次训练动作。

保存后会：

- 写入 `prescription/docs/results/`
- 标记 `runtime_meta.record_role = patient_attempt`
- 记录为当前 session 的 patient attempt

### evaluation_report

评估输出报告。

保存到：

```text
evaluate/reports/
```

它不是动作输入，不能作为模板或患者动作 JSON。

## 4. UI 怎么用

### 第一步：选择动作

当前第三阶段只支持：

```text
knee_flexion
```

页面会显示：

- action_id
- action_name

### 第二步：录入标准动作

点击：

```text
录入标准动作
```

完成动作后点击：

```text
保存为 active template
```

保存成功后，页面会显示当前 active template 文件名。

### 第三步：录入患者动作

点击：

```text
录入患者动作
```

患者完成动作后点击：

```text
保存 patient attempt
```

保存成功后，页面会显示当前 patient attempt 文件名。

### 第四步：结束并评估

点击：

```text
结束并评估
```

后端会：

1. 读取 active template。
2. 读取 patient attempt。
3. 调用 `evaluate/run_evaluate.py`。
4. 生成 `evaluate/reports/report_*.json`。
5. 读取 report。
6. 调用 feedback engine。
7. 返回结果给页面展示。

## 5. 页面会展示什么

评估结果区域展示：

- ROM target
- ROM actual
- ROM diff
- TUT target
- TUT actual
- TUT ratio
- DTW normalized_distance
- Speed ratio
- primary_error
- structured_feedback

反馈区域展示：

- 屏幕反馈标题和文案
- TTS mock 文案
- 马达 mock pattern

## 6. 后端接口

### `GET /api/active_template?action_id=knee_flexion`

查询当前动作的 active template。

### `POST /api/start`

开始录制。

请求示例：

```json
{
  "patient_id": "patient_001",
  "action_name": "knee_flexion",
  "side_mode": "auto",
  "record_role": "doctor_template"
}
```

`record_role` 可为：

- `doctor_template`
- `patient_attempt`

### `POST /api/save`

保存当前录制。

请求示例：

```json
{
  "record_role": "patient_attempt"
}
```

### `POST /api/evaluate`

执行离线评估。

请求示例：

```json
{
  "action_id": "knee_flexion",
  "attempt_file": "prescription/docs/results/patient_attempt_xxx.json"
}
```

`attempt_file` 可以不传。不传时，后端使用当前 session 最近保存的 patient attempt。

如果没有 active template，会返回：

```json
{
  "ok": false,
  "error": "请先录入标准动作"
}
```

如果评估子进程失败，会返回：

```json
{
  "ok": false,
  "error": "评估失败",
  "stdout": "...",
  "stderr": "...",
  "returncode": 1
}
```

## 7. 按钮状态规则

页面会自动处理：

- 录制中不能重复开始。
- 无 active template 时不能评估。
- 无 patient attempt 时不能评估。
- 评估中不能重复提交。
- 保存标准动作后刷新 active template。
- 保存患者动作后刷新 patient attempt。

## 8. 常见问题

### 提示“请先录入标准动作”

说明 `runtime/active_templates.json` 里没有当前 action 的模板。先录入并保存标准动作。

### 提示“请先录入患者动作”

说明当前 session 没有 patient attempt。先录入并保存患者动作。

### 评估失败但不知道原因

查看接口返回里的：

- `stdout`
- `stderr`
- `returncode`

这些来自 `subprocess.run`，用于定位 `evaluate/run_evaluate.py` 的错误。

### 为什么 TTS 和马达没有真实动作

第三阶段只做 mock。页面只显示将要播报的文字和马达 pattern，后续软硬联调阶段再接真实 TTS 和真实马达。

## 9. 验收流程

1. 启动 `prescription/record_prescription_http.py`。
2. 打开浏览器 UI。
3. 选择或保持 `knee_flexion`。
4. 点击“录入标准动作”。
5. 做标准示范动作。
6. 点击“保存为 active template”。
7. 确认页面显示 active template。
8. 点击“录入患者动作”。
9. 做患者动作。
10. 点击“保存 patient attempt”。
11. 确认页面显示 patient attempt。
12. 点击“结束并评估”。
13. 确认生成 `evaluate/reports/report_*.json`。
14. 确认页面展示 metrics、errors、structured_feedback。
15. 确认页面展示 TTS mock 和 motor mock。
