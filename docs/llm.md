# RK3588 训练后 GLM 图文建议使用说明

文件位置：`docs/llm.md`

本文说明 8082 统一训练台新增的“关键帧图片 + 指标卡片 + 对比图”能力怎么上传、启动、调试和验收。

## 1. 需要上传的代码文件

把下面文件按原路径上传到板端项目目录：

- `realtime/training_session.py`
- `prescription/banzi/record_prescription_http.py`
- `evaluate/run_evaluate.py`
- `prescription/common/llm_assistant.py`
- `prescription/common/report_visuals.py`
- `prescription/banzi/static/common.js`
- `prescription/banzi/static/app.css`
- `docs/llm.md`

不要上传或覆盖这些运行数据目录：

- `runtime/`
- `prescription/docs/results/`
- `prescription/docs/summaries/`
- `evaluate/reports/`
- `evaluate/reports/keyframes/`
- `__pycache__/`

## 2. 启动 8082：真实 GLM API 模式

如果要验证“不是预设话术”，不要使用 `echo`。必须让 `/status` 显示：

```json
{
  "provider": "glm4v_api",
  "model": "glm-4v-flash",
  "api_key_configured": true
}
```

### 2.1 启动前检查外网

在 RK3588 板端终端执行：

```bash
date
hostname -I
getent hosts open.bigmodel.cn
python3 -c "import urllib.request; print(urllib.request.urlopen('https://open.bigmodel.cn', timeout=5).status)"
```

判断标准：

- `getent hosts open.bigmodel.cn` 能解析出 IP，说明 DNS 基本正常。
- Python 命令能返回 HTTP 状态码，说明 HTTPS 外网大概率可达。
- 如果失败，先修网络、DNS、网关、代理或系统时间，再启动 8082。

### 2.2 设置真实 API 环境变量

不要把 API Key 写进仓库文件，也不要发到聊天记录里。只在板端当前终端临时设置：

```bash
cd /path/to/project
export ZHIPUAI_API_KEY="你的真实智谱API Key"
export REHAB_LLM_PROVIDER=glm4v_api
export REHAB_LLM_MODEL=glm-4v-flash
export REHAB_LLM_TIMEOUT=60
export RK_CAMERA_DEVICE=auto
export RK_CAMERA_WIDTH=640
export RK_CAMERA_HEIGHT=360
```

确认不是 echo：

```bash
echo $REHAB_LLM_PROVIDER
echo $REHAB_LLM_MODEL
```

应该看到：

```text
glm4v_api
glm-4v-flash
```

### 2.3 启动服务

```bash
python3 prescription/banzi/record_prescription_http.py
```

终端出现下面类似信息后，不要关闭这个终端：

```text
8082 统一训练台已启动: http://板子IP:8082
```

### 2.4 浏览器和接口验证

在板子浏览器打开：

```text
http://127.0.0.1:8082
```

也可以另开一个终端检查 `/status`：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

找到 `llm` 字段，应类似：

```json
{
  "provider": "glm4v_api",
  "model": "glm-4v-flash",
  "api_key_configured": true,
  "endpoint_configured": true,
  "last_error": null,
  "last_latency_ms": null
}
```

如果这里显示 `provider=echo`，说明环境变量没有在启动 8082 的同一个终端里生效。按 `Ctrl + C` 停止服务，重新 export 后再启动。

### 2.5 echo 只用于离线兜底

只有在没有外网或没有 API Key 时，才使用 echo：

```bash
export REHAB_LLM_PROVIDER=echo
python3 prescription/banzi/record_prescription_http.py
```

echo 能验证按钮、接口和 UI，但它不是个性化真实大模型回复，不能作为真实 GLM 验收结果。

## 3. 是否需要重新录动作

旧 report 没有 `keyframes` 字段，所以要重新完成一次患者训练才能验证图片能力。

通常不需要重录 `/doctor` 医生模板；只要 `runtime/active_templates.json` 里的 active template 仍然存在，就可以直接进入 `/train` 重新跑患者训练。只有在 active template 缺失、模板文件不存在或想重新校准动作模板时，才需要先去 `/doctor` 重录。

训练时不会保存完整视频。系统只会为每个有效 rep 保存 1 张 best_peak JPEG。

## 4. 训练后检查

完成 `/train` 后检查：

```text
evaluate/reports/keyframes/<session_id>/
```

目录里应出现类似文件：

```text
seated_knee_extension_rep1_best.jpg
standing_hamstring_curl_rep1_best.jpg
seated_knee_raise_rep1_best.jpg
```

新生成的 report JSON 顶层应包含：

```json
{
  "keyframes": [
    {
      "kind": "best_peak",
      "image_path": "evaluate/reports/keyframes/<session_id>/seated_knee_extension_rep1_best.jpg",
      "rep_index": 1,
      "primary_metric": "knee_extension_angle",
      "signal_value": 123.4
    }
  ]
}
```

report 里只保存相对路径，不保存 base64。

## 5. report_summary 接口

前端点击“生成 AI 建议”时会自动请求：

```http
POST /api/llm/report_summary
Content-Type: application/json
```

请求示例：

```json
{
  "report_id": "latest",
  "audience": "both",
  "include_calorie": true,
  "include_keyframes": true,
  "render_metric_cards": true
}
```

响应会在原有患者版总结、医生版总结、问答朗读字段之外，增加：

```json
{
  "keyframe_notes": [
    "第 1 次动作保存了 best_peak 关键帧；图片只用于辅助观察，精确指标以 report 为准。"
  ],
  "metric_cards": [
    {
      "action_id": "seated_knee_extension",
      "benefit_parts": ["股四头肌", "膝关节伸展控制"],
      "target_joint": "膝关节",
      "calorie_estimate": {
        "value_kcal": null,
        "text": "热量粗略估计，仅供参考；报告缺少体重、心率和真实运动强度，暂不输出精确数值。"
      }
    }
  ],
  "rendered_images": {
    "raw_keyframe_image": {
      "path": "evaluate/reports/keyframes/<session_id>/seated_knee_extension_rep1_best_raw_keyframe.jpg",
      "url": "/report-images/evaluate/reports/keyframes/<session_id>/seated_knee_extension_rep1_best_raw_keyframe.jpg"
    },
    "metric_card_image": {
      "path": "evaluate/reports/keyframes/<session_id>/metric_card_seated_knee_extension_rep1.jpg",
      "url": "/report-images/evaluate/reports/keyframes/<session_id>/metric_card_seated_knee_extension_rep1.jpg"
    },
    "comparison_image": {
      "path": "evaluate/reports/keyframes/<session_id>/comparison_rep1.jpg",
      "url": "/report-images/evaluate/reports/keyframes/<session_id>/comparison_rep1.jpg"
    }
  }
}
```

如果 GLM、图片读取或图片渲染失败，训练、计数、TTS 和 report 生成不受影响。前端会继续显示文本 AI 建议。

## 6. 前端查看

打开：

```text
http://<板端IP>:8082/train
```

完成训练后，在报告区域点击“生成 AI 建议”。AI 区域会保留：

- 患者版总结
- 医生版总结
- 下一步建议
- 风险提醒
- 报告问答
- 朗读 AI 建议
- 朗读回答

并新增：

- AI 图文建议
- 关键帧原图
- 指标卡片图
- 左右对比图，如果渲染成功

## 7. 验证是否真的调用了大模型

真实 GLM API 的验收标准：

- 页面或 `/status` 显示 `provider=glm4v_api`，不是 `echo`。
- `/status` 中 `api_key_configured=true`。
- 点击“生成 AI 建议”后，返回内容里带 `provider: glm4v_api` 和 `latency_ms`。
- `/status` 中 `llm.last_latency_ms` 从 `null` 变成数字。
- 患者版总结、医生版总结、下一步建议、风险提醒会结合当前 report 的动作名、ROM、TUT、DTW、速度或错误码。

验证“不是预设话术”的建议流程：

1. 完成一次 `/train`，生成新的 report。
2. 点击“生成 AI 建议”。
3. 故意换一个动作或做一组不同质量的动作，再生成一次 AI 建议。
4. 在“报告问答”里对同一份 report 提两个不同问题：
   - `我刚才哪里没做好？`
   - `下一次动作重点是什么？`
5. 如果回答能引用当前 report 的问题，并且不同 report 或不同问题的回答有差异，才算真实个性化调用通过。

如果每次都显示 echo、没有 `latency_ms`、没有外网请求延迟，或者内容固定不变，那就是没有走真实 API。

## 8. GLM prompt 安全边界

GLM 只能结合 report 和关键帧做解释：

- 图片只是辅助观察
- 精确角度、ROM、TUT、DTW、速度、热量以 report 为准
- 不要从图片编造准确数值
- 单张图片不能判断持续时间、速度和完整动作轨迹
- 图片不清晰、人体不完整或关键部位遮挡时，要说明无法可靠观察
- 不得诊断疾病、建议用药或调整治疗方案

热量估计必须标注“粗略估计，仅供参考”。

## 9. 常见问题

### 旧 report 没有图片

旧 report 是功能上线前生成的，没有 `keyframes` 字段。重新跑一次 `/train`。

### 没有 keyframes 目录

确认患者训练完成了有效 rep。无效动作不会保存关键帧。

### 前端没有图片

检查 report JSON 是否有 `keyframes`，并确认 `image_path` 指向：

```text
evaluate/reports/keyframes/
```

系统拒绝绝对路径、`..` 路径穿越和非 JPEG 路径。

### GLM API 失败

先检查 `/status`：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

常见原因：

- `api_key_configured=false`：没有在启动 8082 的终端里设置 `ZHIPUAI_API_KEY`。
- `provider=echo`：没有设置 `REHAB_LLM_PROVIDER=glm4v_api`，或设置后没有重启 8082。
- `GLM API 请求超时`：检查外网，或把 `REHAB_LLM_TIMEOUT` 调到 `90`。
- `GLM API 调用失败`：检查 API Key、额度、模型名、endpoint、系统时间和 HTTPS 证书。

真实 API 调试时不要把 echo 当成验收结果。echo 只能证明本地 UI 和接口链路没坏。

### 指标卡片没有生成

检查是否安装 Pillow。没有 Pillow 时文本 AI 建议仍可用，训练主流程不受影响。

### 不要做的事

- 不要保存完整训练视频
- 不要把 base64 写进 report JSON 或日志
- 不要覆盖运行数据目录
- 不要把 report 输出目录当作医生模板或患者动作输入
