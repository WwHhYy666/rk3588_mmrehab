# GLM 大模型接入使用说明

本文说明 8082 统一训练台中的第一版 GLM 大模型功能。

## 功能说明

GLM 只用于训练后的增强能力：

- 训练报告 AI 总结
- 基于评估报告的患者问答
- 可选朗读患者版短文本

GLM 不参与：

- 实时姿态检测
- 实时动作计数
- 实时规则判断
- 原有纠错 TTS 主链路

主链路仍然是：

```text
摄像头 -> 姿态识别 -> rep 计数 -> 规则反馈 -> TTS
```

训练后增强链路是：

```text
evaluate report -> LLM 总结/问答 -> UI 展示 -> 可选 TTS 朗读
```

## 依赖安装

主服务的 GLM 接入使用 Python 标准库 `urllib`，不需要额外安装 `requests`。

最小主服务依赖不变。只要现有 8082 训练台能启动，`REHAB_LLM_PROVIDER=echo` 就能测试 AI 区域。

`api_use/` 目录仍可作为独立 GLM-4V 摄像头示例使用。如果要单独运行 `api_use/glm4v_rehab_camera.py`，再安装：

```bash
pip install -r api_use/requirements-glm4v.txt
```

## 环境变量

默认 smoke test 模式：

```bash
REHAB_LLM_PROVIDER=echo
```

真实 GLM API：

```bash
export ZHIPUAI_API_KEY="你的 API Key"
export REHAB_LLM_PROVIDER=glm4v_api
```

可选配置：

```bash
export GLM_API_KEY="你的 API Key"
export REHAB_LLM_MODEL="glm-4v-flash"
export REHAB_LLM_ENDPOINT="https://open.bigmodel.cn/api/paas/v4/chat/completions"
export REHAB_LLM_TIMEOUT=30
export REHAB_LLM_MAX_TOKENS=768
```

说明：

- `ZHIPUAI_API_KEY` 优先于 `GLM_API_KEY`。
- 不要把 API Key 写入仓库文件。
- 没有 API Key 时请使用 `REHAB_LLM_PROVIDER=echo`。

## 启动命令

CPU/MediaPipe 稳定演示 + echo 模式：

```bash
REHAB_LLM_PROVIDER=echo RK_CAMERA_DEVICE=auto RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=360 python3 prescription/banzi/record_prescription_http.py
```

真实 GLM API：

```bash
export ZHIPUAI_API_KEY="你的 API Key"
export REHAB_LLM_PROVIDER=glm4v_api
RK_CAMERA_DEVICE=auto RK_CAMERA_WIDTH=640 RK_CAMERA_HEIGHT=360 python3 prescription/banzi/record_prescription_http.py
```

浏览器访问：

```text
http://板子IP:8082
```

## 页面使用方法

1. 打开 `/doctor`。
2. 依次录入并保存医生标准模板：
   - `seated_knee_extension`
   - `standing_hamstring_curl`
   - `seated_knee_raise`
3. 打开 `/train`。
4. 点击“开始完整训练”，完成三动作训练。
5. 训练结束后报告卡片会显示基础评估结果。
6. 点击“生成 AI 建议”，生成患者版总结、医生版总结、下一步建议、风险提醒和热量粗略估计。
7. 在“报告问答”输入框提问，例如“我刚才哪里没做好？”。
8. 点击“朗读 AI 建议”或“朗读回答”，系统只会朗读 `spoken_text` 短文本。

AI 朗读不会在实时训练进行中抢占计数和纠错语音。如果系统提示“当前正在实时训练中，暂不朗读 AI 内容”，请等训练结束后再朗读。

## 常见问题

### 没 API Key 怎么测试？

使用默认 echo 模式：

```bash
REHAB_LLM_PROVIDER=echo python3 prescription/banzi/record_prescription_http.py
```

页面会显示“当前为 echo 模式，仅用于测试”，但可以完整验证按钮、接口和 UI。

### API 超时怎么办？

页面会显示“GLM API 请求超时，请稍后重试”。这只影响 AI 区域，不影响 `/doctor`、`/train`、摄像头流、姿态检测、规则反馈和原 TTS。

可以适当增大：

```bash
export REHAB_LLM_TIMEOUT=60
```

板端排查命令：

```bash
getent hosts open.bigmodel.cn
python3 -c "import urllib.request; print(urllib.request.urlopen('https://open.bigmodel.cn', timeout=5).status)"
date
```

如果域名解析失败，优先检查 DNS；如果 HTTPS 证书失败，优先检查系统时间和 CA 证书；如果提示网络不可达，优先检查网关和外网出口。

### 板端没网怎么办？

使用 echo 模式演示，或者先只展示本地规则报告。真实 GLM API 需要板端能访问外网接口。

临时演示可以继续使用：

```bash
REHAB_LLM_PROVIDER=echo python3 prescription/banzi/record_prescription_http.py
```

### 为什么 AI 不参与实时计数？

实时计数需要低延迟和稳定性。大模型网络请求可能超时或变慢，所以第一版只在训练后解释报告，避免拖慢摄像头、姿态识别和 rep 计数。

### 为什么热量只是粗略估计？

当前报告缺少体重、心率和真实运动强度等信息。热量估算只能作为参考，不能作为医学依据。

### 为什么不能问诊断和用药问题？

系统只能解释本次训练报告，不能替代医生。涉及诊断、用药、是否加量、是否停药、是否手术、是否痊愈等问题时，应咨询医生或康复师。

## 板端上传清单

需要上传：

```text
prescription/banzi/record_prescription_http.py
prescription/common/llm_assistant.py
realtime/tts_worker.py
prescription/banzi/static/common.js
prescription/banzi/static/doctor.js
prescription/banzi/static/train.js
docs/GLM readme.md
```

如果后续修改了 CSS，也上传：

```text
prescription/banzi/static/app.css
```

可选上传，不是主服务必需：

```text
api_use/glm4v_rehab_camera.py
api_use/README (1).md
api_use/requirements-glm4v.txt
```

严禁覆盖板端已有运行数据：

```text
runtime/active_templates.json
docs/results/
prescription/docs/results/
evaluate/reports/
prescription/docs/summaries/
```

如果板端实际结果路径以 `prescription/docs/results/` 为准，请以现有代码为准，不要误删或覆盖任何运行数据。

## 验证命令

Python 语法检查：

```bash
python3 -m py_compile prescription/common/llm_assistant.py prescription/banzi/record_prescription_http.py realtime/tts_worker.py
```

JavaScript 语法检查：

```bash
node --check prescription/banzi/static/common.js
node --check prescription/banzi/static/doctor.js
node --check prescription/banzi/static/train.js
```
