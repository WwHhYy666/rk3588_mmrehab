# RK3588 训练后问答：GLM 与 Qwen RKLLM

8085 主训练不在 HTTP 进程内加载大模型。问答链路与摄像头、姿态线程隔离：

```text
8085 页面
-> speech/llm_worker.py
-> rehab_app/services/llm_assistant.py
   -> 有 Key：GLM API
   -> 无网/强制本地：127.0.0.1:18080 RKLLM proxy
      -> 127.0.0.1:8080/rkllm_chat
      -> Qwen RKLLM server
```

## 本地 Qwen 文件

默认路径：

```text
模型       models/language/qwen/qwen1_5b.rkllm
server     /home/elf/qwen_server/rkllm_server/flask_server.py
运行库     /home/elf/qwen_server/rkllm_server/lib/librkllmrt.so
```

模型和 Rockchip 运行库不进入 Git。请根据 RKLLM Toolkit 版本使用匹配的转换环境、量化配置和板端运行库。

## 配置

复制 `.env.example`：

```bash
cp .env.example runtime/llm.env
```

自动路由示例：

```bash
REHAB_LLM_PROVIDER=auto
REHAB_LLM_ONLINE_PROVIDER=glm4v_api
REHAB_LLM_OFFLINE_PROVIDER=local_qwen_rkllm
GLM_API_KEY=
QWEN_RKLLM_MODEL=models/language/qwen/qwen1_5b.rkllm
QWEN_SERVER_DIR=/home/elf/qwen_server/rkllm_server
```

强制本地：

```bash
REHAB_LLM_PROVIDER=local_qwen_rkllm
```

强制在线：

```bash
REHAB_LLM_PROVIDER=glm4v_api
GLM_API_KEY=<your-key>
```

## 启动

`scripts/start_npu_rehab_8085.sh` 会加载 `runtime/llm.env`。若 Qwen 模型、server 和运行库完整，它会先启动 8080 server，再启动 `llm/rkllm_proxy.py` 的 18080 proxy；任何一项缺失时只警告，不阻塞 8085 康复训练。

```bash
./scripts/start_npu_rehab_8085.sh
```

检查：

```bash
BASE_URL=http://127.0.0.1:8085 ./scripts/check_ai_assistant_status.sh
curl -s http://127.0.0.1:18080/health | python3 -m json.tool
```

## 资源互斥

姿态识别和本地 Qwen 都可能占用 NPU。8085 服务通过训练状态和资源快照避免问答抢占正在进行的姿态任务：

- 医生录入和患者训练期间不启动慢问答；
- 本地 Qwen 生成时，NPU 调试页不会强制加载姿态；
- GLM/Qwen 不可用时返回明确错误，不伪造回答；
- 问答失败不影响计数、纠错、TTS 和报告生成。

## API

```text
GET  /api/voice/status
POST /api/voice/ask
GET  /api/voice/ask_result
GET  /status
```

问题会携带当前动作报告的精简上下文。图片只在配置允许且确有关键帧时发送。

## 验收

1. 8085 未训练时 `/api/voice/status` 能区分 GLM、本地 Qwen 和不可用状态。
2. 本地模式下 8080 与 18080 健康检查通过。
3. 训练中提交问答会被保护逻辑拒绝或排队，不降低姿态帧率。
4. 训练结束后问题能引用当前报告，而不是返回与动作无关的通用文本。
5. 连续问题不会被旧 job 的迟到结果覆盖。
6. 删除 Key 或停掉 Qwen 后，康复主闭环仍正常。
