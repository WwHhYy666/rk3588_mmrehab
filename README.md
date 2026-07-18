# RK3588 多模态居家康复终端（NPU 8085）

面向 RK3588 的端侧康复训练闭环。系统在 `8085` 端口提供医生模板录入、患者训练、实时姿态识别、动作计数与纠错、TTS 提醒、训练报告、完成度评分及训练后问答。

当前唯一维护的姿态主线是：

```text
USB 摄像头
  -> YOLOv5n raw RKNN 人体检测
  -> RTMPose RKNN 关键点
  -> 三动作训练状态机
  -> 计数 / ROM_LOW / TUT_LOW / TTS
  -> attempt 与关键帧
  -> 评估报告与完成度评分
  -> GLM 或本地 Qwen 解释
```

## 功能范围

- 医生页面：`http://<RK3588-IP>:8085/doctor`
- 患者页面：`http://<RK3588-IP>:8085/train`
- NPU 调试：`http://<RK3588-IP>:8085/npu-debug`
- 健康状态：`http://<RK3588-IP>:8085/status`
- 默认动作：坐站、站姿屈膝后勾腿、坐姿抬膝
- 姿态模型：YOLOv5n raw + RTMPose，均由 RKNNLite 在 RK3588 NPU 上运行
- 可选能力：GLM、本地 Qwen RKLLM、Paraformer ASR、自然女声 TTS、ONNX 完成度评分

8082/MediaPipe 双模式、YOLOv8、RTMDet、固定 ROI 和各类 Windows/legacy 示例不再属于本分支维护范围。

## 仓库结构

```text
rehab_app/          8085 HTTP 服务、前端页面和应用服务
pose_estimation/    YOLOv5n + RTMPose 推理与关键点适配
training/           三动作状态机、TTS、音频和设备状态
evaluation/         训练后指标计算和报告生成
action_feedback/    可解释纠错规则与板端 TTS 适配
action_scoring/     轻量级时序评分模型的训练、导出和推理
speech/             Paraformer ASR 与异步问答 worker
llm/                RKLLM HTTP 代理
models/             按视觉、音频、语言和动作评分分类的模型注册表
data/               本地模板、attempt、报告和关键帧（不提交）
scripts/            8085 启停、检查、基准和部署工具
docs/               部署、验收和设计文档
tests/              跨模块契约测试
```

运行产生的模板、患者记录、报告、关键帧、日志、PID、模型权重和本地密钥均被 `.gitignore` 排除。

## 模型准备

模型文件不进入 Git。启动前将模型放到以下默认位置：

| 用途 | 默认路径 |
| --- | --- |
| 人体检测 | `models/vision/yolov5n_raw_fp.rknn` |
| 姿态估计 | `models/vision/rtmpose_m_256x192_fp.rknn` |
| 本地 Qwen（可选） | `models/language/qwen/qwen1_5b.rkllm` |
| Paraformer ASR（可选） | `models/audio/asr/paraformer/` |
| 自然女声 TTS（可选） | `models/audio/tts/sherpa_vits/` |

模型转换、板端放置和验收方法见 [docs/npu_rehab_8085_guide.md](docs/npu_rehab_8085_guide.md) 与 [docs/rk3588_qwen_rkllm_rknn_conversion_guide.md](docs/rk3588_qwen_rkllm_rknn_conversion_guide.md)。

模块间的数据流与职责边界见 [docs/architecture.md](docs/architecture.md)。

## 快速启动

RK3588 板端需要 Python 3、OpenCV、NumPy、PyYAML、RKNNLite2 及摄像头/GStreamer 运行环境。RKNNLite2 应使用 Rockchip 随板工具链提供的 wheel，不能用桌面端替代包冒充。

```bash
python3 -m pip install -r requirements.txt
cp .env.example runtime/llm.env
chmod +x scripts/*.sh

./scripts/start_npu_rehab_8085.sh
```

另开终端检查：

```bash
./scripts/check_npu_rehab_8085.sh
python3 scripts/benchmark_npu_rehab_8085.py --scenario train
```

停止服务：

```bash
./scripts/stop_npu_rehab_8085.sh
```

安装 systemd 与桌面自启动：

```bash
./scripts/install_npu_rehab_8085_autostart.sh
```

## 配置

复制 `.env.example` 到 `runtime/llm.env` 后按需填写。该文件只保存在本机，不会被 Git 跟踪。常用环境变量：

| 变量 | 作用 |
| --- | --- |
| `RKNN_DET_MODEL` | 检测模型路径 |
| `RKNN_RTMPOSE_MODEL` | 姿态模型路径 |
| `RK_CAMERA_DEVICE` | 摄像头设备路径 |
| `REHAB_LLM_PROVIDER` | `auto`、`glm4v_api` 或 `local_qwen_rkllm` |
| `GLM_API_KEY` | 在线 GLM 密钥 |
| `QWEN_RKLLM_MODEL` | 本地 Qwen RKLLM 模型路径 |
| `REHAB_AUDIO_OUTPUT_DEVICE` | ALSA 输出设备 |

完整参数说明见 [docs/README.md](docs/README.md)。

## 测试

桌面端可运行不依赖真实 NPU 的单元和契约测试：

```bash
python -m pip install -r requirements.txt
python -m pytest
```

真实摄像头、RKNNLite、NPU 性能和 systemd 启动必须在 RK3588 上使用检查脚本验收。

## 数据与隐私

- 不提交 `runtime/`、患者 attempt、医生模板、报告、关键帧、日志和录音。
- 不提交 `.env`、API Key、访问令牌或设备私有配置。
- 示例和测试数据应先匿名化；真实患者数据不得进入公开仓库。
- 模型文件由使用者按其许可证单独取得和部署。

## 文档

文档总索引见 [docs/README.md](docs/README.md)。8085 的部署、双模型校验、摄像头检查、性能基准、故障排查和验收口径以 [docs/npu_rehab_8085_guide.md](docs/npu_rehab_8085_guide.md) 为准。提交代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
