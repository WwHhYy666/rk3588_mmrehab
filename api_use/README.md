# GLM-4V-Flash 摄像头康复动作分析方案

这个目录用于先在笔记本上跑通“本地摄像头 + 网络 API 大模型 + 语音交互”的方案，后续再迁移到 RK3588 板卡。

主脚本：

```text
api_use/glm4v_rehab_camera.py
```

它完成的闭环：

1. 用 OpenCV 读取本地摄像头。
2. 用帧差状态机检测一个动作循环，并提取最可能处于动作顶点的关键帧。
3. 通过 HTTP `POST https://open.bigmodel.cn/api/paas/v4/chat/completions` 调用 `glm-4v-flash`。
4. 把关键帧和初始化提示词一起传给模型，让模型分析动作标准度、关键角度、风险和纠正建议。
5. 一组动作结束后，让模型根据动作、次数、总时长、体重和历史分析估计卡路里。
6. 可选开启麦克风语音命令，用户可以说“总结”“估计卡路里”“我哪里不标准”“退出程序”等，模型回复会打印在终端。

## 安装

在项目根目录执行：

```powershell
pip install -r .\api_use\requirements-glm4v.txt
```

如果要启用语音交互，并使用离线中文识别，继续安装原项目的 Whisper 依赖：

```powershell
pip install -r .\requirements-whisper.txt
```

设置智谱 API Key：

```powershell
$env:ZHIPUAI_API_KEY="你的 API Key"
```

## 运行

先查看摄像头编号：

```powershell
python .\api_use\glm4v_rehab_camera.py --list-cameras
```

笔记本摄像头实时分析：

```powershell
python .\api_use\glm4v_rehab_camera.py `
  --camera-index 0 `
  --exercise-name "缓慢高抬腿" `
  --target-reps 8 `
  --weight-kg 65 `
  --show-preview `
  --stop-after-set
```

开启语音命令：

```powershell
python .\api_use\glm4v_rehab_camera.py `
  --camera-index 0 `
  --exercise-name "缓慢高抬腿" `
  --target-reps 8 `
  --weight-kg 65 `
  --show-preview `
  --enable-voice `
  --asr-backend faster-whisper `
  --whisper-model tiny
```

预览窗口按键：

- `space`：手动发送当前帧给 GLM。
- `s`：立即总结本组动作并估计卡路里。
- `q`：退出程序；默认会在退出前总结已完成动作。

不用摄像头，先拿一张本地图片测试 API：

```powershell
python .\api_use\glm4v_rehab_camera.py `
  --image-file .\test.jpg `
  --exercise-name "缓慢高抬腿" `
  --target-reps 1
```

只检查流程、不访问网络：

```powershell
python .\api_use\glm4v_rehab_camera.py --image-file .\test.jpg --dry-run
```

## 关键参数

- `--motion-threshold`：动作开始的帧差阈值。环境光变化大或误触发时调高；动作幅度小但检测不到时调低。
- `--stable-frames`：动作结束前需要连续稳定的帧数。动作慢时可以调大。
- `--min-cycle-seconds` / `--max-cycle-seconds`：动作循环的最短和最长时间。
- `--image-max-width` / `--jpeg-quality`：控制发给 API 的图片大小。板卡部署时可降低到 `480` 和 `75`。
- `--image-mode`：默认 `data_uri`。如果 API 对 base64 图片格式有不同要求，可尝试 `raw_base64`。

## 板卡迁移建议

- 笔记本阶段先用 `--show-preview` 调阈值；板卡无桌面环境时去掉该参数。
- 板卡上优先降低 `--camera-width`、`--camera-height`、`--sample-fps`、`--image-max-width`，减少编码和网络开销。
- 如果后续接入姿态估计模型，可以把当前 `MotionKeyFrameExtractor` 替换成基于人体关键点的关键帧提取器，GLM 调用层不需要改。
- 模型反馈仅用于训练辅助，不代替医生或康复师判断。
