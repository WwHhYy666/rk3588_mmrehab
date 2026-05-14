# RK3588 处方模块操作说明

这份文档用于指导你把 `prescription/` 模块跑在 RK3588 板子上，并把每次录制结果直接保存到 Windows 本机当前项目目录下的 `docs/results/` 和 `docs/summaries/`，而不是保存到板子里。

适用场景：

- 板子通过 SSH 登录
- 浏览器在 Windows 本机打开
- RK3588 只负责采集、推理、预览和在内存中组装 JSON
- 结果文件只落在本机 `Path(__file__).resolve().parent / "docs" / "results"`

## 1. 当前目录下每个脚本的定位

- `record_prescription_json_improved.py`
  - 旧的 Windows 本地窗口版录制脚本
  - 继续保留，用于本地开发回归
- `record_prescription_http.py`
  - RK3588 浏览器版录制服务
  - 板子运行这个脚本
- `local_result_sink.py`
  - Windows 本机结果接收器
  - 浏览器保存时会把板子返回的 JSON 转存到本机 `docs/results/`
  - 同时生成人看的中文摘要到 `docs/summaries/`
- `read_prescription_json.py`
  - 读取本机最新 JSON 结果摘要

## 2. 板端先确认什么

先在 RK3588 上确认摄像头和 Python 环境仍然可用。

```bash
ls /dev/video21
v4l2-ctl --device=/dev/video21 --all
python3 --version
pip3 --version
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

如果这里已经失败，不要先跑处方脚本，先回到视觉环境把摄像头和 MediaPipe 确认好。

## 3. Windows 本机先启动什么

先在 Windows 本机启动结果接收器。

```powershell
cd D:\rk3588\project\prescription
python .\local_result_sink.py
```

启动成功后，可以访问：

```text
http://127.0.0.1:8090/health
```

如果返回 `ok: true`，说明本机保存端已经准备好。

## 4. RK3588 上启动什么

在 RK3588 板子上进入 `prescription/` 目录后执行：

```bash
cd /你的项目路径/prescription
python3 record_prescription_http.py
```

默认访问地址：

```text
http://板子IP:8082
```

## 5. 浏览器里如何开始录制

在 Windows 浏览器里打开：

```text
http://板子IP:8082
```

然后按这个顺序操作：

1. 填写患者编号，例如 `patient_001`
2. 填写动作名称，例如 `knee_flexion`
3. 选择侧别模式
4. 点击“开始录制”
5. 完成一组标准动作
6. 点击“保存到本地”

保存流程是：

1. 板子先在内存里生成完整 JSON
2. 浏览器收到 JSON
3. 浏览器把 JSON 发送到 Windows 本机 `http://127.0.0.1:8090/api/save_result`
4. Windows 本机把模板 JSON 写入 `docs/results/`
5. Windows 本机同时生成中文摘要到 `docs/summaries/`
6. 浏览器再调用 RK3588 的 `/api/ack_saved`
7. 只有 `/api/ack_saved` 成功后，板端才清空本次录制缓存和 `last_export`

## 6. 如何确认结果已经写到本地

保存成功后，页面会显示：

- 模板 JSON 路径
- 摘要文件路径

你也可以直接检查：

```powershell
Get-ChildItem D:\rk3588\project\prescription\docs\results
Get-ChildItem D:\rk3588\project\prescription\docs\summaries
Get-Content D:\rk3588\project\prescription\docs\results_log.md
```

## 7. 平时应该看哪个文件

日常优先看：

- `docs/summaries/`

因为这里是短摘要，适合快速抓重点。

模板 JSON 在：

- `docs/results/`

它是给后续算法、模板复用和对比留的，所以按帧保存，行数天然会比较多，不适合直接人工阅读。

## 8. 为什么画面里不再显示中文叠字

这次故意把视频帧上的文字叠加去掉了，原因是当前预览链路用的是 `cv2.putText`，它默认不支持中文字体，所以中文会变成 `????`。

这次的处理方式是：

- 视频里只保留原始画面和骨架
- 所有状态统一放到网页右侧中文面板里显示

这样更稳定，也不依赖板端字体环境。

## 9. 这是磁盘占用，不是板端一直吃内存

多录几次以后，增长的主要是：

- Windows 本机 `docs/results/` 的磁盘占用

不是 RK3588 持续涨内存。

板端内存只会暂时保留：

- 当前录制缓存
- 尚未 `ack_saved` 的 `last_export`

一旦本机保存成功并且 `/api/ack_saved` 成功，板端这部分内存就会清掉。

## 10. 如何读取最新结果

在 Windows 本机执行：

```powershell
cd D:\rk3588\project\prescription
python .\read_prescription_json.py
```

这个脚本默认会从：

```text
D:\rk3588\project\prescription\docs\results\
```

里找最新 JSON 并输出摘要信息。

## 11. 常见问题排查

### 11.1 浏览器能打开页面，但本机保存失败

优先检查：

- `local_result_sink.py` 是否已经在 Windows 本机启动
- `http://127.0.0.1:8090/health` 是否可访问
- Windows 防火墙或安全软件是否拦截了本地监听

如果板端已经生成结果但本机保存失败：

- 页面会提示“板端结果已生成，但本机保存失败”
- 板端会继续保留 `last_export`
- 修复本机接收器后，点击“重试导出最近结果”

### 11.2 本机已保存，但板端确认失败

这种情况表示：

- Windows 本机已经落盘成功
- 但 RK3588 还没有完成清理

此时：

- 页面会提示“本机已保存，但板端确认失败”
- 板端仍保留 `last_export`
- 可以继续点“重试导出最近结果”，或者补一次确认

### 11.3 画面有骨架，但摘要还是不理想

优先看 `docs/summaries/` 里的中文摘要，重点关注：

- 有效帧数
- 无效帧数
- ROM
- 结果判断

如果 ROM 很小或者无效帧太多，通常需要重新录制一版更稳定的动作。

## 12. 当前边界说明

这一步只处理：

- RK3588 浏览器版处方录制
- 浏览器把结果转存到 Windows 本机
- 本机模板 JSON 与中文摘要归档

这一步不包含：

- GPIO 联动
- TTS 联动
- 主线程总装
- 板端历史结果归档
