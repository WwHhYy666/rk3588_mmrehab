# RK3588 处方模块操作说明

这份文档用于指导你把 `prescription/` 模块跑在 RK3588 板子上，并把每次录制结果同时保存到：

- RK3588 板子本地的 `docs/results/` 和 `docs/summaries/`
- Windows 本机当前项目目录下的 `docs/results/` 和 `docs/summaries/`

适用场景：

- 板子通过 SSH 登录
- 浏览器在 Windows 本机打开
- RK3588 负责采集、推理、预览，并先把结果落盘到板端
- 浏览器再把同一份 JSON 转存到 Windows 本机

## 1. 当前目录下每个脚本的定位

- `record_prescription_json_improved.py`
  - 旧的 Windows 本地窗口版录制脚本
  - 继续保留，用于本地开发回归
- `banzi/record_prescription_http.py`
  - RK3588 浏览器版录制服务
  - 板子运行这个脚本
- `windows/local_result_sink.py`
  - Windows 本机结果接收器
  - 浏览器保存时会把板子返回的 JSON 转存到本机 `docs/results/`
  - 同时生成人看的中文摘要到 `docs/summaries/`
- `common/result_storage.py`
  - 双端共用的保存逻辑
  - 统一负责创建目录、写 JSON、生成摘要、更新 `results_log.md`
- `windows/read_prescription_json.py`
  - 读取当前机器上最新 JSON 结果摘要

根目录下的 `record_prescription_http.py`、`local_result_sink.py`、`read_prescription_json.py` 仍保留为兼容入口，但新文档和上板流程优先使用 `banzi/`、`windows/` 下的路径。

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
python .\windows\local_result_sink.py
```

启动成功后，可以访问：

```text
http://127.0.0.1:8090/health
```

如果返回 `ok: true`，说明本机保存端已经准备好。

## 4. RK3588 上启动什么

在 RK3588 板子上进入项目根目录后执行：

```bash
cd /你的项目路径
python3 prescription/banzi/record_prescription_http.py
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
4. 点击“双端同步保存”
5. 完成一组标准动作
6. 观察页面返回的板端和 Windows 路径

保存流程是：

1. 板子先在本地写入完整 JSON 到 `docs/results/`
2. 板子同时生成中文摘要到 `docs/summaries/`
3. 浏览器收到同一份 JSON
4. 浏览器把 JSON 发送到 Windows 本机 `http://127.0.0.1:8090/api/save_result`
5. Windows 本机把模板 JSON 写入 `docs/results/`
6. Windows 本机同时生成中文摘要到 `docs/summaries/`
7. 浏览器再调用 RK3588 的 `/api/ack_saved`
8. `ack_saved` 成功后，板端只清理“待同步导出缓存”，不会删除已经落盘的板端文件

## 6. 如何确认结果已经写到板端和本机

保存成功后，页面会显示：

- 板端模板 JSON 路径
- 板端摘要文件路径
- Windows 模板 JSON 路径
- Windows 摘要文件路径

你也可以直接检查：

```bash
ls /你的项目路径/prescription/docs/results
ls /你的项目路径/prescription/docs/summaries
cat /你的项目路径/prescription/docs/results_log.md
```

以及 Windows 本机：

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

- RK3588 板子 `docs/results/` 的磁盘占用
- Windows 本机 `docs/results/` 的磁盘占用

不是 RK3588 持续涨内存。

板端内存只会暂时保留：

- 当前录制缓存
- 尚未同步完成时的最近一次导出信息

一旦板端本地保存完成，录制缓存就可以结束本轮使命；如果 Windows 同步成功并且 `/api/ack_saved` 成功，板端剩余的“待同步导出缓存”也会清掉。

## 10. 如何读取最新结果

在任意一端执行：

```powershell
cd D:\rk3588\project\prescription
python .\windows\read_prescription_json.py
```

这个脚本默认会从当前机器上的：

```text
D:\rk3588\project\prescription\docs\results\
```

里找最新 JSON 并输出摘要信息。

## 11. 常见问题排查

### 11.1 浏览器能打开页面，但 Windows 本机保存失败

优先检查：

- `prescription/windows/local_result_sink.py` 是否已经在 Windows 本机启动
- `http://127.0.0.1:8090/health` 是否可访问
- Windows 防火墙或安全软件是否拦截了本地监听

如果板端已经生成并落盘结果，但 Windows 本机保存失败：

- 页面会提示“板端已完成本地保存，Windows 同步失败”
- 板端文件已经保底落盘，不会丢
- 板端会继续保留 `last_export`
- 修复本机接收器后，点击“重试导出最近结果”

### 11.2 Windows 已保存，但板端确认失败

这种情况表示：

- Windows 本机已经落盘成功
- 但 RK3588 还没有完成“待同步导出缓存”的清理

此时：

- 页面会提示“Windows 已保存，但板端导出清理失败”
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
- 板端模板 JSON 与中文摘要归档
- 浏览器把结果转存到 Windows 本机
- Windows 本机模板 JSON 与中文摘要归档

这一步不包含：

- GPIO 联动
- TTS 联动
- 主线程总装
