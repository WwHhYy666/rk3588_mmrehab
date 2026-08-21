# RK3588 视觉模块上板操作指南

这份 `guide.md` 是给后续通过 SSH 登录 RK3588 板子的人准备的。

目标很明确：

- 确认摄像头节点还在
- 确认 Python 环境可用
- 安装或确认依赖库
- 按顺序运行 `camera_test.py`
- 再运行 `pose_mediapipe_demo.py`

如果你按下面顺序执行，一般不会走弯路。

## 1. 当前已经确认的摄像头信息

目前这套脚本是按下面这些板端实际参数写的：

- 设备节点：`/dev/video21`
- 图像格式：`MJPG`
- 分辨率：`1280x720`

当前已知验证结果：

- 用 `ffmpeg` 成功拍照
- 用 OpenCV 成功读取并保存图片

所以当前重点不是“摄像头能不能识别到”，而是“能不能把实时预览和 MediaPipe Pose 跑起来”。

## 2. 先确认摄像头设备节点是否存在

先 SSH 到板子上，然后执行：

```bash
ls /dev/video21
```

如果正常，会看到：

```text
/dev/video21
```

如果这里就报错，说明：

- 摄像头没有正确连接
- 驱动没有正常加载
- 设备节点已经变化，不再是 `/dev/video21`

这时候不要急着跑 Python 脚本，先把摄像头节点问题确认清楚。

## 3. 再确认摄像头格式和分辨率

执行：

```bash
v4l2-ctl --device=/dev/video21 --all
```

重点看这些信息：

- 当前设备是否真的是 `/dev/video21`
- 是否支持 `MJPG`
- 是否支持 `1280x720`

如果你想看更明确的格式列表，也可以执行：

```bash
v4l2-ctl --device=/dev/video21 --list-formats-ext
```

这一步的目的很简单：

- 确认脚本里的参数和板子当前真实能力一致

## 4. 确认 Python 和 pip 环境

先看 Python 版本：

```bash
python3 --version
```

再看 pip：

```bash
pip3 --version
```

建议优先保证：

- 有 `python3`
- 有 `pip3`
- 后面装库时都在同一个环境里操作

如果你们之前已经有一个能成功用 OpenCV 读取摄像头的 Python 环境，建议继续沿用，不要先折腾重装 OpenCV。

## 5. 如果 `sudo apt update` 失败，先排查网络链路

有一种很常见的情况是：

- 板子一直连着网
- 也能通过 SSH 连进去
- 但是 `sudo apt update` 一直失败

这种情况通常不表示“完全没网”，更像是：

- 只有局域网，没有真正出外网
- 默认路由缺失
- 网关或 NAT 没有真正放行
- DNS 或软件源只是后续问题，不一定是第一矛盾

所以这时不要先急着换软件源，优先按下面顺序判断：

```text
本地链路 -> 默认路由 -> 网关 -> 公网 IP -> DNS -> apt 源
```

建议先执行：

```bash
ip addr
ip route
cat /etc/resolv.conf
ping -c 3 223.5.5.5
ping -c 3 8.8.8.8
getent hosts deb.debian.org
```

怎么理解这些结果：

- `ip addr`
  - 看有没有有效 IP
- `ip route`
  - 看有没有 `default via ...`
- `ping` 公网 IP
  - 看是不是根本没出去
- `getent hosts deb.debian.org`
  - 看 DNS 是否正常

如果你现在卡在这一步，请优先看这份详细排查文档：

- [`apt_update_troubleshooting.md`](D:\rk3588\project\vision\apt_update_troubleshooting.md)

如果最后确认板子只是“假联网”，不要把进度卡死在 `apt` 上，优先考虑：

1. 换一个真实外网做一次环境初始化，例如手机热点
2. 用电脑下载依赖后拷到板子上
3. 先复用当前已经能跑 Python 和 OpenCV 的环境

## 6. 安装或确认系统工具

### 6.1 安装 `v4l-utils`

如果板子上没有 `v4l2-ctl`，先装：

```bash
sudo apt update
sudo apt install -y v4l-utils
```

如果这里的 `sudo apt update` 已经失败，不要反复重试，先回到上一节排查网络链路。

安装完成后，可以再执行：

```bash
v4l2-ctl --version
```

### 6.2 安装 `ffmpeg`

如果板子上没有 `ffmpeg`，执行：

```bash
sudo apt install -y ffmpeg
```

安装完成后，可以执行：

```bash
ffmpeg -version
```

## 7. 安装或确认 Python 依赖库

### 7.1 先确认 OpenCV

执行：

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

如果能正常打印版本号，说明当前环境里的 OpenCV 可用。

### 7.2 安装或确认 `numpy`

执行：

```bash
pip3 install numpy
```

然后可以验证：

```bash
python3 -c "import numpy as np; print(np.__version__)"
```

### 7.3 安装或确认 `mediapipe`

先执行：

```bash
pip3 install mediapipe
```

再验证：

```bash
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

这里有一个很重要的现实风险：

- RK3588 是 `arm64`
- `mediapipe` 的官方预编译 wheel 可能不直接支持这个平台

如果你看到类似下面这种错误：

```text
No matching distribution found for mediapipe
```

优先判断为：

- 平台 wheel 支持问题

不要先怀疑 `camera_test.py` 或 `pose_mediapipe_demo.py` 的代码本身有问题。

### 7.4 预备安装 `python-periphery`

这一步不是视觉验证必须的，但为了后续 GPIO 联调，建议先记住这个依赖：

```bash
pip3 install python-periphery
```

它是给后面的震动马达 GPIO 控制模块用的，不是当前这两个视觉脚本必需项。

## 8. 按顺序验证视觉脚本

下面这个顺序不要跳。

如果你是通过纯 SSH 从 Windows 笔记本连板子，而且又想保留 `cv2.imshow()` 的实时预览窗口，那要先解决“窗口显示到哪里”的问题。

先明确一点：

- 笔记本通常不能直接当开发板的物理显示器
- 想把板子的实时预览显示到笔记本屏幕上，通常要走远程显示

当前最推荐的路线是：

- `X11` 转发到 Windows

如果你正处在这种场景，请先看这份专门说明，再继续下面的第 8 步：

- [`x11_windows_preview.md`](D:\rk3588\project\vision\x11_windows_preview.md)

如果 `X11` 暂时没打通，但你又急着先确认“摄像头到底有没有画面”，可以临时使用：

- `camera_http.py`

但要注意：

- 它只是浏览器里的临时网页预览
- 不是当前项目的正式低延迟主方案
- 出现明显延迟是正常现象，不要把它当成最终体验标准

### 8.1 先验证 OpenCV 预览脚本

进入 `vision/` 目录后执行：

```bash
python3 camera_test.py
```

成功标准：

- 能看到实时画面
- 窗口正常刷新
- 按 `ESC` 或 `q` 能正常退出

如果这一步失败，先不要跑 Pose。

### 8.2 再单独验证 MediaPipe 是否能导入

执行：

```bash
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

成功标准：

- 不报导入错误
- 能打印版本号

### 8.3 最后运行 Pose 演示脚本

执行：

```bash
python3 pose_mediapipe_demo.py
```

成功标准：

- 能看到实时画面
- 识别到人体时能画出骨架
- 终端会周期性打印左膝坐标和可见度
- 按 `ESC` 或 `q` 能正常退出

## 9. 常见问题排查

## 9.1 摄像头打不开

优先检查：

- `/dev/video21` 是否还存在
- 摄像头节点是不是变成了别的号
- 当前用户是否有访问视频设备的权限
- `MJPG` 和 `1280x720` 是否仍被设备支持

如果设备节点变了，优先修改脚本里的：

- `CAMERA_DEVICE`

## 9.2 OpenCV 能拍照，但不能实时预览

优先检查：

- 当前板子是否接了显示屏
- 图形环境是否正常
- OpenCV 是否带 GUI 支持
- 设备能否稳定连续输出视频流，而不只是单次抓图

如果是纯 SSH 无图形界面环境，`cv2.imshow()` 本身就可能跑不起来。

如果你是：

- 纯 SSH 连板子
- 但又想保留实时窗口预览

那优先处理方向不是改脚本，而是先把远程显示链路搭起来。

推荐顺序：

1. 先试 `X11` 转发到 Windows
2. 如果板子有桌面环境，再考虑 `VNC / 远程桌面`

对应文档：

- [`x11_windows_preview.md`](D:\rk3588\project\vision\x11_windows_preview.md)

如果只是想临时确认有没有画面，而不是追求低延迟窗口体验，也可以先用：

- `camera_http.py`

但要明确：

- 它只是 fallback
- 不是正式主路径
- 浏览器方案有明显延迟时，不优先怀疑摄像头链路

## 9.3 `mediapipe` 安装失败

如果错误类似：

```text
No matching distribution found for mediapipe
```

优先判断为：

- RK3588 `arm64` 平台没有合适的官方 wheel

这时处理方向应该是：

- 查可用 wheel
- 或改走本地构建 MediaPipe Python wheel

而不是先反复修改视觉脚本。

## 9.4 Pose 能运行，但特别卡

先不要立刻大改结构。

建议优先调整：

- 降低推理输入分辨率
- 减少显示窗口开销
- 确认板子 CPU 占用情况

当前最先该改的通常是：

- `FRAME_WIDTH`
- `FRAME_HEIGHT`

后续也可以进一步做“采集高分辨率、推理低分辨率”的分离优化。

## 9.5 没检测到人体

优先检查：

- 光线是否足够
- 人体是否完整进入画面
- 摄像头视角是否合适
- 运行过程中画面有没有明显卡顿或拖影

如果程序能正常出画面，但总是没有骨架，说明更可能是识别条件问题，不一定是脚本出错。

## 10. 当前这两个脚本的定位

`camera_test.py` 的定位是：

- 验证摄像头实时预览链路

`pose_mediapipe_demo.py` 的定位是：

- 验证 MediaPipe Pose 在板子上能不能实时工作

它们现在都还是“板端功能验证脚本”，还没有进入主程序联调阶段。

所以当前最合理的推进顺序就是：

1. 先把摄像头预览跑稳
2. 再把 Pose 跑通
3. 最后再接动作评估和 GPIO 反馈
