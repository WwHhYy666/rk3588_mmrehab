# RK3588 实时预览显示到 Windows 笔记本的操作说明

这份文档专门解决下面这种情况：

- 板子通过 SSH 登录
- 板子没有本地接显示器
- 你又想保留 `cv2.imshow()` 的实时预览窗口

## 1. 先说清楚一个前提

你的笔记本通常**不能直接当开发板的物理显示器**。

原因是：

- 大多数笔记本的 HDMI / Type-C 口是视频输出
- 不是视频输入

所以你现在想实现的是：

- 把板子上的图形窗口，通过远程显示映射到 Windows 屏幕上

而不是：

- 把笔记本当普通外接屏幕直接插上就显示

## 2. 当前最推荐的路线

在你现在这个项目里，最推荐优先试的是：

- `X11` 转发到 Windows

因为它最贴近你现有脚本：

- `camera_test.py`
- `pose_mediapipe_demo.py`

都已经使用了 `cv2.imshow()`，所以优先不改代码，而是先把显示环境打通。

当前目录里虽然还有一个：

- `camera_http.py`

但它的定位只是：

- X11 暂时没打通时的临时网页预览

它不是正式的低延迟主方案。

## 3. 什么情况下适合用 X11 转发

满足下面这些条件时，优先用 X11：

- 你主要通过 SSH 连板子
- 你想实时看到 OpenCV 窗口
- 你不想先给板子装完整桌面环境

如果板子只是纯命令行系统，X11 通常比 VNC / 远程桌面更合适。

## 4. Windows 侧需要准备什么

因为你用的是 Windows，所以本机必须有一个 X Server。

简单理解：

- 板子负责运行 Python 脚本
- Windows 上的 X Server 负责接收并显示图形窗口

推荐两种路线：

### 4.1 一体化路线

用自带 X11 forwarding 能力的 SSH 客户端。

优点：

- 配置更集中
- 不容易漏步骤

### 4.2 组合路线

- 单独安装 Windows X Server
- 再用单独的 SSH 客户端连板子

优点：

- 更灵活

不管你走哪条路线，最终目标都一样：

- Windows 上先有 X Server 在跑
- SSH 登录时开启 X11 forwarding

## 5. 板子侧要满足什么条件

板子侧至少要满足这些前提：

- 已安装并运行 `ssh` 服务
- `sshd` 允许 `X11Forwarding`
- 系统里有 `xauth`
- 当前用户能正常 SSH 登录

你现在已经出现过这种错误：

```text
qt.qpa.xcb: could not connect to display
Could not load the Qt platform plugin "xcb"
```

这说明：

- OpenCV 的 GUI 依赖大概率已经装在板子上了
- 真正缺的是一个可用的 `DISPLAY`

所以当前方向是对的，不是优先怀疑摄像头。

## 6. 推荐的最短验证顺序

不要一上来就直接跑 `camera_test.py`。

推荐顺序固定如下。

### 第一步：在 Windows 启动 X Server

你先确保：

- Windows 上的 X Server 已经运行

这一步没完成，后面的 `DISPLAY` 基本不会正常。

### 第二步：用支持 X11 forwarding 的 SSH 客户端登录板子

登录时要明确开启：

- `X11 forwarding`

如果客户端里有“Enable X11 forwarding”一类选项，要勾上。

如果你是命令行 SSH 客户端，也要用对应的 X11 转发参数。

### 第三步：在板子上先看 `DISPLAY`

登录成功后，先执行：

```bash
echo $DISPLAY
```

成功标准：

- 不是空的

如果这里是空，说明：

- X11 forwarding 还没建立成功

这时候不要先跑 OpenCV。

### 第四步：先跑一个最小图形程序

先不要急着跑摄像头脚本，先验证最小图形链路。

如果板子系统里有简单的 X11 小程序，优先先跑一个最基础窗口测试。

这一层的目标是：

- 确认“板子图形窗口能不能映到 Windows”

只要这一步不通，OpenCV 窗口也不会通。

### 第五步：再运行 `camera_test.py`

确认 `DISPLAY` 没问题后，再进入 `vision/` 目录执行：

```bash
python3 camera_test.py
```

成功标准：

- Windows 上能看到实时摄像头画面
- 窗口能持续刷新
- 按 `ESC` 或 `q` 能退出

### 第六步：最后运行 `pose_mediapipe_demo.py`

执行：

```bash
python3 pose_mediapipe_demo.py
```

成功标准：

- Windows 上能看到实时画面
- 识别到人体时能画出骨架
- 终端会周期性打印左膝坐标和可见度

## 7. 如果失败，怎么分层判断

## 7.1 `echo $DISPLAY` 是空的

说明：

- X11 forwarding 没有建立成功

优先检查：

- Windows X Server 是否已启动
- SSH 客户端是否真的开启了 X11 forwarding
- 板子侧 `sshd` 是否允许 X11 转发

## 7.2 `DISPLAY` 有值，但最小图形程序不弹窗

说明：

- 转发链路本身还有问题

优先检查：

- Windows X Server 是否正在监听
- SSH 客户端和 X Server 是否配套
- 本地安全软件是否拦截

## 7.3 最小图形程序能弹，但 OpenCV 还是报 `xcb`

说明：

- X11 主链路通了
- 更可能是 OpenCV / Qt 相关 GUI 依赖问题

这时才开始优先看：

- Qt 插件
- `xcb` 相关依赖
- OpenCV GUI 安装形态

## 7.4 `camera_test.py` 能显示，`pose_mediapipe_demo.py` 不行

说明：

- 显示环境基本没问题
- 才开始考虑 MediaPipe 或脚本层问题

## 8. VNC / 远程桌面什么时候再考虑

如果 X11 转发一直不稳定，再考虑：

- `VNC`
- `xrdp`
- 其他远程桌面

但这条路有一个前提：

- 板子本身要有图形桌面环境

如果板子现在只是纯命令行系统，那它不是第一选择。

## 9. 当前项目里最实用的建议

在你现在这个阶段，最合理的顺序是：

1. 先把 Windows 上的 X11 转发跑通
2. 先让 `camera_test.py` 的窗口能出来
3. 再跑 `pose_mediapipe_demo.py`

不要一开始就切到 VNC，也不要先重构 Python 脚本。

如果你只是临时想看一眼画面、又不在意明显延迟，可以短暂退回：

- `camera_http.py`

但它只适合：

- 快速确认画面是否出来

不适合：

- 低延迟操作预览
- 作为最终演示主界面

因为你当前的核心问题不是摄像头，也不是 MediaPipe，而是：

- 图形窗口没有被正确显示到 Windows
