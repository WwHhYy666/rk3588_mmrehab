# RK3588 骨科居家数字康复终端项目总说明

这份 `readme.md` 是当前项目的总入口文档。

它的用途不是只做展示，而是给下面两类人快速接手：

- 新开的 Codex 对话
- 第一次接手这个项目的新同学

如果你后面在 `vision/`、`hardware/`、`prescription/` 或其他新文件夹里单独开对话，建议先把这个文件发给 Codex 或先让它阅读这个文件，再继续具体任务。

## 1. 项目是做什么的

本项目的目标是做一套：

**基于 RK3588 的骨科居家数字康复终端**

核心思路是把下面这条闭环跑通：

```text
视觉感知 -> 动作评估 -> 数字处方录入/读取 -> 文本转语音反馈 -> 物理反馈
```

更具体一点，就是：

- 用摄像头采集人体动作
- 用姿态识别提取骨骼关键点
- 根据康复动作标准判断动作是否达标
- 如果动作不规范，就通过语音和震动提醒患者

这个项目的整体方向来自 `嵌赛开发流程_v2.pdf`，当前已经明确的五大功能模块是：

- 视觉感知层：摄像头 + MediaPipe 提取骨骼关键点
- 数字处方层：保存患者的动作模板
- 临床评估层：DTW、ROM、TUT 等动作评估逻辑
- 智能反馈层：端侧大模型 + TTS 语音反馈
- 物理反馈层：GPIO 驱动震动马达

## 2. 当前目录结构

当前项目根目录是：

`D:\rk3588\project`

目前已经看到的结构如下：

```text
project/
├── hardware/
│   └── motro_control/
│       ├── motor_controller.py
│       ├── explain.md
│       └── readme.md
├── prescription/
├── feedback/
│   └── tts/
├── vision/
│   ├── camera_test.py
│   ├── pose_mediapipe_demo.py
│   ├── camera_http.py
│   ├── pose_http.py
│   ├── guide.md
│   ├── explain.md
│   ├── x11_windows_preview.md
│   └── apt_update_troubleshooting.md
└── readme.md
```

说明：

- `readme.md`
  - 当前这个文件
  - 是全项目总入口
- `hardware/motro_control/`
  - 目前是硬件马达控制模块的主要工作目录
- `vision/`
  - 目前是摄像头、骨骼识别和板端预览验证目录
- `prescription/`
  - 当前已完成一阶段主链路的处方模块目录
  - 已具备 RK3588 浏览器版录制、板端本地保存、Windows 同步保存和结果读取
  - 当前结果默认保存在 `prescription/docs/results/`，中文摘要保存在 `prescription/docs/summaries/`
- `feedback/tts/`
  - 当前板端文本转语音模块目录

注意：

- 当前目录名写的是 `motro_control`，不是 `motor_control`
- 这是现有事实，后续如果你想统一命名，可以再单独整理

## 3. 当前开发进度总览

为了避免新对话误判，下面统一使用这几种状态：

- 已完成
- 已验证 Demo
- 待联调
- 待硬件
- 待建立

### 3.1 hardware 模块现状

状态总结：

- `motor_controller.py`：已完成（软件逻辑）
- Mock 模式本地运行：已验证 Demo
- 真实硬件 GPIO 联调：待硬件

目前已完成内容：

- 已实现 `MotorController` 类
- 已实现三种震动模式：
  - `short_buzz()`
  - `long_buzz()`
  - `rapid_buzz()`
- 已实现 `gpio_worker()` 队列工作线程函数
- 已支持：
  - `mock_mode=True` 模拟模式
  - `mock_mode=False` 真实 GPIO 模式
- 已实现 `cleanup()` 资源清理
- 已加 `threading.Lock` 保护 GPIO 操作
- 已写模块说明文档 `explain.md`

当前已经验证过的事情：

- 在普通电脑上直接运行 `motor_controller.py`，Mock 模式可以正常打印：
  - `short_buzz`
  - `long_buzz`
  - `rapid_buzz`
  - `cleanup`
- 程序可以正常退出，软件流程没有报错

当前还没验证的事情：

- 真实开发板能否成功打开 `/dev/gpiochip0`
- 真实接线后是否能正确驱动 MOS 模块和马达
- 当前高电平有效假设是否与真实驱动板一致

### 3.2 vision 模块现状

状态总结：

- `camera_test.py`：代码已完成，窗口预览路径待补验证
- `pose_mediapipe_demo.py`：代码已完成，窗口预览路径待补验证
- `camera_http.py`：已验证 Demo
- `pose_http.py`：已验证 Demo
- 与主程序联动：待联调

目前已完成内容：

- `camera_test.py`
  - 用 OpenCV 打开摄像头
  - 代码层面已支持显示实时画面
  - 代码层面已支持按 `ESC` 或 `q` 退出
  - 但在当前“纯 SSH + 笔记本不能直接当板端显示器”的环境下，这条本地窗口路径还没有真正完成验证
- `pose_mediapipe_demo.py`
  - 用 MediaPipe Pose 识别人体骨架
  - 绘制姿态骨架
  - 打印左膝关键点坐标与可见度
  - 当前同样受限于本地窗口显示路径，尚未完成最终的窗口预览验证
- `camera_http.py`
  - 提供浏览器版摄像头实时预览
  - 用于在 X11 暂时没打通时快速确认画面
- `pose_http.py`
  - 提供浏览器版骨骼识别实时预览
  - 可在浏览器中查看骨架识别结果

当前说明：

- 视觉链路已经能在板端独立跑起来
- 板端摄像头链路已经能工作
- 板端骨骼识别已经能工作
- 当前主要通过浏览器方案进行实时查看
- 当前“本地窗口 / X11 弹窗预览”仍未完全打通，因此不能把 `camera_test.py` 和 `pose_mediapipe_demo.py` 的窗口显示路径算作最终已验证
- X11 / 本地窗口仍然是可选优化方向，不再是当前必须先解决的主阻塞项
- 这些脚本当前仍属于“功能验证脚本”，还没有和主线程架构、动作评估逻辑、硬件反馈逻辑做正式集成

截至当前阶段，同学 B 第 1 周的骨骼识别任务仍可以视为已完成，但这个“完成”主要是指浏览器方案已经够用，不是指本地窗口预览路径已经完全打通。

### 3.3 处方模块现状

状态总结：

- `prescription/`：已完成当前阶段核心实现
- 录入、读取、Windows 同步保存、板端本地保存：已完成
- 浏览器录制 + 最新结果读取：已验证可用

当前更细的状态判断如下：

- `prescription/`
  - 已完成当前阶段核心实现
  - 已完成处方录入、读取、Windows 同步保存和板端本地保存
  - 已支持板端采集、推理、浏览器录制、结果读取与双端保存
  - 当前输出分为模板 JSON 和中文摘要
  - 后续进入结果质量优化、动作扩展或与主程序联调阶段

重点说明：

- `prescription/` 不是“未来可能有”的模块
- 它已经完成了当前阶段浏览器录制与双端保存主链路
- 当前保存方式是：板端本地先落盘，同时浏览器保存流程继续同步到 Windows 本机
- 板端页面入口：`http://板子IP:8082`
- 本机结果接收器：`http://127.0.0.1:8090`
- 模板 JSON 在 `prescription/docs/results/`
- 中文摘要在 `prescription/docs/summaries/`

### 3.4 TTS 模块现状

状态总结：

- `feedback/tts/`：已完成并可用于板端播报
- `tts_board.py`：已可读取板端最新摘要并播报
- `pyttsx3` 优先链路和兜底播报链路：已跑通

当前说明：

- `feedback/tts/banzi/tts_board.py` 负责从板端最新处方摘要里取内容并播报
- 当前播报链路已能工作
- 后续还可以继续优化为更自然的播报语气、数字读法和模型整合

### 3.5 其他模块现状

从项目总目标来看，后续还应有这些模块：

- `evaluate/`
- `llm/`
- `main.py` 或等价的主程序集成入口

当前更细的状态判断如下：

- `evaluate/`
  - 待建立
- `llm/`
  - 待建立
- `main.py`
  - 待建立或待联调

## 4. 关键文件索引

如果你是新开对话，最推荐先读这些文件。

### 4.1 总入口

- `D:\rk3588\project\readme.md`
  - 全项目总说明

### 4.2 硬件模块

- `D:\rk3588\project\hardware\motro_control\motor_controller.py`
  - 震动马达控制源码
- `D:\rk3588\project\hardware\motro_control\explain.md`
  - 硬件模块详细说明
  - 解释了参数、方法、Mock 模式、硬件到货后要改哪些地方
- `D:\rk3588\project\hardware\motro_control\readme.md`
  - 硬件模块最初背景说明
  - 内容较简略，且有部分旧状态描述

### 4.3 视觉模块

- `D:\rk3588\project\vision\camera_test.py`
  - 摄像头基础验证脚本
- `D:\rk3588\project\vision\pose_mediapipe_demo.py`
  - MediaPipe Pose 骨架识别验证脚本
- `D:\rk3588\project\vision\camera_http.py`
  - 浏览器版摄像头预览脚本
- `D:\rk3588\project\vision\pose_http.py`
  - 浏览器版骨骼识别预览脚本
- `D:\rk3588\project\vision\guide.md`
  - 视觉模块上板操作指南
- `D:\rk3588\project\vision\explain.md`
  - 视觉模块代码说明
- `D:\rk3588\project\vision\x11_windows_preview.md`
  - 把板端实时预览映射到 Windows 的说明文档

### 4.4 当前主任务目录

- `D:\rk3588\project\prescription\`
  - 当前已完成一阶段主链路的处方模块目录
- `D:\rk3588\project\prescription\record_prescription_http.py`
  - RK3588 板端浏览器录制入口
- `D:\rk3588\project\prescription\local_result_sink.py`
  - Windows 本机结果保存入口
- `D:\rk3588\project\prescription\read_prescription_json.py`
  - 最新结果读取入口
- `D:\rk3588\project\prescription\docs\rk3588_prescription_guide.md`
  - 处方模块操作文档入口

### 4.5 TTS 模块

- `D:\rk3588\project\feedback\tts\banzi\tts_board.py`
  - 板端文本转语音播报入口
- `D:\rk3588\project\feedback\tts\banzi\explain.md`
  - TTS 模块说明文档

## 5. 团队分工

根据当前 PDF 中的项目规划，三人分工如下：

### 同学 A

定位：

- 核心架构与大模型部署

主责方向：

- 本地 Qwen 模型部署
- 多线程架构
- DTW / 角度计算核心函数

### 同学 B

定位：

- 视觉数据与交互逻辑

主责方向：

- MediaPipe 骨骼点提取
- JSON 数字处方读写
- Prompt 模板
- OpenCV 交互提示
- TTS 语音合成

当前焦点：

- 已完成：
  - MediaPipe / 骨骼识别基础验证
  - 板端浏览器实时预览链路
- 当前主任务：
  - JSON 处方读写和结果输出整理
- 紧接下一任务：
  - `pyttsx3` 文本转语音和更自然播报优化

### 同学 C

定位：

- 硬件驱动与实物总装

主责方向：

- RK3588 环境准备
- GPIO + MOS 驱动马达
- 护具缝制与走线
- 实物装配与联调

当前你已经完成的内容，主要落在：

- 同学 C 的硬件驱动方向
- 同时 `vision/` 目录也已经完成了当前阶段够用的骨骼识别验证

## 6. 8 周计划摘要

下面是从 PDF 浓缩出来的项目推进主线，便于新对话理解当前阶段。

### 第 1 周

目标：

- 基础设施点亮

典型任务：

- 跑通 MediaPipe 示例
- 板子装系统
- 阅读 GPIO 例程

### 第 2 周

目标：

- 核心模块跑通

典型任务：

- DTW 脚本验证
- JSON 处方读写
- TTS 初步验证
- 用 `periphery` 跑通震动马达三种模式

### 第 3 周

目标：

- 业务逻辑成型

典型任务：

- 角度计算 API
- 多线程骨架
- 视觉引导 UI
- 把代码往板子上迁移

### 第 4 周

目标：

- 智能助教发声

典型任务：

- LLM 输出康复指导
- TTS 接队列
- 实物护具与马达固定

### 第 5 周

目标：

- 软硬大缝合

典型任务：

- 主线程接管视觉流和硬件流
- 真实摄像头测试
- 马达反馈延迟测试

### 第 6 周

目标：

- 业务逻辑联调

典型任务：

- 线程锁与竞态处理
- 反馈冷却机制
- 故意错误动作联调
- 处理物理接触问题

### 第 7 周

目标：

- 系统优化排雷

典型任务：

- 控帧率和 CPU 占用
- UI 和 Prompt 微调
- 训练报告页
- 外壳和展示包装

### 第 8 周

目标：

- 冲刺答辩

典型任务：

- 停止大改代码
- 写文档与 PPT
- 录 Demo
- 准备实物演示

## 7. 当前阶段判断

结合现有代码，当前项目可以做两层判断。

### 7.1 从项目整体看

- 仍处在第 2 周附近
- 更准确一点说，是第 2 周向第 3 周过渡的阶段

原因：

- 硬件马达控制的软件逻辑已经写出来了
- 板端摄像头和骨骼识别的独立验证已经够用
- 处方录入 / 读取 / 保存链路已经完成
- TTS 也已经完成
- 但大模块还没有被主程序整合到同一个多线程框架里

所以当前最适合的整体状态判断是：

- 基础 demo 已经点亮
- 处方和 TTS 主链路已完成
- 集成主框架尚未开始或尚未完成

### 7.2 从同学 B 的任务视角看

- 第 1 周的骨骼识别目标已经完成
- 当前进入第 2 周的数字处方与 TTS 阶段，并且这部分已经跑通

## 8. 当前主任务

如果你现在要继续推进项目，当前最合理的顺序是：

1. `vision` 当前够用，不再作为主阻塞项
2. `prescription/` 当前一阶段主链路已打通，先继续做结果质量优化或动作扩展
3. 建立 `evaluate/` 模块并开始评估逻辑接入
4. 然后推进更自然的语音与主程序框架联调

也就是说：

- 现在新开 `prescription/` 对话，不需要重做视觉验证
- 也不需要再从零开始做 JSON 处方读写
- 应该直接在现有浏览器录制 + 双端保存链路上继续优化和扩展

## 9. 当前阶段后的建议

### 9.1 先完成当前主任务

顺序建议：

1. 在 `prescription/` 目录里继续优化结果质量、动作扩展和摘要输出
2. 建立 `evaluate/` 模块，明确评估指标和接口
3. 再推进更自然的 TTS 和模型整合

### 9.2 再接入主线程

后续主程序应该做的事情：

- 建立主线程 + 多个工作线程
- 让视觉线程输出判断结果
- 让 GPIO 线程消费 `Motor_Queue`
- 让数字处方模块提供动作模板数据
- 让 TTS 模块提供语音反馈

### 9.3 最后做软硬联调

最终要联起来的是：

```text
摄像头识别 -> 动作判断 -> 读取处方 -> 生成语音/震动反馈 -> GPIO / TTS 输出
```
