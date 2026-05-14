# `motor_controller.py` 使用说明

## 1. 这个模块是干什么的

`motor_controller.py` 是一个专门控制震动马达的 Python 模块。

它的职责很简单：

- 在没有硬件时，支持 `mock_mode=True` 做纯软件调试
- 在有硬件时，通过 `python-periphery` 控制 RK3588 的 GPIO 输出高低电平
- 提供三种震动模式：
  - `short_buzz()`：短震
  - `long_buzz()`：长震
  - `rapid_buzz()`：急促三连震
- 提供一个 `gpio_worker()` 线程函数，方便主程序通过队列发命令控制马达

这个模块的设计目标是：主程序不用关心 GPIO 细节，只需要发 `"short"`、`"long"`、`"rapid"` 这样的命令即可。

## 2. 代码结构总览

当前模块主要由下面几部分组成：

### 2.1 常量

- `GPIO_PIN = 107`
  - 当前占位 GPIO 编号
  - 对应 README 里的示例引脚 `GPIO3_B3`
  - 硬件到货后最可能需要修改的就是它

- `GPIO_CHIP_PATH = "/dev/gpiochip0"`
  - `python-periphery` 打开 GPIO 控制器时使用的设备路径
  - 通常在 Linux 开发板上会看到 `/dev/gpiochip0`、`/dev/gpiochip1` 之类的设备

- `SHORT_BUZZ_SECONDS = 0.2`
  - 短震持续时间，单位是秒
  - 0.2 秒 = 200 毫秒

- `LONG_BUZZ_SECONDS = 0.8`
  - 长震持续时间，单位是秒
  - 0.8 秒 = 800 毫秒

- `RAPID_PULSE_SECONDS = 0.1`
  - 急促震动模式里单次高电平或低电平的持续时间
  - 0.1 秒 = 100 毫秒

### 2.2 类 `MotorController`

这是核心控制类，负责：

- 初始化 GPIO
- 执行三种震动模式
- 在退出时释放 GPIO
- 用锁保证线程安全

### 2.3 函数 `gpio_worker(...)`

这是给主程序线程直接调用的工作函数。

主程序里可以专门开一个 GPIO 线程，让它一直阻塞等待队列消息。只要收到字符串命令，就调用 `MotorController` 的对应方法。

### 2.4 自测入口

文件末尾的：

```python
if __name__ == "__main__":
```

用于本地快速验证模块逻辑。当前写法会以 `mock_mode=True` 运行，因此即使没有硬件也可以执行。

## 3. 每个参数和成员都是什么意思

## 3.1 `MotorController(mock_mode: bool = False)`

这是控制器的构造函数。

参数说明：

- `mock_mode`
  - 类型：`bool`
  - 默认值：`False`
  - 含义：
    - `False`：尝试使用真实 GPIO 控制马达
    - `True`：不访问硬件，只打印日志

典型场景：

- 你在普通 PC 上开发、没有板子时：
  - 用 `MotorController(mock_mode=True)`
- 你在 ELF 2 开发板上、硬件已经接好时：
  - 用 `MotorController(mock_mode=False)`

注意：

- 即使你传了 `mock_mode=False`，如果系统里没有安装 `periphery`，或者 GPIO 打不开，代码也会自动退回 Mock 模式

## 3.2 `gpio_worker(motor_queue, stop_event, mock_mode=False)`

这是工作线程函数。

参数说明：

- `motor_queue`
  - 类型：`queue.Queue[str]`
  - 含义：主线程往里面塞命令字符串，GPIO 线程从里面拿命令执行
  - 合法命令有：
    - `"short"`
    - `"long"`
    - `"rapid"`
    - `"stop"`

- `stop_event`
  - 类型：`threading.Event`
  - 含义：用于通知线程退出
  - 当收到 `"stop"` 命令时，函数会调用 `stop_event.set()`

- `mock_mode`
  - 类型：`bool`
  - 含义：决定这个工作线程内部创建的 `MotorController` 是否跑在模拟模式

## 3.3 类内部成员变量

这些变量虽然是私有的，但你读代码时会看到，理解它们很重要。

- `self._lock`
  - 类型：`threading.Lock`
  - 作用：给 GPIO 操作加锁，防止多个线程同时改 GPIO 状态
  - 为什么要有它：
    - 比如一个线程正在长震，另一个线程又突然调用短震，如果不加锁，GPIO 高低电平可能会乱掉

- `self._mock_mode`
  - 类型：`bool`
  - 作用：记录当前控制器是不是模拟模式

- `self._gpio`
  - 类型：真实运行时是 `periphery.GPIO` 对象；未初始化或进入 Mock 时是 `None`
  - 作用：保存已经打开的 GPIO 句柄

- `self._closed`
  - 类型：`bool`
  - 作用：表示这个控制器是否已经执行过 `cleanup()`
  - 为什么要有它：
    - 防止资源已经释放后还继续操作 GPIO
    - 也能让 `cleanup()` 支持重复调用而不报错

## 3.4 `PeripheryGPIO` 是什么

代码里有这样一段：

```python
try:
    from periphery import GPIO as PeripheryGPIO
except ImportError:
    PeripheryGPIO = None
```

含义是：

- 如果系统安装了 `python-periphery`，就把它的 `GPIO` 类导入进来
- 如果没有安装，程序不会直接崩掉，而是把 `PeripheryGPIO` 记成 `None`
- 后续初始化时发现 `PeripheryGPIO is None`，就会打印警告并自动切到 Mock 模式

这样做的好处是：你在没有硬件、没有依赖的电脑上也能先把主程序逻辑跑起来。

## 4. 每个方法做什么

## 4.1 `short_buzz()`

作用：

- 触发一次短震
- GPIO 拉高 200 毫秒后，再拉低

业务含义：

- 可以表示“动作角度轻微不足”

内部调用关系：

- `short_buzz()`
- `_run_buzz("short_buzz", SHORT_BUZZ_SECONDS)`
- `_pulse(0.2)`
- `_write_gpio(True/False)`

## 4.2 `long_buzz()`

作用：

- 触发一次长震
- GPIO 拉高 800 毫秒后，再拉低

业务含义：

- 可以表示“动作完成，给予正向反馈”

## 4.3 `rapid_buzz()`

作用：

- 连续输出三次急促脉冲

实际时序：

```text
高 100ms -> 低 100ms
高 100ms -> 低 100ms
高 100ms -> 低 100ms
```

业务含义：

- 可以表示“动作速度过快”

注意：

- 这个方法没有走 `_run_buzz()`，因为它不是单次脉冲，而是 3 组高低电平组合
- 它也使用同一把锁，所以不会和其他 GPIO 操作并发冲突

## 4.4 `cleanup()`

作用：

- 在退出前把 GPIO 拉低
- 关闭 GPIO 句柄
- 释放控制器资源

为什么必须调用：

- 如果程序退出时不拉低 GPIO，可能会让马达保持在错误状态
- 如果不关闭句柄，资源释放就不完整

为什么重复调用也安全：

- 代码里用 `self._closed` 做了保护
- 第一次调用后，再次调用会直接返回，不会重复关闭

## 4.5 `_run_buzz(action_name, duration)`

这是私有辅助方法，用来处理“单次震动”的共用逻辑。

它做的事情是：

- 先加锁
- 检查控制器是否已关闭
- 如果是 Mock，则打印日志
- 如果是真实硬件，就执行一次 `_pulse(duration)`

## 4.6 `_pulse(duration)`

这是最底层的单次脉冲逻辑：

```text
GPIO 高电平
等待 duration 秒
GPIO 低电平
```

## 4.7 `_write_gpio(value)`

这是最底层的 GPIO 写入方法。

参数说明：

- `value`
  - 类型：`bool`
  - `True`：输出高电平
  - `False`：输出低电平

如果 `self._gpio is None`，它会抛出异常，说明 GPIO 并没有成功初始化。

## 5. `gpio_worker()` 是怎么工作的

这个函数适合跑在一个守护线程里。

工作流程如下：

1. 先创建一个 `MotorController`
2. 进入循环，持续检查 `stop_event`
3. 用 `motor_queue.get(timeout=0.1)` 阻塞取命令
4. 根据命令调用：
   - `"short"` -> `short_buzz()`
   - `"long"` -> `long_buzz()`
   - `"rapid"` -> `rapid_buzz()`
   - `"stop"` -> 设置 `stop_event` 并退出
5. 如果收到未知命令，就打印警告
6. 最后在 `finally` 里执行 `cleanup()`

这里 `timeout=0.1` 很重要。

它的作用是：

- 线程不会永远卡死在 `queue.get()`
- 即使当前队列里没有命令，线程也能每隔 0.1 秒醒来检查一次 `stop_event`

## 6. 硬件到货后你要改哪些地方

这一节最重要，真正上板子时主要看这里。

## 6.1 改 `GPIO_PIN`

当前代码里写的是：

```python
GPIO_PIN = 107
```

这是占位值，对应示例 `GPIO3_B3`。

你拿到实际接线后，要按官方公式换算：

```text
引脚编号 = n × 32 + (字母序号 - 1) × 8 + 数字
A=1, B=2, C=3, D=4
```

例如：

- `GPIO3_B3`
- `3 × 32 + (2 - 1) × 8 + 3`
- `96 + 8 + 3 = 107`

如果以后你实际接的是别的脚，比如 `GPIO4_C2`，就重新按公式算，然后把 `GPIO_PIN` 改掉。

## 6.2 确认 `GPIO_CHIP_PATH`

当前写的是：

```python
GPIO_CHIP_PATH = "/dev/gpiochip0"
```

大多数情况下这可以直接用，但上板后最好确认一下系统里实际暴露的是哪个 gpiochip。

你可以在开发板上看：

```bash
ls /dev/gpiochip*
```

如果不是 `gpiochip0`，就把常量改成真实设备路径。

## 6.3 安装依赖

在开发板上安装：

```bash
pip install python-periphery
```

如果你不用虚拟环境，也可能要用：

```bash
pip3 install python-periphery
```

如果没有安装这个库，代码会自动回退到 Mock 模式，因此你会看到：

```text
[MOTOR WARNING] 未安装 periphery，已自动回退到模拟模式。
```

## 6.4 检查接线

硬件链路应该是：

```text
RK3588 GPIO -> MOS 驱动模块 -> 外部 5V 电源 -> 震动马达
```

要点：

- GPIO 只负责输出 3.3V 逻辑信号，不直接给马达供电
- 马达电流走 MOS 模块和外部 5V 电源
- 开发板地线、MOS 模块地线、5V 电源地线必须共地
- GPIO 输入到 MOS 模块的控制端时，不能超过 3.3V

## 6.5 确认高低电平逻辑是不是反的

当前代码假设：

- `GPIO = True` 时，马达启动
- `GPIO = False` 时，马达停止

这叫“高电平有效”。

但某些驱动板可能是反相逻辑，也就是：

- 低电平启动
- 高电平停止

如果你发现代码运行了但马达行为正好反过来，就要改这里的逻辑：

- `_pulse()`
- `rapid_buzz()`
- `cleanup()`

本质上就是把 `True` / `False` 的含义翻转。

## 6.6 第一次联调建议

建议按这个顺序来：

1. 先确认接线和共地没问题
2. 先安装 `python-periphery`
3. 先只测试 `short_buzz()`
4. 确认短震正常后，再测试 `long_buzz()`
5. 最后测试 `rapid_buzz()`
6. 退出前确认 `cleanup()` 已执行

这样做比较安全，不容易一上来就让马达长时间误动作。

## 7. 这段代码该怎么用

下面给你三个最常见的使用方法。

## 7.1 用 Mock 模式在普通电脑上测试

适合场景：

- 没有板子
- 没有接硬件
- 只是想先验证主程序逻辑和调用关系

示例：

```python
from motor_controller import MotorController

motor = MotorController(mock_mode=True)
motor.short_buzz()
motor.long_buzz()
motor.rapid_buzz()
motor.cleanup()
```

你会看到类似输出：

```text
[MOTOR MOCK] 已启用模拟模式。
[MOTOR MOCK] short_buzz 已触发
[MOTOR MOCK] long_buzz 已触发
[MOTOR MOCK] rapid_buzz 已触发
[MOTOR MOCK] cleanup 已执行
```

## 7.2 在真实开发板上直接调用

适合场景：

- 已经接好 GPIO 和 MOS 模块
- 想做最小化硬件联调

示例：

```python
from motor_controller import MotorController

motor = MotorController(mock_mode=False)
motor.short_buzz()
motor.cleanup()
```

如果一切正常，马达会短震一下。

如果没有震动，优先检查：

- `python-periphery` 是否安装
- `GPIO_PIN` 是否正确
- `GPIO_CHIP_PATH` 是否正确
- 接线是否共地
- 驱动板是否高电平有效

## 7.3 在主程序里用工作线程 + 队列

这才是最接近你项目最终形态的用法。

示例：

```python
import queue
import threading

from motor_controller import gpio_worker

Motor_Queue: queue.Queue[str] = queue.Queue()
stop_event = threading.Event()

gpio_thread = threading.Thread(
    target=gpio_worker,
    args=(Motor_Queue, stop_event),
    kwargs={"mock_mode": True},
    daemon=True,
)
gpio_thread.start()

Motor_Queue.put("short")
Motor_Queue.put("long")
Motor_Queue.put("rapid")

# 程序结束时通知线程退出
Motor_Queue.put("stop")
gpio_thread.join()
```

这一套的意义是：

- 主程序其他线程只管发命令
- GPIO 操作统一放在一个线程里执行
- 结构清晰，也更安全

## 7.4 直接运行文件自测

当前文件末尾已经写好了：

```python
if __name__ == "__main__":
    motor = MotorController(mock_mode=True)
    motor.short_buzz()
    motor.long_buzz()
    motor.rapid_buzz()
    motor.cleanup()
```

所以你可以直接运行：

```bash
python motor_controller.py
```

它会自动用 Mock 模式跑一遍三种震动逻辑。

## 8. 常见问题排查

## 8.1 为什么会打印“未安装 periphery”

因为你当前环境没有安装 `python-periphery`。

解决：

```bash
pip install python-periphery
```

## 8.2 为什么我明明传了 `mock_mode=False`，还是进了 Mock

通常有两种原因：

- `python-periphery` 没装
- GPIO 初始化失败

GPIO 初始化失败可能来自：

- 权限不够
- `GPIO_CHIP_PATH` 不对
- `GPIO_PIN` 不对
- 当前机器不是目标开发板

## 8.3 为什么程序运行了，但马达不震动

常见原因：

- MOS 模块没接好
- 外部 5V 电源没接好
- 没有共地
- 真实引脚编号算错
- 驱动板是低电平有效，而代码按高电平有效写的

## 8.4 为什么还要 `stop_event`

因为线程退出不能只靠队列。

`stop_event` 的作用是：

- 作为统一的“停止信号”
- 让线程即使暂时没有新命令，也能周期性检查是否要退出

## 8.5 为什么 `queue.get()` 要加 `timeout=0.1`

因为如果不加超时，线程可能永久卡在等待队列上，无法及时结束。

加上 `timeout=0.1` 后：

- 没有命令时，它最多等 0.1 秒
- 醒来后就能检查 `stop_event`

## 9. 你以后最常改的地方

如果后面你继续做这个模块，最常改动的大概率只有这些：

1. `GPIO_PIN`
2. `GPIO_CHIP_PATH`
3. 三种震动时长常量
4. 高低电平有效逻辑
5. 主程序里往 `Motor_Queue` 发送命令的时机

除了这些，控制器结构本身一般不用大改。

## 10. 一句话总结

把它理解成一个“震动马达驱动器”就行：

- `MotorController` 负责底层 GPIO 控制
- `gpio_worker()` 负责在线程里接收命令
- 没硬件时用 Mock
- 有硬件后主要改 `GPIO_PIN`、检查 `GPIO_CHIP_PATH` 和接线
