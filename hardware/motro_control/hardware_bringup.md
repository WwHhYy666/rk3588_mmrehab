# ELF 2 震动马达硬件联调清单

这份文档用于硬件到货后的第一次上板联调。目标是让硬件新手也能按步骤完成接线，先确认 GPIO 电平正常，再确认 `MOS-1` 驱动链路正常，最后运行 Python 代码驱动圆片震动马达。

本次固定采用下面这条主方案：

```text
ELF2 GPIO -> MOS-1 模块 -> 圆片震动马达
5V 电源  -> HW-832 面包板电源模块 -> 面包板电源轨
```

本次不把继电器作为主方案。

## 1. 当前代码要不要改

本次已经实测使用手册示例引脚 `GPIO3_B3`，但这里要区分两套编号：

- sysfs 手动测试继续使用全局编号 `GPIO107`
- `python-periphery` 代码使用 `/dev/gpiochip3` 里的 line `11`

当前 `motor_controller.py` 里已经是：

```python
GPIO_PIN = 11
GPIO_CHIP_PATH = "/dev/gpiochip3"
```

手册中 `GPIO3_B3` 的 sysfs 全局编号计算为：

```text
GPIO3_B3 = 3 * 32 + (2 - 1) * 8 + 3 = 107
```

其中 `(2 - 1) * 8 + 3 = 11` 是 `GPIO3_B3` 在 `gpiochip3` 内部的 line offset，`python-periphery` 要用这个值。

如果你后面换了别的 GPIO 引脚，需要同时重新确认：

```text
sysfs 全局编号 = n * 32 + (字母序号 - 1) * 8 + 数字
periphery line offset = (字母序号 - 1) * 8 + 数字
A=1, B=2, C=3, D=4
```

## 2. 先认识你手上的器材

### 图 1：面包板

- 两侧带红蓝线的长条区域是电源轨
- 红线这一轨当作 `5V`
- 蓝线这一轨当作 `GND`
- 中间白色大区域暂时不用，第一次联调只把它当电源分配板

注意：

- 这块面包板左右两侧电源轨中间是断开的，不是整条全通
- 第一次联调先只用上半段电源轨，避免被中间断点坑到
- 如果以后想把上下全长都供电，需要自己再用跳线桥接

### 图 2：杜邦线

你这包线里有带金属针的一端，也有母头的一端。第一次联调按下面规则选：

- 接开发板 `P26` 排针的一端，必须用母头
- 接面包板孔位的一端，要用公针
- 接螺丝端子模块时，优先把带金属针的一端压进端子
- 如果某根线两端都不合适，可以临时剥一小段线皮再压进螺丝端子，但这只是补充方案

简单记忆：

- 开发板排针 -> 用母头
- 面包板孔位 -> 用公针
- 螺丝端子 -> 用公针或裸导体

### 图 3：HW-832 面包板电源模块

这个模块的作用不是控制马达，而是给面包板提供稳定的 `5V` 电源。

你这块板上能看到：

- `Micro-USB` 输入接口
- `3.3V / 5V` 拨档
- `OFF / ON` 电源开关

本次固定这样用：

- `Micro-USB` 接外部 `USB 5V`
- 电压拨到 `5V`
- 接线完成前先保持 `OFF`
- 最后一步再拨到 `ON`

### 图 4：圆片震动马达

你这批马达默认按下面理解：

- 红线：正极
- 蓝线：负极

本次接法固定为：

- 马达红线接外部 `5V`
- 马达蓝线接 `MOS-1` 的负载输出端

### 图 5：继电器模块

这块继电器板这次先不要用。

原因不是它完全不能用，而是它不适合这次场景：

- 继电器更适合低频开关
- 震动马达后面要做短震、长震、连震，更适合用 MOS 控制
- 继电器有机械触点，动作慢，也有噪声

### 图 6：MOS-1 模块

这块才是本次唯一推荐的驱动模块。

因为图片里的丝印不够清楚，这份文档不把端子名字写死，而是按功能说明：

- 控制输入端：接开发板 GPIO
- 控制地端：接开发板 GND
- 负载电源正端：接外部 `5V`
- 负载电源负端：接外部 `GND`
- 负载输出端：接马达负极

## 3. P26 排针推荐接线

第一次推荐使用 `GPIO3_B3`。

关键引脚：

- `P26-7`：`GPIO3_B3`，sysfs 手动测试对应 `GPIO107`，Python 代码对应 `/dev/gpiochip3` line `11`
- `P26-6`：`GND`
- `P26-2` / `P26-4`：`5V`
- `P26-1` / `P26-17`：`3.3V`

这里要区分两个编号：

- `GPIO3_B3 = 107` 是 sysfs 手动测试时使用的全局 GPIO 编号
- `/dev/gpiochip3` line `11` 是 `python-periphery` 代码里使用的编号
- `P26-7` 是排针上的物理脚位

注意：

- GPIO 电平范围只有 `0~3.3V`
- 不能把 `5V` 接到 GPIO 引脚
- GPIO 不能直接带马达，必须经过 `MOS-1` 驱动模块

## 4. 这次 5V 从哪里来

第一次联调，推荐直接用外部 `USB 5V`，不要直接拿开发板排针的 `5V` 给马达供电。

推荐来源：

- 手机充电头
- 充电宝
- 稳定的 USB 5V 电源口

推荐接法：

```text
USB 5V -> HW-832 的 Micro-USB -> 面包板 5V/GND 电源轨
```

为什么这样更稳：

- GPIO 只负责出 `3.3V` 控制信号
- 马达电流不经过 GPIO
- 面包板电源轨更适合分发 `5V` 和 `GND`
- 新手更不容易把 `5V` 误接到 GPIO

## 5. 接线总原则

在真正开始接线前，先记住这 5 条：

- `5V` 绝不能直接接到 GPIO 引脚
- GPIO 只负责控制 `MOS-1`，不直接给马达供电
- 马达供电来自外部 `USB 5V`
- 开发板 `GND`、外部 `5V` 负极、`MOS-1` 地线必须共地
- 接线全部完成后再上电，不要边通电边乱插

## 6. 最简接线图

按功能理解，这次的最简接线图就是：

```text
USB 5V -> HW-832 -> 面包板 5V/GND 电源轨

ELF2 P26-7(GPIO3_B3) -> MOS-1 控制输入
ELF2 GND             -> MOS-1 控制地
ELF2 GND             -> 面包板 GND 蓝轨

面包板 5V 红轨       -> MOS-1 负载电源正端
面包板 GND 蓝轨      -> MOS-1 负载电源负端

马达红线             -> 面包板 5V 红轨
马达蓝线             -> MOS-1 负载输出端
```

如果你要一句话理解整个电流路径，就是：

```text
GPIO 只管“发指令”
USB 5V 只管“给马达供电”
MOS-1 负责“按 GPIO 指令接通或断开马达负极”
```

## 7. 可以照抄的接线顺序

按下面顺序做，不要跳步：

1. 把 `HW-832` 插到面包板一端电源轨位置，电压拨到 `5V`，开关先保持 `OFF`
2. 用 `USB` 线把外部 `5V` 接到 `HW-832` 的 `Micro-USB`
3. 用母头接开发板 `P26-7(GPIO3_B3)`，另一端接到 `MOS-1` 的控制输入端
4. 用母头接开发板任意 `GND`，另一端接到 `MOS-1` 的控制地端
5. 再从开发板 `GND` 或 `MOS-1` 地端，拉一根线到面包板蓝色 `GND` 电源轨，形成共地
6. 从面包板红色 `5V` 电源轨，拉一根线到 `MOS-1` 的负载电源正端
7. 从面包板蓝色 `GND` 电源轨，拉一根线到 `MOS-1` 的负载电源负端
8. 把马达红线接到面包板红色 `5V` 电源轨
9. 把马达蓝线接到 `MOS-1` 的负载输出端
10. 全部检查一遍，没有短接后，最后再把 `HW-832` 拨到 `ON`

## 8. 推荐测试顺序

### 第一步：断开马达，只测 GPIO 电平

这一步你已经做过了，但文档继续保留，后面复查时还可以直接照用。

先不要接马达，只用万用表测 `P26-7` 对 `GND` 的电压。

建议先进入 root shell：

```bash
sudo -i
```

然后在板子上执行：

```bash
echo 107 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio107/direction
echo 1 > /sys/class/gpio/gpio107/value
echo 0 > /sys/class/gpio/gpio107/value
echo 107 > /sys/class/gpio/unexport
```

预期：

- 写 `1` 时，`P26-7` 对 `GND` 接近 `3.3V`
- 写 `0` 时，`P26-7` 对 `GND` 接近 `0V`

如果不进入 root shell，不要写 `sudo echo 107 > /sys/class/gpio/export`，因为 `>` 重定向仍然会由普通用户执行。可以改用：

```bash
echo 107 | sudo tee /sys/class/gpio/export
echo out | sudo tee /sys/class/gpio/gpio107/direction
echo 1 | sudo tee /sys/class/gpio/gpio107/value
echo 0 | sudo tee /sys/class/gpio/gpio107/value
echo 107 | sudo tee /sys/class/gpio/unexport
```

### 第二步：只接 MOS 控制侧，不接外部 5V，不接马达

这一步的目标不是让马达震，而是先确认 GPIO 能不能让 `MOS-1` 的控制侧状态发生变化。

此时只接：

```text
P26-7 -> MOS-1 控制输入
GND   -> MOS-1 控制地
```

此时先不要接：

- `HW-832`
- 外部 `USB 5V`
- 圆片震动马达

再次执行上面的 `echo 1` / `echo 0`，观察 `MOS-1` 模块是否有状态变化。如果模块上有指示灯，就看指示灯；如果没有，就先不急着上马达。

### 第三步：接 HW-832、外部 5V 和马达，用手册命令短测

先按“可以照抄的接线顺序”把：

- `HW-832`
- 面包板电源轨
- `MOS-1`
- 圆片震动马达

全部接好，再把 `HW-832` 拨到 `ON`。

然后短时间执行：

```bash
echo 107 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio107/direction
echo 1 > /sys/class/gpio/gpio107/value
echo 0 > /sys/class/gpio/gpio107/value
echo 107 > /sys/class/gpio/unexport
```

预期：

- `echo 1` 时马达启动
- `echo 0` 时马达停止

如果刚好相反，说明 `MOS-1` 模块可能是低电平有效，后续需要把代码里的高低电平逻辑反过来。

### 第四步：确认 Python 依赖

你当前已经把 `python-periphery` 安装到了普通用户 `elf` 的目录：

```bash
/home/elf/.local/lib/python3.10/site-packages
```

普通用户可以这样确认：

```bash
python3 -m pip show python-periphery
python3 -c "from periphery import GPIO; print('periphery ok')"
```

因为真实 GPIO 通常需要 root 权限，如果用 `sudo` 运行测试脚本，就让 root 临时使用普通用户已经安装好的包：

```bash
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 -c "from periphery import GPIO; print('periphery ok')"
```

如果板子网络正常，也可以给 root 单独安装一份：

```bash
sudo python3 -m pip install python-periphery
```

如果出现 `Temporary failure in name resolution`，说明板子当前访问不了 pip 网络源，先用上面的 `PYTHONPATH` 方式即可。

### 第五步：确认 GPIO 没被 sysfs 占用

运行 Python 前，建议先执行：

```bash
echo 107 > /sys/class/gpio/unexport
```

如果提示不存在或无效，通常没关系，说明它可能本来就没有被导出。

### 第六步：运行真实硬件测试脚本

先只测短震：

```bash
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 hardware/motro_control/motor_hardware_test.py --real short
```

短震正常后，再测：

```bash
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 hardware/motro_control/motor_hardware_test.py --real long
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 hardware/motro_control/motor_hardware_test.py --real interval
```

最后再测完整序列：

```bash
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 hardware/motro_control/motor_hardware_test.py --real all
```

`rapid` 仍然保留为旧命令兼容写法，和 `interval` 做同样的三次间隔震动。

## 9. 第一次上电前的 4 个提醒

- `5V` 绝不能直接接到 GPIO
- 第一次接马达前，先完成 GPIO 裸测
- 上电顺序放最后，先接线后通电
- 如果一上电马达就一直震，先立刻关掉 `HW-832`，再检查 `MOS-1` 是否低电平有效，或马达负极是不是接错端子

## 10. 写 0 仍然是 3.3V 怎么办

如果 `echo 1` 和 `echo 0` 后万用表都量到 `3.3V`，先不要急着改 Python 代码。优先确认 GPIO 是否真的被系统控制到了。

### 10.1 先断开外设，只裸测 GPIO

测试时先不要接：

- `MOS-1`
- 圆片震动马达
- `HW-832`
- 外部 `USB 5V`

只保留：

- ELF 2 开发板
- 万用表
- 黑表笔接 `GND`
- 红表笔接 `P26-7`

### 10.2 确认 export 成功

如果你看到：

```text
/sys/class/gpio/export: Permission denied
```

说明 GPIO 根本没有导出成功，后面的 `gpio107/direction` 和 `gpio107/value` 都不会存在。

正确做法是先进入 root shell：

```bash
sudo -i
```

再执行：

```bash
echo 107 > /sys/class/gpio/unexport
echo 107 > /sys/class/gpio/export
ls /sys/class/gpio/gpio107
```

如果 `unexport` 提示不存在，可以忽略。

### 10.3 每一步都用 cat 回读

导出成功后，按下面顺序测：

```bash
cat /sys/class/gpio/gpio107/direction
echo out > /sys/class/gpio/gpio107/direction
cat /sys/class/gpio/gpio107/direction

echo 1 > /sys/class/gpio/gpio107/value
cat /sys/class/gpio/gpio107/value

echo 0 > /sys/class/gpio/gpio107/value
cat /sys/class/gpio/gpio107/value
```

判断方法：

- 如果 `cat value` 能从 `1` 变成 `0`，但万用表还是 `3.3V`，优先怀疑测错物理脚、引脚复用没有切出来，或者外部电路把该脚拉高
- 如果 `cat value` 写 `0` 后仍然是 `1`，优先怀疑权限不足、GPIO 被占用，或 sysfs 控制方式不适配当前系统

### 10.4 用板厂脚本交叉验证

手册里给了官方脚本，可以用它交叉验证 `GPIO3_B3`：

```bash
cmddemo_gpio.sh GPIO3_B3 1
cmddemo_gpio.sh GPIO3_B3 0
```

如果官方脚本能把 `P26-7` 拉低，而 sysfs 命令不行，说明脚本内部可能做了额外的 pinmux 或板级配置。

### 10.5 换一个 GPIO 做对照

如果 `GPIO3_B3` 一直无法拉低，可以换一个普通 GPIO 做对照测试。候选脚：

```text
GPIO3_A1 -> 编号 97，对应 P26-29
GPIO3_A5 -> 编号 101，对应 P26-35
GPIO3_B1 -> 编号 105，对应 P26-37
```

例如测试 `GPIO3_B1`：

```bash
echo 105 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio105/direction
echo 1 > /sys/class/gpio/gpio105/value
cat /sys/class/gpio/gpio105/value
echo 0 > /sys/class/gpio/gpio105/value
cat /sys/class/gpio/gpio105/value
echo 105 > /sys/class/gpio/unexport
```

如果换脚后可以正常在 `0V` 和 `3.3V` 之间切换，说明 `GPIO3_B3` 这根脚可能被复用或不适合当前系统直接用。后续再把 `motor_controller.py` 里的 `GPIO_PIN` 改成新编号，并同步修改接线文档。

## 11. 常见问题

### 运行后仍然显示 Mock

可能原因：

- 没安装 `python-periphery`
- GPIO 初始化失败
- `GPIO_CHIP_PATH` 或 `GPIO_PIN` 配置不对
- 权限不足

排查：

```bash
ls /dev/gpiochip*
python3 -m pip show python-periphery
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 -c "from periphery import GPIO; print('periphery ok')"
```

当前已验证 `GPIO3_B3` 的 Python 配置是：

```python
GPIO_CHIP_PATH = "/dev/gpiochip3"
GPIO_PIN = 11
```

如果权限不足，先用：

```bash
sudo PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages python3 hardware/motro_control/motor_hardware_test.py --real short
```

### 马达不震

优先检查：

- 是否共地
- `HW-832` 是否已拨到 `5V` 且已打开 `ON`
- 外部 `USB 5V` 是否真的送到了面包板红蓝电源轨
- `MOS-1` 控制输入端是否接到 `P26-7`
- 马达红线是否接到 `5V`
- 马达蓝线是否接在 `MOS-1` 负载输出端
- sysfs 手动测试编号是否仍是 `107`
- Python 代码是否仍是 `/dev/gpiochip3` line `11`
- `MOS-1` 模块是否低电平有效

### MOS 有变化但马达不动

优先检查：

- 面包板电源轨是不是接在了断开的下半段
- `HW-832` 是否真的输出了 `5V`
- 马达正负极有没有接反或压紧不牢
- `MOS-1` 的负载电源端和负载输出端有没有混接

### 马达一直震或停止不了

立即关掉 `HW-832`，然后检查：

- `MOS-1` 模块是否低电平有效
- `cleanup()` 是否执行
- `GPIO_PIN` 是否接错
- `MOS-1` 控制输入端是否悬空
- 马达蓝线是不是误接到了固定 `GND`

## 12. 结论

如果你使用 `GPIO3_B3`，第一次联调主要不是改代码，而是按顺序确认：

```text
GPIO 电平正常 -> MOS 控制正常 -> 外部 5V 供电正常 -> 马达通断正常 -> Python 能打开 GPIO -> 三种震动模式正常
```
