# motor_controller 模块说明

## 项目背景
骨科居家康复终端，ELF 2 开发板（RK3588），Ubuntu Desktop 系统。
我是同学C，负责硬件驱动模块。

## 我的任务（第二周）
用 periphery 库控制 GPIO → MOS驱动 → 震动马达，
实现三种震动模式：短震 / 长震 / 间隔震动。

## 硬件说明
- GPIO 电压：0~3.3V，严禁超压
- sysfs 手动测试编号公式：n×32 + (字母序号-1)×8 + 数字，A=1 B=2 C=3 D=4
- 当前接线：GPIO3_B3，sysfs 手动测试用 `GPIO107`
- 当前 Python 配置：`GPIO_CHIP_PATH = "/dev/gpiochip3"`，`GPIO_PIN = 11`
- 注意：`python-periphery` 使用 gpiochip 内部 line offset，不能直接把 sysfs 全局编号 `107` 配 `/dev/gpiochip0` 使用
- 驱动链路：GPIO → MOS模块 → 5V电源 → 马达，必须共地

## 当前进度
- [已完成] motor_controller.py 已按 ELF 2 实测配置更新
- [已完成] GPIO3_B3 已验证为 `/dev/gpiochip3` line `11` 可驱动马达

## 下次告诉 Codex 的内容
如果后面换 GPIO，直接说："我换到了 GPIOx_yz，帮我重新换算 sysfs 编号和 periphery 的 gpiochip/line 配置"
