# 国赛 8085 独立动作组部署与验收指南

## 稳定边界

- 板子当前仍是稳定版，扩展文件尚未部署。
- `REHAB_EXTENDED_GROUPS` 默认关闭；关闭时不加载扩展模块，不显示扩展入口。
- 不修改现有 `/train`、下肢三动作状态机、训练后功能、显示管理器和 kiosk 启动方式。
- 不设置开机自启动。正常使用仍是在 Ubuntu 桌面双击 8085 图标。

冻结文件及 SHA-256：

- `scripts/rehab_display_manager.sh`: `5dcd037c9bfa8c0333c105ca6173aa9220d72fc524eff94463211369cb8296e9`
- `scripts/open_rehab_station_kiosk.sh`: `ba53813da3b8d5bd382ee267f350f31827bbf3db448868a4626e756fe779a75f`
- `prescription/banzi/static/train.js`: `f3c741ccc972ad1ef3ab5ab3471a137681359ae584ade12c3e3f1025f9bcd86b`

## 动作组

### 上肢组

1. 坐姿屈肘训练
   - 侧面对准摄像头，坐稳并露出肩、肘、腕。
   - 上臂贴近躯干，只屈肘将手抬起，随后缓慢伸直。
   - 种子 ROM 55 度；启动变化 12 度；最小有效幅度 35 度；返回变化 8 度；单次 1.2-6 秒。
   - 检查上臂摆动、躯干倾斜和腕点可见度。

2. 坐姿前平举
   - 侧面对准摄像头，手臂自然垂下。
   - 肘部尽量伸直，将手臂向前抬起，再缓慢放回体侧。
   - 种子 ROM 50 度；启动变化 12 度；最小有效幅度 45 度；返回变化 8 度；单次 1.2-6 秒。
   - 检查躯干倾斜、肘部弯曲和肩肘腕可见度。

3. 站姿侧平举
   - 正面对准摄像头，双臂自然下垂。
   - 双臂同步向两侧抬起，最高点停稳后放下。
   - 种子 ROM 55 度；启动变化 12 度；最小有效幅度 45 度；返回变化 8 度；单次 1.2-6 秒。
   - 左右差超过 18 度或身体侧倾超过 12 度时记录质量问题。

上肢整组顺序固定为以上顺序，动作间休息 6 秒，并播放与下肢组相同的休息音乐。

### 全身组

1. 半蹲训练
   - 侧面对准摄像头，全身入画，双脚与肩同宽。
   - 髋部向后下方移动完成半蹲，再站直。
   - 种子 ROM 30 度；启动变化 8 度；最小有效幅度 25 度；返回变化 6 度；单次 1.5-7 秒。
   - 检查髋部是否下沉以及躯干倾斜是否超过 20 度。

2. 站姿侧向迈步
   - 正面对准摄像头，完整露出双踝。
   - 一侧脚向外迈开，身体直立，再收回到起始站距。
   - 踝距/肩宽种子变化 0.45；启动变化 0.18；最小有效幅度 0.35；返回变化 0.12；单次 1.5-7 秒。
   - 检查肩线倾斜和双踝可见度。

3. 低冲击开合步
   - 正面对准摄像头，双脚自然站立，双臂在体侧。
   - 一侧脚外迈的同时双臂上举，不跳跃；上下肢同步回位。
   - 种子变化 0.55；启动变化 0.15；最小有效幅度 0.45；返回变化 0.12；单次 2-8 秒。
   - 上下肢进度差超过 0.30 时记录协调性问题。

全身整组顺序固定为以上顺序，动作间休息 6 秒，并播放与下肢组相同的休息音乐。

侧面动作会自动选择关键点可见度更高的一侧。以上阈值只用于首版启动，最终目标 ROM 来自医生模板，不作为临床标准。

## 必须上传的文件

第一次只上传：

- `scripts/create_rehab_extension_baseline.sh`

创建板端基线后，再上传：

- `extension_rehab/` 整个目录，但不上传 `__pycache__` 和 `.pyc`
- `prescription/banzi/record_prescription_http.py`
- `prescription/banzi/npu_rehab_8085.py`
- `prescription/banzi/static/home.js`
- `prescription/banzi/static/extension_train.js`
- `prescription/banzi/static/train-upper.js`
- `prescription/banzi/static/train-full.js`
- `scripts/set_rehab_extensions.sh`
- `scripts/rollback_rehab_extensions.sh`

不要覆盖冻结的三个显示文件，也不需要上传测试文件或本指南。

## 安全部署顺序

板端项目目录假定为 `/home/elf/project/project_system`。

1. 上传基线脚本后，在覆盖共享文件前执行：

```bash
cd /home/elf/project/project_system
bash scripts/create_rehab_extension_baseline.sh
```

2. 上传其余文件后，先保持扩展关闭并只重启 8085：

```bash
cd /home/elf/project/project_system
bash scripts/set_rehab_extensions.sh off
bash scripts/switch_rehab_mode.sh npu_8085
```

3. 验证原有首页、`/train` 和完整下肢三动作流程。此时扩展卡片不应出现。

4. 原流程通过后再开启扩展：

```bash
cd /home/elf/project/project_system
bash scripts/set_rehab_extensions.sh on --restart
```

不需要 `sudo reboot`。以后正常开机仍从 Ubuntu 桌面双击 8085 图标启动。

5. 逐个录制六个医生模板。每个模板建议录 4-6 秒，完成“起始位 -> 最高点 -> 起始位”，界面应显示至少 20 个有效帧和至少 2 秒。

6. 六个模板都健康后，分别点击“开始整组”验证三动作连续流程；“单动作测试”用于现场校准。

## 验收与回退

开启后检查：

```bash
curl -s http://127.0.0.1:8085/status | python3 -m json.tool
curl -s http://127.0.0.1:8085/api/extension/status | python3 -m json.tool
```

应确认：

- `actual_backend` 为 `rknn`；
- `extension_enabled` 为 `true`；
- 原下肢流程仍可完成；
- 扩展训练不会接收复用的 RTMPose 帧；
- 新页面断流时只重连 `/stream.mjpg`，不会刷新整页；
- 六个动作各完成至少 10 次模拟测试，再用于比赛演示。

异常时执行：

```bash
cd /home/elf/project/project_system
bash scripts/rollback_rehab_extensions.sh
```

回退只恢复共享稳定文件并重启 8085，不修改 8082，也不重启整块板。
