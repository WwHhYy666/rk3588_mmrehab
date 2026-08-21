# 8085 独立动作组板端交付手册

## 当前完成状态

- 已完成 `REHAB_EXTENDED_GROUPS` 默认关闭的独立扩展模块。
- 已完成 `/train-upper`、`/train-full`、六动作指标、补偿检测、独立计数/TUT/速度、模板健康和扩展报告。
- 已完成下肢训练互斥：录制、单动作训练、播放列表运行时，扩展入口会拒绝启动；扩展运行时，下肢入口也会拒绝启动。
- 已完成稳定基线创建和一键回退脚本。
- 本地专项测试为 `18 passed`；现有 8085 COCO-17 测试 `37 passed`，双模式合同测试 `15 passed`，NPU 调试/YOLO 测试 `23 passed`。
- 尚未完成板端摄像头、NPU、浏览器和六个医生模板的实机验收，因此默认不能宣称比赛配置已启用。

## 首次部署顺序

1. 先上传并执行 `scripts/create_rehab_extension_baseline.sh`。它会锁定当前 `home.js`、`record_prescription_http.py`、`npu_rehab_8085.py`、显示管理器、kiosk 启动器和 `train.js` 的 SHA-256。基线已存在时不要覆盖。
2. 上传扩展模块和接线文件，保持 `runtime/rehab_extensions.flag` 为 `0`。关闭状态下新模块不导入，首页只显示原下肢入口。
3. 只重启 8085：使用现有 `bash scripts/switch_rehab_mode.sh npu_8085` 或板端服务重启命令；不要重启整板，不改 8082。
4. 检查 `curl -fsS http://127.0.0.1:8085/status`、`curl -i http://127.0.0.1:8085/train`，确认 8085 为 `rknn/yolov5n_rtmpose`，且 `extension_enabled` 为 `false`。
5. 进入受控验收时临时执行 `bash scripts/set_rehab_extensions.sh on --restart`，录模板并测试；当天验收未完成时执行 `bash scripts/set_rehab_extensions.sh off --restart`。这不等于把扩展设为正式比赛配置。

## 六个动作怎么做

医生模板在对应扩展页面选择动作后，先点击“录制医生模板”，按动作完成一组连续轨迹，再点击“保存模板”。每个模板必须至少 `20` 个有效帧、持续至少 `2s`，包含起始位、动作峰值和回到起始位；否则页面会显示模板不健康，患者训练按钮会拒绝启动。

| 动作 | 站位和方向 | 一次动作 |
| --- | --- | --- |
| 坐姿屈肘 `seated_biceps_curl` | 坐稳，身体侧面对镜头，肩、肘、腕和髋在画面内 | 上臂靠近躯干固定，屈肘把手抬向肩部，再控制伸回起始位；避免上臂前后摆、躯干前倾 |
| 坐姿前平举 `seated_shoulder_flexion` | 坐姿侧面，肩、肘、腕和髋完整可见 | 肘部尽量伸直，从身体侧面向前抬臂到模板峰值，再缓慢放回；不要用躯干前倾代偿 |
| 站姿侧平举 `standing_shoulder_abduction` | 站立正面对镜头，双肩、双肘、双腕、双髋完整可见 | 双臂向身体两侧抬起，左右接近同高，再同时放回；身体不要侧倾 |
| 半蹲 `mini_squat` | 身体侧面对镜头，肩、髋、膝、踝完整可见 | 髋和膝同时下降到半蹲，再站回；不能只弯膝而髋几乎不下降 |
| 站姿侧向迈步 `lateral_step_touch` | 正面对镜头，双脚踝和双肩完整可见 | 向一侧迈开至模板站距，另一脚触回，再回到起始站距；保持肩线水平 |
| 低冲击开合步 `low_impact_step_jack` | 正面对镜头，全身和双腕、双踝完整可见 | 不跳跃，手臂上举同时侧向打开双脚，再同步收回；手臂和腿部进度差不能过大 |

## 模板验收

逐个动作保存后检查：

```bash
find extension_rehab/templates -type f -name '*.json' -print
curl -fsS 'http://127.0.0.1:8085/api/extension/status'
```

六个动作都显示 `template_health.ok=true` 后，再做每个动作至少 10 次患者模拟，记录计数、漏检、无效尝试、实际 ROM、单次 TUT、速度、补偿错误和报告文件。`quality_score` 是比赛展示用的可解释质量分，不是临床准确率。

## 打开比赛演示开关

受控验收会临时打开开关。只有六个动作均通过板端验收后，才把开关保持为开启并作为比赛演示配置：

```bash
bash scripts/set_rehab_extensions.sh on --restart
```

异常立即执行：

```bash
bash scripts/rollback_rehab_extensions.sh
```

回退只关闭扩展、恢复基线共享文件并重启 8085；不重启整板，不修改 8082。回退后应看到 `/train` 可用、三项下肢配置存在、`extension_enabled=false`。

## 上传清单

必须上传：

- `extension_rehab/`
- `prescription/banzi/static/extension_train.js`
- `prescription/banzi/static/train-upper.js`
- `prescription/banzi/static/train-full.js`
- `prescription/banzi/static/home.js`
- `prescription/banzi/record_prescription_http.py`
- `prescription/banzi/npu_rehab_8085.py`
- `scripts/create_rehab_extension_baseline.sh`
- `scripts/set_rehab_extensions.sh`
- `scripts/rollback_rehab_extensions.sh`

不要上传或替换：`prescription/banzi/static/train.js`、`scripts/rehab_display_manager.sh`、`scripts/open_rehab_station_kiosk.sh`。医生模板由板端录制生成到 `extension_rehab/templates/upper/` 和 `extension_rehab/templates/full/`，不从 Windows 空目录覆盖板端模板。
