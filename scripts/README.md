# Operations scripts

| 脚本 | 作用 |
| --- | --- |
| `start_npu_rehab_8085.sh` | 启动模型、可选 Qwen proxy 和 8085 服务 |
| `stop_npu_rehab_8085.sh` | 安全停止 8085 入口 |
| `check_npu_rehab_8085.sh` | 检查模型、摄像头、服务、性能和部署哈希 |
| `benchmark_npu_rehab_8085.py` | 按页面场景采集延迟分位数 |
| `check_ai_assistant_status.sh` | 检查 GLM/Qwen/ASR 状态 |
| `regenerate_npu_reports.py` | 从本地 attempt 重新生成报告 |
| `set_npu_pose_execution_mode.sh` | 切换异步/同步姿态执行诊断模式 |
| `install_npu_rehab_8085_autostart.sh` | 安装 systemd 与浏览器自启动 |
| `open_npu_rehab_8085_kiosk.sh` | 打开患者训练全屏页 |
| `open_npu_debug_8085_kiosk.sh` | 打开 NPU 调试全屏页 |
| `install_rknn_runtime.sh` | 安装匹配的 Rockchip 运行库 |
| `restore_rknn_runtime.sh` | 回滚已备份的 Rockchip 运行库 |
