# Real-time training

`training/` 负责每帧动作状态、计数、纠错时序、TTS 和系统状态。

- `training_session.py`：三动作训练编排与异步报告/评分。
- `action_state_machine.py`：单次动作阶段、ROM、TUT 和回位判定。
- `feedback_runtime.py`：纠错规则到页面/TTS 的适配。
- `tts_worker.py`、`natural_tts.py`：优先级语音队列。
- `audio_player.py`、`audio_output.py`：固定音频与 ALSA 输出。
- `system_monitor.py`：设备状态。
- `configs/training_defaults.yaml`：共享训练默认参数。
- `configs/training_defaults_npu.yaml`：RKNN 8085 低可见度阈值与演示参数。
- `configs/rehab_demo_plan_npu.yaml`：8085 三动作训练计划及模块连接关系。
