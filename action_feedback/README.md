# Action feedback

`action_feedback/` 把评估错误转换为可解释的屏幕、TTS 和设备反馈。

- `rule_engine.py`：加载规则并生成统一反馈结构。
- `rules/`：按动作维护 `ROM_LOW`、`TUT_LOW` 等文案和参数。
- `tts/board/`：板端系统 TTS 兜底。

固定演示 WAV 位于 `rehab_app/server/static/assets/tts/`。
