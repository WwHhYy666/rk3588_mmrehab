# 板端 TTS 说明

`tts_board.py` 每次运行都会重新扫描板端本地最新数据，只播报一次就退出。

读取顺序：

1. 优先读 `prescription/docs/summaries/`
2. 如果没有摘要，再读 `prescription/docs/results/`

播报内容：

- `summaries/` 直接提取“患者、动作、帧数、时长、ROM、结果判断”
- `results/` 从完整 JSON 里拼出同样的播报句子

运行方式：

```bash
cd ~/project/project_system/feedback/tts/banzi
python3 tts_board.py
```

如果板子上有新的录制结果，下次重新运行它就会自动播最新一份，不会固定读旧文件。
