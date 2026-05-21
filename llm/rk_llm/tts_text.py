from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import WindowsSapiTts, normalize_device, play_wav_file, read_text, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将输入文本文件合成为 WAV 语音文件。")
    parser.add_argument("--input-text", default=None, help="输入文本文件。")
    parser.add_argument("--output-audio", default="pipeline/tts/tts_output.wav", help="输出 WAV 路径。")
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN", help="TTS 语言。")
    parser.add_argument("--voice", default=None, help="Windows SAPI 语音名称片段，例如 Huihui 或 Zira。")
    parser.add_argument("--rate", type=int, default=0, help="Windows SAPI 语速，范围 -10 到 10。")
    parser.add_argument("--volume", type=float, default=1.0, help="音量，范围 0.0 到 1.0。")
    parser.add_argument("--play", action="store_true", help="保存后通过当前音频设备播放。")
    parser.add_argument("--output-device", default=None, help="--play 使用的 sounddevice 输出设备编号或名称。")
    parser.add_argument("--list-voices", action="store_true", help="列出 Windows SAPI 语音后退出。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_voices:
        WindowsSapiTts.list_voices()
        return 0
    if not args.input_text:
        parser.error("除非使用 --list-voices，否则必须提供 --input-text")

    text = read_text(args.input_text)
    tts = WindowsSapiTts(
        voice=args.voice,
        language=args.language,
        rate=args.rate,
        volume=args.volume,
    )
    output = tts.synthesize_to_file(text, args.output_audio)
    if args.play:
        play_wav_file(output, normalize_device(args.output_device))
    write_json(
        Path(output).with_suffix(".json"),
        {
            "step": "tts_text",
            "input_text_file": args.input_text,
            "language": args.language,
            "voice": args.voice,
            "output_audio": str(output),
            "text": text,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
