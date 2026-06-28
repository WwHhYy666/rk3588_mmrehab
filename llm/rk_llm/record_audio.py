from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import Recorder, list_audio_devices, normalize_device, positive_float, positive_int, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调用设备麦克风录音，并保存为 WAV 文件。")
    parser.add_argument("--duration", type=positive_float, default=5.0, help="录音时长，单位秒。")
    parser.add_argument("--sample-rate", type=positive_int, default=16000, help="录音采样率。")
    parser.add_argument("--input-device", default=None, help="sounddevice 输入设备编号或名称。")
    parser.add_argument("--output", default="pipeline/record_speech/input.wav", help="输出 WAV 路径。")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_audio_devices()
        return 0

    recorder = Recorder(sample_rate=args.sample_rate, input_device=normalize_device(args.input_device))
    output = recorder.record_to_file(args.duration, args.output)
    write_json(
        Path(output).with_suffix(".json"),
        {
            "step": "record_audio",
            "duration": args.duration,
            "sample_rate": args.sample_rate,
            "input_device": args.input_device,
            "output": str(output),
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
