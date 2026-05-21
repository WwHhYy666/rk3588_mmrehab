from __future__ import annotations

import argparse

from voice_qwen_core import list_audio_devices, normalize_device, play_wav_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过当前音频设备播放 PCM WAV 文件。")
    parser.add_argument("--audio-file", required=False, help="输入 WAV 文件。")
    parser.add_argument("--output-device", default=None, help="sounddevice 输出设备编号或名称。")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_audio_devices()
        return 0
    if not args.audio_file:
        raise RuntimeError("除非使用 --list-devices，否则必须提供 --audio-file")
    play_wav_file(args.audio_file, normalize_device(args.output_device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
