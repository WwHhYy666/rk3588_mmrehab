from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import SpeechRecognitionAsr, read_text, write_json, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将输入 WAV 音频文件识别为文本。")
    parser.add_argument("--audio-file", required=True, help="输入 WAV 音频文件。")
    parser.add_argument("--output", default="pipeline/asr/asr_text.txt", help="输出文本路径。")
    parser.add_argument(
        "--backend",
        choices=["google", "sphinx", "whisper", "faster-whisper"],
        default="faster-whisper",
        help="SpeechRecognition ASR 后端。中文默认建议 faster-whisper。",
    )
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN", help="识别语言。")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper/faster-whisper 模型大小或路径。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    asr = SpeechRecognitionAsr(
        backend=args.backend,
        language=args.language,
        whisper_model=args.whisper_model,
    )
    transcript = asr.transcribe_file(args.audio_file)
    output = write_text(args.output, transcript)
    write_json(
        Path(output).with_suffix(".json"),
        {
            "step": "asr_audio",
            "audio_file": args.audio_file,
            "backend": args.backend,
            "language": args.language,
            "whisper_model": args.whisper_model,
            "output": str(output),
            "text": read_text(output),
        },
    )
    print(output)
    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
