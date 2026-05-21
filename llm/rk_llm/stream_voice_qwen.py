from __future__ import annotations

import argparse
import time
from pathlib import Path

from realtime_voice_qwen import build_llm, run_turn
from voice_qwen_core import (
    SpeechRecognitionAsr,
    Recorder,
    WindowsSapiTts,
    normalize_device,
    timestamped_run_dir,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按固定间隔运行流式麦克风 -> ASR -> Qwen -> TTS 循环。")
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN")
    parser.add_argument("--interval", type=float, default=20.0, help="每轮启动间隔，单位秒。")
    parser.add_argument("--duration", type=float, default=5.0, help="每轮录音时长，单位秒。")
    parser.add_argument("--max-cycles", type=int, default=0, help="最大循环次数，0 表示直到 Ctrl+C。")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--asr-backend", choices=["google", "sphinx", "whisper", "faster-whisper"], default="faster-whisper")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--llm-backend", choices=["qwen", "echo"], default="qwen")
    parser.add_argument("--model-path", default=None, help="Qwen 模型目录或 Hugging Face model id。")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", choices=["auto", "none"], default="none")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tts-voice", default=None)
    parser.add_argument("--tts-rate", type=int, default=0)
    parser.add_argument("--tts-volume", type=float, default=1.0)
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--run-prefix", default="stream")
    return parser


def normalize_args(args):
    from voice_qwen_core import DEFAULT_MODEL_PATH, DEFAULT_SYSTEM_PROMPT

    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPT
    return args


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    session_dir = timestamped_run_dir(args.run_prefix)
    cycles_dir = session_dir / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)

    recorder = Recorder(sample_rate=args.sample_rate, input_device=normalize_device(args.input_device))
    asr = SpeechRecognitionAsr(args.asr_backend, args.language, args.whisper_model)
    llm = build_llm(args)
    tts = WindowsSapiTts(args.tts_voice, args.language, args.tts_rate, args.tts_volume)

    cycle_results = []
    cycle = 0
    try:
        while args.max_cycles <= 0 or cycle < args.max_cycles:
            cycle += 1
            started = time.time()
            turn_dir = cycles_dir / f"cycle_{cycle:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n第 {cycle} 轮：录音 {args.duration:g} 秒，间隔 {args.interval:g} 秒")
            result = run_turn(args, turn_dir, recorder, asr, llm, tts)
            cycle_results.append(result)
            write_json(
                session_dir / "manifest.json",
                {
                    "session_dir": str(session_dir),
                    "interval": args.interval,
                    "duration": args.duration,
                    "max_cycles": args.max_cycles,
                    "cycles": cycle_results,
                },
            )

            elapsed = time.time() - started
            sleep_seconds = max(0.0, args.interval - elapsed)
            if args.max_cycles <= 0 or cycle < args.max_cycles:
                print(f"等待 {sleep_seconds:.1f} 秒后进入下一轮...")
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
