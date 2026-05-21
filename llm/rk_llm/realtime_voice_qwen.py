from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    EchoLLM,
    QwenLLM,
    Recorder,
    SpeechRecognitionAsr,
    WindowsSapiTts,
    normalize_device,
    pipeline_paths,
    play_wav_file,
    positive_float,
    timestamped_run_dir,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行实时麦克风 -> ASR -> Qwen -> TTS 单轮或多轮系统。")
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN", help="管线语言。")
    parser.add_argument("--duration", type=positive_float, default=5.0, help="每轮录音秒数。")
    parser.add_argument("--sample-rate", type=int, default=16000, help="录音采样率。")
    parser.add_argument("--input-device", default=None, help="sounddevice 输入设备编号或名称。")
    parser.add_argument("--output-device", default=None, help="sounddevice 输出设备编号或名称。")
    parser.add_argument(
        "--asr-backend",
        choices=["google", "sphinx", "whisper", "faster-whisper"],
        default="faster-whisper",
        help="ASR 后端。中文/英文离线识别建议 faster-whisper。",
    )
    parser.add_argument("--whisper-model", default="tiny", help="Whisper/faster-whisper 模型大小或路径。")
    parser.add_argument("--llm-backend", choices=["qwen", "echo"], default="qwen", help="LLM 后端。")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Qwen 模型目录或 Hugging Face model id。")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="系统提示词。")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="最大生成 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM 温度。")
    parser.add_argument("--device-map", choices=["auto", "none"], default="none", help="CPU 上建议使用 none。")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tts-voice", default=None, help="Windows SAPI 语音名称片段。")
    parser.add_argument("--tts-rate", type=int, default=0, help="Windows SAPI 语速，范围 -10 到 10。")
    parser.add_argument("--tts-volume", type=float, default=1.0, help="TTS 音量，范围 0.0 到 1.0。")
    parser.add_argument("--skip-tts", action="store_true", help="不播放最终 TTS 音频。")
    parser.add_argument("--once", action="store_true", help="只运行一轮后退出。")
    parser.add_argument("--run-prefix", default="realtime", help="运行记录目录后缀。")
    return parser


def build_llm(args):
    if args.llm_backend == "echo":
        return EchoLLM()
    return QwenLLM(
        model_path=args.model_path,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=None if args.device_map == "none" else args.device_map,
        torch_dtype=args.torch_dtype,
    )


def run_turn(args, turn_dir: Path, recorder: Recorder, asr: SpeechRecognitionAsr, llm, tts: WindowsSapiTts) -> dict:
    paths = pipeline_paths(turn_dir)
    print(f"录音 {args.duration:.1f} 秒...")
    recorder.record_to_file(args.duration, paths["input_audio"])

    print("正在进行 ASR...")
    transcript = asr.transcribe_file(paths["input_audio"])
    write_text(paths["asr_text"], transcript)
    print(f"ASR: {transcript or '<空>'}")

    answer = ""
    if transcript:
        print("正在调用 LLM...")
        answer = llm.generate(transcript)
    write_text(paths["llm_text"], answer)
    print(f"LLM: {answer or '<空>'}")

    if answer:
        tts.synthesize_to_file(answer, paths["tts_audio"])
        if not args.skip_tts:
            play_wav_file(paths["tts_audio"], normalize_device(args.output_device))

    result = {
        "run_dir": str(turn_dir),
        "language": args.language,
        "asr_backend": args.asr_backend,
        "whisper_model": args.whisper_model,
        "llm_backend": args.llm_backend,
        "input_audio": str(paths["input_audio"]),
        "asr_text": str(paths["asr_text"]),
        "llm_text": str(paths["llm_text"]),
        "tts_audio": str(paths["tts_audio"]) if answer else None,
        "transcript": transcript,
        "answer": answer,
    }
    write_json(paths["manifest"], result)
    return result


def main() -> int:
    args = build_parser().parse_args()
    recorder = Recorder(sample_rate=args.sample_rate, input_device=normalize_device(args.input_device))
    asr = SpeechRecognitionAsr(args.asr_backend, args.language, args.whisper_model)
    llm = build_llm(args)
    tts = WindowsSapiTts(args.tts_voice, args.language, args.tts_rate, args.tts_volume)

    while True:
        turn_dir = timestamped_run_dir(args.run_prefix)
        run_turn(args, turn_dir, recorder, asr, llm, tts)
        if args.once:
            break
        value = input("按 Enter 开始下一轮，输入 q 退出> ").strip().lower()
        if value in {"q", "quit", "exit"}:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
