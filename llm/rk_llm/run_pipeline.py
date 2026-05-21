from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    TEST_TEXT_DIR,
    EchoLLM,
    QwenLLM,
    Recorder,
    SpeechRecognitionAsr,
    WindowsSapiTts,
    copy_into,
    normalize_device,
    pipeline_paths,
    play_wav_file,
    read_text,
    timestamped_run_dir,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行完整语音 Qwen 管线，并保存全部测试记录。")
    parser.add_argument(
        "--source",
        choices=["test-text", "text-file", "audio-file", "record"],
        default="test-text",
        help="管线输入来源。",
    )
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN", help="管线语言。")
    parser.add_argument("--test-text", default=None, help="测试文本文件，默认 pipeline/test_text/<lang>.txt。")
    parser.add_argument("--text-file", default=None, help="source=text-file 时的输入文本文件。")
    parser.add_argument("--audio-file", default=None, help="source=audio-file 时的输入 WAV 文件。")
    parser.add_argument("--duration", type=float, default=5.0, help="source=record 时的录音秒数。")
    parser.add_argument("--sample-rate", type=int, default=16000, help="录音采样率。")
    parser.add_argument("--input-device", default=None, help="sounddevice 输入设备编号或名称。")
    parser.add_argument(
        "--asr-backend",
        choices=["google", "sphinx", "whisper", "faster-whisper"],
        default="faster-whisper",
        help="ASR 后端。中文默认建议 faster-whisper。",
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
    parser.add_argument("--tts-volume", type=float, default=1.0, help="音量，范围 0.0 到 1.0。")
    parser.add_argument("--output-device", default=None, help="--play-final 使用的 sounddevice 输出设备编号或名称。")
    parser.add_argument("--play-final", action="store_true", help="播放最终 LLM TTS 输出 WAV。")
    parser.add_argument("--run-prefix", default="pipeline", help="运行记录目录后缀。")
    return parser


def default_test_text(language: str) -> Path:
    if language == "zh-CN":
        return TEST_TEXT_DIR / "zh.txt"
    return TEST_TEXT_DIR / "en.txt"


def main() -> int:
    args = build_parser().parse_args()
    run_dir = timestamped_run_dir(args.run_prefix)
    paths = pipeline_paths(run_dir)

    tts = WindowsSapiTts(
        voice=args.tts_voice,
        language=args.language,
        rate=args.tts_rate,
        volume=args.tts_volume,
    )

    if args.source in {"test-text", "text-file"}:
        source_text = Path(args.text_file) if args.source == "text-file" else Path(args.test_text or default_test_text(args.language))
        seed_text = read_text(source_text)
        write_text(paths["seed_text"], seed_text)
        tts.synthesize_to_file(seed_text, paths["input_audio"])
    elif args.source == "audio-file":
        if not args.audio_file:
            raise RuntimeError("source=audio-file 时必须提供 --audio-file")
        copy_into(args.audio_file, paths["input_audio"])
    else:
        recorder = Recorder(sample_rate=args.sample_rate, input_device=normalize_device(args.input_device))
        recorder.record_to_file(args.duration, paths["input_audio"])

    asr = SpeechRecognitionAsr(
        backend=args.asr_backend,
        language=args.language,
        whisper_model=args.whisper_model,
    )
    transcript = asr.transcribe_file(paths["input_audio"])
    write_text(paths["asr_text"], transcript)

    if args.llm_backend == "echo":
        llm = EchoLLM()
    else:
        llm = QwenLLM(
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device_map=None if args.device_map == "none" else args.device_map,
            torch_dtype=args.torch_dtype,
        )
    answer = llm.generate(transcript)
    write_text(paths["llm_text"], answer)

    tts.synthesize_to_file(answer, paths["tts_audio"])
    if args.play_final:
        play_wav_file(paths["tts_audio"], normalize_device(args.output_device))

    write_json(
        paths["manifest"],
        {
            "run_dir": str(run_dir),
            "source": args.source,
            "language": args.language,
            "asr_backend": args.asr_backend,
            "whisper_model": args.whisper_model,
            "llm_backend": args.llm_backend,
            "model_path": args.model_path if args.llm_backend == "qwen" else None,
            "artifacts": {name: str(path) for name, path in paths.items()},
            "transcript": transcript,
            "answer": answer,
        },
    )

    print(run_dir)
    print(f"ASR: {transcript}")
    print(f"LLM: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
