from __future__ import annotations

import argparse
from pathlib import Path

from voice_qwen_core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    EchoLLM,
    QwenLLM,
    looks_like_local_path,
    read_text,
    resolve_model_path,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取输入文本文件，调用本地 Qwen2-VL 进行推理。")
    parser.add_argument("--input-text", required=True, help="输入文本文件。")
    parser.add_argument("--output", default="pipeline/llm/llm_output.txt", help="输出文本路径。")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Qwen 模型目录或 Hugging Face model id。")
    parser.add_argument("--backend", choices=["qwen", "echo"], default="qwen", help="快速测试可使用 echo。")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Qwen 系统提示词。")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度，0 表示贪心解码。")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样 top-p。")
    parser.add_argument("--device-map", choices=["auto", "none"], default="none", help="CPU 上建议使用 none。")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="模型 dtype。CPU 上 auto 会解析为 float32。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_text = read_text(args.input_text)

    if args.backend == "echo":
        llm = EchoLLM()
        model_path = None
    else:
        model_path = resolve_model_path(args.model_path)
        if looks_like_local_path(model_path) and not Path(model_path).is_dir():
            raise RuntimeError(f"模型路径不存在：{model_path}")
        llm = QwenLLM(
            model_path=model_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device_map=None if args.device_map == "none" else args.device_map,
            torch_dtype=args.torch_dtype,
        )

    answer = llm.generate(input_text)
    output = write_text(args.output, answer)
    write_json(
        Path(output).with_suffix(".json"),
        {
            "step": "llm_infer",
            "backend": args.backend,
            "model_path": model_path,
            "input_text_file": args.input_text,
            "input_text": input_text,
            "output": str(output),
            "answer": answer,
        },
    )
    print(output)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
