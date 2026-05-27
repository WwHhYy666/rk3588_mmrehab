from pathlib import Path
import subprocess
import soundfile as sf
import sherpa_onnx


# 当前文件路径：/home/elf/project/project_system/tts/test.py
ROOT = Path(__file__).resolve().parent

# 权重目录：/home/elf/project/project_system/tts/tts_model_pack
MODEL_DIR = ROOT / "tts_model_pack"

# 先用无量化模型跑通
MODEL = MODEL_DIR / "vits-aishell3.onnx"
# 如果无量化模型跑通后想测试 int8，就改成下面这一行
# MODEL = MODEL_DIR / "vits-aishell3.int8.onnx"

LEXICON = MODEL_DIR / "lexicon.txt"
TOKENS = MODEL_DIR / "tokens.txt"

PHONE_FST = MODEL_DIR / "phone.fst"
DATE_FST = MODEL_DIR / "date.fst"
NUMBER_FST = MODEL_DIR / "number.fst"

OUT_DIR = ROOT / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WAV = OUT_DIR / "test_tts.wav"


def check_files():
    files = [
        MODEL,
        LEXICON,
        TOKENS,
        PHONE_FST,
        DATE_FST,
        NUMBER_FST,
    ]

    print("当前 ROOT:", ROOT)
    print("当前 MODEL_DIR:", MODEL_DIR)

    for f in files:
        print("检查文件:", f)
        if not f.exists():
            raise FileNotFoundError(f"找不到文件: {f}")

    print("[OK] 所有必要文件都存在")


def main():
    check_files()

    rule_fsts = f"{PHONE_FST},{DATE_FST},{NUMBER_FST}"

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(MODEL),
                lexicon=str(LEXICON),
                tokens=str(TOKENS),
                data_dir="",
            ),
            num_threads=2,
            debug=True,
            provider="cpu",
        ),
        rule_fsts=rule_fsts,
        max_num_sentences=1,
    )

    print("[INFO] 开始校验 TTS 配置")
    if not config.validate():
        raise RuntimeError(
            "TTS config validate failed，请检查 onnx、lexicon、tokens、fst 文件是否匹配"
        )

    print("[INFO] 开始加载 TTS 模型")
    tts = sherpa_onnx.OfflineTts(config)
    print("[OK] TTS 模型加载完成")

    text = "膝盖稍微向外打开，背部保持挺直。"
    sid = 66
    speed = 1.0

    print("[INFO] 合成文本:", text)

    # 兼容不同版本 sherpa_onnx API
    if hasattr(sherpa_onnx, "GenerationConfig"):
        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = sid
        gen_config.speed = speed
        gen_config.silence_scale = 0.2
        audio = tts.generate(text, gen_config)
    else:
        audio = tts.generate(text, sid=sid, speed=speed)

    if len(audio.samples) == 0:
        raise RuntimeError("生成音频为空")

    sf.write(
        str(OUT_WAV),
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )

    print("[OK] wav 已生成:", OUT_WAV)
    print("[INFO] sample_rate:", audio.sample_rate)
    print("[INFO] samples:", len(audio.samples))

    print("[INFO] 尝试播放")
    subprocess.run(["aplay", str(OUT_WAV)], check=False)


if __name__ == "__main__":
    main()