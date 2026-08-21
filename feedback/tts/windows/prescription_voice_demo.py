import json
from pathlib import Path
import pyttsx3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "prescription_data"


def find_latest_json():
    json_files = list(DATA_DIR.glob("*.json"))

    if not json_files:
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")

    for voice in voices:
        voice_info = (voice.name + " " + voice.id).lower()
        if "chinese" in voice_info or "huihui" in voice_info or "zh" in voice_info:
            engine.setProperty("voice", voice.id)
            break

    engine.say(text)
    engine.runAndWait()


def build_message(data):
    patient_id = data.get("patient_id", "患者")
    action_name = data.get("action_name", "康复动作")

    clinical = data.get("clinical_baseline", {})

    max_flexion = clinical.get("max_knee_flexion_angle")
    min_flexion = clinical.get("min_knee_flexion_angle")
    rom_flexion = clinical.get("rom_flexion")
    duration = clinical.get("duration_seconds")
    frame_count = clinical.get("frame_count")

    if max_flexion is None or rom_flexion is None:
        return "数字处方读取成功，但没有找到膝关节屈曲角数据，请检查 JSON 文件。"

    max_flexion = round(max_flexion, 1)
    min_flexion = round(min_flexion, 1)
    rom_flexion = round(rom_flexion, 1)
    duration = round(duration, 1) if duration is not None else 0

    if rom_flexion >= 60:
        evaluation = "动作幅度很好，可以作为后续训练模板。"
    elif rom_flexion >= 30:
        evaluation = "动作幅度基本可用，但后续可以再录一版更标准的模板。"
    else:
        evaluation = "动作幅度偏小，建议重新录制标准动作。"

    if action_name == "knee_flexion":
        action_name_cn = "膝关节屈伸"
    else:
        action_name_cn = action_name

    message = (
        f"{patient_id}，{action_name_cn}数字处方读取成功。"
        f"本次最小屈曲角{min_flexion}度，"
        f"最大屈曲角{max_flexion}度，"
        f"活动范围{rom_flexion}度，"
        f"动作时长{duration}秒，"
        f"共记录{frame_count}帧。"
        f"{evaluation}"
    )

    return message


def main():
    latest_file = find_latest_json()

    if latest_file is None:
        print("没有找到 JSON 文件。请先录制数字处方。")
        return

    print("读取文件：", latest_file)

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    message = build_message(data)

    print()
    print("生成的康复语音提示：")
    print(message)

    print()
    print("开始播报...")
    speak(message)

    print("JSON + TTS 联动测试完成。")


if __name__ == "__main__":
    main()
