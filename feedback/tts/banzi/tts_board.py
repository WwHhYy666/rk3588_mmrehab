from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "prescription").is_dir():
            return parent
    return current.parents[3]


def latest_file(directory: Path, suffix: str) -> Path | None:
    if not directory.exists():
        return None

    files = [path for path in directory.glob(f"*{suffix}") if path.is_file()]
    if not files:
        return None

    return max(files, key=lambda path: (path.stat().st_mtime, path.name))


def strip_markdown(value: str) -> str:
    text = value.strip()
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1]
    return text.strip()


def parse_summary_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or "：" not in line:
            continue
        key, value = line[2:].split("：", 1)
        fields[key.strip()] = strip_markdown(value)
    return fields


def extract_section(text: str, section_title: str) -> str:
    lines = text.splitlines()
    in_section = False
    collected: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if line == f"## {section_title}":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line:
            collected.append(strip_markdown(line))

    return " ".join(collected).strip()


def format_number(value: object, digits: int = 1) -> str:
    if value is None:
        return "未知"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    text = str(value).strip()
    if not text:
        return "未知"
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return text


def normalize_speech_text(text: str) -> str:
    def decimal_to_speech(match: re.Match[str]) -> str:
        return f"{match.group(1)}点{match.group(2)}"

    normalized = re.sub(r"(\d+)\.(\d+)", decimal_to_speech, text)
    replacements = {
        "`": "",
        "ROM": "活动范围",
        "auto": "自动",
        "left": "左侧",
        "right": "右侧",
        "knee_flexion": "膝关节屈曲",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def build_message_from_summary(summary_text: str, summary_path: Path) -> str:
    fields = parse_summary_fields(summary_text)
    judgment = extract_section(summary_text, "结果判断")

    patient_id = fields.get("患者编号", "患者")
    action_name = fields.get("动作名称", "康复动作")
    side_mode = fields.get("侧别模式")
    frame_count = fields.get("有效帧数")
    invalid_frame_count = fields.get("无效帧数")
    duration_seconds = fields.get("动作时长")
    min_angle = fields.get("最小屈曲角")
    max_angle = fields.get("最大屈曲角")
    rom_flexion = fields.get("ROM")

    parts = [f"{patient_id}，{action_name}数字处方已生成。"]
    if side_mode:
        parts.append(f"侧别模式{side_mode}。")
    if frame_count:
        parts.append(f"有效帧{frame_count}帧。")
    if invalid_frame_count:
        parts.append(f"无效帧{invalid_frame_count}帧。")
    if duration_seconds:
        parts.append(f"动作时长{duration_seconds}秒。")
    if min_angle and max_angle and rom_flexion:
        parts.append(f"最小屈曲角{min_angle}度，最大屈曲角{max_angle}度，活动范围{rom_flexion}度。")
    elif rom_flexion:
        parts.append(f"活动范围{rom_flexion}度。")
    if judgment:
        parts.append(judgment)
    else:
        parts.append(f"摘要文件：{summary_path.name}。")

    return "".join(parts)


def build_message_from_result(payload: dict[str, object], result_path: Path) -> str:
    clinical = payload.get("clinical_baseline", {})
    runtime_meta = payload.get("runtime_meta", {})
    if not isinstance(clinical, dict):
        clinical = {}
    if not isinstance(runtime_meta, dict):
        runtime_meta = {}

    patient_id = str(payload.get("patient_id", "患者"))
    action_name = str(payload.get("action_name", "康复动作"))
    side_mode = str(runtime_meta.get("side_mode", "unknown"))
    frame_count = clinical.get("frame_count")
    invalid_frame_count = runtime_meta.get("invalid_frame_count")
    duration_seconds = clinical.get("duration_seconds")
    min_angle = clinical.get("min_knee_flexion_angle")
    max_angle = clinical.get("max_knee_flexion_angle")
    rom_flexion = clinical.get("rom_flexion")

    judgment = ""
    if isinstance(rom_flexion, (int, float)):
        if rom_flexion < 15:
            judgment = "ROM偏小，建议重新录制。"
        elif rom_flexion < 30:
            judgment = "ROM勉强可用，建议再录一版更稳定的结果。"
        else:
            judgment = "ROM较明显，这份结果可以作为后续模板参考。"

    parts = [f"{patient_id}，{action_name}数字处方已生成。"]
    if side_mode and side_mode != "unknown":
        parts.append(f"侧别模式{side_mode}。")
    if frame_count is not None:
        parts.append(f"有效帧{frame_count}帧。")
    if invalid_frame_count is not None:
        parts.append(f"无效帧{invalid_frame_count}帧。")
    if duration_seconds is not None:
        parts.append(f"动作时长{format_number(duration_seconds)}秒。")
    if min_angle is not None and max_angle is not None and rom_flexion is not None:
        parts.append(
            f"最小屈曲角{format_number(min_angle)}度，"
            f"最大屈曲角{format_number(max_angle)}度，"
            f"活动范围{format_number(rom_flexion)}度。"
        )
    elif rom_flexion is not None:
        parts.append(f"活动范围{format_number(rom_flexion)}度。")
    if judgment:
        parts.append(judgment)
    else:
        parts.append(f"结果文件：{result_path.name}。")

    return "".join(parts)


def choose_voice(engine) -> str | None:
    voices = engine.getProperty("voices") or []
    preferred_markers = (
        "chinese",
        "zh",
        "zh-cn",
        "zh_cn",
        "huihui",
        "kangkang",
        "xiaoxiao",
        "xiaoyi",
        "yunyang",
        "yunjian",
    )

    for voice in voices:
        voice_info = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
        languages = str(getattr(voice, "languages", "")).lower()
        if any(marker in voice_info for marker in preferred_markers) or "zh" in languages:
            try:
                engine.setProperty("voice", voice.id)
                return voice.id
            except Exception:
                continue
    return None


def speak_with_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3
    except Exception as exc:
        print(f"pyttsx3 导入失败：{exc}")
        return False

    try:
        print("正在使用 pyttsx3 播报。")
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        voice_id = choose_voice(engine)
        if voice_id:
            print(f"已切换语音：{voice_id}")
        else:
            print("未找到中文语音，使用系统默认语音。")
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        print("pyttsx3 播报完成。")
        return True
    except Exception as exc:
        print(f"pyttsx3 播报失败：{exc}")
        return False


def speak_with_espeak(text: str) -> bool:
    command = shutil.which("espeak-ng") or shutil.which("espeak")
    aplay = shutil.which("aplay")
    if not command or not aplay:
        return False

    try:
        print("正在使用 espeak-ng + aplay 播报。")
        pipeline = (
            f'printf "%s" {shlex.quote(text)} '
            f'| {shlex.quote(command)} -v zh --stdout '
            f'| {shlex.quote(aplay)} -D plughw:1,0'
        )
        subprocess.run(pipeline, shell=True, check=False)
        return True
    except Exception as exc:
        print(f"espeak 播报失败：{exc}")
        return False


def main() -> int:
    project_root = resolve_project_root()
    prescription_root = project_root / "prescription"
    summaries_dir = prescription_root / "docs" / "summaries"
    results_dir = prescription_root / "docs" / "results"

    summary_path = latest_file(summaries_dir, ".md")
    if summary_path is not None:
        summary_text = summary_path.read_text(encoding="utf-8")
        message = build_message_from_summary(summary_text, summary_path)
        print(f"使用摘要：{summary_path}")
    else:
        result_path = latest_file(results_dir, ".json")
        if result_path is None:
            print(f"没有找到可播报数据：{summaries_dir} / {results_dir}")
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        message = build_message_from_result(payload, result_path)
        print(f"使用结果：{result_path}")

    speech_text = normalize_speech_text(message)
    print("播报内容：")
    print(speech_text)

    if speak_with_pyttsx3(speech_text):
        return 0

    print("pyttsx3 未成功发声，回退到 espeak-ng 播报。")
    if speak_with_espeak(speech_text):
        return 0

    print("当前没有可用的 TTS 后端。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
