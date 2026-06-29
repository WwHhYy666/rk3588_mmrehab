from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESCRIPTION_DIR = PROJECT_ROOT / "prescription"
DOCS_DIR = PRESCRIPTION_DIR / "docs"
RESULTS_DIR = DOCS_DIR / "results"
DOCTOR_TEMPLATES_DIR = DOCS_DIR / "doctor_templates"
PATIENT_ATTEMPTS_DIR = DOCS_DIR / "patient_attempts"
SUMMARIES_DIR = DOCS_DIR / "summaries"
RESULTS_LOG_PATH = DOCS_DIR / "results_log.md"

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
ROLE_DIRS = {
    "doctor_template": ("doctor_templates", "医生模板", "医生模板文件"),
    "patient_attempt": ("patient_attempts", "患者动作", "患者动作文件"),
}


def ensure_dirs(docs_dir: Path | None = None) -> tuple[Path, Path, Path, Path]:
    base_dir = docs_dir or DOCS_DIR
    results_dir = base_dir / "results"
    doctor_templates_dir = base_dir / "doctor_templates"
    patient_attempts_dir = base_dir / "patient_attempts"
    summaries_dir = base_dir / "summaries"
    results_log_path = base_dir / "results_log.md"

    base_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    doctor_templates_dir.mkdir(parents=True, exist_ok=True)
    patient_attempts_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if not results_log_path.exists():
        results_log_path.write_text(
            "# 处方结果索引\n\n"
            "这里记录每次本地保存的简要摘要，方便快速找到对应的 JSON 和中文摘要文件。\n"
            "医生模板保存在 `docs/doctor_templates/`，患者动作保存在 `docs/patient_attempts/`。\n"
            "`docs/results/` 仅保留旧版本兼容文件。\n\n",
            encoding="utf-8",
        )

    return base_dir, results_dir, summaries_dir, results_log_path


def safe_name(value: str, default: str) -> str:
    text = (value or "").strip()
    if not text:
        return default
    return SAFE_NAME_PATTERN.sub("_", text).strip("_") or default


def format_number(value: object, digits: int = 2) -> str:
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


def quality_text(rom_flexion: float | None) -> str:
    if rom_flexion is None:
        return "未能计算出有效 ROM，建议检查关键点可见度后重录。"
    if rom_flexion < 15:
        return "ROM 偏小，建议重新录制一版动作更标准的结果。"
    if rom_flexion < 30:
        return "ROM 勉强可用，如需做模板建议再录一版更稳定的结果。"
    return "ROM 较明显，这份结果可以作为后续模板参考。"


def record_role_from_prescription(prescription: dict[str, object]) -> str:
    runtime_meta = prescription.get("runtime_meta", {})
    if not isinstance(runtime_meta, dict):
        runtime_meta = {}
    role = str(prescription.get("record_role") or runtime_meta.get("record_role") or "doctor_template").strip()
    return role if role in ROLE_DIRS else "doctor_template"


def role_storage_info(record_role: str) -> tuple[str, str, str]:
    return ROLE_DIRS.get(record_role, ROLE_DIRS["doctor_template"])


def render_summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 处方录制摘要",
            "",
            f"- 保存时间：{summary['saved_at']}",
            f"- 记录类型：`{summary['record_role']}`（{summary['role_label']}）",
            f"- 对象编号：`{summary['patient_id']}`",
            f"- 动作名称：`{summary['action_name']}`",
            f"- 侧别模式：`{summary['side_mode']}`",
            f"- 有效帧数：`{summary['frame_count']}`",
            f"- 无效帧数：`{summary['invalid_frame_count']}`",
            f"- 动作时长：`{format_number(summary['duration_seconds'])}` 秒",
            f"- 最小屈曲角：`{format_number(summary['min_knee_flexion_angle'])}` 度",
            f"- 最大屈曲角：`{format_number(summary['max_knee_flexion_angle'])}` 度",
            f"- ROM：`{format_number(summary['rom_flexion'])}` 度",
            f"- 结果格式：`{summary['result_format']}`",
            f"- 来源板子：`{summary['board_ip']}:{summary['board_port']}`",
            f"- 保存来源：`{summary['source']}`",
            f"- {summary['file_label']}：`{summary['saved_path']}`",
            f"- 摘要文件：`{summary['summary_path']}`",
            "",
            "## 结果判断",
            "",
            str(summary["quality_text"]),
            "",
            "## 说明",
            "",
            "- 医生模板 JSON 保存在 `docs/doctor_templates/`，患者动作 JSON 保存在 `docs/patient_attempts/`。",
            "- `docs/results/` 是旧版本兼容目录，新保存结果不会再默认混放到这里。",
            "- 日常查看请优先打开这份摘要，不需要直接翻长 JSON。",
        ]
    ) + "\n"


def append_results_log(summary: dict[str, object], results_log_path: Path) -> None:
    line = (
        f"- 保存时间：{summary['saved_at']} | 类型：`{summary['record_role']}` | 对象：`{summary['patient_id']}` | 动作：`{summary['action_name']}` | "
        f"帧数：`{summary['frame_count']}` | ROM：`{format_number(summary['rom_flexion'])}` | "
        f"文件：`{Path(str(summary['saved_path'])).name}` | 摘要：`{Path(str(summary['summary_path'])).name}` | "
        f"板子：`{summary['board_ip']}` | 来源：`{summary['source']}`\n"
    )
    with results_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def save_prescription_artifacts(
    prescription: dict[str, object],
    *,
    board_ip: str = "unknown",
    board_port: str = "unknown",
    source: str = "unknown",
    docs_dir: Path | None = None,
) -> dict[str, object]:
    _, results_dir, summaries_dir, results_log_path = ensure_dirs(docs_dir)

    record_role = record_role_from_prescription(prescription)
    role_dir_name, role_label, file_label = role_storage_info(record_role)
    output_dir = results_dir.parent / role_dir_name
    patient_id = safe_name(str(prescription.get("patient_id", "")), "patient")
    action_name = safe_name(str(prescription.get("action_name", "")), "action")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{record_role}_{patient_id}_{action_name}_{timestamp}.json"
    summary_name = f"{record_role}_{patient_id}_{action_name}_{timestamp}_summary.md"
    output_path = output_dir / file_name
    summary_path = summaries_dir / summary_name

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(prescription, handle, ensure_ascii=False, indent=2)

    clinical = prescription.get("clinical_baseline", {})
    if not isinstance(clinical, dict):
        clinical = {}

    runtime_meta = prescription.get("runtime_meta", {})
    if not isinstance(runtime_meta, dict):
        runtime_meta = {}

    summary = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "saved_path": str(output_path),
        "summary_name": summary_name,
        "summary_path": str(summary_path),
        "record_role": record_role,
        "role_label": role_label,
        "file_label": file_label,
        "patient_id": prescription.get("patient_id"),
        "action_name": prescription.get("action_name"),
        "frame_count": clinical.get("frame_count"),
        "duration_seconds": clinical.get("duration_seconds"),
        "min_knee_flexion_angle": clinical.get("min_knee_flexion_angle"),
        "max_knee_flexion_angle": clinical.get("max_knee_flexion_angle"),
        "rom_flexion": clinical.get("rom_flexion"),
        "invalid_frame_count": runtime_meta.get("invalid_frame_count"),
        "board_ip": board_ip,
        "board_port": board_port,
        "source": source,
        "side_mode": runtime_meta.get("side_mode", "unknown"),
        "result_format": runtime_meta.get("result_format", "unknown"),
    }
    summary["quality_text"] = quality_text(summary["rom_flexion"])

    summary_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    append_results_log(summary, results_log_path)

    return {
        "saved_path": str(output_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }
