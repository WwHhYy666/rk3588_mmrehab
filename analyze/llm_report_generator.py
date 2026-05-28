from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def build_summary_bundle(report: dict[str, Any]) -> dict[str, Any]:
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    primary_error = str(errors.get("primary_error") or "OK")

    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    speed = metrics.get("speed") if isinstance(metrics.get("speed"), dict) else {}
    dtw = metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {}

    risk_flags: list[str] = []
    if primary_error != "OK":
        risk_flags.append(primary_error)
    if (_as_float(speed.get("ratio")) or 0.0) > 1.35:
        risk_flags.append("SPEED_CONTROL")
    if (_as_float(dtw.get("normalized_distance")) or 0.0) > 0.28:
        risk_flags.append("TRAJECTORY_DEVIATION")

    action_name = str(meta.get("action_name") or meta.get("action_id") or "动作")
    patient_id = str(meta.get("patient_id") or "patient")
    rom_actual = _as_float(rom.get("actual")) or 0.0
    rom_target = _as_float(rom.get("target")) or 0.0
    tut_ratio = _as_float(tut.get("ratio")) or 0.0
    speed_ratio = _as_float(speed.get("ratio")) or 0.0
    dtw_norm = _as_float(dtw.get("normalized_distance")) or 0.0

    if primary_error == "ROM_LOW":
        next_step = "优先提高动作幅度，再保持当前节奏。"
    elif primary_error == "TUT_LOW":
        next_step = "优先延长顶点保持时间。"
    elif primary_error == "TOO_FAST":
        next_step = "优先放慢节奏，避免借力完成。"
    elif primary_error == "SHAPE_BAD":
        next_step = "优先按照标准模板调整动作轨迹。"
    else:
        next_step = "维持当前质量，可进入下一组或增加难度。"

    doctor_summary = (
        f"{patient_id} 的 {action_name} 评估完成。"
        f"ROM {rom_actual:.2f}/{rom_target:.2f}，"
        f"TUT 比例 {tut_ratio:.2f}，"
        f"速度比例 {speed_ratio:.2f}，"
        f"轨迹距离 {dtw_norm:.2f}。"
        f"主错误码为 {primary_error}，建议：{next_step}"
    )
    patient_summary = (
        f"{action_name} 本次训练"
        f"{'完成得不错' if primary_error == 'OK' else '还有提升空间'}。"
        f"{next_step}"
    )
    return {
        "backend": "template",
        "doctor_summary": doctor_summary,
        "patient_summary": patient_summary,
        "risk_flags": risk_flags,
        "next_step": next_step,
    }


def write_summary_bundle(report: dict[str, Any], report_path: Path | None) -> dict[str, Any]:
    bundle = build_summary_bundle(report)
    if report_path is None:
        return bundle

    base = report_path.with_suffix("")
    summary_json = base.with_name(base.name + "_summary.json")
    doctor_txt = base.with_name(base.name + "_doctor_summary.txt")
    patient_txt = base.with_name(base.name + "_patient_summary.txt")

    summary_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    doctor_txt.write_text(str(bundle.get("doctor_summary") or ""), encoding="utf-8")
    patient_txt.write_text(str(bundle.get("patient_summary") or ""), encoding="utf-8")

    bundle["summary_json_file"] = str(summary_json)
    bundle["doctor_summary_file"] = str(doctor_txt)
    bundle["patient_summary_file"] = str(patient_txt)
    return bundle

