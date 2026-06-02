from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional rendering dependency
    Image = None
    ImageDraw = None
    ImageFont = None


KEYFRAME_PREFIX = "evaluate/reports/keyframes/"
SAFE_IMAGE_SUFFIXES = {".jpg", ".jpeg"}

ACTION_RULES: dict[str, dict[str, Any]] = {
    "seated_knee_extension": {
        "benefit_parts": ["股四头肌", "膝关节伸展控制"],
        "target_joint": "膝关节",
        "goal": "膝盖逐步伸直到位",
    },
    "standing_hamstring_curl": {
        "benefit_parts": ["腘绳肌", "膝关节屈曲控制"],
        "target_joint": "膝关节",
        "goal": "小腿稳定向后弯曲",
    },
    "seated_knee_raise": {
        "benefit_parts": ["髋屈肌", "大腿抬高控制"],
        "target_joint": "髋关节 / 膝关节",
        "goal": "膝盖或大腿稳定抬高",
    },
}

ERROR_NEXT_STEPS = {
    "OK": "保持当前节奏，继续按医生模板训练。",
    "ROM_LOW": "下一组先放慢速度，再逐步增加动作幅度。",
    "TUT_LOW": "到达目标位置后先稳住，再缓慢返回。",
    "TOO_FAST": "放慢动作速度，避免借力或突然回落。",
    "SHAPE_BAD": "先控制动作轨迹，减少晃动后再追求幅度。",
}


def is_safe_keyframe_path(value: object) -> bool:
    text = str(value or "").replace("\\", "/").strip()
    if not text or text.startswith("/") or text.startswith("\\"):
        return False
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return False
    return path.as_posix().startswith(KEYFRAME_PREFIX) and path.suffix.lower() in SAFE_IMAGE_SUFFIXES


def resolve_keyframe_path(project_root: Path, value: object) -> Path | None:
    text = str(value or "").replace("\\", "/").strip()
    if not is_safe_keyframe_path(text):
        return None
    resolved = (project_root / text).resolve()
    base = (project_root / KEYFRAME_PREFIX).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def project_relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_metric_cards(report: dict[str, Any]) -> list[dict[str, Any]]:
    action_id = _action_id(report)
    rules = ACTION_RULES.get(action_id, {})
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    error_code = str(errors.get("primary_error") or "OK")
    card = {
        "action_id": action_id,
        "action_name": str(report.get("action_name") or rules.get("goal") or action_id or "本次动作"),
        "primary_error": error_code,
        "benefit_parts": list(rules.get("benefit_parts") or ["目标关节周围肌群"]),
        "target_joint": rules.get("target_joint") or "目标关节",
        "goal": rules.get("goal") or "按医生模板稳定完成动作",
        "metrics": _metric_items(metrics),
        "calorie_estimate": {
            "value_kcal": None,
            "text": "热量粗略估计，仅供参考；报告缺少体重、心率和真实运动强度，暂不输出精确数值。",
        },
        "risk_note": _risk_note(error_code),
        "next_step": ERROR_NEXT_STEPS.get(error_code, "优先修正报告中的主要问题，不要自行增加训练量。"),
    }
    return [card]


def build_keyframe_notes(report: dict[str, Any]) -> list[str]:
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    notes: list[str] = []
    for item in keyframes:
        if not isinstance(item, dict):
            continue
        rep = item.get("rep_index") or "-"
        metric = item.get("primary_metric") or "primary signal"
        unit = item.get("primary_metric_unit") or ""
        value = _fmt(item.get("signal_value"), 2)
        notes.append(
            f"第 {rep} 次动作保存了 best_peak 关键帧；{metric} 峰值约 {value}{unit}。图片只用于辅助观察，精确指标以 report 为准。"
        )
    if not notes:
        notes.append("本次报告没有可用关键帧；AI 图文建议将仅基于 report 指标生成。")
    return notes


def attach_keyframe_urls(report: dict[str, Any], url_prefix: str = "/report-images/") -> list[dict[str, Any]]:
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    payload: list[dict[str, Any]] = []
    for item in keyframes:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").replace("\\", "/")
        if not is_safe_keyframe_path(image_path):
            continue
        row = dict(item)
        row["url"] = f"{url_prefix}{image_path}"
        payload.append(row)
    return payload


def render_report_images(project_root: Path, report: dict[str, Any], metric_cards: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    first_keyframe = next((item for item in keyframes if isinstance(item, dict)), None)
    if not first_keyframe:
        return result

    raw_path = resolve_keyframe_path(project_root, first_keyframe.get("image_path"))
    if raw_path is None or not raw_path.exists():
        result["errors"].append("keyframe_image_missing")
        return result

    try:
        raw_rendered = _render_raw_keyframe(project_root, raw_path, report, first_keyframe)
        result["raw_keyframe_image"] = _image_payload(project_root, raw_rendered or raw_path)
    except Exception as exc:
        result["errors"].append(f"raw_keyframe_render_failed: {exc}")
        result["raw_keyframe_image"] = _image_payload(project_root, raw_path)

    try:
        if metric_cards:
            card_path = _render_metric_card(project_root, raw_path.parent, metric_cards[0], first_keyframe)
            result["metric_card_image"] = _image_payload(project_root, card_path)
    except Exception as exc:
        result["errors"].append(f"metric_card_render_failed: {exc}")

    try:
        raw_item = result.get("raw_keyframe_image")
        card_item = result.get("metric_card_image")
        if isinstance(raw_item, dict) and isinstance(card_item, dict):
            comparison = _render_comparison(
                project_root,
                raw_path.parent,
                Path(str(raw_item["path"])),
                Path(str(card_item["path"])),
                first_keyframe,
            )
            result["comparison_image"] = _image_payload(project_root, comparison)
    except Exception as exc:
        result["errors"].append(f"comparison_render_failed: {exc}")

    if not result["errors"]:
        result.pop("errors", None)
    return result


def _render_raw_keyframe(project_root: Path, source: Path, report: dict[str, Any], keyframe: dict[str, Any]) -> Path | None:
    if Image is None or ImageDraw is None:
        return None
    image = Image.open(source).convert("RGB")
    image.thumbnail((640, 640))
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_title = _font(24, bold=True)
    font_body = _font(18)
    overlay_h = 86
    draw.rectangle((0, canvas.height - overlay_h, canvas.width, canvas.height), fill=(5, 12, 24))
    action = str(report.get("action_name") or keyframe.get("action_name") or "Rehab action")
    rep = keyframe.get("rep_index") or "-"
    metric = keyframe.get("primary_metric") or "primary signal"
    value = _fmt(keyframe.get("signal_value"), 2)
    unit = str(keyframe.get("primary_metric_unit") or "")
    draw.text((18, canvas.height - 74), f"{action}  Rep {rep}", fill=(237, 245, 255), font=font_title)
    draw.text((18, canvas.height - 36), f"{metric}: {value}{unit}  best_peak", fill=(150, 210, 255), font=font_body)
    out = source.with_name(source.stem + "_raw_keyframe.jpg")
    canvas.save(out, "JPEG", quality=88, optimize=True)
    return out


def _render_metric_card(project_root: Path, out_dir: Path, card: dict[str, Any], keyframe: dict[str, Any]) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL unavailable")
    width, height = 900, 520
    image = Image.new("RGB", (width, height), (7, 17, 31))
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    label_font = _font(18)
    value_font = _font(24, bold=True)
    small_font = _font(16)
    draw.rectangle((0, 0, width, height), fill=(7, 17, 31))
    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=18, fill=(14, 27, 48), outline=(68, 117, 166), width=2)
    draw.text((54, 52), str(card.get("action_name") or "Rehab Report"), fill=(237, 245, 255), font=title_font)
    draw.text((54, 98), f"Rep {keyframe.get('rep_index') or '-'}  |  {card.get('goal')}", fill=(150, 190, 220), font=label_font)

    x, y = 54, 145
    for metric in list(card.get("metrics") or [])[:4]:
        draw.rounded_rectangle((x, y, x + 400, y + 76), radius=12, fill=(20, 39, 68), outline=(42, 74, 110))
        draw.text((x + 18, y + 12), str(metric.get("label") or "-"), fill=(150, 190, 220), font=small_font)
        draw.text((x + 18, y + 38), str(metric.get("value") or "-"), fill=(237, 245, 255), font=value_font)
        x += 420
        if x > 500:
            x, y = 54, y + 92

    lower_y = 338
    draw.text((54, lower_y), "受益部位 / 目标关节", fill=(85, 214, 255), font=label_font)
    draw.text((54, lower_y + 30), "、".join(card.get("benefit_parts") or []) + f"；{card.get('target_joint')}", fill=(237, 245, 255), font=small_font)
    draw.text((54, lower_y + 68), "热量", fill=(255, 207, 112), font=label_font)
    draw.text((54, lower_y + 98), str((card.get("calorie_estimate") or {}).get("text") or ""), fill=(237, 220, 180), font=small_font)
    draw.text((54, lower_y + 138), "提醒", fill=(255, 118, 118), font=label_font)
    draw.text((112, lower_y + 138), str(card.get("risk_note") or ""), fill=(245, 205, 205), font=small_font)
    draw.text((54, lower_y + 170), "下一步", fill=(83, 242, 176), font=label_font)
    draw.text((128, lower_y + 170), str(card.get("next_step") or ""), fill=(206, 255, 230), font=small_font)
    out = out_dir / f"metric_card_{card.get('action_id') or 'action'}_rep{keyframe.get('rep_index') or 1}.jpg"
    image.save(out, "JPEG", quality=90, optimize=True)
    return out


def _render_comparison(project_root: Path, out_dir: Path, raw_rel: Path, card_rel: Path, keyframe: dict[str, Any]) -> Path:
    if Image is None:
        raise RuntimeError("PIL unavailable")
    raw = Image.open(project_root / raw_rel).convert("RGB")
    card = Image.open(project_root / card_rel).convert("RGB")
    raw.thumbnail((640, 520))
    card.thumbnail((640, 520))
    height = max(raw.height, card.height)
    canvas = Image.new("RGB", (raw.width + card.width + 24, height), (7, 17, 31))
    canvas.paste(raw, (0, (height - raw.height) // 2))
    canvas.paste(card, (raw.width + 24, (height - card.height) // 2))
    out = out_dir / f"comparison_rep{keyframe.get('rep_index') or 1}.jpg"
    canvas.save(out, "JPEG", quality=90, optimize=True)
    return out


def _image_payload(project_root: Path, path: Path) -> dict[str, str]:
    rel = project_relative(project_root, path)
    return {"path": rel, "url": f"/report-images/{rel}"}


def _metric_items(metrics: dict[str, Any]) -> list[dict[str, str]]:
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    speed = metrics.get("speed") if isinstance(metrics.get("speed"), dict) else {}
    dtw = metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {}
    return [
        {"label": "ROM", "value": f"{_fmt(rom.get('actual'))} / {_fmt(rom.get('target'))}"},
        {"label": "TUT", "value": f"{_fmt(tut.get('actual'))}s / {_fmt(tut.get('target'))}s"},
        {"label": "速度比", "value": _fmt(speed.get("ratio"), 2)},
        {"label": "DTW", "value": _fmt(dtw.get("normalized_distance"), 2)},
    ]


def _risk_note(error_code: str) -> str:
    if error_code == "OK":
        return "未见主要错误；如出现疼痛、肿胀、麻木、头晕或站立不稳，请停止训练并联系医生或康复师。"
    return "发现动作风险提示；请按反馈降低速度或幅度，不要自行调整治疗方案。"


def _action_id(report: dict[str, Any]) -> str:
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    for item in keyframes:
        if isinstance(item, dict) and item.get("action_id"):
            return str(item.get("action_id"))
    config_file = str(report.get("config_file") or "")
    if config_file:
        return Path(config_file).stem
    return str(report.get("action_id") or "")


def _font(size: int, *, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fmt(value: object, digits: int = 1) -> str:
    number = _as_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None
