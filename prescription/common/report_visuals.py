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
CALORIE_WEIGHT_KG = 75.0
DEFAULT_REHAB_MET = 2.3
MET_BY_ACTION = {
    "seated_knee_extension": 2.3,
    "standing_hamstring_curl": 2.3,
    "seated_knee_raise": 2.3,
}
CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
]

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
    metrics = report.get("report_card_metrics") if isinstance(report.get("report_card_metrics"), dict) else report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
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
        "sample_label": _sample_label(metrics),
        "calorie_estimate": estimate_calories(report),
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
    valid_keyframes = [item for item in keyframes if isinstance(item, dict)]
    if not valid_keyframes:
        return result

    rendered_items: list[dict[str, Any]] = []
    card = metric_cards[0] if metric_cards else {}
    for keyframe in valid_keyframes:
        raw_path = resolve_keyframe_path(project_root, keyframe.get("image_path"))
        if raw_path is None or not raw_path.exists():
            result["errors"].append(f"keyframe_image_missing: rep{keyframe.get('rep_index') or '-'}")
            continue

        item: dict[str, Any] = {
            "rep_index": keyframe.get("rep_index"),
            "action_id": keyframe.get("action_id") or _action_id(report),
            "source_keyframe_path": project_relative(project_root, raw_path),
        }
        try:
            raw_rendered = _render_raw_keyframe(project_root, raw_path, report, keyframe)
            raw_item = _image_payload(project_root, raw_rendered or raw_path)
        except Exception as exc:
            result["errors"].append(f"raw_keyframe_render_failed: {exc}")
            raw_item = _image_payload(project_root, raw_path)

        try:
            if card:
                card_path = _render_metric_card(project_root, raw_path.parent, card, keyframe)
                card_item = _image_payload(project_root, card_path)
            else:
                card_item = None
        except Exception as exc:
            result["errors"].append(f"metric_card_render_failed: {exc}")
            card_item = None

        try:
            if isinstance(raw_item, dict) and isinstance(card_item, dict):
                comparison = _render_comparison(
                    project_root,
                    raw_path.parent,
                    Path(str(raw_item["path"])),
                    Path(str(card_item["path"])),
                    keyframe,
                )
                item["comparison_image"] = _image_payload(project_root, comparison)
            elif isinstance(raw_item, dict):
                item["comparison_image"] = raw_item
        except Exception as exc:
            result["errors"].append(f"comparison_render_failed: {exc}")

        rendered_items.append(item)

    if rendered_items:
        result["items"] = rendered_items
        first = rendered_items[0]
        if "comparison_image" in first:
            result["comparison_image"] = first["comparison_image"]

    if not result["errors"]:
        result.pop("errors", None)
    return result


def _render_raw_keyframe(project_root: Path, source: Path, report: dict[str, Any], keyframe: dict[str, Any]) -> Path | None:
    if Image is None:
        return None
    image = Image.open(source).convert("RGB")
    _draw_keyframe_skeleton(image, keyframe)
    canvas = _fit_image(image, (980, 760), (5, 12, 24))
    out = source.with_name(source.stem + "_raw_keyframe.jpg")
    canvas.save(out, "JPEG", quality=88, optimize=True)
    return out


def _render_metric_card(project_root: Path, out_dir: Path, card: dict[str, Any], keyframe: dict[str, Any]) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL unavailable")
    width, height = 980, 760
    has_cjk_font = _cjk_font_available()
    image = Image.new("RGB", (width, height), (7, 17, 31))
    draw = ImageDraw.Draw(image)
    title_font = _font(38, bold=True)
    label_font = _font(20)
    value_font = _font(28, bold=True)
    small_font = _font(18)
    draw.rectangle((0, 0, width, height), fill=(7, 17, 31))
    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=22, fill=(14, 27, 48), outline=(68, 117, 166), width=2)
    action_title = str(card.get("action_name") or card.get("action_id") or "Rehab Report") if has_cjk_font else str(card.get("action_id") or "Rehab Report")
    goal_text = str(card.get("goal") or "") if has_cjk_font else "Controlled rehab movement"
    draw.text((54, 52), action_title, fill=(237, 245, 255), font=title_font)
    sample_label = str(card.get("sample_label") or f"Rep {keyframe.get('rep_index') or '-'}")
    draw.text((54, 98), f"{sample_label}  |  {goal_text}", fill=(150, 190, 220), font=label_font)

    x, y = 54, 150
    for metric in list(card.get("metrics") or [])[:4]:
        draw.rounded_rectangle((x, y, x + 424, y + 92), radius=14, fill=(20, 39, 68), outline=(42, 74, 110))
        metric_label = _ascii_metric_label(metric.get("label")) if not has_cjk_font else str(metric.get("label") or "-")
        draw.text((x + 20, y + 14), metric_label, fill=(150, 190, 220), font=small_font)
        draw.text((x + 20, y + 48), str(metric.get("value") or "-"), fill=(237, 245, 255), font=value_font)
        x += 444
        if x > 520:
            x, y = 54, y + 108

    lower_y = 400
    calorie = card.get("calorie_estimate") if isinstance(card.get("calorie_estimate"), dict) else {}
    if has_cjk_font:
        joint_label = "受益部位 / 目标关节"
        joint_text = "、".join(card.get("benefit_parts") or []) + f"；{card.get('target_joint')}"
        calorie_label = "热量"
        calorie_text = str(calorie.get("text") or "")
        risk_label = "提醒"
        risk_text = str(card.get("risk_note") or "")
        next_label = "下一步"
        next_text = str(card.get("next_step") or "")
    else:
        joint_label = "Target"
        joint_text = str(card.get("action_id") or "rehab action")
        calorie_label = "Calories"
        calorie_text = str(calorie.get("ascii_text") or calorie.get("text") or "")
        risk_label = "Safety"
        risk_text = "Stop and contact a clinician if pain, swelling, numbness, dizziness, or instability occurs."
        next_label = "Next"
        next_text = "Keep the movement slow and controlled; use the report metrics as the source of truth."
    draw.text((54, lower_y), joint_label, fill=(85, 214, 255), font=label_font)
    draw.text((54, lower_y + 36), _fit_text(joint_text, 86), fill=(237, 245, 255), font=small_font)
    draw.text((54, lower_y + 100), calorie_label, fill=(255, 207, 112), font=label_font)
    draw.text((54, lower_y + 136), _fit_text(calorie_text, 96), fill=(237, 220, 180), font=small_font)
    draw.text((54, lower_y + 204), risk_label, fill=(255, 118, 118), font=label_font)
    draw.text((144, lower_y + 204), _fit_text(risk_text, 84), fill=(245, 205, 205), font=small_font)
    draw.text((54, lower_y + 272), next_label, fill=(83, 242, 176), font=label_font)
    draw.text((144, lower_y + 272), _fit_text(next_text, 84), fill=(206, 255, 230), font=small_font)
    out = out_dir / f"{_keyframe_output_stem(keyframe)}_metric_card.jpg"
    image.save(out, "JPEG", quality=90, optimize=True)
    return out


def _render_comparison(project_root: Path, out_dir: Path, raw_rel: Path, card_rel: Path, keyframe: dict[str, Any]) -> Path:
    if Image is None:
        raise RuntimeError("PIL unavailable")
    raw = Image.open(project_root / raw_rel).convert("RGB")
    card = Image.open(project_root / card_rel).convert("RGB")
    raw_panel = _fit_image(raw, (980, 760), (5, 12, 24))
    card_panel = _fit_image(card, (980, 760), (7, 17, 31))
    canvas = Image.new("RGB", (980, 1560), (7, 17, 31))
    canvas.paste(raw_panel, (0, 0))
    canvas.paste(card_panel, (0, 800))
    out = out_dir / f"{_safe_output_token(raw_rel.stem)}_comparison.jpg"
    canvas.save(out, "JPEG", quality=90, optimize=True)
    return out


def _draw_keyframe_skeleton(image, keyframe: dict[str, Any]) -> None:
    if ImageDraw is None:
        return
    keypoints = keyframe.get("rehab_keypoints") if isinstance(keyframe.get("rehab_keypoints"), dict) else {}
    if not keypoints:
        return
    side = str(keyframe.get("selected_side") or "left").lower()
    if side not in {"left", "right"}:
        side = "left"
    width, height = image.size
    draw = ImageDraw.Draw(image)
    thickness = max(5, min(width, height) // 90)
    radius = max(7, min(width, height) // 70)
    names = [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
    pixels = [_keypoint_pixel(keypoints.get(name), width, height) for name in names]
    for start, end in zip(pixels, pixels[1:]):
        if start is None or end is None:
            continue
        draw.line([start, end], fill=(5, 10, 18), width=thickness + 4)
        draw.line([start, end], fill=(255, 210, 60), width=thickness)
    for pixel in pixels:
        if pixel is None:
            continue
        x, y = pixel
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=(5, 10, 18))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(70, 220, 255))


def _keypoint_pixel(point: object, width: int, height: int) -> tuple[int, int] | None:
    if not isinstance(point, dict):
        return None
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    visibility = _as_float(point.get("visibility"))
    if x is None or y is None or (visibility is not None and visibility < 0.01):
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= width
        y *= height
    return (
        int(round(max(0.0, min(float(width - 1), x)))),
        int(round(max(0.0, min(float(height - 1), y)))),
    )


def _keyframe_output_stem(keyframe: dict[str, Any]) -> str:
    image_path = str(keyframe.get("image_path") or "")
    stem = Path(image_path.replace("\\", "/")).stem if image_path else ""
    if not stem:
        stem = f"{keyframe.get('action_id') or 'action'}_rep{keyframe.get('rep_index') or 1}"
    return _safe_output_token(stem)


def _safe_output_token(value: object) -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return safe.strip("_") or "image"


def _image_payload(project_root: Path, path: Path) -> dict[str, str]:
    rel = project_relative(project_root, path)
    return {"path": rel, "url": f"/report-images/{rel}"}



def _sample_label(metrics: dict[str, Any]) -> str:
    source = str(metrics.get("source") or "full_session")
    if source == "best_correct":
        rep = metrics.get("rep_index") or metrics.get("attempt_index") or "-"
        return f"第 {rep} 次正确动作"
    if source == "representative_wrong":
        attempt = metrics.get("attempt_index") or "-"
        return f"第 {attempt} 次错误动作"
    return "整段汇总"

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


def estimate_calories(report: dict[str, Any]) -> dict[str, Any]:
    action_id = _action_id(report)
    met = _as_float(((report.get("runtime_meta") if isinstance(report.get("runtime_meta"), dict) else {}) or {}).get("met"))
    if met is None:
        met = MET_BY_ACTION.get(action_id, DEFAULT_REHAB_MET)
    duration_seconds = _report_duration_seconds(report)
    minutes = duration_seconds / 60.0
    value = met * 3.5 * CALORIE_WEIGHT_KG * minutes / 200.0
    value = max(0.0, value)
    return {
        "value_kcal": round(value, 2),
        "text": f"约 {value:.2f} kcal（MET {met:.1f} × 3.5 × 75kg × {minutes:.2f}min / 200）",
        "ascii_text": f"{value:.2f} kcal (MET {met:.1f} * 3.5 * 75kg * {minutes:.2f}min / 200)",
        "formula": "kcal = MET × 3.5 × body_weight_kg × minutes / 200",
        "met": met,
        "weight_kg": CALORIE_WEIGHT_KG,
        "duration_seconds": round(duration_seconds, 2),
    }


def _report_duration_seconds(report: dict[str, Any]) -> float:
    clinical = report.get("clinical_baseline") if isinstance(report.get("clinical_baseline"), dict) else {}
    duration = _as_float(clinical.get("duration_seconds"))
    if duration is not None and duration > 0:
        return duration
    metrics = report.get("report_card_metrics") if isinstance(report.get("report_card_metrics"), dict) else report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    duration = _as_float(tut.get("actual"))
    if duration is not None and duration > 0:
        return duration
    return 0.0


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


def _fit_image(image, size: tuple[int, int], background: tuple[int, int, int]):
    target_w, target_h = size
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", size, background)
    scale = min(target_w / src_w, target_h / src_h)
    new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    resized = image.resize(new_size, _resample_filter())
    canvas = Image.new("RGB", size, background)
    canvas.paste(resized, ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2))
    return canvas


def _resample_filter():
    resampling = getattr(Image, "Resampling", None)
    return getattr(resampling, "LANCZOS", Image.LANCZOS if hasattr(Image, "LANCZOS") else 1)


def _cjk_font_available() -> bool:
    return any(Path(item).exists() for item in CJK_FONT_CANDIDATES)


def _ascii_metric_label(label: object) -> str:
    text = str(label or "")
    mapping = {
        "速度比": "Speed ratio",
    }
    return mapping.get(text, text if text.isascii() else "Metric")


def _fit_text(text: object, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _font(size: int, *, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        *CJK_FONT_CANDIDATES,
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
