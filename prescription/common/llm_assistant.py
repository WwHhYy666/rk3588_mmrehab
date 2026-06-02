from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from prescription.common.report_visuals import build_keyframe_notes


DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4v-flash"
DEFAULT_PROVIDER = "echo"
MAX_QUESTION_CHARS = 300
MAX_SPOKEN_CHARS = 120

SAFETY_SYSTEM_PROMPT = """你是骨科居家康复训练辅助解释助手。
你只能根据系统提供的训练报告、动作名称、完成次数、ROM、TUT、DTW、速度、错误类型、规则反馈等信息进行解释。
你不能诊断疾病，不能替代医生。
你不能建议患者自行加大训练量、停止治疗、改变药物或改变医生处方。
如果报告信息不足，应明确说明“不确定”，不要编造。
如果患者提到疼痛、肿胀、麻木、头晕、跌倒风险、伤口异常等，应建议停止训练并联系医生或康复师。
患者版回答要短、温和、适合朗读。
医生版回答可以更结构化，保留指标、异常项和改进建议。
热量估算只能作为粗略参考，因为系统缺少体重、心率和真实运动强度等信息。"""

SUMMARY_PROMPT = """请根据下面这份康复训练评估报告生成训练后解释。

要求：
1. 只基于报告内容回答，不要编造病情。
2. 只输出一个 JSON 对象，不要解释，不要 Markdown，不要代码块。
3. JSON 字段必须包含：
   patient_summary: 患者能听懂的中文短总结，1 句话，45 个中文字符以内
   doctor_summary: 医生可看的短字符串，120 个中文字符以内，禁止输出对象或数组
   next_steps: 字符串数组，最多 2 条
   risk_notes: 字符串数组，最多 1 条
   calorie_estimate: 对象，包含 text 和 value_kcal；无法估计时 value_kcal 为 null
   spoken_text: 适合 TTS 朗读的短文本，120 个中文字符以内
4. 禁止复制完整 report、metric 对象、JSON 输入原文或英文字段说明。

评估报告摘要：
{report_summary}"""

SAFETY_SYSTEM_PROMPT += """
如果提供关键帧图片，图片只作为辅助观察；精确角度、ROM、TUT、DTW、速度、热量必须以 report 为准。
不要从单张图片编造准确数值；单张图片不能判断持续时间、速度和完整动作轨迹。
如果图片不清晰、人体不完整或关键部位遮挡，要说明无法可靠观察。
不得诊断疾病、建议用药或调整治疗方案。"""

SUMMARY_PROMPT += """

关键帧补充要求：
1. 如提供图片，只描述可见姿态线索，不要从图片编造准确数值。
2. 输出 JSON 可包含 keyframe_notes 字符串数组，最多 2 条。
3. 精确角度、ROM、TUT、DTW、速度和热量以 report 为准。"""

QUESTION_PROMPT = """患者基于本次康复训练报告提出了问题。

回答要求：
1. 只能基于本次训练报告回答。
2. 如果问题涉及诊断、用药、是否加量、是否停药、是否手术、是否痊愈，不能直接判断，必须建议咨询医生或康复师。
3. 如果问题涉及疼痛、肿胀、麻木、头晕、跌倒风险、伤口异常，应提醒停止训练并联系医生或康复师。
4. 只输出一个 JSON 对象，不要解释，不要 Markdown，不要代码块。
5. JSON 字段必须包含：
   answer: 中文回答，短、温和、适合患者阅读，80 个中文字符以内
   spoken_text: 适合 TTS 朗读的短回答，120 个中文字符以内
6. 禁止复制完整 report、metric 对象、JSON 输入原文或英文字段说明。

患者问题：
{question}

训练报告摘要：
{report_summary}"""

MEDICAL_RISK_WORDS = (
    "疼",
    "疼痛",
    "肿",
    "肿胀",
    "麻",
    "麻木",
    "头晕",
    "摔",
    "跌倒",
    "伤口",
    "出血",
)

MEDICAL_DECISION_WORDS = (
    "诊断",
    "什么病",
    "停药",
    "吃药",
    "用药",
    "加量",
    "加训练",
    "加大",
    "手术",
    "痊愈",
    "康复了吗",
    "要不要去医院",
)


@dataclass
class LLMRuntimeState:
    last_error: str | None = None
    last_success_at: str | None = None
    last_latency_ms: int | None = None


_STATE = LLMRuntimeState()


def get_llm_status() -> dict[str, Any]:
    settings = _settings()
    return {
        "enabled": True,
        "provider": settings["provider"],
        "model": settings["model"],
        "api_key_configured": bool(settings["api_key"]),
        "endpoint_configured": bool(settings["endpoint"]),
        "last_error": _STATE.last_error,
        "last_success_at": _STATE.last_success_at,
        "last_latency_ms": _STATE.last_latency_ms,
    }


def summarize_report(
    report: dict[str, Any],
    audience: str = "both",
    include_calorie: bool = True,
    include_keyframes: bool = False,
    keyframe_frame_b64: str | None = None,
) -> dict[str, Any]:
    settings = _settings()
    started = time.monotonic()
    provider = settings["provider"]
    try:
        if provider == "echo":
            result = _echo_summary(report, include_calorie, include_keyframes)
        elif provider == "glm4v_api":
            result = _glm_summary(report, settings, audience, include_calorie, include_keyframes, keyframe_frame_b64)
        else:
            return _failure("bad_request", f"不支持的 LLM provider: {provider}", provider, settings["model"])
        return _success(result, provider, settings["model"], started)
    except TimeoutError as exc:
        return _failure("timeout", "GLM API 请求超时，请稍后重试。", provider, settings["model"], exc)
    except urllib.error.URLError as exc:
        error_code, message = _network_failure_message(exc)
        return _failure(error_code, message, provider, settings["model"], exc)
    except ValueError as exc:
        return _failure("parse_error", "GLM 返回格式暂时无法解析，请稍后重试。", provider, settings["model"], exc)
    except RuntimeError as exc:
        return _failure("provider_error", "GLM API 调用失败，请检查配置或稍后重试。", provider, settings["model"], exc)
    except Exception as exc:  # pragma: no cover - last-resort guard for web service stability
        return _failure("unknown", "AI 建议生成失败，但训练主流程不受影响。", provider, settings["model"], exc)


def answer_question(
    report: dict[str, Any],
    question: str,
    frame_b64: str | None = None,
) -> dict[str, Any]:
    settings = _settings()
    started = time.monotonic()
    provider = settings["provider"]
    question = str(question or "").strip()
    if not question:
        return _failure("bad_request", "请输入要咨询的问题。", provider, settings["model"])
    if len(question) > MAX_QUESTION_CHARS:
        return _failure("bad_request", f"问题太长，请控制在 {MAX_QUESTION_CHARS} 字以内。", provider, settings["model"])
    safety = _safety_answer_if_needed(question)
    if safety:
        return _success(safety, provider, settings["model"], started)
    try:
        if provider == "echo":
            result = _echo_answer(report, question)
        elif provider == "glm4v_api":
            result = _glm_answer(report, settings, question, frame_b64)
        else:
            return _failure("bad_request", f"不支持的 LLM provider: {provider}", provider, settings["model"])
        return _success(result, provider, settings["model"], started)
    except TimeoutError as exc:
        return _failure("timeout", "GLM API 请求超时，请稍后重试。", provider, settings["model"], exc)
    except urllib.error.URLError as exc:
        error_code, message = _network_failure_message(exc)
        return _failure(error_code, message, provider, settings["model"], exc)
    except ValueError as exc:
        return _failure("parse_error", "GLM 返回格式暂时无法解析，请稍后重试。", provider, settings["model"], exc)
    except RuntimeError as exc:
        return _failure("provider_error", "GLM API 调用失败，请检查配置或稍后重试。", provider, settings["model"], exc)
    except Exception as exc:  # pragma: no cover
        return _failure("unknown", "AI 问答失败，但训练主流程不受影响。", provider, settings["model"], exc)


def _settings() -> dict[str, Any]:
    provider = os.getenv("REHAB_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    model = os.getenv("REHAB_LLM_MODEL", DEFAULT_MODEL if provider == "glm4v_api" else "echo").strip() or "echo"
    return {
        "provider": provider,
        "api_key": os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY") or "",
        "model": model,
        "endpoint": os.getenv("REHAB_LLM_ENDPOINT", DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT,
        "timeout": _float_env("REHAB_LLM_TIMEOUT", 30.0),
        "max_tokens": _int_env("REHAB_LLM_MAX_TOKENS", 768),
    }


def _glm_summary(
    report: dict[str, Any],
    settings: dict[str, Any],
    audience: str,
    include_calorie: bool,
    include_keyframes: bool,
    keyframe_frame_b64: str | None,
) -> dict[str, Any]:
    if not settings["api_key"]:
        raise RuntimeError("missing API key")
    prompt = SUMMARY_PROMPT.format(
        report_summary=json.dumps(_compact_report(report), ensure_ascii=False, indent=2),
    )
    prompt += f"\n输出受众：{audience}。是否包含热量估算：{bool(include_calorie)}。"
    prompt += f"\n是否提供关键帧图片：{bool(include_keyframes and keyframe_frame_b64)}。"
    answer = _chat_text(settings, prompt, frame_b64=keyframe_frame_b64 if include_keyframes else None)
    parsed = _parse_json_object_or_none(answer)
    if parsed is None:
        return _fallback_summary_from_text(answer, report)
    return _normalize_summary_payload(parsed, fallback_text=answer, report=report)


def _glm_answer(
    report: dict[str, Any],
    settings: dict[str, Any],
    question: str,
    frame_b64: str | None,
) -> dict[str, Any]:
    if not settings["api_key"]:
        raise RuntimeError("missing API key")
    prompt = QUESTION_PROMPT.format(
        question=question,
        report_summary=json.dumps(_compact_report(report), ensure_ascii=False, indent=2),
    )
    answer = _chat_text(settings, prompt, frame_b64=frame_b64)
    parsed = _parse_json_object_or_none(answer)
    if parsed is None:
        return _fallback_answer_from_text(answer)
    return _normalize_answer_payload(parsed, fallback_text=answer)


def _chat_text(settings: dict[str, Any], prompt: str, frame_b64: str | None = None) -> str:
    content: str | list[dict[str, Any]]
    if frame_b64:
        image_url = frame_b64 if frame_b64.startswith("data:image/") else f"data:image/jpeg;base64,{frame_b64}"
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": settings["max_tokens"],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        settings["endpoint"],
        data=body,
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings["timeout"]) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    response_json = json.loads(raw)
    try:
        content_value = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response missing choices/message/content") from exc
    if isinstance(content_value, list):
        parts = []
        for item in content_value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content_value).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not JSON")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def _parse_json_object_or_none(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    stripped = _strip_markdown_json_fence(text)
    candidates = [text, stripped]
    if stripped.lower().startswith("json"):
        candidates.append(stripped[4:].strip())
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_markdown_json_fence(text: str) -> str:
    value = str(text or "").strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value.strip("`").strip()


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    structured_feedback = report.get("structured_feedback") if isinstance(report.get("structured_feedback"), dict) else {}
    metric = report.get("metric") if isinstance(report.get("metric"), dict) else {}
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    return {
        "evaluated_at": report.get("evaluated_at") or meta.get("evaluated_at"),
        "action_name": report.get("action_name") or meta.get("action_name") or meta.get("action_id"),
        "metric": metric,
        "metrics": {
            "rom": metrics.get("rom"),
            "tut": metrics.get("tut"),
            "dtw": metrics.get("dtw"),
            "speed": metrics.get("speed"),
            "secondary_metrics": metrics.get("secondary_metrics"),
        },
        "errors": errors,
        "structured_feedback": structured_feedback,
        "fields": report.get("fields"),
        "keypoint_rule": report.get("keypoint_rule"),
        "keyframes": report.get("keyframes") if isinstance(report.get("keyframes"), list) else [],
    }


def _fallback_summary_from_text(text: str, report: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_model_text(text)
    patient_summary = _shorten(_first_sentence(clean), 60) or _local_patient_summary(report)
    return _normalize_summary_payload(
        {
            "patient_summary": patient_summary,
            "doctor_summary": _local_doctor_summary(report),
            "next_steps": _local_next_steps(report),
            "risk_notes": _local_risk_notes(),
            "calorie_estimate": {"text": "GLM 本次未返回结构化热量估算；热量只能作为粗略参考。", "value_kcal": None},
            "spoken_text": patient_summary,
            "raw_text_preview": _shorten(clean, 180),
        },
        fallback_text=patient_summary,
        report=report,
    )


def _fallback_answer_from_text(text: str) -> dict[str, Any]:
    clean = _clean_model_text(text)
    return _normalize_answer_payload(
        {
            "answer": clean or "GLM 已返回内容，但不是结构化 JSON。",
            "spoken_text": clean,
        },
        fallback_text=clean,
    )


def _clean_model_text(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = _strip_markdown_json_fence(value)
    if value.lower().startswith("json"):
        value = value[4:].strip()
    return value.strip()


def _first_sentence(text: str) -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""
    if value.startswith("{") or '"patient_summary"' in value or '"doctor_summary"' in value:
        return ""
    stops = ["。", "！", "？", ".", "!", "?"]
    indexes = [value.find(stop) for stop in stops if value.find(stop) >= 0]
    if indexes:
        return value[: min(indexes) + 1].strip()
    return value


def _local_patient_summary(report: dict[str, Any]) -> str:
    compact = _compact_report(report)
    action_name = str(compact.get("action_name") or "本次动作")
    error_code = _report_error_code(report)
    if error_code == "OK":
        return f"{action_name}完成不错，继续保持慢、稳、到位。"
    if error_code == "ROM_LOW":
        return f"{action_name}幅度还差一点，下一次慢慢做到位。"
    if error_code == "TUT_LOW":
        return f"{action_name}保持时间偏短，到位后先稳住。"
    if error_code == "TOO_FAST":
        return f"{action_name}节奏偏快，下一组请放慢一点。"
    if error_code == "SHAPE_BAD":
        return f"{action_name}轨迹还不够稳定，下一组先做稳。"
    return f"{action_name}还有需要调整的地方，请按报告提示练习。"


def _local_doctor_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "报告信息不足，无法生成医生版摘要。"
    compact = _compact_report(report)
    metrics = compact["metrics"]
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    speed = metrics.get("speed") if isinstance(metrics.get("speed"), dict) else {}
    dtw = metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {}
    action_name = str(compact.get("action_name") or "本次动作")
    return (
        f"{action_name}：主要错误 { _report_error_code(report) }；"
        f"ROM {_fmt(rom.get('actual'))}/{_fmt(rom.get('target'))}，"
        f"TUT {_fmt(tut.get('actual'))}/{_fmt(tut.get('target'))}，"
        f"速度比 {_fmt(speed.get('ratio'), 2)}，DTW {_fmt(dtw.get('normalized_distance'), 2)}。"
    )


def _local_next_steps(report: dict[str, Any] | None) -> list[str]:
    error_code = _report_error_code(report or {})
    if error_code == "OK":
        return ["保持当前节奏，继续按医生模板训练。"]
    if error_code == "ROM_LOW":
        return ["下一组先放慢速度，再逐步增加动作幅度。"]
    if error_code == "TUT_LOW":
        return ["到达目标位置后先停稳，再缓慢返回。"]
    if error_code == "TOO_FAST":
        return ["放慢动作速度，避免借力或突然回落。"]
    if error_code == "SHAPE_BAD":
        return ["先控制动作轨迹，减少晃动后再追求幅度。"]
    return ["优先修正报告中的主要问题，不要自行增加训练量。"]


def _local_risk_notes() -> list[str]:
    return ["如出现疼痛、肿胀、麻木、头晕或站立不稳，请停止训练并联系医生或康复师。"]


def _report_error_code(report: dict[str, Any]) -> str:
    structured = report.get("structured_feedback") if isinstance(report.get("structured_feedback"), dict) else {}
    if structured.get("error_code"):
        return str(structured.get("error_code"))
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    return str(errors.get("primary_error") or "OK")


def _coerce_summary_text(value: Any, report: dict[str, Any] | None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) or isinstance(value, list):
        return _local_doctor_summary(report)
    if value is None:
        return _local_doctor_summary(report) if report else ""
    return str(value).strip()


def _echo_summary(report: dict[str, Any], include_calorie: bool, include_keyframes: bool = False) -> dict[str, Any]:
    compact = _compact_report(report)
    metrics = compact["metrics"]
    error_code = str((compact["errors"] or {}).get("primary_error") or "OK")
    action_name = str(compact.get("action_name") or "本次动作")
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    speed = metrics.get("speed") if isinstance(metrics.get("speed"), dict) else {}
    dtw = metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {}
    rom_actual = _fmt(rom.get("actual"))
    rom_target = _fmt(rom.get("target"))
    tut_actual = _fmt(tut.get("actual"))
    tut_target = _fmt(tut.get("target"))
    speed_ratio = _fmt(speed.get("ratio"), 2)
    dtw_distance = _fmt(dtw.get("normalized_distance"), 2)
    if error_code == "OK":
        patient = f"{action_name}完成得不错，主要指标接近本次模板要求。"
        next_steps = ["保持现在的节奏，继续按医生模板完成动作。"]
    elif error_code == "ROM_LOW":
        patient = f"{action_name}的动作幅度还差一点，先在安全范围内慢慢做到位。"
        next_steps = ["下一组先放慢速度，再逐步增加动作幅度。"]
    elif error_code == "TUT_LOW":
        patient = f"{action_name}到位后的保持时间偏短，下一次可以在最高点稳住一会儿。"
        next_steps = ["到达目标位置后先停稳，再缓慢返回。"]
    elif error_code == "TOO_FAST":
        patient = f"{action_name}节奏偏快，下一组请慢一点，先保证动作稳定。"
        next_steps = ["放慢动作速度，避免借力或突然回落。"]
    else:
        patient = f"{action_name}还有需要调整的地方，请按页面规则反馈继续练习。"
        next_steps = ["优先修正当前报告提示的主要问题。"]
    doctor = (
        f"错误类型：{error_code}；ROM {rom_actual}/{rom_target}；TUT {tut_actual}/{tut_target}；"
        f"速度比例 {speed_ratio}；DTW {dtw_distance}。"
    )
    calorie_text = "热量仅为粗略估计；当前报告缺少体重、心率和真实运动强度，不能作为医学依据。"
    return _normalize_summary_payload(
        {
            "patient_summary": patient,
            "doctor_summary": doctor,
            "next_steps": next_steps,
            "risk_notes": ["如出现疼痛、肿胀、麻木、头晕或站立不稳，请停止训练并联系医生或康复师。"],
            "calorie_estimate": {"text": calorie_text if include_calorie else "本次未启用热量估算。", "value_kcal": None},
            "keyframe_notes": build_keyframe_notes(report)[:2] if include_keyframes else [],
            "spoken_text": patient,
        }
    )


def _echo_answer(report: dict[str, Any], question: str) -> dict[str, Any]:
    compact = _compact_report(report)
    metrics = compact["metrics"]
    error_code = str((compact["errors"] or {}).get("primary_error") or "OK")
    action_name = str(compact.get("action_name") or "本次动作")
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    if "哪里" in question or "不好" in question or "问题" in question:
        if error_code == "OK":
            answer = f"从本次报告看，{action_name}整体完成不错，主要指标接近模板。下一组继续保持慢、稳、到位。"
        elif error_code == "ROM_LOW":
            answer = f"你刚才主要是动作幅度还差一点，ROM 约为 {_fmt(rom.get('actual'))}，目标约为 {_fmt(rom.get('target'))}。下一次慢慢做到安全范围内。"
        elif error_code == "TUT_LOW":
            answer = f"你刚才主要是到位后保持时间偏短，本次约 {_fmt(tut.get('actual'))} 秒，模板约 {_fmt(tut.get('target'))} 秒。下一次到位后先稳住。"
        else:
            answer = f"本次主要提示是 {error_code}，建议先按报告中的主要问题调整，不要急着增加强度。"
    else:
        answer = f"我只能根据这份训练报告回答。{action_name}本次主要结果是 {error_code}，下一组请按页面反馈慢慢调整。"
    return _normalize_answer_payload({"answer": answer, "spoken_text": answer})


def _safety_answer_if_needed(question: str) -> dict[str, Any] | None:
    if any(word in question for word in MEDICAL_RISK_WORDS):
        text = "你提到的情况可能涉及安全风险，请先停止训练，并联系医生或康复师确认后再继续。"
        return {"answer": text, "spoken_text": text}
    if any(word in question for word in MEDICAL_DECISION_WORDS):
        text = "这个问题需要医生或康复师结合病情判断。我只能根据本次训练报告解释动作表现，不能替你决定用药、治疗或训练加量。"
        return {"answer": text, "spoken_text": _shorten(text)}
    return None


def _normalize_summary_payload(payload: dict[str, Any], fallback_text: str = "", report: dict[str, Any] | None = None) -> dict[str, Any]:
    patient_summary = _shorten(str(payload.get("patient_summary") or fallback_text or "AI 已返回建议，但内容较短。").strip(), 80)
    doctor_summary = _coerce_summary_text(payload.get("doctor_summary"), report) or "暂无医生版总结。"
    next_steps = _string_list(payload.get("next_steps"))[:2] or (_local_next_steps(report) if report else ["继续按医生模板完成训练。"])
    risk_notes = _string_list(payload.get("risk_notes"))[:1] or _local_risk_notes()
    calorie = payload.get("calorie_estimate") if isinstance(payload.get("calorie_estimate"), dict) else {}
    spoken = _shorten(str(payload.get("spoken_text") or patient_summary))
    result = {
        "patient_summary": patient_summary,
        "doctor_summary": _shorten(doctor_summary, 160),
        "next_steps": next_steps,
        "risk_notes": risk_notes,
        "calorie_estimate": {
            "text": str(calorie.get("text") or "热量仅为粗略估计，不作为医学依据。"),
            "value_kcal": calorie.get("value_kcal") if isinstance(calorie.get("value_kcal"), (int, float)) else None,
        },
        "keyframe_notes": _string_list(payload.get("keyframe_notes"))[:2] or (build_keyframe_notes(report)[:2] if report else []),
        "spoken_text": spoken,
    }
    if payload.get("raw_text_preview"):
        result["raw_text_preview"] = str(payload.get("raw_text_preview"))
    return result


def _normalize_answer_payload(payload: dict[str, Any], fallback_text: str = "") -> dict[str, Any]:
    answer = str(payload.get("answer") or fallback_text or "暂时没有可展示的 AI 回答。").strip()
    return {"answer": answer, "spoken_text": _shorten(str(payload.get("spoken_text") or answer))}


def _success(payload: dict[str, Any], provider: str, model: str, started: float) -> dict[str, Any]:
    latency = int((time.monotonic() - started) * 1000)
    _STATE.last_error = None
    _STATE.last_latency_ms = latency
    _STATE.last_success_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return {"ok": True, "provider": provider, "model": model, "latency_ms": latency, **payload}


def _failure(
    error_code: str,
    message: str,
    provider: str,
    model: str,
    exc: Exception | None = None,
) -> dict[str, Any]:
    last_error = _sanitize_error(exc or message)
    _STATE.last_error = last_error
    return {
        "ok": False,
        "provider": provider,
        "model": model,
        "error_code": error_code,
        "message": message,
        "last_error": last_error,
    }


def _network_failure_message(exc: urllib.error.URLError) -> tuple[str, str]:
    reason = getattr(exc, "reason", exc)
    text = str(reason or exc)
    lower = text.lower()
    if isinstance(reason, TimeoutError) or "timed out" in lower or "timeout" in lower:
        return "timeout", "GLM API 请求超时，请检查板端网络或调大 REHAB_LLM_TIMEOUT。"
    if "name or service not known" in lower or "temporary failure in name resolution" in lower:
        return "network_error", "GLM API 域名解析失败，请检查板端 DNS/网络。"
    if "getaddrinfo failed" in lower or "nodename nor servname" in lower:
        return "network_error", "GLM API 域名解析失败，请检查板端 DNS/网络。"
    if "certificate_verify_failed" in lower or "certificat" in lower:
        return "network_error", "GLM API HTTPS 证书校验失败，请检查板端系统时间或 CA 证书。"
    if "network is unreachable" in lower or "no route to host" in lower:
        return "network_error", "板端无法访问外网，请检查网关、路由或热点网络。"
    if "connection refused" in lower:
        return "network_error", "GLM API 连接被拒绝，请检查 endpoint、代理或网络出口。"
    if "proxy" in lower:
        return "network_error", "GLM API 代理连接失败，请检查 HTTP_PROXY/HTTPS_PROXY 配置。"
    return "network_error", "GLM API 网络连接失败，请检查板端网络。"


def _sanitize_error(error: Exception | str) -> str:
    text = str(error)
    for secret in (os.getenv("ZHIPUAI_API_KEY"), os.getenv("GLM_API_KEY")):
        if secret:
            text = text.replace(secret, "***")
    return text.replace("\\", "/")[:240]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _shorten(text: str, max_chars: int = MAX_SPOKEN_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
