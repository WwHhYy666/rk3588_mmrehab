from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prescription.common.report_visuals import build_keyframe_notes, estimate_calories


DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4v-flash"
DEFAULT_PROVIDER = "auto"
DEFAULT_LOCAL_QWEN_ENDPOINT = "http://127.0.0.1:18080/generate"
DEFAULT_LOCAL_QWEN_HEALTH_ENDPOINT = "http://127.0.0.1:18080/health"
MAX_QUESTION_CHARS = 300
MAX_SPOKEN_CHARS = 120
QWEN_SMOKE_TTL_SECONDS = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAFETY_SYSTEM_PROMPT = (
    "你是骨科居家康复训练辅助解释助手。只能根据系统提供的训练报告回答，"
    "不能诊断疾病，不能替代医生，不能建议患者自行改变训练量、药物或治疗方案。"
    "如果报告信息不足，要明确说明不确定。回答要短、温和，适合中文语音播报。"
)

SUMMARY_PROMPT = """请根据下面这份康复训练评估报告生成训练后解释。
要求：只基于报告内容回答，不要编造病情；输出 JSON，包含 patient_summary、doctor_summary、next_steps、risk_notes、spoken_text。
训练报告摘要：
{report_summary}"""

QUESTION_PROMPT = """患者基于本次康复训练报告提出了问题。
要求：只基于报告回答；不要诊断疾病或调整治疗；回答短、温和、适合朗读；尽量输出 JSON，包含 answer 和 spoken_text。

患者问题：{question}

训练报告摘要：
{report_summary}"""

MEDICAL_RISK_WORDS = (
    "疼", "疼痛", "痛", "肿", "肿胀", "麻", "麻木", "头晕", "跌倒", "伤口", "出血",
)
MEDICAL_DECISION_WORDS = (
    "诊断", "什么病", "停药", "吃药", "用药", "加量", "加训练", "加大", "手术", "痊愈", "要不要去医院",
)


SUPPORTED_PROVIDERS = {"auto", "glm4v_api", "local_qwen_rkllm", "echo"}


@dataclass
class LLMRuntimeState:
    last_error: str | None = None
    last_success_at: str | None = None
    last_latency_ms: int | None = None
    last_active_provider: str | None = None
    fallback_reason: str | None = None
    provider_override: str | None = None
    qwen_smoke_ok: bool | None = None
    qwen_smoke_at: float | None = None
    qwen_smoke_error: str | None = None


_STATE = LLMRuntimeState()


def get_llm_provider_override() -> str | None:
    return _STATE.provider_override


def set_llm_provider_override(provider: str | None) -> str | None:
    normalized = str(provider or "").strip().lower()
    if not normalized or normalized == "env":
        _STATE.provider_override = None
        return None
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    _STATE.provider_override = normalized
    _STATE.fallback_reason = None
    return normalized


def get_llm_status(check_health: bool = True) -> dict[str, Any]:
    settings = _settings()
    rkllm_reachable = _rkllm_health(settings) if check_health else None
    glm_reachable = _glm_health(settings) if check_health and settings.get("api_key") else None
    qwen_generate_ok = _local_qwen_smoke(settings) if check_health and rkllm_reachable else None
    active_provider = _predict_active_provider(settings, bool(rkllm_reachable))
    return {
        "enabled": True,
        "provider": settings["provider"],
        "provider_override": settings.get("provider_override"),
        "env_provider": settings.get("env_provider"),
        "active_provider": active_provider,
        "model": settings["model"],
        "api_key_configured": bool(settings["api_key"]),
        "endpoint_configured": bool(settings["endpoint"]),
        "glm_endpoint": settings["endpoint"],
        "glm_endpoint_reachable": glm_reachable,
        "rkllm_endpoint": settings["local_qwen_endpoint"],
        "rkllm_server_reachable": rkllm_reachable,
        "qwen_generate_ok": qwen_generate_ok,
        "qwen_generate_cached": _qwen_smoke_cached(),
        "qwen_generate_age_seconds": _qwen_smoke_age_seconds(),
        "qwen_generate_error": _STATE.qwen_smoke_error,
        "health_checked": check_health,
        "last_error": _STATE.last_error,
        "last_success_at": _STATE.last_success_at,
        "last_latency_ms": _STATE.last_latency_ms,
        "fallback_reason": _STATE.fallback_reason,
        "last_active_provider": _STATE.last_active_provider,
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
        if provider in {"glm4v_api", "auto"} and settings["api_key"]:
            result = _glm_summary(report, settings, audience, include_calorie, include_keyframes, keyframe_frame_b64)
            return _success(result, provider, settings["model"], started, active_provider="glm4v_api")
        result = _echo_summary(report, include_calorie, include_keyframes)
        result["fallback_reason"] = "report_summary_uses_local_rules_without_glm"
        return _success(result, provider, settings["model"], started, active_provider="echo")
    except Exception as exc:
        return _failure("provider_error", "AI 总结生成失败，但训练主流程不受影响。", provider, settings["model"], exc)


def answer_question(report: dict[str, Any], question: str, frame_b64: str | None = None) -> dict[str, Any]:
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
        return _success(safety, provider, settings["model"], started, active_provider="local_rules")

    try:
        if provider == "echo":
            result = _echo_answer(report, question)
            return _success(result, provider, settings["model"], started, active_provider="echo")
        if provider == "glm4v_api":
            result = _glm_answer(report, settings, question, frame_b64)
            return _success(result, provider, settings["model"], started, active_provider="glm4v_api")
        if provider == "local_qwen_rkllm":
            result = _local_qwen_answer(report, settings, question)
            return _success(result, provider, settings["model"], started, active_provider="local_qwen_rkllm")
        if provider == "auto":
            result = _auto_answer(report, settings, question, frame_b64)
            active_provider = str(result.pop("active_provider", "echo"))
            return _success(result, provider, settings["model"], started, active_provider=active_provider)
        return _failure("bad_request", f"Unsupported LLM provider: {provider}", provider, settings["model"])
    except TimeoutError as exc:
        return _failure("timeout", "LLM 请求超时，请稍后重试。", provider, settings["model"], exc)
    except urllib.error.URLError as exc:
        error_code, message = _network_failure_message(exc)
        return _failure(error_code, message, provider, settings["model"], exc)
    except Exception as exc:
        return _failure("provider_error", "AI 问答失败，但训练主流程不受影响。请检查 GLM 网络或本地 Qwen proxy。", provider, settings["model"], exc)


def _settings() -> dict[str, Any]:
    env_provider = os.getenv("REHAB_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    if env_provider not in SUPPORTED_PROVIDERS:
        env_provider = DEFAULT_PROVIDER
    provider = _STATE.provider_override or env_provider
    if provider == "glm4v_api":
        model_default = DEFAULT_MODEL
    elif provider == "local_qwen_rkllm":
        model_default = "qwen2.5-1.5b-rkllm"
    else:
        model_default = DEFAULT_MODEL if os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY") else "qwen2.5-1.5b-rkllm"
    return {
        "provider": provider,
        "provider_override": _STATE.provider_override,
        "env_provider": env_provider,
        "online_provider": os.getenv("REHAB_LLM_ONLINE_PROVIDER", "glm4v_api").strip().lower() or "glm4v_api",
        "offline_provider": os.getenv("REHAB_LLM_OFFLINE_PROVIDER", "local_qwen_rkllm").strip().lower() or "local_qwen_rkllm",
        "api_key": os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY") or "",
        "model": os.getenv("REHAB_LLM_MODEL", model_default).strip() or model_default,
        "endpoint": os.getenv("REHAB_LLM_ENDPOINT", DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT,
        "local_qwen_endpoint": os.getenv("REHAB_LOCAL_QWEN_ENDPOINT", DEFAULT_LOCAL_QWEN_ENDPOINT).strip() or DEFAULT_LOCAL_QWEN_ENDPOINT,
        "local_qwen_health_endpoint": os.getenv("REHAB_LOCAL_QWEN_HEALTH_ENDPOINT", DEFAULT_LOCAL_QWEN_HEALTH_ENDPOINT).strip() or DEFAULT_LOCAL_QWEN_HEALTH_ENDPOINT,
        "timeout": _float_env("REHAB_LLM_TIMEOUT", 30.0),
        "local_qwen_timeout": _float_env("REHAB_LOCAL_QWEN_TIMEOUT", 120.0),
        "max_tokens": _int_env("REHAB_LLM_MAX_TOKENS", 256),
    }


def _predict_active_provider(settings: dict[str, Any], rkllm_reachable: bool) -> str:
    provider = settings["provider"]
    if provider != "auto":
        return provider
    if settings["api_key"]:
        return settings["online_provider"]
    if rkllm_reachable:
        return settings["offline_provider"]
    return "echo"


def _auto_answer(report: dict[str, Any], settings: dict[str, Any], question: str, frame_b64: str | None) -> dict[str, Any]:
    failures: list[str] = []
    _STATE.fallback_reason = None

    if settings.get("online_provider") == "glm4v_api" and settings.get("api_key"):
        try:
            result = _glm_answer(report, settings, question, frame_b64)
            result["active_provider"] = "glm4v_api"
            return result
        except Exception as exc:
            failures.append(f"glm4v_api: {_sanitize_error(exc)}")

    if settings.get("offline_provider") == "local_qwen_rkllm":
        try:
            result = _local_qwen_answer(report, settings, question)
            result["active_provider"] = "local_qwen_rkllm"
            return result
        except Exception as exc:
            failures.append(f"local_qwen_rkllm: {_sanitize_error(exc)}")

    reason = "; ".join(failures) or "no GLM API key and local Qwen unavailable"
    _STATE.fallback_reason = reason
    raise RuntimeError(reason)


def _glm_summary(
    report: dict[str, Any],
    settings: dict[str, Any],
    audience: str,
    include_calorie: bool,
    include_keyframes: bool,
    keyframe_frame_b64: str | None,
) -> dict[str, Any]:
    prompt = SUMMARY_PROMPT.format(report_summary=json.dumps(_compact_report(report), ensure_ascii=False, indent=2))
    prompt += f"\n输出受众：{audience}。是否包含热量估算：{bool(include_calorie)}。"
    prompt += f"\n是否提供关键帧图片：{bool(include_keyframes and keyframe_frame_b64)}。"
    answer = _chat_text(settings, prompt, frame_b64=keyframe_frame_b64 if include_keyframes else None)
    parsed = _parse_json_object_or_none(answer)
    if parsed is None:
        return _fallback_summary_from_text(answer, report)
    return _normalize_summary_payload(parsed, fallback_text=answer, report=report)


def _glm_answer(report: dict[str, Any], settings: dict[str, Any], question: str, frame_b64: str | None) -> dict[str, Any]:
    if not settings["api_key"]:
        raise RuntimeError("missing API key")
    prompt = QUESTION_PROMPT.format(question=question, report_summary=json.dumps(_compact_report(report), ensure_ascii=False, indent=2))
    answer = _chat_text(settings, prompt, frame_b64=frame_b64)
    parsed = _parse_json_object_or_none(answer)
    if parsed is None:
        return _fallback_answer_from_text(answer)
    return _normalize_answer_payload(parsed, fallback_text=answer)


def _local_qwen_answer(report: dict[str, Any], settings: dict[str, Any], question: str) -> dict[str, Any]:
    prompts = _local_qwen_prompts(report, question)
    last_raw = ""
    last_response: dict[str, Any] = {}
    for index, prompt in enumerate(prompts):
        payload = {
            "prompt": prompt,
            "max_new_tokens": min(int(settings.get("max_tokens") or 128), 96),
            "temperature": 0.2,
            "request_id": f"rehab_{int(time.time() * 1000)}_{index}",
        }
        try:
            raw = _post_json(settings["local_qwen_endpoint"], payload, timeout=float(settings.get("local_qwen_timeout") or 120.0))
        except RuntimeError as exc:
            response_json = _json_from_http_runtime_error(exc)
            if response_json and response_json.get("ok") is False:
                last_response = response_json
                if _local_qwen_should_try_shorter(response_json) and index < len(prompts) - 1:
                    continue
            _mark_qwen_smoke(False, _sanitize_error(exc))
            raise
        last_raw = raw
        try:
            response_json = json.loads(raw)
        except json.JSONDecodeError:
            response_json = {"ok": True, "text": raw}
        last_response = response_json
        if response_json.get("ok") is False:
            error_message = _local_qwen_error_message(response_json)
            _mark_qwen_smoke(False, error_message)
            if _local_qwen_should_try_shorter(response_json) and index < len(prompts) - 1:
                continue
            raise RuntimeError(error_message)
        text = str(response_json.get("text") or response_json.get("answer") or response_json.get("response") or "").strip()
        if not text:
            continue
        parsed = _parse_json_object_or_none(text)
        result = _fallback_answer_from_text(text) if parsed is None else _normalize_answer_payload(parsed, fallback_text=text)
        result["rkllm_latency_ms"] = response_json.get("latency_ms")
        result["rkllm_model"] = response_json.get("model")
        result["qwen_prompt_mode"] = "compact" if index == 0 else "minimal_retry"
        result["qwen_queue_wait_ms"] = response_json.get("queue_wait_ms")
        result["qwen_retry_count"] = response_json.get("retry_count")
        result["qwen_empty_retry_count"] = response_json.get("empty_retry_count")
        _mark_qwen_smoke(True, None)
        return result
    preview = str(last_response.get("upstream_error_preview") or last_raw or "")[:220]
    raise ValueError(f"local qwen returned empty text after compact retry: {preview}")


def _local_qwen_error_message(response_json: dict[str, Any]) -> str:
    message = str(response_json.get("error") or response_json.get("message") or "local qwen failed")
    upstream_status = response_json.get("upstream_status")
    preview = str(response_json.get("upstream_error_preview") or "").strip()
    retry_count = response_json.get("retry_count")
    empty_retry_count = response_json.get("empty_retry_count")
    queue_wait_ms = response_json.get("queue_wait_ms")
    if upstream_status is not None:
        message += f"; upstream_status={upstream_status}"
    if retry_count is not None:
        message += f"; retry_count={retry_count}"
    if empty_retry_count is not None:
        message += f"; empty_retry_count={empty_retry_count}"
    if queue_wait_ms is not None:
        message += f"; queue_wait_ms={queue_wait_ms}"
    if preview:
        message += f"; preview={preview[:180]}"
    return message


def _local_qwen_should_try_shorter(response_json: dict[str, Any]) -> bool:
    message = _local_qwen_error_message(response_json).lower()
    return "missing text" in message or "empty text" in message or "空" in message


def _json_from_http_runtime_error(exc: Exception) -> dict[str, Any] | None:
    text = str(exc)
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _local_qwen_prompts(report: dict[str, Any], question: str) -> list[str]:
    summary_text = _summary_markdown_for_qwen(report)
    brief = summary_text or _compact_report_brief_text(report)
    safe = (
        "你是康复训练报告解释助手。只根据给出的训练报告回答，不诊断疾病，不替代医生。"
        "请直接用中文回答1到2句，最多80字，不要输出JSON，不要重复题目。"
    )
    source_label = "训练摘要" if summary_text else "训练报告"
    compact = f"{safe}\n{source_label}：\n{brief}\n患者问题：{question}\n回答："
    minimal = f"请用一句中文回答患者问题，只基于{source_label}。{source_label}：{_one_line(brief, 520)}\n问题：{question}\n回答："
    return [compact[:1800], minimal[:900]]


def _summary_markdown_for_qwen(report: dict[str, Any]) -> str:
    inline = report.get("_summary_markdown") or report.get("summary_markdown")
    if isinstance(inline, str) and inline.strip():
        return _trim_summary_markdown(inline)
    for path in _summary_markdown_candidates(report):
        try:
            if path.exists() and path.is_file():
                return _trim_summary_markdown(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return ""


def _summary_markdown_candidates(report: dict[str, Any]) -> list[Path]:
    values: list[object] = []
    for key in ("summary_file", "summary_path", "attempt_file"):
        value = report.get(key)
        if value:
            values.append(value)
    runtime_meta = report.get("runtime_meta") if isinstance(report.get("runtime_meta"), dict) else {}
    for key in ("summary_file", "summary_path", "attempt_file"):
        value = runtime_meta.get(key)
        if value:
            values.append(value)
    candidates: list[Path] = []
    for value in values:
        raw = Path(str(value))
        stems = []
        if raw.suffix.lower() == ".md":
            stems.append(raw.stem.removesuffix("_summary"))
            candidates.append(raw)
        else:
            stems.append(raw.stem)
        for stem in stems:
            if not stem:
                continue
            name = f"{stem}_summary.md"
            if raw.is_absolute():
                candidates.append(raw.parent.parent / "summaries" / name)
            candidates.append(PROJECT_ROOT / "prescription" / "docs" / "summaries" / name)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _trim_summary_markdown(text: str, limit: int = 1400) -> str:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("#"):
            continue
        if "患者动作文件" in value or "摘要文件" in value or "docs/results" in value:
            continue
        lines.append(value)
    return "\n".join(lines)[:limit].strip()


def _one_line(text: str, limit: int = 520) -> str:
    return " ".join(str(text or "").split())[:limit]


def _compact_report_brief_text(report: dict[str, Any]) -> str:
    compact = _compact_report(report)
    action_name = str(compact.get("action_name") or "本次动作")
    error_code = _report_error_code(report)
    metrics = compact.get("metrics") if isinstance(compact.get("metrics"), dict) else {}
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    quality_attempts = compact.get("quality_attempts") if isinstance(compact.get("quality_attempts"), list) else []
    last_attempt = quality_attempts[-1] if quality_attempts else {}
    parts = [f"动作={action_name}", f"结果={error_code}"]
    if rom:
        parts.append(f"幅度={_fmt(rom.get('actual'))}/{_fmt(rom.get('target'))}")
    if tut:
        parts.append(f"保持={_fmt(tut.get('actual'))}秒/{_fmt(tut.get('target'))}秒")
    if isinstance(last_attempt, dict):
        score = last_attempt.get("quality_score") or last_attempt.get("completion_percent")
        if score is not None:
            parts.append(f"完成度={_fmt(score)}%")
        reason = last_attempt.get("reason") or last_attempt.get("primary_error")
        if reason:
            parts.append(f"提示={reason}")
    return "；".join(parts)

def _chat_text(settings: dict[str, Any], prompt: str, frame_b64: str | None = None) -> str:
    content: str | list[dict[str, Any]]
    if frame_b64 and isinstance(frame_b64, str):
        image_url = frame_b64 if frame_b64.startswith("data:image/") else f"data:image/jpeg;base64,{frame_b64}"
        content = [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": prompt}]
    else:
        content = prompt
    payload = {
        "model": settings["model"],
        "messages": [{"role": "system", "content": SAFETY_SYSTEM_PROMPT}, {"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": settings["max_tokens"],
    }
    raw = _post_json(settings["endpoint"], payload, timeout=float(settings["timeout"]), headers={"Authorization": f"Bearer {settings['api_key']}"})
    response_json = json.loads(raw)
    try:
        content_value = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response missing choices/message/content") from exc
    if isinstance(content_value, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content_value).strip()
    return str(content_value).strip()


def _post_json(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None) -> str:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(str(url), data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _rkllm_health(settings: dict[str, Any] | None = None) -> bool:
    settings = settings or _settings()
    request = urllib.request.Request(str(settings["local_qwen_health_endpoint"]), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            if response.status >= 400:
                return False
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:
        return False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return bool(raw.strip())
    return bool(payload.get("ok", True))


def _glm_health(settings: dict[str, Any] | None = None) -> bool:
    settings = settings or _settings()
    if not settings.get("api_key"):
        return False
    payload = {
        "model": settings["model"],
        "messages": [{"role": "user", "content": "请只回复 ok"}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        raw = _post_json(settings["endpoint"], payload, timeout=min(float(settings.get("timeout") or 30.0), 8.0), headers={"Authorization": f"Bearer {settings['api_key']}"})
        response_json = json.loads(raw)
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content")
        return bool(str(content or "").strip())
    except Exception as exc:
        _STATE.last_error = _sanitize_error(exc)
        return False


def _local_qwen_smoke(settings: dict[str, Any] | None = None) -> bool:
    settings = settings or _settings()
    cached = _cached_qwen_smoke_value()
    if cached is not None:
        return cached
    payload = {
        "prompt": "请只回复 ok",
        "max_new_tokens": 8,
        "temperature": 0.0,
        "request_id": f"health_{int(time.time() * 1000)}",
    }
    try:
        raw = _post_json(settings["local_qwen_endpoint"], payload, timeout=min(float(settings.get("local_qwen_timeout") or 120.0), 10.0))
        try:
            response_json = json.loads(raw)
        except json.JSONDecodeError:
            ok = bool(raw.strip())
            _mark_qwen_smoke(ok, None if ok else "local qwen smoke returned empty text")
            return ok
        if response_json.get("ok") is False:
            error_message = _local_qwen_error_message(response_json)
            _mark_qwen_smoke(False, error_message)
            _STATE.last_error = _sanitize_error(error_message)
            return False
        text = str(response_json.get("text") or response_json.get("answer") or response_json.get("response") or "").strip()
        ok = bool(text)
        _mark_qwen_smoke(ok, None if ok else "local qwen smoke returned empty text")
        return ok
    except Exception as exc:
        error = _sanitize_error(exc)
        if _STATE.qwen_smoke_ok is True:
            _STATE.qwen_smoke_error = error
            return True
        _mark_qwen_smoke(False, error)
        _STATE.last_error = error
        return False


def _mark_qwen_smoke(ok: bool, error: str | None) -> None:
    _STATE.qwen_smoke_ok = bool(ok)
    _STATE.qwen_smoke_at = time.monotonic()
    _STATE.qwen_smoke_error = error


def _cached_qwen_smoke_value() -> bool | None:
    if _STATE.qwen_smoke_ok is None or _STATE.qwen_smoke_at is None:
        return None
    if time.monotonic() - float(_STATE.qwen_smoke_at) > QWEN_SMOKE_TTL_SECONDS:
        return None
    return bool(_STATE.qwen_smoke_ok)


def _qwen_smoke_cached() -> bool:
    return _cached_qwen_smoke_value() is not None


def _qwen_smoke_age_seconds() -> float | None:
    if _STATE.qwen_smoke_at is None:
        return None
    return round(max(0.0, time.monotonic() - float(_STATE.qwen_smoke_at)), 2)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("report_card_metrics") if isinstance(report.get("report_card_metrics"), dict) else report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    full_metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    structured_feedback = report.get("structured_feedback") if isinstance(report.get("structured_feedback"), dict) else {}
    metric = report.get("metric") if isinstance(report.get("metric"), dict) else {}
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    compact = {
        "evaluated_at": report.get("evaluated_at") or meta.get("evaluated_at"),
        "action_name": report.get("action_name") or meta.get("action_name") or meta.get("action_id"),
        "metric": metric,
        "metrics": {
            "rom": metrics.get("rom"),
            "tut": metrics.get("tut"),
            "dtw": metrics.get("dtw"),
            "speed": metrics.get("speed"),
            "secondary_metrics": metrics.get("secondary_metrics"),
            "source": metrics.get("source"),
            "attempt_index": metrics.get("attempt_index"),
            "rep_index": metrics.get("rep_index"),
        },
        "errors": errors,
        "structured_feedback": structured_feedback,
        "report_card_metrics": metrics,
        "full_session_metrics": {
            "rom": full_metrics.get("rom"),
            "tut": full_metrics.get("tut"),
            "dtw": full_metrics.get("dtw"),
            "speed": full_metrics.get("speed"),
        },
        "fields": report.get("fields"),
        "keypoint_rule": report.get("keypoint_rule"),
        "keyframes": report.get("keyframes") if isinstance(report.get("keyframes"), list) else [],
    }
    compact["overall_quality"] = report.get("overall_quality")
    compact["quality_model"] = _compact_quality_model(report.get("quality_model"))
    compact["selected_attempts"] = _compact_selected_attempts(report.get("selected_attempts"))
    compact["quality_attempts"] = _compact_quality_attempts(report.get("quality_attempts"))
    compact["reps"] = _compact_quality_attempts(report.get("reps"))
    return compact



def _compact_selected_attempts(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    payload: dict[str, Any] = {}
    for key in ("best_correct", "representative_wrong"):
        item = value.get(key)
        if isinstance(item, dict):
            rows = _compact_quality_attempts([item], limit=1)
            payload[key] = rows[0] if rows else None
        else:
            payload[key] = None
    return payload

def _compact_quality_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "available": value.get("available"),
        "backend": value.get("backend"),
        "action_id": value.get("action_id"),
        "model_path": value.get("model_path"),
        "last_score_time_ms": value.get("last_score_time_ms"),
        "last_error": value.get("last_error"),
    }


def _compact_quality_attempts(value: Any, limit: int = 12) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "attempt_index": item.get("attempt_index"),
                "rep_index": item.get("rep_index"),
                "countable": item.get("countable"),
                "primary_error": item.get("primary_error"),
                "reason": item.get("reason"),
                "quality_score": item.get("quality_score"),
                "quality_grade": item.get("quality_grade"),
                "backend": item.get("quality_backend") or item.get("backend"),
                "rom": item.get("rom"),
                "rom_target": item.get("rom_target"),
                "rom_diff": item.get("rom_diff"),
                "tut_seconds": item.get("tut_seconds"),
                "tut_target": item.get("tut_target"),
                "missing_seconds": item.get("missing_seconds"),
                "tut_ratio": item.get("tut_ratio"),
            }
        )
    return rows


def _fallback_summary_from_text(text: str, report: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_model_text(text)
    patient_summary = _shorten(_first_sentence(clean), 80) or _local_patient_summary(report)
    return _normalize_summary_payload(
        {
            "patient_summary": _clean_answer_text(patient_summary),
            "doctor_summary": _local_doctor_summary(report),
            "next_steps": _local_next_steps(report),
            "risk_notes": _local_risk_notes(),
            "calorie_estimate": estimate_calories(report),
            "spoken_text": patient_summary,
            "raw_text_preview": _shorten(clean, 180),
        },
        fallback_text=patient_summary,
        report=report,
    )


def _fallback_answer_from_text(text: str) -> dict[str, Any]:
    clean = _clean_model_text(text)
    return _normalize_answer_payload({"answer": clean or "模型已返回内容，但暂时没有可展示的文本。", "spoken_text": clean}, fallback_text=clean)


def _echo_summary(report: dict[str, Any], include_calorie: bool, include_keyframes: bool = False) -> dict[str, Any]:
    patient = _local_patient_summary(report)
    return _normalize_summary_payload(
        {
            "patient_summary": patient,
            "doctor_summary": _local_doctor_summary(report),
            "next_steps": _local_next_steps(report),
            "risk_notes": _local_risk_notes(),
            "calorie_estimate": estimate_calories(report) if include_calorie else {"text": "本次未启用热量估算。", "value_kcal": None},
            "keyframe_notes": build_keyframe_notes(report)[:2] if include_keyframes else [],
            "spoken_text": patient,
        },
        report=report,
    )


def _echo_answer(report: dict[str, Any], question: str) -> dict[str, Any]:
    compact = _compact_report(report)
    metrics = compact["metrics"]
    error_code = _report_error_code(report)
    action_name = str(compact.get("action_name") or "本次动作")
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    if "哪里" in question or "不好" in question or "问题" in question:
        if error_code == "OK":
            answer = f"从本次报告看，{action_name}整体完成不错，主要指标接近模板要求。下一组继续保持慢、稳、到位。"
        elif error_code == "ROM_LOW":
            answer = f"刚才主要是动作幅度还差一点，ROM 约为 {_fmt(rom.get('actual'))}，目标约为 {_fmt(rom.get('target'))}。下一次慢慢做到安全范围内。"
        elif error_code == "TUT_LOW":
            answer = f"刚才主要是到位后保持时间偏短，本次约 {_fmt(tut.get('actual'))} 秒，模板约 {_fmt(tut.get('target'))} 秒。下一次到位后先稳住。"
        else:
            answer = f"本次主要提示是 {error_code}，建议先按页面反馈调整，不要急着增加强度。"
    else:
        answer = f"我只能根据这份训练报告回答。{action_name}本次主要结果是 {error_code}，下一组请按页面反馈慢慢调整。"
    return _normalize_answer_payload({"answer": answer, "spoken_text": answer})


def _safety_answer_if_needed(question: str) -> dict[str, Any] | None:
    if any(word in question for word in MEDICAL_RISK_WORDS):
        text = "你提到的情况可能涉及安全风险，请先停止训练，并联系医生或康复师确认后再继续。"
        return {"answer": text, "spoken_text": text}
    if any(word in question for word in MEDICAL_DECISION_WORDS):
        text = "这个问题需要医生或康复师结合病情判断。我只能根据本次训练报告解释动作表现，不能替你决定用药、治疗或训练加减。"
        return {"answer": text, "spoken_text": _shorten(text)}
    return None


def _normalize_summary_payload(payload: dict[str, Any], fallback_text: str = "", report: dict[str, Any] | None = None) -> dict[str, Any]:
    patient_summary = _shorten(str(payload.get("patient_summary") or fallback_text or "AI 已返回建议，但内容较短。").strip(), 100)
    doctor_summary = _coerce_summary_text(payload.get("doctor_summary"), report) or "暂无医生版总结。"
    next_steps = _string_list(payload.get("next_steps"))[:2] or (_local_next_steps(report) if report else ["继续按医生模板完成训练。"])
    risk_notes = _string_list(payload.get("risk_notes"))[:1] or _local_risk_notes()
    local_calorie = estimate_calories(report) if report else None
    calorie = local_calorie or (payload.get("calorie_estimate") if isinstance(payload.get("calorie_estimate"), dict) else {})
    result = {
        "patient_summary": patient_summary,
        "doctor_summary": _shorten(_clean_answer_text(doctor_summary), 180),
        "next_steps": [_clean_answer_text(item) for item in next_steps],
        "risk_notes": [_clean_answer_text(item) for item in risk_notes],
        "calorie_estimate": {
            "text": str(calorie.get("text") or "热量仅为粗略估计，不作为医学依据。"),
            "ascii_text": calorie.get("ascii_text"),
            "value_kcal": calorie.get("value_kcal") if isinstance(calorie.get("value_kcal"), (int, float)) else None,
            "formula": calorie.get("formula"),
            "met": calorie.get("met"),
            "weight_kg": calorie.get("weight_kg"),
            "duration_seconds": calorie.get("duration_seconds"),
        },
        "keyframe_notes": _string_list(payload.get("keyframe_notes"))[:2] or (build_keyframe_notes(report)[:2] if report else []),
        "spoken_text": _shorten(_clean_answer_text(str(payload.get("spoken_text") or patient_summary)), 96),
    }
    if payload.get("raw_text_preview"):
        result["raw_text_preview"] = str(payload.get("raw_text_preview"))
    return result


def _normalize_answer_payload(payload: dict[str, Any], fallback_text: str = "") -> dict[str, Any]:
    answer = _clean_answer_text(str(payload.get("answer") or fallback_text or "暂时没有可展示的 AI 回答。"))
    spoken = _clean_answer_text(str(payload.get("spoken_text") or answer))
    return {"answer": answer, "spoken_text": _shorten(spoken, 96)}


def _success(payload: dict[str, Any], provider: str, model: str, started: float, active_provider: str | None = None) -> dict[str, Any]:
    latency = int((time.monotonic() - started) * 1000)
    active = active_provider or provider
    _STATE.last_error = None
    _STATE.last_latency_ms = latency
    _STATE.last_success_at = time.strftime("%Y-%m-%d %H:%M:%S")
    _STATE.last_active_provider = active
    if active == "local_qwen_rkllm":
        _mark_qwen_smoke(True, None)
    if payload.get("fallback_reason"):
        _STATE.fallback_reason = str(payload.get("fallback_reason"))
    elif active != "echo":
        _STATE.fallback_reason = None
    return {"ok": True, "provider": provider, "active_provider": active, "model": model, "latency_ms": latency, **payload}


def _failure(error_code: str, message: str, provider: str, model: str, exc: Exception | str | None = None) -> dict[str, Any]:
    last_error = _sanitize_error(exc or message)
    _STATE.last_error = last_error
    return {"ok": False, "provider": provider, "active_provider": provider, "model": model, "error_code": error_code, "message": message, "last_error": last_error}


def _parse_json_object_or_none(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None
    stripped = _strip_markdown_json_fence(value)
    candidates = [value, stripped]
    if stripped.lower().startswith("json"):
        candidates.append(stripped[4:].strip())
    start = value.find("{")
    end = value.rfind("}")
    if 0 <= start < end:
        candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
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


def _clean_model_text(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = _strip_markdown_json_fence(value)
    if value.lower().startswith("json"):
        value = value[4:].strip()
    return _clean_answer_text(value)


def _clean_answer_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"^[ \t>*-]+", "", value, flags=re.MULTILINE)
    replacements = {
        "ROM_LOW": "幅度不够",
        "TUT_LOW": "保持时间不够",
        "ROM": "动作幅度",
        "TUT": "保持时间",
        "sit_to_stand": "坐站训练",
        "standing_hamstring_curl": "站姿屈膝后勾腿",
        "seated_knee_raise": "坐姿抬膝",
        "local_qwen_rkllm": "本地千问",
        "glm4v_api": "智谱GLM",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = _dedupe_repeated_paragraphs(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _dedupe_repeated_paragraphs(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]
    if len(paragraphs) <= 1:
        return text
    kept: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        key_text = re.sub(r"^回答[：:]", "", paragraph.strip())
        key = re.sub(r"\s+", "", key_text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept)


def _first_sentence(text: str) -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value or value.startswith("{") or '"patient_summary"' in value:
        return ""
    indexes = [value.find(stop) for stop in ["。", "；", ";", ".", "!", "?"] if value.find(stop) >= 0]
    return value[: min(indexes) + 1].strip() if indexes else value


def _local_patient_summary(report: dict[str, Any]) -> str:
    action_name = str(_compact_report(report).get("action_name") or "本次动作")
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
        f"{action_name}: 主要错误 {_report_error_code(report)}; "
        f"ROM {_fmt(rom.get('actual'))}/{_fmt(rom.get('target'))}; "
        f"TUT {_fmt(tut.get('actual'))}/{_fmt(tut.get('target'))}; "
        f"速度比 {_fmt(speed.get('ratio'), 2)}; DTW {_fmt(dtw.get('normalized_distance'), 2)}。"
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
    card_metrics = report.get("report_card_metrics") if isinstance(report.get("report_card_metrics"), dict) else {}
    if card_metrics.get("primary_error"):
        return str(card_metrics.get("primary_error"))
    structured = report.get("structured_feedback") if isinstance(report.get("structured_feedback"), dict) else {}
    if structured.get("error_code"):
        return str(structured.get("error_code"))
    errors = report.get("errors") if isinstance(report.get("errors"), dict) else {}
    return str(errors.get("primary_error") or "OK")


def _coerce_summary_text(value: Any, report: dict[str, Any] | None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return _local_doctor_summary(report)
    if value is None:
        return _local_doctor_summary(report) if report else ""
    return str(value).strip()


def _network_failure_message(exc: urllib.error.URLError) -> tuple[str, str]:
    reason = getattr(exc, "reason", exc)
    text = str(reason or exc).lower()
    if "timed out" in text or "timeout" in text:
        return "timeout", "LLM 请求超时，请检查网络或调大超时时间。"
    if "connection refused" in text:
        return "network_error", "LLM 服务连接被拒绝，请检查 GLM endpoint 或本地 Qwen proxy 是否启动。"
    if "network is unreachable" in text or "no route to host" in text:
        return "network_error", "板端无法访问网络，请检查热点、网关或路由。"
    if "name or service not known" in text or "temporary failure" in text or "getaddrinfo failed" in text:
        return "network_error", "外网域名解析失败，请检查板端 DNS 或热点网络。"
    return "network_error", "LLM 网络连接失败，请检查板端网络。"


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
    value = " ".join(str(text or "").split())
    return value if len(value) <= max_chars else value[: max_chars - 1] + "…"


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



