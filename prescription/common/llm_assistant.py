from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
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
MAX_SPOKEN_CHARS = 220
QWEN_SMOKE_TTL_SECONDS = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEATED_KNEE_RAISE_GUIDANCE = (
    "坐姿抬膝标准要点：患者坐稳，躯干保持直立，骨盆尽量稳定；目标腿膝盖朝前，"
    "大腿主动上抬到医生模板高度，到位后短暂停住，再控制速度慢慢放下；"
    "避免身体后仰、耸肩、骨盆晃动或用惯性甩腿。"
)
ACTION_GUIDANCE = {
    "坐站训练": (
        "选择稳固、有靠背且不滑动的椅子，双脚与髋同宽并稍收至膝下，身体轻微前倾后用脚跟发力站起。"
        "站稳时保持膝盖朝向脚尖，坐下时髋部向后、缓慢控制下降，避免膝内扣、憋气或直接跌坐；疼痛、头晕时立即停止。"
    ),
    "站姿屈膝后勾腿": (
        "扶稳固定支撑物，躯干直立、骨盆保持中立，两膝尽量并拢，再让训练侧脚跟缓慢向臀部方向靠近。"
        "动作只在舒适范围内完成，避免大腿向前摆、身体前倾或借惯性甩腿，到位后稳住再慢慢放下。"
    ),
    "坐姿抬膝": (
        "坐在稳固椅面上，双脚落地，躯干直立并保持骨盆稳定，让训练侧膝盖朝前、主动抬向医生设定高度。"
        "到位后短暂停稳再缓慢放下，避免身体后仰、耸肩、骨盆晃动或借惯性甩腿。"
    ),
}
ACTION_PURPOSE = {
    "坐站训练": "主要练习下肢伸展力量、重心转移和日常坐下站起的控制能力。",
    "站姿屈膝后勾腿": "主要练习膝关节屈曲控制和大腿后侧肌群的主动发力。",
    "坐姿抬膝": "主要练习髋关节主动屈曲、抬腿控制以及坐位骨盆稳定。",
}
PROJECT_QA_CONTEXT = (
    "本系统运行在RK3588边缘设备上；8085患者训练使用YOLOv5n检测单人，RTMPose输出COCO-17关键点。"
    "医生先录制个体化动作模板，患者训练再比较动作次数、幅度ROM、保持时间TUT、速度和主要错误。"
    "当前稳定演示包含坐站训练、站姿屈膝后勾腿、坐姿抬膝三个下肢动作，动作可通过独立配置和医生模板继续扩展。"
    "训练识别和本地Qwen问答均在板端运行；本地路径不要求上传原始训练视频，报告保存结构化指标和必要关键帧。"
    "系统用于康复训练辅助评估和解释，不进行疾病诊断，不代替医生制定或修改处方。"
    "热量属于估算展示而非临床测量；没有提供明确公式时不得编造计算依据。"
)
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
要求：只基于报告回答；严格使用报告中锁定的动作名称，不得讨论其他动作；严格遵守 qa_answer_policy；第一句点明本次回答对应的动作名称；如果是坐姿抬膝，要说明坐稳、躯干直立、骨盆稳定、膝盖朝前、抬到模板高度、稳住后慢慢放下；不要诊断疾病或调整治疗；回答短、温和、适合朗读；尽量输出 JSON，包含 answer 和 spoken_text。

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
    prompt = QUESTION_PROMPT.format(question=question, report_summary=json.dumps(_question_report_context(report, question), ensure_ascii=False, indent=2))
    answer = _chat_text(settings, prompt, frame_b64=frame_b64)
    parsed = _parse_json_object_or_none(answer)
    if parsed is None:
        return _lock_answer_to_report_action(_fallback_answer_from_text(answer), report)
    return _lock_answer_to_report_action(_normalize_answer_payload(parsed, fallback_text=answer), report)


def _local_qwen_answer(report: dict[str, Any], settings: dict[str, Any], question: str) -> dict[str, Any]:
    intent = _qa_answer_intent(question)
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
            return _qa_provider_fallback(report, question, f"{intent}_provider_error")
        except Exception as exc:
            _mark_qwen_smoke(False, _sanitize_error(exc))
            return _qa_provider_fallback(report, question, f"{intent}_provider_error")
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
            return _qa_provider_fallback(report, question, f"{intent}_provider_error")
        text = str(response_json.get("text") or response_json.get("answer") or response_json.get("response") or "").strip()
        if not text:
            continue
        parsed = _parse_json_object_or_none(text)
        result = _fallback_answer_from_text(text) if parsed is None else _normalize_answer_payload(parsed, fallback_text=text)
        if intent != "general":
            result = _lock_answer_to_report_action(result, report)
        if _qa_answer_needs_rule_fallback(result, report, question):
            result = _qa_provider_fallback(report, question, f"{intent}_response_guard")
        elif _qa_answer_needs_evidence(result, report, question):
            result = _qa_append_evidence(result, report, question)
            result["qwen_fallback_reason"] = "structured_evidence_appended"
        result = _qa_limit_answer_payload(result)
        result["rkllm_latency_ms"] = response_json.get("latency_ms")
        result["rkllm_model"] = response_json.get("model")
        result["qwen_prompt_mode"] = "compact" if index == 0 else "minimal_retry"
        result["qwen_queue_wait_ms"] = response_json.get("queue_wait_ms")
        result["qwen_retry_count"] = response_json.get("retry_count")
        result["qwen_empty_retry_count"] = response_json.get("empty_retry_count")
        _mark_qwen_smoke(True, None)
        return result
    preview = str(last_response.get("upstream_error_preview") or last_raw or "")[:220]
    _mark_qwen_smoke(False, preview or "local qwen returned empty text")
    return _qa_provider_fallback(report, question, f"{intent}_empty_response")


def _qa_provider_fallback(report: dict[str, Any], question: str, reason: str) -> dict[str, Any]:
    intent = _qa_answer_intent(question)
    result = _qa_rule_answer(report, question)
    result["qwen_prompt_mode"] = f"{intent}_local_rules"
    result["qwen_fallback_reason"] = reason
    return result


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
    action_name = _report_action_name(report)
    intent = _qa_answer_intent(question)
    if intent == "guidance":
        guidance = _qa_action_guidance(report, question)
        safe = (
            f"你是康复动作科普助手，只回答《{action_name}》的动作要领。"
            "结合你已有的康复训练常识和下列经审核要点，用自然中文回答2句；不读取训练报告，不提次数、完成度、模板数值或患者刚才表现。"
            "不得诊断、不得调整处方或训练量；最多3句，不要列表，不要输出JSON。"
        )
        compact = f"{safe}\n经审核要点：{guidance}\n患者问题：{question}\n回答："
        minimal = f"只简要说明《{action_name}》动作要领，不读报告、不列数据。参考：{guidance}\n问题：{question}\n回答："
        return [compact[:1200], minimal[:700]]
    if intent == "general":
        facts_text = _qa_session_facts_text(report)
        attempts_text = _qa_attempt_details_text(report)
        safe = (
            "你是国赛康复训练系统的现场问答助手。请直接理解用户原问题的真实意图，不依赖固定关键词库。"
            "如果用户是在用新的说法点评、复盘或追问本次训练，就只依据当前动作的结构化记录回答；"
            "如果用户询问工程、算法、部署、隐私、动作扩展或医疗科普，就依据项目事实和可靠常识回答，不要套用训练次数话术。"
            "不得诊断、不得调整处方；没有提供的项目细节要明确说暂未提供，禁止编造。"
            "回答2到3句，简洁自然，不输出JSON。"
        )
        compact = (
            f"{safe}\n项目事实：{PROJECT_QA_CONTEXT}\n当前动作：{action_name}\n"
            f"本次统计：{facts_text}\n逐次记录：{attempts_text}\n用户原问题：{question}\n回答："
        )
        minimal = (
            f"理解用户问题后直接回答，不套固定话术、不编造。项目事实：{PROJECT_QA_CONTEXT}\n"
            f"当前动作：{action_name}；训练统计：{facts_text}\n问题：{question}\n回答："
        )
        return [compact[:1800], minimal[:1000]]
    facts_text = _qa_session_facts_text(report)
    attempts_text = _qa_attempt_details_text(report)
    answer_policy = _qa_answer_policy(question)
    seated_guidance = f"{SEATED_KNEE_RAISE_GUIDANCE}" if _is_seated_knee_raise_report(report) else ""
    safe = (
        f"你是康复训练报告解释助手。本题锁定动作是《{action_name}》，只能回答这个动作，严禁提及或借用其他动作。"
        "只根据下面的结构化统计事实回答，不得使用或猜测任何摘要、模板残句、其他动作或user字样。"
        "不诊断疾病，不替代医生。请直接用自然中文回答2到3句，第一句点明锁定动作名称，至少引用一个真实数字。不要输出JSON，不要重复题目。"
    )
    compact = f"{safe}\n统计事实：{facts_text}\n逐次记录：{attempts_text}\n回答口径：{answer_policy}\n{seated_guidance}\n患者问题：{question}\n回答："
    minimal = f"锁定动作《{action_name}》，不得提其他动作或摘要文字。统计：{facts_text}\n逐次记录：{attempts_text}\n回答口径：{answer_policy} {seated_guidance}\n问题：{question}\n回答："
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
        "action_name": _report_action_name(report),
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
    if _is_seated_knee_raise_report(report):
        compact["action_guidance"] = SEATED_KNEE_RAISE_GUIDANCE
    compact["overall_quality"] = report.get("overall_quality")
    compact["quality_model"] = _compact_quality_model(report.get("quality_model"))
    compact["selected_attempts"] = _compact_selected_attempts(report.get("selected_attempts"))
    compact["quality_attempts"] = _compact_quality_attempts(report.get("quality_attempts"))
    compact["reps"] = _compact_quality_attempts(report.get("reps"))
    return compact


def _report_action_name(report: dict[str, Any]) -> str:
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    runtime_meta = report.get("runtime_meta") if isinstance(report.get("runtime_meta"), dict) else {}
    action_id = str(
        report.get("action_id")
        or meta.get("action_id")
        or runtime_meta.get("action_id")
        or ""
    )
    return str(
        report.get("action_name")
        or meta.get("action_name")
        or {
            "sit_to_stand": "坐站训练",
            "standing_hamstring_curl": "站姿屈膝后勾腿",
            "seated_knee_raise": "坐姿抬膝",
            "seated_knee_extension": "坐姿伸膝",
            "knee_flexion": "屈膝训练",
        }.get(action_id)
        or action_id
        or "本次动作"
    )


def _qa_session_facts(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("quality_attempts")
    if not isinstance(source, list) or not source:
        source = report.get("reps")
    attempts = _compact_quality_attempts(source, limit=50)
    passed = [item for item in attempts if bool(item.get("countable"))]
    failed = [item for item in attempts if not bool(item.get("countable"))]
    error_counts = Counter(
        str(item.get("primary_error") or "").upper()
        for item in failed
        if str(item.get("primary_error") or "").upper() not in {"", "OK"}
    )
    most_common_error = error_counts.most_common(1)[0] if error_counts else ("OK", 0)
    selected = _compact_selected_attempts(report.get("selected_attempts"))
    overall = report.get("overall_completion")
    if overall is None:
        overall = report.get("overall_quality")
    return {
        "action_name": _report_action_name(report),
        "attempt_count": len(attempts),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "pass_rate": round(len(passed) * 100.0 / len(attempts), 1) if attempts else None,
        "overall_completion": overall,
        "most_common_error": most_common_error[0],
        "most_common_error_count": most_common_error[1],
        "last_attempt": attempts[-1] if attempts else None,
        "best_correct": selected.get("best_correct") or (max(passed, key=_attempt_score) if passed else None),
        "representative_wrong": selected.get("representative_wrong") or (min(failed, key=_attempt_score) if failed else None),
        "attempts": attempts,
    }


def _attempt_score(item: dict[str, Any]) -> float:
    for key in ("completion_percent", "quality_score"):
        try:
            return float(item.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _qa_answer_intent(question: str) -> str:
    value = re.sub(r"\s+", "", str(question or ""))
    if any(word in value for word in (
        "注意什么", "注意事项", "动作要领", "训练要领", "怎么做", "如何做", "正确做法",
        "怎样做", "标准动作", "动作标准", "有什么要领", "注意哪些", "注意点", "姿势要求",
        "正确姿势", "怎么才标准", "介绍一下", "讲讲动作", "训练目的", "有什么作用", "锻炼哪里",
    )):
        return "guidance"
    if any(word in value for word in ("最后一次", "最后一遍", "刚刚那次")):
        return "last_attempt"
    if any(word in value for word in ("最好一次", "最好的一次", "哪次最好", "最佳")):
        return "best_attempt"
    if any(word in value for word in (
        "下一步", "怎么提高", "如何提高", "怎么提升", "如何提升", "改进", "改善", "以后怎么练", "接下来怎么练",
    )):
        return "improvement"
    if (
        any(word in value for word in (
            "哪里不好", "哪儿不好", "什么问题", "哪里做错", "哪里没做好", "错在哪", "问题在哪", "不足",
        ))
        or (("哪里" in value or "哪儿" in value) and ("不好" in value or "问题" in value or "错" in value))
    ):
        return "weakness"
    if any(word in value for word in (
        "做得怎么样", "做的怎么样", "表现怎么样", "练得怎么样", "训练怎么样", "完成情况",
        "训练结果", "训练效果", "上一组", "刚才做得", "刚才练得", "合格吗", "完成得如何",
    )):
        return "overall"
    return "general"


def _qa_answer_policy(question: str) -> str:
    intent = _qa_answer_intent(question)
    return {
        "guidance": "只讲该动作通用要领和安全注意事项，不读取、不引用训练报告数据。",
        "last_attempt": "只评价 last_attempt，不得用最佳动作或整段平均替代最后一次。",
        "best_attempt": "只评价 best_correct，并明确这是最好的一次，不代表每次都一样。",
        "improvement": "根据 most_common_error 和 representative_wrong 给出这个动作的下一步改进，不得引用其他动作。",
        "weakness": "优先说明 most_common_error，再用 representative_wrong 举例；若多数动作合格，也要先肯定合格次数。",
        "overall": "必须汇总全部 attempts：先说总次数、合格次数和未合格次数，再说总体完成度与最常见问题；不得只评价最佳一次或最后一次。",
        "general": "根据已确认的项目事实直接回答开放问题；不要强行套用训练评价，不知道的内容必须明确说明。",
    }[intent]


def _qa_action_guidance(report: dict[str, Any], question: str = "") -> str:
    action_name = _report_action_name(report)
    guidance = ACTION_GUIDANCE.get(
        action_name,
        f"{action_name}应在安全、稳定和无痛范围内缓慢完成，保持身体对齐并避免借力；具体动作幅度和训练量以医生处方为准。",
    )
    compact_question = re.sub(r"\s+", "", str(question or ""))
    if any(word in compact_question for word in ("训练目的", "有什么作用", "锻炼哪里", "练哪里", "为什么做")):
        return f"{ACTION_PURPOSE.get(action_name, '')}{guidance}"
    return guidance


def _qa_limit_answer_text(text: str, max_sentences: int = 3, max_chars: int = 200) -> str:
    clean = " ".join(_clean_answer_text(text).split()).strip()
    if not clean:
        return ""
    sentences = [part.strip() for part in re.findall(r"[^。！？；]+[。！？；]?", clean) if part.strip()]
    limited = "".join(sentences[:max_sentences]).strip()
    if len(limited) <= max_chars:
        return limited
    clipped = limited[:max_chars]
    boundary = max(clipped.rfind(mark) for mark in "。！？；，")
    if boundary >= max_chars // 2:
        clipped = clipped[: boundary + 1]
    else:
        clipped = clipped.rstrip("，。；、 ") + "。"
    return clipped


def _qa_limit_answer_payload(result: dict[str, Any]) -> dict[str, Any]:
    limited = dict(result)
    answer = _qa_limit_answer_text(str(limited.get("answer") or ""))
    spoken = _qa_limit_answer_text(str(limited.get("spoken_text") or answer))
    limited["answer"] = answer
    limited["spoken_text"] = spoken or answer
    return limited


def _question_report_context(report: dict[str, Any], question: str) -> dict[str, Any]:
    intent = _qa_answer_intent(question)
    if intent == "guidance":
        return {
            "qa_locked_action_name": _report_action_name(report),
            "qa_answer_intent": "guidance",
            "qa_answer_policy": _qa_answer_policy(question),
            "action_guidance": _qa_action_guidance(report, question),
        }
    if intent == "general":
        return {
            "qa_answer_intent": "general",
            "qa_answer_policy": _qa_answer_policy(question),
            "project_context": PROJECT_QA_CONTEXT,
            "current_action_name": _report_action_name(report),
        }
    context = _compact_report(report)
    context["qa_locked_action_name"] = _report_action_name(report)
    context["qa_answer_intent"] = _qa_answer_intent(question)
    context["qa_answer_policy"] = _qa_answer_policy(question)
    context["qa_session_facts"] = _qa_session_facts(report)
    return context


def _qa_session_facts_text(report: dict[str, Any]) -> str:
    facts = _qa_session_facts(report)
    total = int(facts.get("attempt_count") or 0)
    passed = int(facts.get("passed_count") or 0)
    failed = int(facts.get("failed_count") or 0)
    parts = [f"动作={facts.get('action_name')}，共{total}次尝试，{passed}次合格，{failed}次未合格"]
    completion = facts.get("overall_completion")
    if isinstance(completion, (int, float)):
        parts.append(f"总体完成度={float(completion):.2f}%")
    scores = [
        item.get("completion_percent") if item.get("completion_percent") is not None else item.get("quality_score")
        for item in facts.get("attempts", [])
    ]
    scores = [score for score in scores if isinstance(score, (int, float))]
    if scores:
        parts.append("逐次完成度=" + "/".join(_fmt(score, 2) + "%" for score in scores))
    error_code = str(facts.get("most_common_error") or "OK")
    error_count = int(facts.get("most_common_error_count") or 0)
    if error_code != "OK" and error_count:
        parts.append(f"最常见问题={_qa_error_label(error_code)}({error_count}次)")
    last_attempt = facts.get("last_attempt") if isinstance(facts.get("last_attempt"), dict) else None
    if last_attempt:
        parts.append(f"最后一次={_qa_error_label(str(last_attempt.get('primary_error') or 'OK'))}")
    return "；".join(parts) + "。"


def _qa_attempt_detail_text(item: dict[str, Any]) -> str:
    index = item.get("attempt_index") or item.get("rep_index") or "?"
    status = "合格" if bool(item.get("countable")) else "未合格"
    parts = [f"第{index}次{status}"]
    completion = item.get("completion_percent")
    if completion is None:
        completion = item.get("quality_score")
    if completion is not None:
        parts.append(f"完成度{_fmt(completion, 2)}%")
    rom = item.get("rom")
    rom_target = item.get("rom_target")
    if rom is not None or rom_target is not None:
        parts.append(f"幅度{_fmt(rom, 2)}（模板{_fmt(rom_target, 2)}）")
    tut = item.get("tut_seconds")
    tut_target = item.get("tut_target")
    if tut is not None or tut_target is not None:
        parts.append(f"保持{_fmt(tut, 2)}秒（模板{_fmt(tut_target, 2)}秒）")
    error = str(item.get("primary_error") or "OK").upper()
    if error != "OK":
        parts.append(f"问题{_qa_error_label(error)}")
    return "，".join(parts)


def _qa_attempt_details_text(report: dict[str, Any]) -> str:
    attempts = _qa_session_facts(report).get("attempts") or []
    return "；".join(_qa_attempt_detail_text(item) for item in attempts) or "暂无逐次动作数据"


def _qa_error_label(error_code: str) -> str:
    return {
        "OK": "动作达标",
        "ROM_LOW": "动作幅度不足",
        "TUT_LOW": "保持时间不足",
        "TOO_FAST": "动作速度过快",
        "EARLY_RETURN": "回落过早",
        "SHAPE_BAD": "动作轨迹不稳定",
        "VISIBILITY_LOW": "关键点可见性不足",
    }.get(str(error_code or "").upper(), "动作需要改进")


def _qa_error_advice(error_code: str, report: dict[str, Any]) -> str:
    code = str(error_code or "").upper()
    if _is_seated_knee_raise_report(report):
        return _seated_knee_raise_answer(code)
    return {
        "OK": "继续保持当前动作幅度、保持时间和稳定节奏。",
        "ROM_LOW": "下一次在安全范围内把动作幅度做得更完整。",
        "TUT_LOW": "到达目标位置后先稳定保持，再缓慢返回。",
        "TOO_FAST": "下一次放慢速度，避免借力或突然回落。",
        "EARLY_RETURN": "到位后不要马上返回，确认保持完成后再回到起点。",
        "SHAPE_BAD": "先控制动作轨迹和身体稳定，再追求动作幅度。",
        "VISIBILITY_LOW": "保持身体和目标关节完整入画，避免关键点中途丢失。",
    }.get(code, "请优先修正报告中最常见的问题。")


def _qa_general_fallback(question: str) -> str:
    value = re.sub(r"\s+", "", str(question or ""))
    if any(word in value for word in ("模型", "算法", "识别", "YOLO", "RTMPose", "关键点")):
        return "系统先用YOLOv5n定位单个训练者，再由RTMPose提取COCO-17关键点，并结合医生模板计算次数、动作幅度、保持时间和速度。"
    if any(word in value for word in ("落地", "部署", "硬件", "板端", "边缘", "离线")):
        return "系统部署在RK3588边缘设备上，姿态识别和本地Qwen问答都可在板端完成，减少网络依赖并便于现场独立运行。"
    if any(word in value for word in ("隐私", "数据", "上传", "视频")):
        return "本地训练路径在板端处理摄像头画面，不要求上传原始训练视频；系统主要保存结构化训练指标、报告和必要关键帧。"
    if any(word in value for word in ("动作少", "几个动作", "扩展动作", "增加动作", "更多动作")):
        return "当前稳定演示包含三个下肢动作，动作指标、反馈规则和医生模板采用独立配置，后续可以按同样边界扩展上肢和全身动作。"
    if any(word in value for word in ("热量", "卡路里", "能量消耗")):
        return "报告中的热量属于展示性估算，不是临床测量，也不能替代代谢设备结果；没有明确公式和个体参数时不应把它解释为精确消耗。"
    return "这个问题需要调用本地Qwen结合项目事实回答；当前模型不可用时，我只能确认系统用于康复训练辅助评估，不进行疾病诊断，也不替代医生处方。"


def _lock_answer_to_report_action(payload: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    action_name = _report_action_name(report)
    answer = _clean_answer_text(str(result.get("answer") or ""))
    if answer and action_name not in answer:
        answer = f"{action_name}：{answer}"
    spoken = _clean_answer_text(str(result.get("spoken_text") or answer))
    if spoken and action_name not in spoken:
        spoken = f"{action_name}：{spoken}"
    result["answer"] = answer
    result["spoken_text"] = _shorten(spoken)
    result["qa_action_name"] = action_name
    return result


def _qa_answer_needs_rule_fallback(result: dict[str, Any], report: dict[str, Any], question: str) -> bool:
    """Reject only contaminated or action-mismatched model text."""
    answer = " ".join(
        part for part in (str(result.get("answer") or ""), str(result.get("spoken_text") or "")) if part
    )
    action_name = _report_action_name(report)
    intent = _qa_answer_intent(question)
    if not answer:
        return True
    if intent != "general" and action_name not in answer:
        return True
    if intent != "general":
        other_actions = set(ACTION_GUIDANCE) - {action_name}
        if any(other_action in answer for other_action in other_actions):
            return True
    bad_fragments = (
        "根据摘要",
        "摘要内容",
        "多数 user",
        "user坐",
        "重新录制一版动作更标准的结果是",
        "坐不对呢",
        "锛",
        "鏂",
        "璇",
        "浼",
        "绔",
        "鐨",
        "杩",
        "鍒",
        "�",
    )
    if any(fragment in answer for fragment in bad_fragments):
        return True
    if intent == "guidance":
        if any(fragment in answer for fragment in ("本次", "合格", "未合格", "完成度", "第1次", "第2次", "第3次")):
            return True
    if intent == "general" and action_name in answer and any(
        fragment in answer for fragment in ("次合格", "次未合格", "次不合格", "合格率")
    ):
        if not _qa_overall_answer_preserves_counts(answer, report):
            return True
    if intent == "overall" and not _qa_overall_answer_preserves_counts(answer, report):
        return True
    if intent in {"weakness", "improvement"}:
        facts = _qa_session_facts(report)
        if int(facts.get("failed_count") or 0) > 0:
            expected_error = _qa_error_label(str(facts.get("most_common_error") or "OK"))
            if expected_error not in answer:
                return True
    return False


def _qa_overall_answer_preserves_counts(answer: str, report: dict[str, Any]) -> bool:
    facts = _qa_session_facts(report)
    total = int(facts.get("attempt_count") or 0)
    if total <= 0:
        return True
    passed = int(facts.get("passed_count") or 0)
    failed = int(facts.get("failed_count") or 0)
    compact = re.sub(r"\s+", "", str(answer or ""))

    def has_count(count: int, labels: tuple[str, ...]) -> bool:
        patterns = []
        for label in labels:
            patterns.extend((rf"{count}次{label}", rf"{label}{count}次"))
        return any(re.search(pattern, compact) for pattern in patterns)

    total_ok = has_count(total, ("尝试", "动作", "训练")) or f"共{total}次" in compact
    passed_ok = has_count(passed, ("合格", "达标", "通过"))
    failed_ok = has_count(failed, ("未合格", "不合格", "未达标", "需要改进"))
    return total_ok and passed_ok and failed_ok


def _qa_answer_needs_evidence(result: dict[str, Any], report: dict[str, Any], question: str) -> bool:
    answer = str(result.get("answer") or "")
    facts = _qa_session_facts(report)
    if facts.get("attempt_count") and _qa_answer_intent(question) in {
        "overall", "last_attempt", "best_attempt", "weakness", "improvement"
    }:
        if not re.search(r"\d", answer):
            return True
        if not any(token in answer for token in ("完成度", "幅度", "保持", "模板", "合格率", "秒")):
            return True
    return False


def _qa_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _qa_metric_comparison(actual: Any, target: Any, label: str, unit: str = "") -> str:
    actual_value = _qa_number(actual)
    target_value = _qa_number(target)
    if actual_value is None and target_value is None:
        return ""
    if actual_value is None or target_value is None:
        return f"{label}{_fmt(actual_value, 2)}{unit}"
    gap = target_value - actual_value
    if gap > 0.005:
        return f"{label}{actual_value:.2f}{unit}，模板{target_value:.2f}{unit}，少{gap:.2f}{unit}"
    return f"{label}{actual_value:.2f}{unit}，已达到模板{target_value:.2f}{unit}"


def _qa_evidence_attempt(report: dict[str, Any], question: str) -> dict[str, Any] | None:
    facts = _qa_session_facts(report)
    attempts = facts.get("attempts") or []
    intent = _qa_answer_intent(question)
    if intent == "last_attempt":
        return facts.get("last_attempt") if isinstance(facts.get("last_attempt"), dict) else None
    if intent == "best_attempt":
        return facts.get("best_correct") if isinstance(facts.get("best_correct"), dict) else None
    if intent in {"weakness", "improvement", "overall"} and isinstance(facts.get("representative_wrong"), dict):
        return facts.get("representative_wrong")
    if intent == "weakness" and attempts:
        return min(attempts, key=_attempt_score)
    return facts.get("best_correct") if isinstance(facts.get("best_correct"), dict) else (attempts[-1] if attempts else None)


def _qa_evidence_sentence(report: dict[str, Any], question: str) -> str:
    item = _qa_evidence_attempt(report, question)
    if not isinstance(item, dict):
        return ""
    index = item.get("attempt_index") or item.get("rep_index") or "?"
    completion = item.get("completion_percent")
    if completion is None:
        completion = item.get("quality_score")
    error = str(item.get("primary_error") or "OK").upper()
    rom = _qa_metric_comparison(item.get("rom"), item.get("rom_target"), "动作幅度")
    tut = _qa_metric_comparison(item.get("tut_seconds"), item.get("tut_target"), "保持", "秒")
    ordered = [tut, rom] if error == "TUT_LOW" else [rom, tut]
    metrics = "；".join(part for part in ordered if part)
    score_text = f"完成度{_fmt(completion, 2)}%" if completion is not None else ""
    prefix = f"第{index}次{score_text}" if score_text else f"第{index}次"
    return f"{prefix}，{metrics}" if metrics else prefix


def _qa_append_evidence(result: dict[str, Any], report: dict[str, Any], question: str) -> dict[str, Any]:
    evidence = _qa_evidence_sentence(report, question)
    if not evidence:
        return result
    answer = _clean_answer_text(str(result.get("answer") or "")).rstrip("。；; ")
    combined = f"{answer}。具体数据是：{evidence}。"
    enriched = dict(result)
    enriched["answer"] = combined
    enriched["spoken_text"] = _shorten(combined)
    return enriched


def _qa_natural_next_step(report: dict[str, Any], item: dict[str, Any] | None, error: str) -> str:
    action_name = _report_action_name(report)
    code = str(error or "OK").upper()
    item = item if isinstance(item, dict) else {}
    if code == "TUT_LOW":
        actual = _qa_number(item.get("tut_seconds"))
        target = _qa_number(item.get("tut_target"))
        gap = max(0.0, target - actual) if actual is not None and target is not None else None
        extra = f"约{gap:.2f}秒" if gap is not None and gap > 0.005 else "片刻"
        if action_name == "坐站训练":
            return f"下一次站起到位后多稳住{extra}，再缓慢坐下。"
        if action_name == "站姿屈膝后勾腿":
            return f"下一次脚跟勾到位后多稳住{extra}，再控制小腿放下。"
        if action_name == "坐姿抬膝":
            return f"下一次膝盖抬到位后多稳住{extra}，再慢慢放下。"
        return f"下一次到位后多稳住{extra}，再缓慢返回。"
    if code == "ROM_LOW":
        if action_name == "坐站训练":
            return "下一次把髋膝伸展做完整，站稳后再坐下。"
        if action_name == "站姿屈膝后勾腿":
            return "下一次让脚跟再向后上方勾一些，躯干保持稳定。"
        if action_name == "坐姿抬膝":
            return "下一次保持坐稳，让膝盖再主动抬高一些。"
    return _qa_error_advice(code, report)


def _qa_rule_answer(report: dict[str, Any], question: str) -> dict[str, Any]:
    facts = _qa_session_facts(report)
    action_name = str(facts.get("action_name") or _report_action_name(report))
    attempts = facts.get("attempts") or []
    intent = _qa_answer_intent(question)
    total = int(facts.get("attempt_count") or 0)
    passed = int(facts.get("passed_count") or 0)
    failed = int(facts.get("failed_count") or 0)
    evidence_item = _qa_evidence_attempt(report, question)
    evidence = _qa_evidence_sentence(report, question)
    scores = [
        item.get("completion_percent") if item.get("completion_percent") is not None else item.get("quality_score")
        for item in attempts
    ]
    scores = [score for score in scores if isinstance(score, (int, float))]
    average_score = sum(float(score) for score in scores) / len(scores) if scores else None
    if intent == "guidance":
        answer = f"{action_name}：{_qa_action_guidance(report, question)}"
    elif intent == "general":
        answer = _qa_general_fallback(question)
    elif not total:
        answer = f"{action_name}暂时没有可用的逐次动作数据，请先完成一轮训练后再问我。"
    elif intent == "overall":
        answer = f"{action_name}本次{total}次尝试中，{passed}次合格、{failed}次未合格。"
        if average_score is not None:
            answer += f"平均完成度{average_score:.2f}%。"
        if failed:
            answer += f"未合格主要因为{_qa_error_label(str(facts.get('most_common_error') or 'OK'))}。"
        else:
            answer += "这一组整体完成稳定，继续保持当前节奏。"
    elif intent == "last_attempt" and isinstance(facts.get("last_attempt"), dict):
        last_error = str(facts["last_attempt"].get("primary_error") or "OK").upper()
        answer = f"{action_name}最后一次{'已经合格' if last_error == 'OK' else '需要改进'}：{evidence}。"
        if last_error != "OK":
            answer += _qa_natural_next_step(report, facts["last_attempt"], last_error)
    elif intent == "best_attempt" and isinstance(facts.get("best_correct"), dict):
        answer = f"{action_name}最好的是{evidence}。这是本次记录中的最高完成度，但仍要保持每次动作同样稳定。"
    elif intent == "weakness":
        if failed:
            error = str(facts.get("most_common_error") or "OK")
            answer = f"{action_name}这次主要问题是{_qa_error_label(error)}：{evidence}。"
            if passed:
                answer += f"其余{passed}次已经合格。"
        else:
            answer = f"{action_name}本次{total}次都合格，没有发现重复性错误。"
            if evidence:
                answer += f"相对较弱的一次是{evidence}，但仍达到计数标准。"
    elif intent == "improvement":
        if failed:
            error = str(facts.get("most_common_error") or "OK")
            answer = f"{action_name}下一组先把重点放在{_qa_error_label(error)}上。"
            answer += _qa_natural_next_step(report, evidence_item, error)
            rom = _qa_number((evidence_item or {}).get("rom"))
            rom_target = _qa_number((evidence_item or {}).get("rom_target"))
            if rom is not None and rom_target is not None and rom >= rom_target:
                answer += "你的动作幅度已经达到模板，保持这个基础，把每次动作做得更一致。"
            else:
                answer += "先完成这个小目标，不必着急加快速度，你会越来越稳定。"
        else:
            answer = f"{action_name}这一组已经全部合格。下一组继续保持慢、稳、到位，并把每次动作的节奏做得更一致。"
    elif evidence:
        answer = f"{action_name}本次记录显示：{evidence}。请继续以医生模板和页面反馈为准。"
    else:
        answer = f"{action_name}本次没有足够的逐次数据，暂时无法给出更具体的判断。"
    result = _normalize_answer_payload({"answer": answer, "spoken_text": answer})
    if intent != "general":
        result = _lock_answer_to_report_action(result, report)
    return _qa_limit_answer_payload(result)



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
                "completion_percent": item.get("completion_percent"),
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


def _report_action_text(report: dict[str, Any]) -> str:
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    runtime_meta = report.get("runtime_meta") if isinstance(report.get("runtime_meta"), dict) else {}
    values = [
        report.get("action_id"),
        report.get("action_name"),
        meta.get("action_id"),
        meta.get("action_name"),
        runtime_meta.get("action_id"),
        runtime_meta.get("action_name"),
    ]
    return " ".join(str(item) for item in values if item)


def _is_seated_knee_raise_report(report: dict[str, Any]) -> bool:
    action_text = _report_action_text(report)
    return "seated_knee_raise" in action_text or "坐姿抬膝" in action_text or "抬膝" in action_text


def _seated_knee_raise_answer(error_code: str) -> str:
    if error_code == "OK":
        return "坐姿抬膝这组整体完成不错。标准要点是坐稳、躯干直立、骨盆稳定，膝盖朝前抬到模板高度，稳住后再慢慢放下。"
    if error_code == "ROM_LOW":
        return "坐姿抬膝主要是抬膝高度还不够。下一次先坐稳、躯干别后仰，骨盆保持稳定，目标腿膝盖朝前，大腿主动抬到医生模板高度。"
    if error_code == "TUT_LOW":
        return "坐姿抬膝主要是到位后保持时间偏短。膝盖抬到医生模板高度后先稳住，别马上回落，再控制速度慢慢放下。"
    if error_code in {"SHAPE_BAD", "TOO_FAST", "EARLY_RETURN"}:
        return "坐姿抬膝要避免身体后仰或用惯性甩腿。请保持躯干直立、骨盆稳定，膝盖朝前主动抬起，抬到位后再控制放下。"
    return "坐姿抬膝还有需要调整的地方。下一组先坐稳、躯干直立、骨盆稳定，膝盖朝前抬到模板高度，稳住后再慢慢放下。"

def _echo_answer(report: dict[str, Any], question: str) -> dict[str, Any]:
    compact = _compact_report(report)
    metrics = compact["metrics"]
    facts = _qa_session_facts(report)
    intent = _qa_answer_intent(question)
    if intent in {"guidance", "overall", "weakness", "improvement", "last_attempt", "best_attempt", "general"}:
        return _qa_rule_answer(report, question)
    error_code = str(facts.get("most_common_error") or _report_error_code(report))
    action_name = _report_action_name(report)
    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    if intent == "overall" and facts.get("attempt_count"):
        answer = (
            f"{action_name}本次共完成{facts['attempt_count']}次尝试，"
            f"其中{facts['passed_count']}次合格、{facts['failed_count']}次需要改进。"
        )
        if error_code != "OK":
            answer += f"最常见的问题是{_qa_error_label(error_code)}。"
        elif isinstance(facts.get("overall_completion"), (int, float)):
            answer += f"总体完成度约为{float(facts['overall_completion']):.1f}%。"
    elif intent == "last_attempt" and isinstance(facts.get("last_attempt"), dict):
        last_error = str(facts["last_attempt"].get("primary_error") or "OK")
        answer = f"{action_name}最后一次的结果是{_qa_error_label(last_error)}。"
    elif intent == "best_attempt" and isinstance(facts.get("best_correct"), dict):
        score = _attempt_score(facts["best_correct"])
        answer = f"{action_name}最好的一次动作达标，完成度约为{score:.1f}%。"
    elif intent in {"weakness", "improvement"}:
        answer = f"{action_name}最常见的问题是{_qa_error_label(error_code)}。{_qa_error_advice(error_code, report)}"
    elif _is_seated_knee_raise_report(report) and ("哪里" in question or "不好" in question or "问题" in question or "注意" in question or "介绍" in question or "标准" in question or "怎么" in question or "要领" in question):
        answer = _seated_knee_raise_answer(error_code)
    elif "哪里" in question or "不好" in question or "问题" in question:
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
    return _lock_answer_to_report_action(_normalize_answer_payload({"answer": answer, "spoken_text": answer}), report)


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
    return {"answer": answer, "spoken_text": _shorten(spoken)}


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
    if _is_seated_knee_raise_report(report):
        return _seated_knee_raise_answer(error_code)
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
    if report and _is_seated_knee_raise_report(report):
        if error_code == "ROM_LOW":
            return ["下一组先坐稳、骨盆保持稳定，再把目标腿膝盖主动抬到医生模板高度。"]
        if error_code == "TUT_LOW":
            return ["抬到模板高度后先稳住，再慢慢放下，不要马上回落。"]
        return ["保持躯干直立、膝盖朝前，按医生模板完成抬膝和控制放下。"]
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






