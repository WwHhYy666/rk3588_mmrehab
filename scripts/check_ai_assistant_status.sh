#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8085}"
LLM_STATUS_URL="${LLM_STATUS_URL:-${BASE_URL}/api/llm/status?force=1}"
VOICE_STATUS_URL="${VOICE_STATUS_URL:-${BASE_URL}/api/voice/status}"

python3 - "$LLM_STATUS_URL" "$VOICE_STATUS_URL" <<'PY'
import json
import sys
import urllib.request

llm_url = sys.argv[1]
voice_url = sys.argv[2]

def read_json(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))

try:
    llm_payload = read_json(llm_url)
except Exception as exc:
    print(f"[FAIL] cannot read {llm_url}: {exc}")
    sys.exit(1)
try:
    voice_payload = read_json(voice_url)
except Exception as exc:
    voice_payload = {"ok": False, "voice": {}, "error": str(exc)}

llm = llm_payload.get("llm") or {}
voice = voice_payload.get("voice") or {}
llm_jobs = voice.get("llm_jobs") or {}
current_job = llm_jobs.get("current_job") or {}
last_job = llm_jobs.get("last_completed_job") or {}
if not last_job:
    recent = llm_jobs.get("recent_jobs") or []
    for item in reversed(recent):
        if item.get("status") in {"done", "failed", "blocked_training"}:
            last_job = item
            break

def show(key, value):
    if value is None or value == "":
        value = "-"
    print(f"{key} = {value}")

show("llm.provider", llm.get("provider"))
show("llm.expected_provider_now", llm.get("active_provider"))
show("llm.api_key_configured", llm.get("api_key_configured"))
show("llm.glm_endpoint", llm.get("glm_endpoint"))
show("llm.glm_endpoint_reachable", llm.get("glm_endpoint_reachable"))
show("llm.qwen_health_ok", llm.get("rkllm_server_reachable"))
show("llm.qwen_generate_ok", llm.get("qwen_generate_ok"))
show("llm.qwen_generate_cached", llm.get("qwen_generate_cached"))
show("llm.qwen_generate_age_seconds", llm.get("qwen_generate_age_seconds"))
show("llm.qwen_generate_error", llm.get("qwen_generate_error"))
show("llm.status_cached", llm.get("cached"))
show("llm.last_active_provider", llm.get("last_active_provider"))
show("llm.fallback_reason", llm.get("fallback_reason"))
show("llm.last_error", llm.get("last_error"))
show("voice.qa_allowed", voice.get("qa_allowed"))
show("voice.training_status", voice.get("training_status"))
show("voice.llm_queue_size", llm_jobs.get("queue_size"))
show("current_job.status", current_job.get("status"))
show("current_job.active_provider", current_job.get("active_provider"))
show("current_job.report_file", current_job.get("report_file") or current_job.get("source_report_file"))
show("last_done.status", last_job.get("status"))
show("last_done.active_provider", last_job.get("active_provider"))
show("last_done.report_file", last_job.get("report_file") or last_job.get("source_report_file"))
show("last_done.error", last_job.get("error") or llm_jobs.get("last_error"))

if llm.get("api_key_configured") and llm.get("glm_endpoint_reachable") is not True:
    sys.exit(2)
if not llm.get("api_key_configured") and llm.get("qwen_generate_ok") is not True:
    sys.exit(3)
PY
