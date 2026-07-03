from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_CHAT_ENDPOINT = "http://127.0.0.1:8080/rkllm_chat"
DEFAULT_TIMEOUT_SECONDS = 120.0
MODEL_NAME = "qwen2.5-1.5b-rkllm"
BUSY_BACKOFF_SECONDS = (0.8, 1.2, 1.8)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _parse_upstream_location(endpoint: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported upstream scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _tcp_reachable(endpoint: str, timeout: float = 1.0) -> tuple[bool, str | None]:
    try:
        host, port = _parse_upstream_location(endpoint)
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except Exception as exc:
        return False, str(exc)


def _extract_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, (int, float, bool)):
        return str(payload).strip()
    if isinstance(payload, list):
        parts = [_extract_text(item) for item in payload]
        return "\n".join(part for part in parts if part).strip()
    if not isinstance(payload, dict):
        return str(payload or "").strip()

    for key in ("text", "answer", "response", "content", "result", "output", "generated_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            nested = _extract_text(value)
            if nested:
                return nested

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if isinstance(message, dict):
        nested = _extract_text(message)
        if nested:
            return nested

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            nested = _extract_text(choice)
            if nested:
                return nested
            if isinstance(choice, dict):
                delta = choice.get("delta")
                nested = _extract_text(delta)
                if nested:
                    return nested

    data = payload.get("data")
    if isinstance(data, (dict, list, str)):
        return _extract_text(data)
    return ""


def _extract_text_from_stream(raw: str) -> str:
    parts: list[str] = []
    for line in str(raw or "").splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("data:"):
            value = value[5:].strip()
        if value in {"[DONE]", "DONE"}:
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            if not value.startswith("<"):
                parts.append(value)
            continue
        text = _extract_text(parsed)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _raw_response_has_text(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return bool(_extract_text_from_stream(raw))
    return bool(_extract_text(parsed))


def _short_retry_prompt(prompt: str) -> str:
    value = " ".join(str(prompt or "").split())
    for marker in ("患者问题：", "问题："):
        index = value.rfind(marker)
        if index >= 0:
            question = value[index + len(marker):].strip()[:120]
            if question:
                return f"请只用一句中文直接回答这个康复问题：{question}"
    return f"请只用一句中文回答：{value[:180]}"


def _looks_busy(status: int, detail: str) -> bool:
    value = str(detail or "").lower()
    return status == 503 and ("busy" in value or "try again" in value or "稍后" in value)


class UpstreamHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = int(status)
        self.detail = str(detail or "")
        super().__init__(f"upstream HTTP {self.status}: {self.detail}")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RKLLMProxyHandler(BaseHTTPRequestHandler):
    server_version = "rehab-rkllm-proxy/1.0"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] != "/health":
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        reachable, error = _tcp_reachable(self.server.chat_endpoint)
        self._send_json(
            {
                "ok": reachable,
                "model": MODEL_NAME,
                "upstream": self.server.chat_endpoint,
                "upstream_reachable": reachable,
                "error": error,
                "generate_busy": self.server.generate_lock.locked(),
                "last_generate_ok": self.server.last_generate_ok,
                "last_generate_at": self.server.last_generate_at,
                "last_generate_error": self.server.last_generate_error,
            },
            status=200 if reachable else 503,
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path != "/generate":
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return

        started = time.monotonic()
        queue_wait_ms = 0
        retry_count = 0
        empty_retry_count = 0
        request_id = ""
        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json({"ok": False, "error": "missing prompt"}, status=400)
                return

            request_id = str(payload.get("request_id") or f"proxy_{int(time.time() * 1000)}")
            upstream_payload: dict[str, Any] = {
                "prompt": prompt,
                "query": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if payload.get("max_new_tokens") is not None:
                upstream_payload["max_new_tokens"] = payload.get("max_new_tokens")
                upstream_payload["max_tokens"] = payload.get("max_new_tokens")
            if payload.get("temperature") is not None:
                upstream_payload["temperature"] = payload.get("temperature")

            queue_started = time.monotonic()
            with self.server.generate_lock:
                queue_wait_ms = int((time.monotonic() - queue_started) * 1000)
                upstream_status, raw_response, retry_count, empty_retry_count = self._generate_upstream(prompt, upstream_payload)

            try:
                upstream_json = json.loads(raw_response)
            except json.JSONDecodeError:
                text = _extract_text_from_stream(raw_response)
                upstream_format = "stream_or_plain_text"
                upstream_keys: list[str] = []
            else:
                text = _extract_text(upstream_json)
                upstream_format = "json"
                upstream_keys = sorted(upstream_json.keys()) if isinstance(upstream_json, dict) else []

            latency_ms = int((time.monotonic() - started) * 1000)
            if not text:
                self.server.last_generate_ok = False
                self.server.last_generate_at = time.time()
                self.server.last_generate_error = "upstream response missing text"
                self._send_json(
                    {
                        "ok": False,
                        "error": "upstream response missing text",
                        "upstream_status": upstream_status,
                        "upstream_error_preview": raw_response[:500],
                        "upstream_keys": upstream_keys,
                        "upstream_format": upstream_format,
                        "queue_wait_ms": queue_wait_ms,
                        "retry_count": retry_count,
                        "empty_retry_count": empty_retry_count,
                        "latency_ms": latency_ms,
                        "model": MODEL_NAME,
                        "request_id": request_id,
                    },
                    status=502,
                )
                return

            self.server.last_generate_ok = True
            self.server.last_generate_at = time.time()
            self.server.last_generate_error = None
            self._send_json(
                {
                    "ok": True,
                    "text": text,
                    "latency_ms": latency_ms,
                    "model": MODEL_NAME,
                    "request_id": request_id,
                    "upstream_status": upstream_status,
                    "upstream_format": upstream_format,
                    "queue_wait_ms": queue_wait_ms,
                    "retry_count": retry_count,
                    "empty_retry_count": empty_retry_count,
                }
            )
        except UpstreamHTTPError as exc:
            self.server.last_generate_ok = False
            self.server.last_generate_at = time.time()
            self.server.last_generate_error = str(exc)
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "upstream_status": exc.status,
                    "upstream_error_preview": exc.detail[:500],
                    "queue_wait_ms": queue_wait_ms,
                    "retry_count": retry_count,
                    "empty_retry_count": empty_retry_count,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                    "request_id": request_id,
                },
                status=502,
            )
        except TimeoutError as exc:
            self.server.last_generate_ok = False
            self.server.last_generate_at = time.time()
            self.server.last_generate_error = str(exc)
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "queue_wait_ms": queue_wait_ms,
                    "retry_count": retry_count,
                    "empty_retry_count": empty_retry_count,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                    "request_id": request_id,
                },
                status=504,
            )
        except Exception as exc:
            self.server.last_generate_ok = False
            self.server.last_generate_at = time.time()
            self.server.last_generate_error = str(exc)
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "queue_wait_ms": queue_wait_ms,
                    "retry_count": retry_count,
                    "empty_retry_count": empty_retry_count,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                    "request_id": request_id,
                },
                status=500,
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rkllm-proxy] {self.address_string()} - {fmt % args}")

    def _generate_upstream(self, prompt: str, payload: dict[str, Any]) -> tuple[int, str, int, int]:
        retry_count = 0
        empty_retry_count = 0
        status, raw, busy_retries = self._post_upstream_with_retries(payload)
        retry_count += busy_retries
        if _raw_response_has_text(raw):
            return status, raw, retry_count, empty_retry_count

        retry_payload = dict(payload)
        short_prompt = _short_retry_prompt(prompt)
        retry_payload["prompt"] = short_prompt
        retry_payload["query"] = short_prompt
        retry_payload["messages"] = [{"role": "user", "content": short_prompt}]
        retry_payload["temperature"] = 0.0
        retry_payload["max_new_tokens"] = min(int(retry_payload.get("max_new_tokens") or 64), 64)
        retry_payload["max_tokens"] = retry_payload["max_new_tokens"]
        empty_retry_count += 1
        status, raw, busy_retries = self._post_upstream_with_retries(retry_payload)
        retry_count += busy_retries
        return status, raw, retry_count, empty_retry_count

    def _post_upstream_with_retries(self, payload: dict[str, Any]) -> tuple[int, str, int]:
        busy_retries = 0
        for attempt in range(len(BUSY_BACKOFF_SECONDS) + 1):
            try:
                status, raw = self._post_upstream(payload)
                return status, raw, busy_retries
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if _looks_busy(int(exc.code), detail) and attempt < len(BUSY_BACKOFF_SECONDS):
                    busy_retries += 1
                    time.sleep(BUSY_BACKOFF_SECONDS[attempt])
                    continue
                raise UpstreamHTTPError(int(exc.code), detail) from exc
        raise TimeoutError("upstream busy retry exhausted")

    def _post_upstream(self, payload: dict[str, Any]) -> tuple[int, str]:
        body = _json_bytes(payload)
        request = urllib.request.Request(
            self.server.chat_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.server.timeout_seconds) as response:
            return int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace")


class RKLLMProxyServer(ThreadedHTTPServer):
    chat_endpoint: str
    timeout_seconds: float
    generate_lock: threading.Lock
    last_generate_ok: bool | None
    last_generate_at: float | None
    last_generate_error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy official RKLLM Flask /rkllm_chat to /generate.")
    parser.add_argument("--host", default=os.getenv("REHAB_RKLLM_PROXY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("REHAB_RKLLM_PROXY_PORT", str(DEFAULT_PORT))))
    parser.add_argument(
        "--chat-endpoint",
        default=os.getenv("REHAB_RKLLM_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("REHAB_RKLLM_PROXY_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = RKLLMProxyServer((args.host, args.port), RKLLMProxyHandler)
    server.chat_endpoint = args.chat_endpoint
    server.timeout_seconds = args.timeout
    server.generate_lock = threading.Lock()
    server.last_generate_ok = None
    server.last_generate_at = None
    server.last_generate_error = None
    print(f"[rkllm-proxy] listening on http://{args.host}:{args.port}")
    print(f"[rkllm-proxy] upstream {args.chat_endpoint}")
    server.serve_forever()


if __name__ == "__main__":
    main()
