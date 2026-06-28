from __future__ import annotations

import argparse
import json
import os
import socket
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
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return str(payload or "").strip()
    for key in ("text", "answer", "response", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text") or item.get("content") or ""))
                        else:
                            parts.append(str(item))
                    return "\n".join(part for part in parts if part).strip()
            value = first.get("text") or first.get("content")
            if isinstance(value, str):
                return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_text(data)
    return ""


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
            },
            status=200 if reachable else 503,
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path != "/generate":
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        started = time.monotonic()
        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json({"ok": False, "error": "missing prompt"}, status=400)
                return
            request_id = str(payload.get("request_id") or f"proxy_{int(time.time() * 1000)}")
            upstream_payload = {
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if payload.get("max_new_tokens") is not None:
                upstream_payload["max_new_tokens"] = payload.get("max_new_tokens")
            if payload.get("temperature") is not None:
                upstream_payload["temperature"] = payload.get("temperature")
            raw_response = self._post_upstream(upstream_payload)
            upstream_json = json.loads(raw_response)
            text = _extract_text(upstream_json)
            if not text:
                raise ValueError("upstream response missing text")
            latency_ms = int((time.monotonic() - started) * 1000)
            self._send_json(
                {
                    "ok": True,
                    "text": text,
                    "latency_ms": latency_ms,
                    "model": MODEL_NAME,
                    "request_id": request_id,
                }
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            self._send_json(
                {
                    "ok": False,
                    "error": f"upstream HTTP {exc.code}: {detail}",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                },
                status=502,
            )
        except TimeoutError as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                },
                status=504,
            )
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "model": MODEL_NAME,
                },
                status=500,
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rkllm-proxy] {self.address_string()} - {fmt % args}")

    def _post_upstream(self, payload: dict[str, Any]) -> str:
        body = _json_bytes(payload)
        request = urllib.request.Request(
            self.server.chat_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.server.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")


class RKLLMProxyServer(ThreadedHTTPServer):
    chat_endpoint: str
    timeout_seconds: float


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
    print(f"[rkllm-proxy] listening on http://{args.host}:{args.port}")
    print(f"[rkllm-proxy] upstream {args.chat_endpoint}")
    server.serve_forever()


if __name__ == "__main__":
    main()
