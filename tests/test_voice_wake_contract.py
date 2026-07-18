from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8-sig")


def test_voice_ui_uses_one_explicit_start_stop_listen_button() -> None:
    source = read_text("rehab_app/server/static/train.js")

    assert 'id="voice-listen-btn"' in source
    assert 'id="voice-record-btn"' not in source
    assert 'id="voice-wake-btn"' not in source
    assert 'listenBtn.textContent = voiceListenBusy ? "正在处理..." : (voiceListening ? "结束监听" : "唤醒监听")' in source
    assert 'listenBtn.addEventListener("click", toggleVoiceListening)' in source
    assert "wakeListening" not in source
    assert "wakeQuestionCaptureMs" not in source


def test_manual_listen_records_until_stop_then_replaces_question_with_asr_text() -> None:
    source = read_text("rehab_app/server/static/train.js")

    assert 'UI.postJSON("/api/voice/listen_start", {})' in source
    assert 'UI.postJSON("/api/voice/listen_stop", { session_id: voiceListenSessionId })' in source
    assert "const text = await pollAsrResult(result.job_id, 240)" in source
    assert "setVoiceQuestionText(text)" in source
    assert 'setVoiceQuestionText("")' in source
    assert "正在监听并录音；说完后点击“结束监听”" in source


def test_backend_manual_capture_has_identity_status_and_resource_guards() -> None:
    backend = read_text("rehab_app/server/rehab_http_server.py")

    assert "def start_manual_voice_capture(" in backend
    assert "def stop_manual_voice_capture(" in backend
    assert "manual_voice_capture_session_id" in backend
    assert "subprocess.Popen(command" in backend
    assert "process.terminate()" in backend
    assert "requested_session_id != session_id" in backend
    assert '"manual_capture": manual_voice_capture_snapshot()' in backend
    assert 'return "manual_voice_capture_active"' in backend
    assert 'return "manual_voice_capture"' in backend
    assert 'self.path == "/api/voice/listen_start"' in backend
    assert 'self.path == "/api/voice/listen_stop"' in backend


def test_voice_submit_uses_backend_as_authority_and_hides_transient_provider_warning() -> None:
    source = read_text("rehab_app/server/static/train.js")

    assert "GLM/Qwen 未就绪" not in source
    assert "GLM/Qwen 都未就绪" not in source
    assert "const allowed = Boolean(voice.qa_allowed) && !trainingBusy;" in source
    assert 'message.hidden = !idleMessage' in source


def test_legacy_timed_capture_keeps_trailing_silence_support_for_compatibility() -> None:
    backend = read_text("rehab_app/server/rehab_http_server.py")

    assert "def _capture_voice_audio_until_silence(" in backend
    assert 'stop_reason = "trailing_silence"' in backend
    assert 'REHAB_ASR_VAD_RMS' in backend
