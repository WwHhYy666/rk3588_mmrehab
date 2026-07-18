from speech.llm_worker import VoiceLLMWorker


def make_worker(status: str) -> VoiceLLMWorker:
    return VoiceLLMWorker(
        answer_fn=lambda report, question, allow_local: {"ok": True, "answer": "ok"},
        training_status_fn=lambda: status,
        speak_fn=lambda text, event_type: {"ok": True},
    )


def test_voice_worker_blocks_all_training_and_rest_audio_phases() -> None:
    for status in (
        "running",
        "paused",
        "resting",
        "awaiting_orientation",
        "awaiting_return",
        "awaiting_care_response",
        "awaiting_action_audio",
        "awaiting_rep_feedback",
    ):
        assert make_worker(status)._training_block() is not None


def test_voice_worker_allows_questions_after_training_completion() -> None:
    assert make_worker("completed")._training_block() is None
    assert make_worker("idle")._training_block() is None
