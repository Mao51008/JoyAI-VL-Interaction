import asyncio

import pytest

from joy_interaction_webui import server
from joy_interaction_webui.omni.decision import (
    ActionKind,
    DecisionAction,
    DecisionRecord,
)
from joy_interaction_webui.omni.orchestrator import (
    clear_orchestrators,
    get_orchestrator,
)
from joy_interaction_webui.tts import (
    cancel_tts_generation,
    get_active_tts_generation,
    record_tts_playback_progress,
    register_tts_generation,
)


@pytest.mark.asyncio
async def test_generation_registry_cancels_only_matching_generation() -> None:
    started = asyncio.Event()

    async def wait_forever() -> None:
        started.set()
        await asyncio.Future()

    task = asyncio.create_task(wait_forever())
    register_tts_generation("s", "generation-1", task)
    await started.wait()

    mismatch = await cancel_tts_generation("s", "generation-old")
    assert mismatch["cancelled"] is False
    assert mismatch["reason"] == "generation_mismatch"
    assert get_active_tts_generation("s") == "generation-1"

    cancelled = await cancel_tts_generation("s", "generation-1")
    assert cancelled["cancelled"] is True
    assert get_active_tts_generation("s") is None
    assert task.cancelled()


@pytest.mark.asyncio
async def test_browser_playback_progress_is_written_to_timeline() -> None:
    await clear_orchestrators()
    try:
        await record_tts_playback_progress(
            {
                "session_id": "playback",
                "request_id": "request-1",
                "generation_id": "generation-1",
                "status": "interrupted",
                "played_audio_ms": 420,
                "played_text": "已经播放",
                "text": "已经播放的完整计划文本",
            }
        )

        orchestrator = get_orchestrator("playback")
        assert orchestrator is not None
        event = orchestrator.timeline.snapshot()[-1]
        assert event.kind == "tts_playback"
        assert event.payload["status"] == "interrupted"
        assert event.payload["played_audio_ms"] == pytest.approx(420)
        assert event.payload["played_text"] == "已经播放"
    finally:
        await clear_orchestrators()


@pytest.mark.asyncio
async def test_omni_interrupt_cancels_tts_vlm_and_notifies_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notifications: list[dict] = []

    class FakeVLM:
        async def cancel_active_requests(self) -> int:
            return 2

    async def wait_forever() -> None:
        await asyncio.Future()

    task = asyncio.create_task(wait_forever())
    register_tts_generation("interrupt", "generation-1", task)
    server.sessions["interrupt"] = {"vlm_service": FakeVLM()}
    monkeypatch.setattr(
        server,
        "notify_session_json",
        lambda session_id, payload: notifications.append({"session_id": session_id, **payload}),
    )
    record = DecisionRecord(
        snapshot_sequence=1,
        snapshot_trigger="priority",
        trigger_reason="user_speech_during_tts",
        context_fingerprint="context",
        action=DecisionAction(ActionKind.INTERRUPT),
        latency_ms=1,
    )

    try:
        await server.handle_omni_decision("interrupt", record)
    finally:
        server.sessions.pop("interrupt", None)

    assert task.cancelled()
    assert notifications[-1]["decision"]["action"] == "interrupt"
    assert notifications[-1]["interruption"]["tts_generation"]["cancelled"] is True
    assert notifications[-1]["interruption"]["cancelled_vlm_tasks"] == 2
    assert notifications[-1]["interruption"]["browser_playback"] == "stop_requested"
