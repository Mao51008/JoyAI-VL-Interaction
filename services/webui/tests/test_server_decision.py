import asyncio
import time

import pytest

from joy_interaction_webui import server
from joy_interaction_webui.omni.decision import get_decision_engine
from joy_interaction_webui.omni.orchestrator import get_orchestrator
from joy_interaction_webui.omni.timeline import TimelineEvent


@pytest.mark.asyncio
async def test_fake_decision_mode_installs_notifies_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "fake-decision-server"
    notifications: list[dict] = []
    monkeypatch.setenv("OMNI_DECISION_MODE", "fake")
    monkeypatch.setattr(
        server,
        "notify_session_json",
        lambda sid, payload: notifications.append({"session_id": sid, **payload}),
    )

    session = server.get_or_create_session(session_id)
    engine = get_decision_engine(session_id)
    orchestrator = get_orchestrator(session_id)

    try:
        assert engine is not None
        assert orchestrator is not None
        assert session["omni_decision_engine"] is engine

        now_ms = time.monotonic() * 1000
        await orchestrator.record_event(
            TimelineEvent(
                session_id,
                "audio",
                "audio_event",
                now_ms,
                now_ms,
                {"label": "fire_alarm", "confidence": 0.95},
                priority=100,
            )
        )
        await asyncio.wait_for(engine._queue.join(), timeout=1)

        assert notifications[-1]["session_id"] == session_id
        assert notifications[-1]["type"] == "omni_decision"
        assert notifications[-1]["decision"]["action"] == "response"
        assert notifications[-1]["decision"]["confidence"] == pytest.approx(0.95)
    finally:
        result = await server.cleanup_session(session_id, reset_adapter=False)

    assert result["decision_engine_removed"] is True
    assert result["orchestrator_removed"] is True
    assert get_decision_engine(session_id) is None
    assert get_orchestrator(session_id) is None
