import asyncio
import time

import pytest

from joy_interaction_webui.omni.orchestrator import (
    clear_orchestrators,
    get_orchestrator,
)
from joy_interaction_webui.server import (
    get_session_callback,
    get_video_timeline_callback,
    record_session_event,
)


@pytest.mark.asyncio
async def test_server_records_text_video_and_model_action_on_one_timeline() -> None:
    await clear_orchestrators()
    try:
        base_ms = time.monotonic() * 1000
        await record_session_event(
            "s",
            modality="text",
            kind="user_query",
            payload={"text": "发生了什么？"},
            priority=60,
            monotonic_ms=base_ms - 100,
        )
        video_callback = get_video_timeline_callback("s")
        await video_callback(
            {
                "server_monotonic_ms": base_ms - 50,
                "frame_index": 1,
                "width": 640,
                "height": 360,
            }
        )
        model_callback = get_session_callback("s")
        model_callback("</response> 请立即撤离。", {"total_inferences": 1})
        await asyncio.sleep(0)

        orchestrator = get_orchestrator("s")
        assert orchestrator is not None
        events = orchestrator.timeline.snapshot()
        assert [(event.modality, event.kind) for event in events] == [
            ("text", "user_query"),
            ("video", "frame"),
            ("model", "model_action"),
        ]
        assert events[-1].payload["action"] == "response"
    finally:
        await clear_orchestrators()
