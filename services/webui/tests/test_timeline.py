import asyncio

import pytest

from joy_interaction_webui.omni.events import AudioTimelineEvent
from joy_interaction_webui.omni.orchestrator import (
    OmniOrchestrator,
    OrchestratorConfig,
)
from joy_interaction_webui.omni.timeline import TimelineBuffer, TimelineEvent


def test_timeline_is_bounded_and_snapshot_is_time_ordered() -> None:
    timeline = TimelineBuffer(max_age_seconds=2, max_events=3)
    timeline.append(TimelineEvent("s", "video", "frame", 2000, 2000))
    timeline.append(TimelineEvent("s", "audio", "speech_partial", 1000, 1500))
    timeline.append(TimelineEvent("s", "system", "action", 3000, 3000))
    timeline.append(TimelineEvent("s", "audio", "audio_event", 3500, 3500))

    snapshot = timeline.snapshot(now_ms=3500, lookback_seconds=2)

    assert len(timeline) == 3
    assert [event.kind for event in snapshot] == ["frame", "action", "audio_event"]
    assert [event.kind for event in timeline.snapshot(modalities={"audio"})] == [
        "audio_event"
    ]


def test_audio_event_can_enter_generic_timeline_without_losing_confidence() -> None:
    event = TimelineEvent.from_audio(
        AudioTimelineEvent(
            "s",
            "audio_event",
            100,
            200,
            label="smoke_alarm",
            confidence=0.503,
        )
    )

    assert event.modality == "audio"
    assert event.priority == 100
    assert event.payload == {
        "label": "smoke_alarm",
        "confidence": 0.503,
        "source": "microphone",
    }


def test_timeline_rejects_backwards_intervals() -> None:
    with pytest.raises(ValueError, match="end_ms"):
        TimelineEvent("s", "audio", "invalid", 200, 100)


@pytest.mark.asyncio
async def test_orchestrator_ticks_and_priority_event_triggers_immediately() -> None:
    clock_ms = 10_000.0
    snapshots = []
    orchestrator = OmniOrchestrator(
        "s",
        config=OrchestratorConfig(
            tick_seconds=0.01,
            lookback_seconds=5,
            urgent_priority=80,
        ),
        clock=lambda: clock_ms,
    )
    orchestrator.add_listener(snapshots.append)
    orchestrator.start()
    try:
        await orchestrator.record_event(
            TimelineEvent("s", "video", "frame", 9000, 9000, priority=10)
        )
        urgent = await orchestrator.record_event(
            TimelineEvent(
                "s",
                "audio",
                "audio_event",
                9500,
                9500,
                payload={"label": "smoke_alarm", "confidence": 0.9},
                priority=100,
            )
        )
        await asyncio.sleep(0.025)
    finally:
        await orchestrator.stop()

    assert urgent is not None
    assert urgent.trigger == "priority"
    assert urgent.trigger_event_kind == "audio_event"
    assert [event.modality for event in urgent.events] == ["video", "audio"]
    assert any(snapshot.trigger == "tick" for snapshot in snapshots)
    assert orchestrator.stats["priority_triggers"] == 1
    assert orchestrator.stats["ticks"] >= 1


@pytest.mark.asyncio
async def test_orchestrator_snapshot_replays_multimodal_events_in_time_order() -> None:
    orchestrator = OmniOrchestrator("s", clock=lambda: 5000)
    for event in (
        TimelineEvent("s", "model", "model_action", 4000, 4000),
        TimelineEvent("s", "audio", "speech_final", 2000, 2500),
        TimelineEvent("s", "video", "frame", 3000, 3000),
        TimelineEvent("s", "audio", "tts_playback", 4100, 4500),
    ):
        await orchestrator.record_event(event)

    snapshot = await orchestrator.create_snapshot(trigger="test")

    assert [(event.modality, event.kind) for event in snapshot.events] == [
        ("audio", "speech_final"),
        ("video", "frame"),
        ("model", "model_action"),
        ("audio", "tts_playback"),
    ]
