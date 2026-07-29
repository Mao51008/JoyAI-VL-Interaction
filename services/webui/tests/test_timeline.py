import pytest

from joy_interaction_webui.omni.events import AudioTimelineEvent
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
    assert event.payload == {"label": "smoke_alarm", "confidence": 0.503}


def test_timeline_rejects_backwards_intervals() -> None:
    with pytest.raises(ValueError, match="end_ms"):
        TimelineEvent("s", "audio", "invalid", 200, 100)
