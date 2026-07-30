import pytest

from joy_interaction_webui.omni.echo_filter import EchoFilterConfig, TTSEchoFilter
from joy_interaction_webui.omni.events import AudioTimelineEvent
from joy_interaction_webui.omni.timeline import TimelineEvent


def tts_reference(
    *,
    text: str = "请立即撤离危险区域",
    status: str = "playing",
    end_ms: float = 1000,
) -> TimelineEvent:
    return TimelineEvent(
        "s",
        "audio",
        "tts_playback",
        1000,
        end_ms,
        {
            "source": "system_output",
            "generation_id": "generation-1",
            "status": status,
            "text": text,
        },
    )


def speech(text: str, kind: str = "speech_partial") -> AudioTimelineEvent:
    return AudioTimelineEvent("s", kind, 1100, 1400, text=text)


def test_exact_tts_transcript_is_marked_as_echo() -> None:
    event = TTSEchoFilter().annotate(
        speech("请立即撤离危险区域"),
        (tts_reference(),),
        now_ms=1400,
    )

    assert event.metadata["echo_checked"] is True
    assert event.metadata["likely_tts_echo"] is True
    assert event.metadata["echo_similarity"] == pytest.approx(1)
    assert event.metadata["echo_reference_generation_id"] == "generation-1"


def test_partial_tts_prefix_is_marked_but_new_user_content_is_preserved() -> None:
    echo = TTSEchoFilter().annotate(
        speech("请立即撤离"),
        (tts_reference(),),
        now_ms=1400,
    )
    user = TTSEchoFilter().annotate(
        speech("不要说了，帮我报警"),
        (tts_reference(),),
        now_ms=1400,
    )

    assert echo.metadata["likely_tts_echo"] is True
    assert user.metadata["echo_checked"] is True
    assert user.metadata["likely_tts_echo"] is False


def test_recent_finished_playback_uses_tail_but_old_playback_is_ignored() -> None:
    echo_filter = TTSEchoFilter(EchoFilterConfig(playback_tail_seconds=1))
    recent = echo_filter.annotate(
        speech("请立即撤离危险区域"),
        (tts_reference(status="done", end_ms=1000),),
        now_ms=1800,
    )
    expired = echo_filter.annotate(
        speech("请立即撤离危险区域"),
        (tts_reference(status="done", end_ms=1000),),
        now_ms=2101,
    )

    assert recent.metadata["likely_tts_echo"] is True
    assert expired.metadata == {}


def test_short_common_text_is_not_suppressed() -> None:
    event = TTSEchoFilter().annotate(
        speech("请问"),
        (tts_reference(text="请问现在需要什么帮助"),),
        now_ms=1400,
    )

    assert event.metadata["echo_checked"] is True
    assert event.metadata["likely_tts_echo"] is False
