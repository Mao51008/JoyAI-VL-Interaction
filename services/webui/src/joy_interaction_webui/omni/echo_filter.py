"""Text-reference echo detection that preserves real barge-in speech."""

import os
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import ClassVar

from .events import AudioTimelineEvent
from .timeline import TimelineEvent


@dataclass(frozen=True)
class EchoFilterConfig:
    similarity_threshold: float = 0.82
    min_text_chars: int = 4
    playback_tail_seconds: float = 1.5


def echo_filter_config_from_env() -> EchoFilterConfig:
    return EchoFilterConfig(
        similarity_threshold=float(os.getenv("TTS_ECHO_SIMILARITY_THRESHOLD", "0.82")),
        min_text_chars=int(os.getenv("TTS_ECHO_MIN_TEXT_CHARS", "4")),
        playback_tail_seconds=float(os.getenv("TTS_ECHO_TAIL_SECONDS", "1.5")),
    )


class TTSEchoFilter:
    """Annotate ASR text when it is likely a transcription of recent TTS."""

    speech_kinds: ClassVar[set[str]] = {
        "speech_partial",
        "speech_stable",
        "speech_final",
    }

    def __init__(self, config: EchoFilterConfig | None = None):
        self.config = config or EchoFilterConfig()

    @staticmethod
    def normalize(text: str) -> str:
        return "".join(re.findall(r"[\w\u3400-\u9fff]+", str(text or "").casefold()))

    @staticmethod
    def similarity(observed: str, reference: str) -> float:
        observed_normalized = TTSEchoFilter.normalize(observed)
        reference_normalized = TTSEchoFilter.normalize(reference)
        if not observed_normalized or not reference_normalized:
            return 0.0
        matcher = SequenceMatcher(None, observed_normalized, reference_normalized)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        observed_coverage = matched / len(observed_normalized)
        return max(matcher.ratio(), observed_coverage)

    def _latest_reference(
        self,
        timeline_events: tuple[TimelineEvent, ...],
        now_ms: float,
    ) -> TimelineEvent | None:
        tail_ms = self.config.playback_tail_seconds * 1000
        for event in reversed(timeline_events):
            if event.kind != "tts_playback":
                continue
            payload = event.payload
            if payload.get("source") not in {"system_output", "browser_output"}:
                continue
            if not (payload.get("text") or payload.get("played_text")):
                continue
            status = payload.get("status")
            if status == "playing" or now_ms - event.end_ms <= tail_ms:
                return event
            return None
        return None

    def annotate(
        self,
        event: AudioTimelineEvent,
        timeline_events: tuple[TimelineEvent, ...],
        *,
        now_ms: float,
    ) -> AudioTimelineEvent:
        if event.kind not in self.speech_kinds or not event.text:
            return event
        reference = self._latest_reference(timeline_events, now_ms)
        if reference is None:
            return event

        reference_text = str(
            reference.payload.get("played_text") or reference.payload.get("text") or ""
        )
        observed_text = self.normalize(event.text)
        score = self.similarity(event.text, reference_text)
        likely_echo = (
            len(observed_text) >= self.config.min_text_chars
            and score >= self.config.similarity_threshold
        )
        metadata = {
            **event.metadata,
            "echo_checked": True,
            "likely_tts_echo": likely_echo,
            "echo_similarity": round(score, 4),
            "echo_reference_generation_id": str(reference.payload.get("generation_id") or ""),
            "echo_reference_status": str(reference.payload.get("status") or ""),
        }
        return replace(event, metadata=metadata)
