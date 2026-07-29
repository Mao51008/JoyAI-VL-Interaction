"""Bounded, replayable timeline shared by continuous modalities."""

from dataclasses import dataclass, field
from typing import Any

from .events import AudioTimelineEvent


@dataclass(frozen=True)
class TimelineEvent:
    session_id: str
    modality: str
    kind: str
    start_ms: float
    end_ms: float
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError("timeline event end_ms cannot be before start_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modality": self.modality,
            "kind": self.kind,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "priority": self.priority,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_audio(cls, event: AudioTimelineEvent) -> "TimelineEvent":
        payload = {
            key: value
            for key, value in (
                ("text", event.text),
                ("stable_prefix", event.stable_prefix),
                ("label", event.label),
                ("confidence", event.confidence),
                ("metadata", dict(event.metadata) if event.metadata else None),
            )
            if value not in ("", None)
        }
        priority = 100 if event.kind == "audio_event" else 20
        return cls(
            session_id=event.session_id,
            modality="audio",
            kind=event.kind,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            payload=payload,
            priority=priority,
        )


class TimelineBuffer:
    """Retain only a bounded recent history and return deterministic snapshots."""

    def __init__(self, *, max_age_seconds: float = 120, max_events: int = 2000):
        if max_age_seconds <= 0 or max_events <= 0:
            raise ValueError("timeline bounds must be positive")
        self.max_age_ms = max_age_seconds * 1000
        self.max_events = max_events
        self._events: list[TimelineEvent] = []
        self._latest_ms = 0.0

    def append(self, event: TimelineEvent) -> None:
        self._events.append(event)
        self._events.sort(key=lambda item: (item.end_ms, item.start_ms))
        self._latest_ms = max(self._latest_ms, event.end_ms)
        self._prune()

    def _prune(self) -> None:
        cutoff = self._latest_ms - self.max_age_ms
        while self._events and (
            len(self._events) > self.max_events or self._events[0].end_ms < cutoff
        ):
            self._events.pop(0)

    def snapshot(
        self,
        *,
        now_ms: float | None = None,
        lookback_seconds: float | None = None,
        modalities: set[str] | None = None,
    ) -> tuple[TimelineEvent, ...]:
        end_ms = self._latest_ms if now_ms is None else now_ms
        lookback_ms = self.max_age_ms if lookback_seconds is None else lookback_seconds * 1000
        cutoff = end_ms - lookback_ms
        events = (
            event
            for event in self._events
            if event.end_ms >= cutoff
            and event.start_ms <= end_ms
            and (modalities is None or event.modality in modalities)
        )
        return tuple(sorted(events, key=lambda event: (event.start_ms, event.end_ms)))

    def replay(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.snapshot()]

    def __len__(self) -> int:
        return len(self._events)
