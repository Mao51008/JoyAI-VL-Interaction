"""Unified timestamped events produced from continuous audio."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioWindow:
    pcm: bytes
    sample_rate: int
    start_ms: float
    end_ms: float
    last_sequence: int
    voice_ratio: float = 0.0
    latest_voice_active: bool = False
    last_voice_ms: float | None = None


@dataclass(frozen=True)
class AudioTimelineEvent:
    session_id: str
    kind: str
    start_ms: float
    end_ms: float
    text: str = ""
    stable_prefix: str = ""
    label: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "audio_event",
            "session_id": self.session_id,
            "modality": "audio",
            "kind": self.kind,
            "start_sec": round(self.start_ms / 1000, 3),
            "end_sec": round(self.end_ms / 1000, 3),
        }
        for key, value in (
            ("text", self.text),
            ("stable_prefix", self.stable_prefix),
            ("label", self.label),
            ("confidence", self.confidence),
        ):
            if value not in ("", None):
                data[key] = value
        if self.metadata:
            data["metadata"] = self.metadata
        return data
