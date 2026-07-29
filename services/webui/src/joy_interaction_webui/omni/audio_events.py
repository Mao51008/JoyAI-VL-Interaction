"""Pluggable non-speech audio event detection interface."""

from dataclasses import dataclass
from typing import Protocol

from .events import AudioWindow


@dataclass(frozen=True)
class AudioEventDetection:
    label: str
    confidence: float
    start_ms: float
    end_ms: float


class AudioEventDetector(Protocol):
    async def detect(self, window: AudioWindow) -> list[AudioEventDetection]: ...


class NullAudioEventDetector:
    async def detect(self, window: AudioWindow) -> list[AudioEventDetection]:
        return []
