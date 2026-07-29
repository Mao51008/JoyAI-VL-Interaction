"""Sliding-window ASR scheduler for non-turn-based audio."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from .audio_events import AudioEventDetector, NullAudioEventDetector
from .events import AudioTimelineEvent, AudioWindow
from .stable_prefix import StablePrefixTracker


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float | None = None


class WindowTranscriber(Protocol):
    async def transcribe(self, window: AudioWindow) -> TranscriptionResult: ...


@dataclass(frozen=True)
class StreamingASRConfig:
    interval_seconds: float = 0.4
    window_seconds: float = 6.0
    speech_end_silence_seconds: float = 0.8
    stable_observations: int = 2


class StreamingASRCoordinator:
    """Recognize the newest window without ever queuing stale inference work."""

    def __init__(
        self,
        session_id: str,
        snapshot: Callable[[float], AudioWindow | None],
        transcriber: WindowTranscriber,
        emit: Callable[[AudioTimelineEvent], Awaitable[None]],
        *,
        detector: AudioEventDetector | None = None,
        config: StreamingASRConfig | None = None,
    ):
        self.session_id = session_id
        self.snapshot = snapshot
        self.transcriber = transcriber
        self.emit = emit
        self.detector = detector or NullAudioEventDetector()
        self.config = config or StreamingASRConfig()
        self.tracker = StablePrefixTracker(self.config.stable_observations)
        self.speech_active = False
        self.speech_start_ms = 0.0
        self.last_voice_ms: float | None = None
        self.last_sequence = -1
        self.running = False
        self._task: asyncio.Task | None = None
        self.stats = {"ticks": 0, "requests": 0, "skipped": 0, "events": 0}

    async def _emit(self, event: AudioTimelineEvent) -> None:
        self.stats["events"] += 1
        await self.emit(event)

    async def process_once(self) -> None:
        self.stats["ticks"] += 1
        window = self.snapshot(self.config.window_seconds)
        if window is None or window.last_sequence == self.last_sequence:
            self.stats["skipped"] += 1
            return
        self.last_sequence = window.last_sequence

        if window.latest_voice_active:
            self.last_voice_ms = window.end_ms
            if not self.speech_active:
                self.speech_active = True
                self.speech_start_ms = window.end_ms
                await self._emit(
                    AudioTimelineEvent(
                        self.session_id, "speech_start", window.end_ms, window.end_ms
                    )
                )

        for detection in await self.detector.detect(window):
            await self._emit(
                AudioTimelineEvent(
                    self.session_id,
                    "audio_event",
                    detection.start_ms,
                    detection.end_ms,
                    label=detection.label,
                    confidence=detection.confidence,
                )
            )

        if self.speech_active:
            self.stats["requests"] += 1
            result = await self.transcriber.transcribe(window)
            previous_stable = self.tracker.stable
            stable, _ = self.tracker.update(result.text)
            await self._emit(
                AudioTimelineEvent(
                    self.session_id,
                    "speech_partial",
                    self.speech_start_ms,
                    window.end_ms,
                    text=result.text,
                    stable_prefix=stable,
                    confidence=result.confidence,
                )
            )
            if stable and stable != previous_stable:
                await self._emit(
                    AudioTimelineEvent(
                        self.session_id,
                        "speech_stable",
                        self.speech_start_ms,
                        window.end_ms,
                        text=stable,
                        stable_prefix=stable,
                        confidence=result.confidence,
                    )
                )

            silence_ms = window.end_ms - self.last_voice_ms if self.last_voice_ms is not None else 0
            if silence_ms >= self.config.speech_end_silence_seconds * 1000:
                await self._emit(
                    AudioTimelineEvent(
                        self.session_id,
                        "speech_final",
                        self.speech_start_ms,
                        window.end_ms,
                        text=result.text,
                        stable_prefix=stable,
                        confidence=result.confidence,
                    )
                )
                await self._emit(
                    AudioTimelineEvent(
                        self.session_id,
                        "speech_end",
                        self.speech_start_ms,
                        window.end_ms,
                    )
                )
                self.speech_active = False
                self.last_voice_ms = None
                self.tracker.reset()

    async def _run(self) -> None:
        while self.running:
            started = asyncio.get_running_loop().time()
            await self.process_once()
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0, self.config.interval_seconds - elapsed))

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            await self._task
            self._task = None
