"""Deterministic CPU fakes for exercising Omni shapes, masks, and timing."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .schema import OmniSample
from .timeline import build_decision_steps

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FakeAudioEncoderConfig:
    hidden_size: int = 8
    tokens_per_second: int = 13
    commit_ms: int = 1000
    active_window_ms: int = 8000

    def validate(self) -> None:
        if (
            min(
                self.hidden_size,
                self.tokens_per_second,
                self.commit_ms,
                self.active_window_ms,
            )
            <= 0
        ):
            raise ValueError("fake audio encoder configuration must be positive")
        if self.active_window_ms < self.commit_ms:
            raise ValueError("active_window_ms must be at least commit_ms")


@dataclass(frozen=True)
class EncodedAudio:
    hidden_states: FloatArray
    attention_mask: BoolArray
    token_times_ms: IntArray
    committed_length: int
    history_token_count: int = 0


class FakeAudioEncoder:
    """Generate reproducible small vectors without reading audio or model weights."""

    def __init__(self, config: FakeAudioEncoderConfig | None = None) -> None:
        self.config = config or FakeAudioEncoderConfig()
        self.config.validate()

    def encode_sample(self, sample: OmniSample) -> EncodedAudio:
        if not sample.audio:
            return self._empty()
        start_ms = min(segment.start_ms for segment in sample.audio)
        end_ms = max(segment.end_ms for segment in sample.audio)
        return self.encode_range(sample.sample_id, start_ms, end_ms, final=True)

    def encode_range(
        self,
        key: str,
        start_ms: int,
        end_ms: int,
        *,
        final: bool,
        history_token_count: int = 0,
    ) -> EncodedAudio:
        duration_ms = max(end_ms - start_ms, 0)
        if duration_ms == 0:
            return self._empty(history_token_count)
        token_count = max(
            math.ceil(duration_ms * self.config.tokens_per_second / 1000),
            1,
        )
        token_times = (
            start_ms
            + (np.arange(token_count, dtype=np.float64) + 0.5)
            * duration_ms
            / token_count
        ).astype(np.int64)
        seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()[:8]
        phase = int.from_bytes(seed_bytes, "little") / (2**64)
        columns = np.arange(self.config.hidden_size, dtype=np.float32) + 1
        hidden = np.sin(
            token_times[:, None].astype(np.float32) / 1000 * columns + phase
        ).astype(np.float32)
        committed_ms = (
            duration_ms
            if final
            else duration_ms // self.config.commit_ms * (self.config.commit_ms)
        )
        committed_length = min(
            token_count,
            math.floor(committed_ms * self.config.tokens_per_second / 1000),
        )
        return EncodedAudio(
            hidden_states=hidden,
            attention_mask=np.ones(token_count, dtype=np.bool_),
            token_times_ms=token_times,
            committed_length=committed_length,
            history_token_count=history_token_count,
        )

    def _empty(self, history_token_count: int = 0) -> EncodedAudio:
        return EncodedAudio(
            hidden_states=np.zeros((0, self.config.hidden_size), dtype=np.float32),
            attention_mask=np.zeros(0, dtype=np.bool_),
            token_times_ms=np.zeros(0, dtype=np.int64),
            committed_length=0,
            history_token_count=history_token_count,
        )


class FakeStreamingAudioState:
    """Bound active-window recomputation while tracking committed history."""

    def __init__(self, encoder: FakeAudioEncoder, stream_id: str) -> None:
        self.encoder = encoder
        self.stream_id = stream_id
        self.total_ms = 0
        self.history_token_count = 0

    def append(self, duration_ms: int, *, final: bool = False) -> EncodedAudio:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        previous_window_start = max(
            0,
            self.total_ms - self.encoder.config.active_window_ms,
        )
        self.total_ms += duration_ms
        window_start = max(0, self.total_ms - self.encoder.config.active_window_ms)
        evicted_ms = window_start - previous_window_start
        if evicted_ms > 0:
            self.history_token_count += math.floor(
                evicted_ms * self.encoder.config.tokens_per_second / 1000
            )
        return self.encoder.encode_range(
            self.stream_id,
            window_start,
            self.total_ms,
            final=final,
            history_token_count=self.history_token_count,
        )

    def reset(self) -> None:
        self.total_ms = 0
        self.history_token_count = 0


@dataclass(frozen=True)
class OmniBatch:
    sample_ids: tuple[str, ...]
    audio_hidden_states: FloatArray
    audio_attention_mask: BoolArray
    audio_token_times_ms: IntArray
    modality_presence: FloatArray
    step_mask: BoolArray
    action_labels: IntArray
    step_end_times_ms: IntArray


def collate_samples(
    samples: list[OmniSample],
    encoder: FakeAudioEncoder,
    *,
    step_ms: int = 1000,
) -> OmniBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    encoded = [encoder.encode_sample(sample) for sample in samples]
    timelines = [build_decision_steps(sample, step_ms=step_ms) for sample in samples]
    batch_size = len(samples)
    max_audio = max((item.hidden_states.shape[0] for item in encoded), default=0)
    max_steps = max(len(steps) for steps in timelines)
    hidden_size = encoder.config.hidden_size

    audio_hidden = np.zeros((batch_size, max_audio, hidden_size), dtype=np.float32)
    audio_mask = np.zeros((batch_size, max_audio), dtype=np.bool_)
    audio_times = np.full((batch_size, max_audio), -1, dtype=np.int64)
    step_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
    action_labels = np.full((batch_size, max_steps), -100, dtype=np.int64)
    step_end_times = np.zeros((batch_size, max_steps), dtype=np.int64)
    modality_presence = np.asarray(
        [sample.modality_presence for sample in samples],
        dtype=np.float32,
    )

    for index, (audio, steps) in enumerate(zip(encoded, timelines, strict=True)):
        audio_count = audio.hidden_states.shape[0]
        audio_hidden[index, :audio_count] = audio.hidden_states
        audio_mask[index, :audio_count] = audio.attention_mask
        audio_times[index, :audio_count] = audio.token_times_ms
        step_count = len(steps)
        step_mask[index, :step_count] = True
        action_labels[index, :step_count] = [int(step.action) for step in steps]
        step_end_times[index, :step_count] = [step.end_ms for step in steps]

    return OmniBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        audio_hidden_states=audio_hidden,
        audio_attention_mask=audio_mask,
        audio_token_times_ms=audio_times,
        modality_presence=modality_presence,
        step_mask=step_mask,
        action_labels=action_labels,
        step_end_times_ms=step_end_times,
    )
