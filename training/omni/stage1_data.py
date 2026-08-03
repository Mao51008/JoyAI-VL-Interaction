"""CPU-only construction of projector stage-one token sequences.

The module deliberately does not import Transformers or load model weights.
It turns tokenized context/transcript IDs and an audio-token count into the
exact input/label masks consumed by a later model-specific collator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import OmniSample


IGNORE_INDEX = -100


@dataclass(frozen=True)
class Stage1Sequence:
    """One padded-free sequence before model-specific tensor conversion."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    audio_placeholder_mask: tuple[bool, ...]
    target_start: int

    def validate(self) -> None:
        length = len(self.input_ids)
        if length == 0:
            raise ValueError("stage-one sequence must not be empty")
        if not (len(self.labels) == len(self.attention_mask) == len(self.audio_placeholder_mask) == length):
            raise ValueError("stage-one sequence fields must have equal lengths")
        if not 0 <= self.target_start < length:
            raise ValueError("target_start must point inside the sequence")
        if any(label != IGNORE_INDEX for label in self.labels[: self.target_start]):
            raise ValueError("non-target labels before target_start")
        if not any(label != IGNORE_INDEX for label in self.labels[self.target_start :]):
            raise ValueError("sequence has no supervised target labels")


def build_stage1_sequence(
    *,
    context_ids: Sequence[int],
    target_ids: Sequence[int],
    audio_token_count: int,
    audio_start_id: int,
    audio_placeholder_id: int,
    audio_end_id: int,
    assistant_prefix_ids: Sequence[int] = (),
    eos_id: int | None = None,
) -> Stage1Sequence:
    """Build ``context + audio span + assistant target`` labels.

    Only target IDs (and the optional EOS) receive loss labels. Audio
    placeholders are later replaced by projected audio embeddings while their
    labels remain ``-100``.
    """
    if audio_token_count <= 0:
        raise ValueError("audio_token_count must be positive")
    if not target_ids:
        raise ValueError("target_ids must not be empty")

    prefix = tuple(int(value) for value in context_ids)
    audio_span = (int(audio_start_id),) + (int(audio_placeholder_id),) * audio_token_count + (
        int(audio_end_id),
    )
    assistant_prefix = tuple(int(value) for value in assistant_prefix_ids)
    target = tuple(int(value) for value in target_ids)
    suffix = target + ((int(eos_id),) if eos_id is not None else ())
    input_ids = prefix + audio_span + assistant_prefix + suffix
    target_start = len(prefix) + len(audio_span) + len(assistant_prefix)
    labels = (IGNORE_INDEX,) * target_start + suffix
    sequence = Stage1Sequence(
        input_ids=input_ids,
        labels=labels,
        attention_mask=(1,) * len(input_ids),
        audio_placeholder_mask=(
            (False,) * (len(prefix) + 1)
            + (True,) * audio_token_count
            + (False,) * (1 + len(assistant_prefix) + len(suffix))
        ),
        target_start=target_start,
    )
    sequence.validate()
    return sequence


def target_text(sample: OmniSample) -> str:
    """Return the explicit assistant transcription target from a sample."""
    value = str(sample.metadata.get("assistant_target_text", "")).strip()
    if not value:
        raise ValueError(f"sample {sample.sample_id!r} has no assistant_target_text")
    return value


def pad_sequences(
    sequences: Iterable[Stage1Sequence],
    *,
    pad_token_id: int,
) -> dict[str, list[list[int]] | list[list[bool]]]:
    """Right-pad CPU sequences and preserve ``-100`` label masking."""
    values = list(sequences)
    if not values:
        raise ValueError("cannot pad an empty sequence list")
    for sequence in values:
        sequence.validate()
    width = max(len(sequence.input_ids) for sequence in values)
    return {
        "input_ids": [
            list(sequence.input_ids) + [pad_token_id] * (width - len(sequence.input_ids))
            for sequence in values
        ],
        "labels": [
            list(sequence.labels) + [IGNORE_INDEX] * (width - len(sequence.labels))
            for sequence in values
        ],
        "attention_mask": [
            list(sequence.attention_mask) + [0] * (width - len(sequence.attention_mask))
            for sequence in values
        ],
        "audio_placeholder_mask": [
            list(sequence.audio_placeholder_mask)
            + [False] * (width - len(sequence.audio_placeholder_mask))
            for sequence in values
        ],
    }
