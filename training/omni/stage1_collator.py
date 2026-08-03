"""Transformers-facing batch construction for projector-only audio alignment.

JoyAI-VL has no native audio token.  ``<|vision_pad|>`` is therefore used only
as an immutable embedding replacement slot; it is never passed as an image or
video input and never receives a language-model label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .schema import OmniSample
from .stage1_data import Stage1Sequence, build_stage1_sequence, pad_sequences, target_text


SYSTEM_PROMPT = "You are a precise speech transcription assistant."
USER_PROMPT = "Transcribe the provided audio."


@dataclass(frozen=True)
class JoyAIStage1TokenLayout:
    """Existing JoyAI special IDs used to delimit injected audio embeddings."""

    audio_start_id: int
    audio_placeholder_id: int
    audio_end_id: int
    pad_token_id: int
    eos_token_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "JoyAIStage1TokenLayout":
        required = {
            "<|vision_start|>": "audio_start_id",
            "<|vision_pad|>": "audio_placeholder_id",
            "<|vision_end|>": "audio_end_id",
        }
        values: dict[str, int] = {}
        for token, field in required.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"JoyAI tokenizer has no required token {token!r}")
            values[field] = int(token_id)
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("JoyAI tokenizer must define pad_token_id and eos_token_id")
        return cls(
            **values,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )


def build_chat_parts(tokenizer: Any, *, user_prompt: str = USER_PROMPT) -> tuple[list[int], list[int]]:
    """Build token IDs before and after the injected audio span.

    The explicit Qwen chat delimiters avoid invoking multimodal chat-template
    expansion for the synthetic audio replacement slots.
    """
    encode = lambda value: list(tokenizer(value, add_special_tokens=False).input_ids)
    context = encode(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n"
    )
    assistant_prefix = encode("<|im_end|>\n<|im_start|>assistant\n")
    return context, assistant_prefix


def build_sample_sequence(
    sample: OmniSample,
    *,
    tokenizer: Any,
    layout: JoyAIStage1TokenLayout,
    audio_token_count: int,
) -> Stage1Sequence:
    """Turn one manifest row and measured ASR token count into a loss sequence."""
    context_ids, assistant_prefix_ids = build_chat_parts(tokenizer)
    target_ids = list(tokenizer(target_text(sample), add_special_tokens=False).input_ids)
    return build_stage1_sequence(
        context_ids=context_ids,
        target_ids=target_ids,
        audio_token_count=audio_token_count,
        audio_start_id=layout.audio_start_id,
        audio_placeholder_id=layout.audio_placeholder_id,
        audio_end_id=layout.audio_end_id,
        assistant_prefix_ids=assistant_prefix_ids,
        eos_id=layout.eos_token_id,
    )


class ProjectorStage1Collator:
    """Load audio, obtain official ASR features, and pad JoyAI projector batches."""

    def __init__(
        self,
        *,
        asr_processor: Any,
        joyai_tokenizer: Any,
        audio_loader: Callable[[str], tuple[Any, int]] | None = None,
    ) -> None:
        self.asr_processor = asr_processor
        self.joyai_tokenizer = joyai_tokenizer
        self.layout = JoyAIStage1TokenLayout.from_tokenizer(joyai_tokenizer)
        self.audio_loader = audio_loader or _load_mono_audio

    def __call__(self, samples: Sequence[OmniSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        audios = []
        for sample in samples:
            if len(sample.audio) != 1:
                raise ValueError(f"sample {sample.sample_id!r} must contain exactly one audio segment")
            waveform, sample_rate = self.audio_loader(sample.audio[0].path)
            if sample_rate != 16_000:
                raise ValueError(f"sample {sample.sample_id!r} is not 16 kHz")
            audios.append(waveform)

        asr_text = [self.asr_processor.audio_token] * len(samples)
        audio_batch = self.asr_processor(
            text=asr_text, audio=audios, sampling_rate=16_000, return_tensors="pt", padding=True
        )
        audio_token_id = self.asr_processor.tokenizer.convert_tokens_to_ids(self.asr_processor.audio_token)
        counts = _count_tokens(audio_batch["input_ids"], int(audio_token_id))
        sequences = [
            build_sample_sequence(
                sample,
                tokenizer=self.joyai_tokenizer,
                layout=self.layout,
                audio_token_count=count,
            )
            for sample, count in zip(samples, counts, strict=True)
        ]
        padded = pad_sequences(sequences, pad_token_id=self.layout.pad_token_id)
        import torch

        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
            "audio_placeholder_mask": torch.tensor(padded["audio_placeholder_mask"], dtype=torch.bool),
            "input_features": audio_batch["input_features"],
            "feature_attention_mask": audio_batch["feature_attention_mask"],
            "sample_ids": [sample.sample_id for sample in samples],
        }


def _count_tokens(input_ids: Any, token_id: int) -> list[int]:
    rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    counts = [sum(int(token) == token_id for token in row) for row in rows]
    if any(count <= 0 for count in counts):
        raise ValueError("ASR processor did not produce audio placeholder tokens")
    return counts


def _load_mono_audio(path: str) -> tuple[Any, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to load stage-one audio") from exc
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return waveform.mean(axis=1), int(sample_rate)
