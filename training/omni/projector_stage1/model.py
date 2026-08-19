"""Composable projector-only training graph with frozen backbone contracts."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .projector import AudioProjector, freeze_for_projector_training


def replace_audio_placeholders(
    text_embeddings: Tensor,
    audio_embeddings: Tensor,
    audio_placeholder_mask: Tensor,
    audio_attention_mask: Tensor,
) -> Tensor:
    """Replace per-sample placeholder positions with valid projected audio tokens."""
    if text_embeddings.ndim != 3 or audio_embeddings.ndim != 3:
        raise ValueError("text_embeddings and audio_embeddings must be rank 3")
    if audio_placeholder_mask.shape != text_embeddings.shape[:2]:
        raise ValueError("audio_placeholder_mask does not match text embeddings")
    if audio_attention_mask.shape != audio_embeddings.shape[:2]:
        raise ValueError("audio_attention_mask does not match audio embeddings")
    if text_embeddings.shape[0] != audio_embeddings.shape[0]:
        raise ValueError("text and audio batch sizes differ")
    if text_embeddings.shape[-1] != audio_embeddings.shape[-1]:
        raise ValueError("text and projected audio embedding sizes differ")

    output = text_embeddings.clone()
    for batch_index in range(text_embeddings.shape[0]):
        placeholder_count = int(audio_placeholder_mask[batch_index].sum().item())
        audio_count = int(audio_attention_mask[batch_index].sum().item())
        if placeholder_count != audio_count:
            raise ValueError(
                f"sample {batch_index} has {placeholder_count} placeholders "
                f"but {audio_count} valid audio tokens"
            )
        output[batch_index, audio_placeholder_mask[batch_index]] = audio_embeddings[
            batch_index, audio_attention_mask[batch_index]
        ]
    return output


class ProjectorStage1Model(nn.Module):
    """Run a frozen audio encoder and LLM while training only the projector."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        language_model: nn.Module,
        audio_projector: AudioProjector,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.language_model = language_model
        self.audio_projector = audio_projector
        freeze_for_projector_training(self, self.audio_projector)

    def forward(
        self,
        *,
        audio_inputs: Tensor,
        audio_attention_mask: Tensor,
        text_embeddings: Tensor,
        audio_placeholder_mask: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **language_model_kwargs: Any,
    ) -> Any:
        # Encoder activations do not need a graph because its inputs are not trainable.
        with torch.no_grad():
            encoded = self.audio_encoder(audio_inputs, attention_mask=audio_attention_mask)
            audio_hidden_states = _last_hidden_state(encoded)
            audio_feature_mask = _feature_attention_mask(
                encoded, audio_attention_mask, audio_hidden_states
            )

        projected_audio = self.audio_projector(audio_hidden_states)
        inputs_embeds = replace_audio_placeholders(
            text_embeddings,
            projected_audio,
            audio_placeholder_mask.bool(),
            audio_feature_mask,
        )
        # Do not wrap this call in no_grad: input gradients must cross the frozen LLM.
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **language_model_kwargs,
        )


class CachedProjectorStage1Model(nn.Module):
    """Train the projector from frozen, precomputed ASR features.

    Caching is valid for stage one because the ASR encoder and its native
    projector are frozen.  It lets the GPU training loop load only JoyAI-VL
    and the small trainable projector.
    """

    def __init__(self, language_model: nn.Module, audio_projector: AudioProjector) -> None:
        super().__init__()
        self.language_model = language_model
        self.audio_projector = audio_projector
        freeze_for_projector_training(self, self.audio_projector)

    def forward(
        self,
        *,
        audio_features: Tensor,
        audio_attention_mask: Tensor,
        text_embeddings: Tensor,
        audio_placeholder_mask: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **language_model_kwargs: Any,
    ) -> Any:
        projected = (
            self.audio_projector(audio_features, audio_attention_mask)
            if hasattr(self.audio_projector, "k")
            else self.audio_projector(audio_features)
        )
        if isinstance(projected, tuple):
            projected_audio, projected_mask = projected
        else:
            projected_audio, projected_mask = projected, audio_attention_mask
        inputs_embeds = replace_audio_placeholders(
            text_embeddings,
            projected_audio,
            audio_placeholder_mask.bool(),
            projected_mask.bool(),
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **language_model_kwargs,
        )


def _last_hidden_state(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("audio encoder output does not expose last_hidden_state")


def _feature_attention_mask(
    output: Any, input_attention_mask: Tensor, hidden_states: Tensor
) -> Tensor:
    feature_mask = getattr(output, "attention_mask", None)
    if feature_mask is not None:
        if feature_mask.shape != hidden_states.shape[:2]:
            raise ValueError("encoder feature attention mask does not match hidden states")
        return feature_mask.bool()
    if input_attention_mask.shape == hidden_states.shape[:2]:
        return input_attention_mask.bool()
    raise ValueError(
        "audio encoder changed sequence length but did not return a feature attention mask"
    )
