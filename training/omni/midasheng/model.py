"""Projector-only model assembly for task-aware cached MiDasheng features."""

from __future__ import annotations

from typing import Any


def build_phase1_model(
    llm: Any, *, projector_in_features: int = 1280, projector_out_features: int = 4096,
    audio_projector: Any | None = None,
) -> Any:
    """Build a cached-feature model with exactly the project projector trainable.

    The caller supplies batches from
    ``projector_stage2.collate_cached_audio_conversations`` so VoiceAssistant,
    LibriSpeech and Clotho-AQA retain their distinct prompt contracts.
    """
    from torch import nn

    from training.omni.projector_stage1.model import CachedProjectorStage1Model
    from training.omni.projector_stage1.projector import (
        AudioProjector,
        AudioProjectorConfig,
        freeze_for_projector_training,
    )

    projector = audio_projector or AudioProjector(AudioProjectorConfig(
        input_size=projector_in_features, output_size=projector_out_features,
        hidden_size=projector_out_features,
    ))
    core = CachedProjectorStage1Model(llm, projector)

    class Phase1Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_encoder = nn.Identity()
            self.core = core

        def forward(self, batch: Any) -> Any:
            required = (
                "input_ids", "labels", "attention_mask", "audio_features",
                "audio_attention_mask", "audio_placeholder_mask",
            )
            if any(getattr(batch, name, None) is None for name in required):
                raise TypeError("phase-one batch must contain cached audio features and replacement masks")
            input_ids = batch.input_ids.to(next(self.parameters()).device)
            projector_dtype = next(self.core.audio_projector.parameters()).dtype
            return self.core(
                audio_features=batch.audio_features.to(input_ids.device, dtype=projector_dtype),
                audio_attention_mask=batch.audio_attention_mask.to(input_ids.device),
                text_embeddings=self.core.language_model.get_input_embeddings()(input_ids),
                audio_placeholder_mask=batch.audio_placeholder_mask.to(input_ids.device),
                attention_mask=batch.attention_mask.to(input_ids.device),
                labels=batch.labels.to(input_ids.device),
            )

    model = Phase1Model()
    freeze_for_projector_training(model, model.core.audio_projector)
    return model
