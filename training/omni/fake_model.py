"""Tiny NumPy decision model used only to prove CPU training plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .batching import FloatArray, OmniBatch
from .schema import ActionKind


@dataclass(frozen=True)
class FakeModelConfig:
    audio_hidden_size: int = 8
    bridge_hidden_size: int = 6
    seed: int = 20260730


class FakeOmniDecisionModel:
    """Train a bridge and action head; never represents real model quality."""

    def __init__(self, config: FakeModelConfig | None = None) -> None:
        self.config = config or FakeModelConfig()
        rng = np.random.default_rng(self.config.seed)
        self.bridge = rng.normal(
            0,
            0.05,
            (self.config.audio_hidden_size, self.config.bridge_hidden_size),
        ).astype(np.float32)
        feature_size = self.config.bridge_hidden_size + 4
        self.action_head = rng.normal(
            0,
            0.05,
            (feature_size, len(ActionKind)),
        ).astype(np.float32)
        self.action_bias = np.zeros(len(ActionKind), dtype=np.float32)

    def train_step(self, batch: OmniBatch, learning_rate: float) -> float:
        features, labels = self._features_and_labels(batch)
        logits = features @ self.action_head + self.action_bias
        probabilities = _softmax(logits)
        count = labels.shape[0]
        loss = float(-np.log(probabilities[np.arange(count), labels] + 1e-12).mean())

        grad_logits = probabilities
        grad_logits[np.arange(count), labels] -= 1
        grad_logits /= count
        grad_head = features.T @ grad_logits
        grad_bias = grad_logits.sum(axis=0)
        grad_features = grad_logits @ self.action_head.T
        grad_bridge = (
            self._audio_means(batch).T
            @ grad_features[:, : self.config.bridge_hidden_size]
        )

        self.action_head -= learning_rate * grad_head
        self.action_bias -= learning_rate * grad_bias
        self.bridge -= learning_rate * grad_bridge
        return loss

    def predict(self, batch: OmniBatch) -> NDArray[np.int64]:
        features, _labels = self._features_and_labels(batch)
        logits = features @ self.action_head + self.action_bias
        return np.argmax(logits, axis=1).astype(np.int64)

    def _features_and_labels(
        self, batch: OmniBatch
    ) -> tuple[FloatArray, NDArray[np.int64]]:
        audio_means = self._audio_means(batch)
        bridged = audio_means @ self.bridge
        rows = []
        labels = []
        flat_index = 0
        for batch_index in range(len(batch.sample_ids)):
            valid_steps = np.flatnonzero(batch.step_mask[batch_index])
            max_time = max(
                int(batch.step_end_times_ms[batch_index, valid_steps[-1]]), 1
            )
            for step_index in valid_steps:
                time_feature = (
                    float(batch.step_end_times_ms[batch_index, step_index]) / max_time
                )
                rows.append(
                    np.concatenate(
                        (
                            bridged[flat_index],
                            batch.modality_presence[batch_index],
                            np.asarray([time_feature], dtype=np.float32),
                        )
                    )
                )
                labels.append(batch.action_labels[batch_index, step_index])
                flat_index += 1
        return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def _audio_means(self, batch: OmniBatch) -> FloatArray:
        rows = []
        for batch_index in range(len(batch.sample_ids)):
            valid_steps = np.flatnonzero(batch.step_mask[batch_index])
            for step_index in valid_steps:
                step_end = batch.step_end_times_ms[batch_index, step_index]
                visible = batch.audio_attention_mask[batch_index] & (
                    batch.audio_token_times_ms[batch_index] < step_end
                )
                if np.any(visible):
                    rows.append(
                        batch.audio_hidden_states[batch_index, visible].mean(axis=0)
                    )
                else:
                    rows.append(
                        np.zeros(self.config.audio_hidden_size, dtype=np.float32)
                    )
        return np.asarray(rows, dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "audio_hidden_size": self.config.audio_hidden_size,
                "bridge_hidden_size": self.config.bridge_hidden_size,
                "seed": self.config.seed,
            },
            "bridge": self.bridge.tolist(),
            "action_head": self.action_head.tolist(),
            "action_bias": self.action_bias.tolist(),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> FakeOmniDecisionModel:
        model = cls(FakeModelConfig(**value["config"]))
        model.bridge = np.asarray(value["bridge"], dtype=np.float32)
        model.action_head = np.asarray(value["action_head"], dtype=np.float32)
        model.action_bias = np.asarray(value["action_bias"], dtype=np.float32)
        return model


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
