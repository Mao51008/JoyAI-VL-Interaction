"""Deterministic audio-feature ablations shared by stage-one evaluators."""
from __future__ import annotations

import random
from collections.abc import Sequence


def sample_exchange_order(sample_count: int, randomizer: random.Random) -> list[int]:
    """Return a deterministic derangement for cross-sample feature exchange."""
    if sample_count < 2:
        raise ValueError("sample exchange requires at least two samples")
    order = list(range(sample_count))
    for _attempt in range(100):
        randomizer.shuffle(order)
        if all(index != donor for index, donor in enumerate(order)):
            return order.copy()
    # A non-zero cyclic shift is a guaranteed fallback derangement.
    shift = randomizer.randrange(1, sample_count)
    return [(index + shift) % sample_count for index in range(sample_count)]


def sample_temporal_order(frame_count: int, randomizer: random.Random) -> list[int]:
    """Return a deterministic derangement for within-sample time-axis shuffling."""
    if frame_count < 2:
        raise ValueError("temporal shuffle requires at least two frames")
    return sample_exchange_order(frame_count, randomizer)


def apply_temporal_shuffle(feature, randomizer: random.Random):
    """Permute only the time axis while preserving shape, dtype, and values."""
    order = sample_temporal_order(feature.shape[0], randomizer)
    return feature[order]


def apply_feature_ablation(feature, mode: str):
    """Apply feature-space ablations; waveform-zero is intentionally separate."""
    if mode == "none":
        return feature
    if mode == "feature-zero":
        return feature.new_zeros(feature.shape)
    if mode == "projected-zero":
        return feature
    raise ValueError(f"unsupported feature ablation: {mode}")


def validate_exchange(features: Sequence, order: Sequence[int]) -> None:
    if len(features) != len(order):
        raise ValueError("sample exchange order length does not match features")
    if sorted(order) != list(range(len(features))):
        raise ValueError("sample exchange order must be a permutation")
    if any(index == donor for index, donor in enumerate(order)):
        raise ValueError("sample exchange order must not contain fixed points")
