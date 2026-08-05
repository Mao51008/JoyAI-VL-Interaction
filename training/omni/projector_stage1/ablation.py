"""Deterministic audio-feature ablations shared by stage-one evaluators."""
from __future__ import annotations

import random
from collections.abc import Sequence


def sample_exchange_order(sample_count: int, randomizer: random.Random) -> list[int]:
    """Return a deterministic derangement for cross-sample feature exchange."""
    if sample_count < 2:
        raise ValueError("sample exchange requires at least two samples")
    order = list(range(sample_count))
    randomizer.shuffle(order)
    # A cyclic successor in a shuffled cycle cannot map a sample to itself.
    return order[1:] + order[:1]


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
