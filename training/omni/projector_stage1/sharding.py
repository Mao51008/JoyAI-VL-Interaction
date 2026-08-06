"""Deterministic manifest sharding shared by stage-one evaluators."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Sequence

from ..schema import load_samples
from .ablation import sample_exchange_order, validate_exchange


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def shard_indices(total_samples: int, num_shards: int, shard_index: int) -> list[int]:
    validate_shard_args(num_shards, shard_index)
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    return list(range(shard_index, total_samples, num_shards))


def load_sharded_samples(manifest: Path, max_samples: int | None, num_shards: int, shard_index: int):
    """Load the complete manifest, then select deterministic global indices."""
    validate_shard_args(num_shards, shard_index)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    manifest_samples = load_samples(manifest)
    samples = manifest_samples if max_samples is None else manifest_samples[:max_samples]
    indices = shard_indices(len(samples), num_shards, shard_index)
    return manifest_samples, samples, indices


def global_exchange_order(sample_count: int, seed: int) -> list[int]:
    if sample_count < 2:
        raise ValueError("cross-sample shuffle requires at least two samples")
    order = sample_exchange_order(sample_count, random.Random(seed))
    validate_exchange(list(range(sample_count)), order)
    return order


def temporal_seed(seed: int, global_index: int) -> int:
    """Derive a stable per-sample seed independent of shard count and order."""
    return (seed + global_index) & 0xFFFFFFFF


def result_metadata(
    *,
    manifest: Path,
    manifest_sample_count: int,
    evaluation_sample_count: int,
    indices: Sequence[int],
    num_shards: int,
    shard_index: int,
    checkpoint: Path,
    ablation: str,
    seed: int,
) -> dict:
    validate_shard_args(num_shards, shard_index)
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_sample_count": manifest_sample_count,
        "total_samples": evaluation_sample_count,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selected_samples": len(indices),
        "global_indices": list(indices),
        "checkpoint": str(checkpoint),
        "audio_ablation": ablation,
        "seed": seed,
    }
