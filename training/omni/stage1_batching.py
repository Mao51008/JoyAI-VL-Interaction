"""Length-aware batch planning for cached projector-stage-one features."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def padded_attention_cost(token_counts: Sequence[int]) -> int:
    """Estimate the padded attention work for one batch from audio token lengths."""
    if not token_counts:
        raise ValueError("token_counts must not be empty")
    if any(count <= 0 for count in token_counts):
        raise ValueError("audio token counts must be positive")
    return len(token_counts) * max(token_counts) ** 2


def plan_length_aware_batches(
    sample_ids: Sequence[str],
    token_counts: Mapping[str, int],
    *,
    max_batch_size: int,
    max_attention_cost: int,
) -> list[list[str]]:
    """Keep every sample while making oversized pairs single-sample batches.

    Sorting by length minimizes padding. A sample larger than the pair budget is
    still retained as a single batch, because dropping it would silently change
    the training corpus.
    """
    if max_batch_size <= 0 or max_attention_cost <= 0:
        raise ValueError("batch size and attention cost budget must be positive")
    ordered = sorted(sample_ids, key=lambda sample_id: token_counts[sample_id], reverse=True)
    batches: list[list[str]] = []
    current: list[str] = []
    for sample_id in ordered:
        candidate = [*current, sample_id]
        if current and (
            len(candidate) > max_batch_size
            or padded_attention_cost([token_counts[item] for item in candidate]) > max_attention_cost
        ):
            batches.append(current)
            current = [sample_id]
        else:
            current = candidate
        if len(current) == max_batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def distribute_batches(
    batches: Sequence[Sequence[str]], world_size: int) -> list[list[list[str]]]:
    """Round-robin batches across ranks and pad ranks by repeating their own work."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not batches:
        raise ValueError("batches must not be empty")
    per_rank = [list(map(list, batches[rank::world_size])) for rank in range(world_size)]
    longest = max(len(rank_batches) for rank_batches in per_rank)
    for rank_batches in per_rank:
        if not rank_batches:
            raise ValueError("world size exceeds planned batches")
        source = list(rank_batches)
        while len(rank_batches) < longest:
            rank_batches.append(list(source[(len(rank_batches) - len(source)) % len(source)]))
    return per_rank
