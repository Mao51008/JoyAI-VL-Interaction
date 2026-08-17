"""Deterministic explicit task sampling for MiDasheng phase-one alignment."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


PHASE1_TASK_WEIGHTS = {
    "voiceassistant": 0.50,
    "librispeech": 0.25,
    "clotho_aqa": 0.25,
}


def task_draw_counts(total_examples: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate a fixed draw count using largest remainder rounding."""
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("task weights must be non-empty and positive")
    weight_total = sum(weights.values())
    raw = {task: total_examples * weight / weight_total for task, weight in weights.items()}
    counts = {task: math.floor(value) for task, value in raw.items()}
    remaining = total_examples - sum(counts.values())
    for task in sorted(weights, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:remaining]:
        counts[task] += 1
    return counts


def sample_task_rows(
    rows_by_task: Mapping[str, Sequence[dict]], *, total_examples: int, seed: int
) -> list[tuple[str, int, dict]]:
    """Sample each task independently with replacement, never by manifest row count."""
    unknown = set(rows_by_task) - set(PHASE1_TASK_WEIGHTS)
    missing = set(PHASE1_TASK_WEIGHTS) - set(rows_by_task)
    if unknown or missing:
        raise ValueError(f"task sources must match phase-one tasks; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if any(not rows for rows in rows_by_task.values()):
        raise ValueError("each phase-one task source must contain at least one row")
    draws: list[tuple[str, int, dict]] = []
    counts = task_draw_counts(total_examples, PHASE1_TASK_WEIGHTS)
    for task in PHASE1_TASK_WEIGHTS:
        generator = random.Random(f"{seed}:{task}")
        source_rows = rows_by_task[task]
        for draw_index in range(counts[task]):
            draws.append((task, draw_index, dict(generator.choice(source_rows))))
    random.Random(seed).shuffle(draws)
    return draws
