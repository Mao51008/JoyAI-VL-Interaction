"""Convert sparse Omni targets into dense per-second action supervision."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ActionKind, OmniSample


@dataclass(frozen=True)
class DecisionStep:
    index: int
    start_ms: int
    end_ms: int
    action: ActionKind
    text: str
    semantic_tags: tuple[str, ...]


def build_decision_steps(
    sample: OmniSample,
    *,
    step_ms: int = 1000,
) -> tuple[DecisionStep, ...]:
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    step_count = (sample.duration_ms + step_ms - 1) // step_ms
    targets_by_step = {}
    for target in sample.targets:
        step_index = target.timestamp_ms // step_ms
        if step_index in targets_by_step:
            raise ValueError(
                f"sample {sample.sample_id!r} has multiple targets in step {step_index}"
            )
        targets_by_step[step_index] = target

    steps = []
    for index in range(step_count):
        target = targets_by_step.get(index)
        steps.append(
            DecisionStep(
                index=index,
                start_ms=index * step_ms,
                end_ms=min((index + 1) * step_ms, sample.duration_ms),
                action=target.action if target else ActionKind.SILENCE,
                text=target.text if target else "",
                semantic_tags=target.semantic_tags if target else (),
            )
        )
    return tuple(steps)
