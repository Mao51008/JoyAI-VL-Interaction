"""Checkpoint scheduling and compatibility helpers for stage-one training."""
from __future__ import annotations

from pathlib import Path


def should_save_last(
    step: int,
    *,
    total_steps: int,
    every_steps: int,
    steps_per_epoch: int | None,
) -> bool:
    """Save periodically, at epoch boundaries, and at the final training step."""
    if step <= 0 or total_steps <= 0 or every_steps <= 0:
        raise ValueError("steps and checkpoint interval must be positive")
    return (
        step == total_steps
        or step % every_steps == 0
        or (steps_per_epoch is not None and step % steps_per_epoch == 0)
    )


def resume_checkpoint_path(output_dir: Path) -> Path:
    """Prefer the new last checkpoint while accepting pre-schedule runs."""
    last = output_dir / "last.pt"
    if last.is_file():
        return last
    legacy = output_dir / "latest.pt"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"no last.pt or legacy latest.pt in {output_dir}")
