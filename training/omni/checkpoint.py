"""Recoverable JSON checkpoints for the CPU-only B2 training skeleton."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .fake_model import FakeOmniDecisionModel


@dataclass
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    data_cursor: int = 0
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    scaler: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "scale": 1.0}
    )
    rng: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""
    code_commit: str = ""


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    directory: Path,
    model: FakeOmniDecisionModel,
    state: TrainingState,
) -> Path:
    """Create an immutable step checkpoint, then atomically update latest.json."""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"checkpoint-{state.global_step:08d}.json"
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
    payload = {
        "format": "omni-b2-fake-checkpoint-v1",
        "training_state": asdict(state),
        "model": model.state_dict(),
    }
    _atomic_write_json(checkpoint_path, payload)
    _atomic_write_json(
        directory / "latest.json",
        {
            "format": "omni-b2-latest-v1",
            "checkpoint": checkpoint_path.name,
            "global_step": state.global_step,
        },
    )
    return checkpoint_path


def load_checkpoint(path: Path) -> tuple[FakeOmniDecisionModel, TrainingState]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "omni-b2-fake-checkpoint-v1":
        raise ValueError(f"unsupported checkpoint format in {path}")
    model = FakeOmniDecisionModel.from_state_dict(payload["model"])
    state = TrainingState(**payload["training_state"])
    return model, state


def load_latest(directory: Path) -> tuple[FakeOmniDecisionModel, TrainingState, Path]:
    latest_path = directory / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_path = directory / latest["checkpoint"]
    model, state = load_checkpoint(checkpoint_path)
    return model, state, checkpoint_path


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
