"""Run the B2 CPU fake training loop without loading model weights or media."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .batching import FakeAudioEncoder, FakeAudioEncoderConfig, collate_samples
from .checkpoint import (
    TrainingState,
    fingerprint_file,
    load_latest,
    save_checkpoint,
)
from .fake_model import FakeModelConfig, FakeOmniDecisionModel
from .schema import load_samples


def run(
    data_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    samples = load_samples(data_path)
    encoder_config = FakeAudioEncoderConfig(**config["fake_audio_encoder"])
    encoder = FakeAudioEncoder(encoder_config)
    batch = collate_samples(samples, encoder, step_ms=int(config["step_ms"]))
    learning_rate = float(config["learning_rate"])
    total_steps = int(config["total_steps"])
    save_every = int(config["save_every"])
    data_fingerprint = fingerprint_file(data_path)

    if resume and (output_dir / "latest.json").is_file():
        model, state, resumed_from = load_latest(output_dir)
        if state.data_fingerprint != data_fingerprint:
            raise ValueError("training data changed since the checkpoint was created")
    else:
        model = FakeOmniDecisionModel(
            FakeModelConfig(
                audio_hidden_size=encoder_config.hidden_size,
                **config["fake_model"],
            )
        )
        state = TrainingState(
            optimizer={"name": "sgd", "learning_rate": learning_rate},
            scheduler={"name": "constant", "step": 0},
            rng={"seed": model.config.seed},
            run_config=config,
            data_fingerprint=data_fingerprint,
            code_commit=_git_commit(),
        )
        resumed_from = None

    initial_step = state.global_step
    losses = []
    while state.global_step < total_steps:
        loss = model.train_step(batch, learning_rate)
        losses.append(loss)
        state.global_step += 1
        state.data_cursor = (state.data_cursor + len(samples)) % len(samples)
        state.epoch += int(state.data_cursor == 0)
        state.scheduler["step"] = state.global_step
        if state.global_step % save_every == 0:
            save_checkpoint(output_dir, model, state)

    if state.global_step > initial_step and state.global_step % save_every:
        save_checkpoint(output_dir, model, state)
    return {
        "mode": "cpu_fake_only",
        "samples": len(samples),
        "decision_steps": int(batch.step_mask.sum()),
        "initial_step": initial_step,
        "final_step": state.global_step,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "resumed_from": str(resumed_from) if resumed_from else None,
        "latest_checkpoint": str((output_dir / "latest.json").resolve()),
        "gpu_used": False,
        "media_read": False,
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(args.data, args.output_dir, config, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
