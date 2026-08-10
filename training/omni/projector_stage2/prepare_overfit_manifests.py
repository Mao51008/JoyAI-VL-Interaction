"""Create deterministic, dialogue-disjoint manifests for Stage2 overfit experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .train import load_manifest


def select_one_turn_per_dialogue(
    rows: Sequence[dict[str, Any]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    """Choose one reproducible turn from each selected dialogue."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dialogue_id"]), []).append(row)
    if sample_count > len(grouped):
        raise ValueError(
            f"requested {sample_count} samples but manifest has only {len(grouped)} dialogues"
        )
    rng = random.Random(seed)
    dialogue_ids = sorted(grouped)
    rng.shuffle(dialogue_ids)
    selected = [rng.choice(grouped[dialogue_id]) for dialogue_id in dialogue_ids[:sample_count]]
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def assign_mixed_tasks(
    rows: Sequence[dict[str, Any]], asr_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Annotate a fixed ASR/dialogue mixture without changing selected examples."""
    if not 0 <= asr_samples <= len(rows):
        raise ValueError(f"asr_samples must be in [0, {len(rows)}]")
    asr_indices = set(random.Random(seed).sample(range(len(rows)), asr_samples))
    return [
        dict(
            row,
            training_task=(
                "asr_transcription" if index in asr_indices else "dialogue_response"
            ),
        )
        for index, row in enumerate(rows)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--asr-train-samples", type=int, default=16)
    parser.add_argument("--dev-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    train_rows = select_one_turn_per_dialogue(
        load_manifest(args.train_manifest), args.train_samples, args.seed
    )
    train_rows = assign_mixed_tasks(train_rows, args.asr_train_samples, args.seed + 2)
    dev_rows = select_one_turn_per_dialogue(
        load_manifest(args.dev_manifest), args.dev_samples, args.seed + 1
    )
    train_dialogues = {str(row["dialogue_id"]) for row in train_rows}
    dev_dialogues = {str(row["dialogue_id"]) for row in dev_rows}
    overlap = train_dialogues & dev_dialogues
    if overlap:
        raise ValueError(f"train/dev dialogue leakage: {sorted(overlap)[:5]}")
    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)
    metadata = {
        "format": "projector-stage2-overfit-manifests-v1",
        "seed": args.seed,
        "train": {
            "source": str(args.train_manifest.resolve()),
            "samples": len(train_rows),
            "dialogues": len(train_dialogues),
            "tasks": {
                "asr_transcription": args.asr_train_samples,
                "dialogue_response": len(train_rows) - args.asr_train_samples,
            },
            "manifest": str(train_path.resolve()),
            "sha256": _sha256(train_path),
        },
        "dev": {
            "source": str(args.dev_manifest.resolve()),
            "samples": len(dev_rows),
            "dialogues": len(dev_dialogues),
            "manifest": str(dev_path.resolve()),
            "sha256": _sha256(dev_path),
        },
        "selection": "one_random_turn_per_dialogue_with_fixed_task_assignment",
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
