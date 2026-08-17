"""Build a deterministic 50/25/25 MiDasheng phase-one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sampling import PHASE1_TASK_WEIGHTS, sample_task_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_manifest(
    *, voiceassistant_manifest: Path, librispeech_manifest: Path, clotho_aqa_manifest: Path,
    output_manifest: Path, total_examples: int, seed: int,
) -> dict[str, Any]:
    """Write an explicit sampler result without modifying any source manifest."""
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_manifest}")
    sources = {
        "voiceassistant": voiceassistant_manifest,
        "librispeech": librispeech_manifest,
        "clotho_aqa": clotho_aqa_manifest,
    }
    rows = {task: _read_jsonl(path) for task, path in sources.items()}
    draws = sample_task_rows(rows, total_examples=total_examples, seed=seed)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as handle:
        for task, draw_index, row in draws:
            metadata = dict(row.get("metadata", {}))
            source_sample_id = str(row.get("sample_id", "")).strip()
            if not source_sample_id:
                raise ValueError(f"{task} sampler source has no sample_id")
            row["sample_id"] = f"midasheng-{task}-{draw_index:08d}-{source_sample_id}"
            metadata["midasheng_phase1_sampler"] = {
                "task": task,
                "draw_index": draw_index,
                "seed": seed,
                "source_sample_id": source_sample_id,
            }
            row["metadata"] = metadata
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    task_counts = {task: sum(task == draw[0] for draw in draws) for task in PHASE1_TASK_WEIGHTS}
    return {
        "output_manifest": str(output_manifest),
        "samples": len(draws),
        "seed": seed,
        "task_counts": task_counts,
        "task_weights": PHASE1_TASK_WEIGHTS,
        "sources": {task: {"path": str(path), "sha256": _sha256(path)} for task, path in sources.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voiceassistant-manifest", type=Path, required=True)
    parser.add_argument("--librispeech-manifest", type=Path, required=True)
    parser.add_argument("--clotho-aqa-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    print(json.dumps(prepare_manifest(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
