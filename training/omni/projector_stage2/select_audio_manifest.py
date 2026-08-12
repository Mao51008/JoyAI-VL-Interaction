"""Select formal audio-manifest rows by source dataset or training task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_manifests(
    *,
    train_manifest: Path,
    dev_manifest: Path,
    output_dir: Path,
    provenance_dataset: str | None = None,
    training_task: str | None = None,
) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    if provenance_dataset is None and training_task is None:
        raise ValueError("at least one selector is required")

    def matches(row: dict[str, Any]) -> bool:
        if provenance_dataset is not None:
            provenance = row.get("provenance", {})
            if not isinstance(provenance, dict) or provenance.get("dataset") != provenance_dataset:
                return False
        return training_task is None or row.get("training_task") == training_task

    output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    for split, manifest in (("train", train_manifest), ("dev", dev_manifest)):
        rows = [row for row in _rows(manifest) if matches(row)]
        if not rows:
            raise ValueError(f"selector produced no {split} rows")
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-dataset")
    parser.add_argument("--training-task")
    print(json.dumps(select_manifests(**vars(parser.parse_args())), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
