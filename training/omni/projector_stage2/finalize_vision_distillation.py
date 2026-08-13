"""Validate sharded JoyAI visual teacher labels and write one ordered manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def finalize(
    source_manifest: Path | list[Path],
    shards: list[Path],
    output_manifest: Path,
    teacher_model: str,
) -> dict[str, int]:
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite: {output_manifest}")
    source_paths = source_manifest if isinstance(source_manifest, list) else [source_manifest]
    source_rows = [row for path in source_paths for row in _load_jsonl(path)]
    teacher_rows = [row for shard in shards for row in _load_jsonl(shard)]
    source_ids = [str(row["sample_id"]) for row in source_rows]
    teacher_ids = [str(row.get("sample_id", "")) for row in teacher_rows]
    if len(teacher_ids) != len(set(teacher_ids)):
        raise ValueError("duplicate teacher sample_id")
    if set(source_ids) != set(teacher_ids):
        raise ValueError("teacher sample_ids do not exactly match the source manifest")
    by_id = {str(row["sample_id"]): row for row in teacher_rows}
    ordered = [by_id[sample_id] for sample_id in source_ids]
    for row in ordered:
        if not str(row.get("teacher_response", "")).strip():
            raise ValueError(f"empty teacher response: {row['sample_id']}")
        if not Path(str(row.get("image_path", ""))).is_file():
            raise FileNotFoundError(f"missing teacher image: {row.get('image_path')}")
        if row.get("provenance", {}).get("teacher_model") != teacher_model:
            raise ValueError(f"unexpected teacher model: {row['sample_id']}")
    with output_manifest.open("x", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"source_samples": len(source_rows), "written_samples": len(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    args = parser.parse_args()
    source_manifest: Path | list[Path] = (
        args.source_manifest[0] if len(args.source_manifest) == 1 else args.source_manifest
    )
    print(
        json.dumps(
            finalize(
                source_manifest=source_manifest,
                shards=args.shard,
                output_manifest=args.output_manifest,
                teacher_model=args.teacher_model,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
