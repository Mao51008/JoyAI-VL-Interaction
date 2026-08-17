"""Create a deterministic, bounded ACAVCaps YouTube audio manifest."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = ("00A", "0M0", "0MA", "S0A", "SM0", "SMA")


def parse_sample_key(key: str) -> tuple[str, float, float]:
    """Parse ACAVCaps keys whose timestamp decimal components may be omitted."""
    parts = key.rsplit("_", 4)
    if len(parts) == 5:
        video_id, start_int, start_decimal, end_int, end_decimal = parts
        start = float(f"{start_int}.{start_decimal}")
        end = float(f"{end_int}.{end_decimal}")
    elif len(parts) == 4:
        video_id, start_int, start_decimal, end_int = parts
        start = float(f"{start_int}.{start_decimal}")
        end = float(end_int)
    elif len(parts) == 3:
        video_id, start_int, end_int = parts
        start = float(start_int)
        end = float(end_int)
    else:
        raise ValueError(f"invalid ACAVCaps sample key: {key}")
    if len(video_id) != 11 or start < 0 or end <= start:
        raise ValueError(f"invalid ACAVCaps sample range: {key}")
    return video_id, start, end


def build_subset(metadata_dir: Path, per_category: int, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    rows: list[dict[str, Any]] = []
    for category in categories:
        source = metadata_dir / f"{category}.jsonl.gz"
        if not source.is_file():
            raise FileNotFoundError(source)
        selected = 0
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                entry = json.loads(line)
                if len(entry) != 1:
                    raise ValueError(f"{source}:{line_number}: expected one sample per line")
                sample_key, annotations = next(iter(entry.items()))
                video_id, start, end = parse_sample_key(sample_key)
                rows.append(
                    {
                        "sample_key": sample_key,
                        "category": category,
                        "video_id": video_id,
                        "start_seconds": start,
                        "end_seconds": end,
                        "annotations": annotations,
                    }
                )
                selected += 1
                if selected == per_category:
                    break
        if selected != per_category:
            raise ValueError(f"{source}: only {selected} samples, need {per_category}")
    return rows


def write_manifest(rows: list[dict[str, Any]], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10_000)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = parser.parse_args(argv)
    rows = build_subset(args.metadata_dir, args.per_category, tuple(args.categories))
    write_manifest(rows, args.output_manifest)
    print(json.dumps({"manifest": str(args.output_manifest), "samples": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
