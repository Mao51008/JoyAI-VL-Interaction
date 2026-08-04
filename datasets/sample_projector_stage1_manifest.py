"""Deterministically select a bounded projector-stage-one training subset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def select(source: Path, output: Path, *, seed: int, max_samples: int | None, max_duration_ms: int | None) -> dict[str, int]:
    if max_samples is None and max_duration_ms is None:
        raise ValueError("set --max-samples, --max-duration-ms, or both")
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    selected = []
    total_duration = 0
    for row in rows:
        if max_samples is not None and len(selected) >= max_samples:
            break
        duration = int(row["duration_ms"])
        if max_duration_ms is not None and total_duration + duration > max_duration_ms:
            continue
        selected.append(row)
        total_duration += duration
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    return {"samples": len(selected), "duration_ms": total_duration, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-duration-ms", type=int)
    args = parser.parse_args()
    print(json.dumps(select(args.input, args.output, seed=args.seed, max_samples=args.max_samples, max_duration_ms=args.max_duration_ms), indent=2))


if __name__ == "__main__":
    main()
