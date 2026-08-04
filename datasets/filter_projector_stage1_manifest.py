"""Create a bounded-duration projector manifest without altering the source JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(source: Path, output: Path, max_duration_ms: int) -> dict[str, int]:
    if max_duration_ms <= 0:
        raise ValueError("max duration must be positive")
    kept = removed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            row = json.loads(line)
            if int(row["duration_ms"]) > max_duration_ms:
                removed += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    result = {"kept": kept, "removed_over_duration": removed, "max_duration_ms": max_duration_ms}
    output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration-ms", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.max_duration_ms), indent=2))


if __name__ == "__main__":
    main()
