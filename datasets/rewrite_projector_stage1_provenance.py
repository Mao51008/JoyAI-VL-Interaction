"""Copy a projector-stage-one manifest while correcting LibriSpeech provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rewrite(source: Path, output: Path, *, version: str) -> dict[str, int | str]:
    if not version.strip():
        raise ValueError("version must not be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8", newline="\n") as writer:
        for line in reader:
            row = json.loads(line)
            if row.get("provenance", {}).get("dataset") != "LibriSpeech":
                raise ValueError(f"{row.get('sample_id')}: expected LibriSpeech provenance")
            source_version = row.get("metadata", {}).get("dataset_version")
            if source_version is not None and source_version != version:
                raise ValueError(f"{row.get('sample_id')}: source/version mismatch")
            row["provenance"]["version"] = version
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return {"samples": count, "version": version, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(rewrite(args.input, args.output, version=args.version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
