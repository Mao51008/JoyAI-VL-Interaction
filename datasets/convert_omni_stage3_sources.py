"""Convert audited media for multiple JoyAI sources into Omni JSONL files.

The provenance registry is a JSON object mapping each JoyAI ``source`` to a
provenance JSON file accepted by ``omni-training-v1``.  A mixed media-audit
JSONL is supported when each row includes its ``source`` field; legacy audit
files without that field continue to work when converted one source at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from convert_audit_to_omni import convert


def safe_file_stem(source: str) -> str:
    value = "".join(character if character.isalnum() else "_" for character in source)
    return value.strip("_") or "source"


def load_registry(path: Path) -> dict[str, Path]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provenance registry must map source names to file paths")
    registry: dict[str, Path] = {}
    for source, provenance in value.items():
        if not isinstance(source, str) or not isinstance(provenance, str):
            raise ValueError("provenance registry entries must be string -> string")
        resolved = (path.parent / provenance).resolve()
        if not resolved.is_file():
            raise ValueError(f"provenance file not found for {source}: {resolved}")
        registry[source] = resolved
    return registry


def run(
    annotations: Path,
    media_audit: Path,
    registry: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    used_names: set[str] = set()
    for source in sorted(registry):
        stem = safe_file_stem(source)
        if stem in used_names:
            raise ValueError(f"source names collide after filename normalization: {source}")
        used_names.add(stem)
        summaries.append(
            convert(
                annotations,
                media_audit,
                registry[source],
                output_dir / f"{stem}.jsonl",
                source=source,
            )
        )
    total = sum(int(summary["converted_samples"]) for summary in summaries)
    result = {"sources": summaries, "converted_samples": total}
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("media_audit", type=Path)
    parser.add_argument("--provenance-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.annotations,
                args.media_audit,
                load_registry(args.provenance_registry),
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
