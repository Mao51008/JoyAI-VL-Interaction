"""Build validated projector stage-one JSONL from local audio-text metadata.

This tool never downloads media. The input JSONL must already contain local
audio paths and transcript text. Files larger than the project download limit
are rejected so a mistaken path cannot silently pull a large archive into a
training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Support both ``python -m datasets...`` and direct execution from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.omni.schema import OmniSample

MAX_FILE_BYTES = 1_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(value: dict[str, Any]) -> dict[str, Any]:
    required = (
        "dataset",
        "version",
        "source_uri",
        "license_name",
        "license_tier",
        "allows_training",
        "allows_modification",
        "allows_redistribution",
        "allows_commercial_use",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"provenance missing fields: {', '.join(missing)}")
    return value


def convert(input_path: Path, output_path: Path, provenance_path: Path) -> dict[str, Any]:
    provenance = _provenance(json.loads(provenance_path.read_text(encoding="utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped: dict[str, int] = {}

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                source_version = row.get("dataset_version")
                if source_version is not None and source_version != provenance["version"]:
                    raise ValueError("provenance_version_mismatch")
                audio_path = Path(str(row["audio_path"])).expanduser().resolve()
                target_text = str(row["target_text"] if "target_text" in row else row["text"]).strip()
                input_text = str(row.get("input_text", "")).strip()
                duration_ms = int(row["duration_ms"])
                sample_rate = int(row.get("sample_rate", 16_000))
                num_samples = int(row["num_samples"])
                if not audio_path.is_file():
                    raise ValueError("audio_missing")
                if audio_path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("audio_file_over_1gb")
                if not target_text or duration_ms <= 0 or sample_rate <= 0 or num_samples <= 0:
                    raise ValueError("invalid_audio_text_fields")
                sample_id = str(row.get("sample_id") or f"stage1-{line_number:07d}")
                sample = {
                    "schema_version": "omni-training-v1",
                    "sample_id": sample_id,
                    "duration_ms": duration_ms,
                    "provenance": provenance,
                    "audio": [
                        {
                            "path": str(audio_path),
                            "start_ms": 0,
                            "end_ms": duration_ms,
                            "sample_rate": sample_rate,
                            "num_samples": num_samples,
                            "channel": "user_audio",
                        }
                    ],
                    "text": ([
                        {
                            "text": input_text,
                            "timestamp_ms": 0,
                            "channel": "user_text",
                            "auxiliary": False,
                        }
                    ] if input_text else []),
                    "targets": [],
                    "metadata": {
                        "stage": "projector_only_audio_alignment",
                        "target_kind": "assistant_transcription",
                        "assistant_target_text": target_text,
                        "dataset_version": source_version,
                        "split": str(row.get("split", "train")),
                        "source_record": str(row.get("source_record", "")),
                        "media_sha256": _sha256(audio_path),
                    },
                }
                OmniSample.from_dict(sample)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                key = str(exc) if str(exc) in {
                    "audio_missing",
                    "audio_file_over_1gb",
                    "invalid_audio_text_fields",
                    "provenance_version_mismatch",
                } else "invalid_record"
                skipped[key] = skipped.get(key, 0) + 1
                continue
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            converted += 1

    summary = {
        "schema_version": "omni-training-v1",
        "stage": "projector_only_audio_alignment",
        "converted_samples": converted,
        "skipped": skipped,
        "output": str(output_path.resolve()),
        "max_media_file_bytes": MAX_FILE_BYTES,
        "note": "No media is downloaded by this command.",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="local audio-text metadata JSONL")
    parser.add_argument("--output", type=Path, required=True, help="output omni JSONL")
    parser.add_argument("--provenance", type=Path, required=True, help="provenance JSON")
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.provenance), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
