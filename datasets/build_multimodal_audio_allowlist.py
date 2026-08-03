"""Build a per-file audio allowlist from local FFprobe CSV reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


KNOWN_SILENT = {"-BCoVGruruc"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(report: Path, source: str, media_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # FFprobe reports produced by PowerShell may contain a UTF-8 BOM.
    with report.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("file", "")).strip().strip('"')
            if not name or name == "file":
                continue
            media = media_dir / f"{name}.mp4"
            audio_codec = str(row.get("audio", "") or row.get("audio_codec", "")).strip()
            sample_rate = row.get("rate") or row.get("sample_rate")
            channels = row.get("channels")
            audio_duration = row.get("audio_s")
            silent = name in KNOWN_SILENT
            rows.append(
                {
                    "schema_version": "multimodal-audio-allowlist-v1",
                    "source": source,
                    "video_name": media.name,
                    "media_path": str(media.resolve()),
                    "media_exists": media.is_file(),
                    "media_size_bytes": media.stat().st_size if media.is_file() else 0,
                    "media_sha256": _sha256(media) if media.is_file() else "",
                    "audio_stream_status": "present" if audio_codec else "missing",
                    "audible_content_status": "silent" if silent else "unknown",
                    "audio_codec": audio_codec,
                    "audio_sample_rate": int(float(sample_rate)) if sample_rate else 0,
                    "audio_channels": int(channels) if channels else 0,
                    "audio_duration_s": float(audio_duration) if audio_duration else 0.0,
                    "allowlist_status": (
                        "exclude_silent" if silent else
                        "confirmed_audio_stream" if audio_codec and media.is_file()
                        else "unusable"
                    ),
                    "audio_semantics": "original_scene_audio_not_user_speech",
                    "training_role": "multimodal_auxiliary",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--media-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.report) == len(args.source) == len(args.media_dir)):
        parser.error("--report, --source and --media-dir must have equal counts")
    rows: list[dict[str, Any]] = []
    for report, source, media_dir in zip(args.report, args.source, args.media_dir):
        rows.extend(build(report, source, media_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "records": len(rows),
        "confirmed_audio_stream": sum(row["allowlist_status"] == "confirmed_audio_stream" for row in rows),
        "excluded_silent": sum(row["allowlist_status"] == "exclude_silent" for row in rows),
        "unusable": sum(row["allowlist_status"] == "unusable" for row in rows),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
