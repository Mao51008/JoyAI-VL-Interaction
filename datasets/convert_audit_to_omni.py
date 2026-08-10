"""Convert media-audited JoyAI records into validated omni-training-v1 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.omni.schema import OmniSample

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}


def canonical_video_key(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    return (
        name[: -len(suffix)].casefold() if suffix in MEDIA_SUFFIXES else name.casefold()
    )


def load_audit_rows(path: Path, *, source: str) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audited_source = str(row.get("source", "")).strip()
            if audited_source and audited_source != source:
                continue
            key = canonical_video_key(str(row["video_name"]))
            if key in rows:
                raise ValueError(f"duplicate audited video_name: {row['video_name']}")
            rows[key] = row
    return rows


def load_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    # Reuse the production validator instead of maintaining a second license contract.
    probe = {
        "schema_version": "omni-training-v1",
        "sample_id": "provenance-validation",
        "duration_ms": 1,
        "provenance": value,
        "text": [{"text": "validation", "timestamp_ms": 0}],
    }
    OmniSample.from_dict(probe)
    return value


def convert(
    annotations_path: Path,
    audit_path: Path,
    provenance_path: Path,
    output_path: Path,
    *,
    source: str,
) -> dict[str, Any]:
    records = json.loads(annotations_path.read_text(encoding="utf-8"))
    audit_rows = load_audit_rows(audit_path, source=source)
    provenance = load_provenance(provenance_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    skipped: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    video_sample_counts: Counter[str] = Counter()

    for record_index, record in enumerate(records):
        if record.get("source") != source:
            continue
        video_name = str(record.get("video_name", ""))
        audit = audit_rows.get(canonical_video_key(video_name))
        if audit is None:
            skipped["not_in_media_audit"] += 1
            continue
        if not audit.get("media_usable"):
            skipped["media_not_usable"] += 1
            continue
        if audit.get("frame_status") != "ok" or not audit.get("frames"):
            skipped["frames_not_extracted"] += 1
            continue
        duration_s = float(audit.get("duration_s", 0) or 0)
        duration_ms = math.ceil(duration_s * 1000)
        if duration_ms <= 0:
            skipped["invalid_duration"] += 1
            continue

        text_inputs = _convert_text(record.get("question", []), duration_ms)
        targets = _convert_targets(record.get("response", []), duration_ms)
        if text_inputs is None or targets is None:
            skipped["annotation_timestamp_outside_media"] += 1
            continue
        frames = [
            {
                "path": str(frame["path"]),
                "timestamp_ms": min(int(frame["timestamp_ms"]), duration_ms - 1),
            }
            for frame in audit["frames"]
            if int(frame["timestamp_ms"]) < duration_ms
        ]
        if not frames:
            skipped["no_frames_inside_duration"] += 1
            continue

        audio = []
        if audit.get("audio_usable"):
            sample_rate = int(audit["audio_sample_rate"])
            audio_duration_s = min(
                float(audit.get("audio_duration_s", 0) or duration_s),
                duration_s,
            )
            audio_end_ms = min(math.floor(audio_duration_s * 1000), duration_ms)
            if audio_end_ms > 0 and sample_rate > 0:
                audio.append(
                    {
                        "path": str(audit["media_path"]),
                        "start_ms": 0,
                        "end_ms": audio_end_ms,
                        "sample_rate": sample_rate,
                        "num_samples": round(audio_end_ms * sample_rate / 1000),
                        "channel": "environment_audio",
                    }
                )

        sample_id = _sample_id(record_index, video_name, text_inputs, targets)
        sample = {
            "schema_version": "omni-training-v1",
            "sample_id": sample_id,
            "duration_ms": duration_ms,
            "provenance": provenance,
            "audio": audio,
            "video": frames,
            "text": text_inputs,
            "targets": targets,
            "metadata": {
                "joy_source": source,
                "joy_video_name": video_name,
                "joy_task_type": str(record.get("task_type", "")),
                "media_path": str(audit["media_path"]),
                "audio_semantics": (
                    "original_scene_audio_not_user_speech"
                    if audio
                    else "no_usable_audio_stream"
                ),
                "conversion_note": (
                    "JoyAI question text remains user_text; original video audio is "
                    "environment_audio and must not be treated as spoken question."
                ),
            },
        }
        try:
            OmniSample.from_dict(sample)
        except ValueError:
            skipped["omni_schema_validation_failed"] += 1
            continue
        converted.append(sample)
        video_sample_counts[video_name] += 1
        presence = ("audio+" if audio else "") + "video+text"
        modality_counts[presence] += 1

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in converted:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "omni-training-v1",
        "source": source,
        "converted_samples": len(converted),
        "converted_unique_videos": len(video_sample_counts),
        "modality_counts": dict(modality_counts),
        "skipped": dict(skipped),
        "output": str(output_path.resolve()),
        "important_note": (
            "Original CharadesEgo audio is labelled environment_audio. It is not "
            "evidence that the JoyAI question was spoken in the clip."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _convert_text(
    values: list[dict[str, Any]],
    duration_ms: int,
) -> list[dict[str, Any]] | None:
    converted = []
    for value in values:
        for timestamp_ms in _parse_times_ms(value.get("time")):
            if not 0 <= timestamp_ms < duration_ms:
                return None
            converted.append(
                {
                    "text": str(value["content"]),
                    "timestamp_ms": timestamp_ms,
                    "channel": "user_text",
                    "auxiliary": False,
                }
            )
    return converted


def _convert_targets(
    values: list[Any],
    duration_ms: int,
) -> list[dict[str, Any]] | None:
    flattened = []
    for value in values:
        flattened.extend(value if isinstance(value, list) else [value])
    converted = []
    occupied_steps = set()
    for value in flattened:
        for timestamp_ms in _parse_times_ms(value.get("time")):
            if not 0 <= timestamp_ms < duration_ms:
                return None
            step = timestamp_ms // 1000
            if step in occupied_steps:
                return None
            occupied_steps.add(step)
            converted.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "action": "response",
                    "text": str(value["content"]),
                }
            )
    return converted


def _parse_times_ms(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    return [
        round(float(part.strip()) * 1000)
        for part in str(value).split(",")
        if part.strip()
    ]


def _sample_id(
    record_index: int,
    video_name: str,
    text: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        [video_name, text, targets],
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"joy-{record_index:07d}-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("media_audit", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="CharadesEgo")
    args = parser.parse_args()
    summary = convert(
        args.annotations,
        args.media_audit,
        args.provenance,
        args.output,
        source=args.source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
