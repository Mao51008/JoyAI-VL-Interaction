"""Selectively audit JoyAI media inside a tar archive without full extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}


@dataclass(frozen=True)
class ArchiveMatch:
    video_name: str
    member_name: str
    member_size: int
    annotation_count: int
    max_annotation_time_s: float
    selection_reason: str = "stable_sample"


def canonical_video_key(value: str) -> str:
    """Match annotation names with archive members regardless of media extension."""
    name = PurePosixPath(value.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    return (
        name[: -len(suffix)].casefold() if suffix in MEDIA_SUFFIXES else name.casefold()
    )


def load_annotation_index(
    path: Path,
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("source") != source:
            continue
        video_name = str(record.get("video_name", ""))
        key = canonical_video_key(video_name)
        entry = grouped.setdefault(
            key,
            {
                "video_name": video_name,
                "records": [],
                "max_annotation_time_s": 0.0,
            },
        )
        entry["records"].append(record)
        times = [
            float(node.get("time", 0))
            for branch in ("question", "response")
            for node in record.get(branch, [])
        ]
        entry["max_annotation_time_s"] = max(
            entry["max_annotation_time_s"],
            max(times, default=0.0),
        )
    return grouped


def scan_archive(
    archive_path: Path,
    annotation_index: dict[str, dict[str, Any]],
) -> tuple[list[ArchiveMatch], dict[str, Any]]:
    media_members = 0
    duplicate_keys: Counter[str] = Counter()
    matches = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(PurePosixPath(member.name).name).suffix.lower()
            if suffix not in MEDIA_SUFFIXES:
                continue
            media_members += 1
            key = canonical_video_key(member.name)
            duplicate_keys[key] += 1
            annotation = annotation_index.get(key)
            if annotation:
                matches.append(
                    ArchiveMatch(
                        video_name=annotation["video_name"],
                        member_name=member.name,
                        member_size=member.size,
                        annotation_count=len(annotation["records"]),
                        max_annotation_time_s=annotation["max_annotation_time_s"],
                    )
                )
    summary = {
        "archive": str(archive_path.resolve()),
        "archive_size_bytes": archive_path.stat().st_size,
        "media_member_count": media_members,
        "matched_member_count": len(matches),
        "matched_unique_video_names": len({match.video_name for match in matches}),
        "archive_media_name_collisions": sum(
            count > 1 for count in duplicate_keys.values()
        ),
    }
    return matches, summary


def select_matches(
    matches: list[ArchiveMatch],
    sample_size: int,
    *,
    seed: str,
    priority_names: set[str] | None = None,
) -> list[ArchiveMatch]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    priority_keys = {canonical_video_key(name) for name in (priority_names or set())}
    unique: dict[str, ArchiveMatch] = {}
    collisions: set[str] = set()
    for match in matches:
        key = canonical_video_key(match.video_name)
        if key in unique:
            collisions.add(key)
        else:
            unique[key] = match
    for key in collisions:
        unique.pop(key, None)

    def sort_key(item: tuple[str, ArchiveMatch]) -> tuple[int, str]:
        key, _match = item
        digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        return (0 if key in priority_keys else 1, digest)

    selected = []
    for key, match in sorted(unique.items(), key=sort_key)[:sample_size]:
        reason = "priority_candidate" if key in priority_keys else "stable_sample"
        selected.append(
            ArchiveMatch(
                **{
                    **asdict(match),
                    "selection_reason": reason,
                }
            )
        )
    return selected


def load_priority_names(path: Path | None, source: str) -> set[str]:
    if path is None:
        return set()
    names = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == source:
                names.add(str(row["video_name"]))
    return names


def extract_selected(
    archive_path: Path,
    matches: list[ArchiveMatch],
    extract_dir: Path,
) -> dict[str, Path]:
    """Extract by stream copy into controlled names; never trust tar paths."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    wanted = {match.member_name: match for match in matches}
    extracted = {}
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            match = wanted.get(member.name)
            if match is None:
                continue
            if not member.isfile():
                raise ValueError(
                    f"selected member is not a regular file: {member.name}"
                )
            suffix = Path(PurePosixPath(member.name).name).suffix.lower() or ".mp4"
            destination = (
                extract_dir / f"{canonical_video_key(match.video_name)}{suffix}"
            )
            resolved = destination.resolve()
            if extract_dir.resolve() not in resolved.parents:
                raise ValueError(f"unsafe extraction destination: {destination}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.stat().st_size != member.size:
                raise ValueError(f"extracted size mismatch: {member.name}")
            extracted[match.video_name] = destination
    missing = {match.video_name for match in matches} - set(extracted)
    if missing:
        raise ValueError(
            f"selected members disappeared during extraction: {sorted(missing)}"
        )
    return extracted


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    if not _tool_available(ffprobe):
        return {"probe_status": "tool_missing", "probe_error": ffprobe}
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        return {
            "probe_status": "failed",
            "probe_error": result.stderr.strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"probe_status": "failed", "probe_error": f"invalid JSON: {exc}"}
    return parse_probe_payload(payload)


def parse_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams", [])
    video_streams = [
        stream for stream in streams if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    video = video_streams[0] if video_streams else {}
    audio = audio_streams[0] if audio_streams else {}
    format_info = payload.get("format", {})
    duration = _first_float(
        format_info.get("duration"),
        video.get("duration"),
        audio.get("duration"),
    )
    audio_duration = _first_float(audio.get("duration"), format_info.get("duration"))
    return {
        "probe_status": "ok",
        "probe_error": "",
        "duration_s": duration,
        "video_stream_count": len(video_streams),
        "video_codec": str(video.get("codec_name", "")),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "frame_rate": str(video.get("avg_frame_rate", "")),
        "audio_stream_count": len(audio_streams),
        "audio_codec": str(audio.get("codec_name", "")),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": int(audio.get("channels", 0) or 0),
        "audio_duration_s": audio_duration,
    }


def decode_media(path: Path, *, ffmpeg: str = "ffmpeg") -> tuple[str, str]:
    if not _tool_available(ffmpeg):
        return "tool_missing", ffmpeg
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        check=False,
        text=True,
    )
    return ("ok", "") if result.returncode == 0 else ("failed", result.stderr.strip())


def analyze_silence(
    path: Path,
    duration_s: float,
    *,
    ffmpeg: str = "ffmpeg",
) -> tuple[str, float | None, str]:
    if not _tool_available(ffmpeg):
        return "tool_missing", None, ffmpeg
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "info",
            "-nostdin",
            "-i",
            str(path),
            "-vn",
            "-af",
            "silencedetect=noise=-50dB:d=0.5",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        return "failed", None, result.stderr.strip()
    durations = [
        float(value)
        for value in re.findall(r"silence_duration:\s*([0-9.]+)", result.stderr)
    ]
    ratio = min(sum(durations) / duration_s, 1.0) if duration_s > 0 else 0.0
    return "ok", round(ratio, 6), ""


def extract_frames(
    path: Path,
    frame_dir: Path,
    *,
    ffmpeg: str = "ffmpeg",
    fps: float = 1.0,
) -> tuple[str, list[dict[str, Any]], str]:
    if not _tool_available(ffmpeg):
        return "tool_missing", [], ffmpeg
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%06d.jpg"
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-vf",
            f"fps={fps}",
            "-start_number",
            "0",
            "-q:v",
            "5",
            "-y",
            str(pattern),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        return "failed", [], result.stderr.strip()
    frames = [
        {"path": str(frame.resolve()), "timestamp_ms": round(index * 1000 / fps)}
        for index, frame in enumerate(sorted(frame_dir.glob("frame_*.jpg")))
    ]
    return ("ok", frames, "") if frames else ("failed", [], "no frames extracted")


def audit_selected(
    selected: list[ArchiveMatch],
    extracted: dict[str, Path],
    *,
    decode: bool,
    silence: bool,
    frames_dir: Path | None,
    frame_fps: float,
    ffprobe: str,
    ffmpeg: str,
) -> list[dict[str, Any]]:
    rows = []
    for match in selected:
        media_path = extracted[match.video_name]
        row = {
            **asdict(match),
            "media_path": str(media_path.resolve()),
            "extracted_size_bytes": media_path.stat().st_size,
            **probe_media(media_path, ffprobe=ffprobe),
        }
        duration = float(row.get("duration_s", 0) or 0)
        audio_duration = float(row.get("audio_duration_s", 0) or 0)
        row["audio_video_duration_delta_s"] = (
            round(abs(duration - audio_duration), 6)
            if duration > 0 and audio_duration > 0
            else ""
        )
        row["annotation_within_duration"] = (
            row["probe_status"] == "ok"
            and match.max_annotation_time_s <= duration + 0.25
        )
        if decode:
            row["decode_status"], row["decode_error"] = decode_media(
                media_path,
                ffmpeg=ffmpeg,
            )
        else:
            row["decode_status"], row["decode_error"] = "not_run", ""
        if silence and row.get("audio_stream_count", 0) > 0:
            (
                row["silence_status"],
                row["silence_ratio"],
                row["silence_error"],
            ) = analyze_silence(
                media_path,
                audio_duration or duration,
                ffmpeg=ffmpeg,
            )
        else:
            row["silence_status"] = (
                "not_applicable"
                if row.get("probe_status") == "ok"
                and row.get("audio_stream_count", 0) == 0
                else "not_run"
            )
            row["silence_ratio"], row["silence_error"] = None, ""
        row["frame_status"], row["frames"], row["frame_error"] = (
            extract_frames(
                media_path,
                frames_dir / canonical_video_key(match.video_name),
                ffmpeg=ffmpeg,
                fps=frame_fps,
            )
            if frames_dir
            else ("not_run", [], "")
        )
        row["media_usable"] = (
            row["probe_status"] == "ok"
            and row.get("video_stream_count", 0) > 0
            and row["annotation_within_duration"]
            and row["decode_status"] in {"ok", "not_run"}
        )
        row["audio_usable"] = (
            row["media_usable"]
            and row.get("audio_stream_count", 0) > 0
            and row.get("audio_sample_rate", 0) > 0
        )
        rows.append(row)
    return rows


def run(
    archive_path: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    source: str,
    sample_size: int,
    seed: str,
    priority_path: Path | None,
    decode: bool,
    analyze_audio_silence: bool,
    extract_frame_images: bool,
    frame_fps: float,
    ffprobe: str,
    ffmpeg: str,
    expected_size: int | None = None,
) -> dict[str, Any]:
    actual_size = archive_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(
            f"archive size mismatch: expected {expected_size}, got {actual_size}"
        )
    annotations = load_annotation_index(annotations_path, source=source)
    matches, archive_summary = scan_archive(archive_path, annotations)
    priorities = load_priority_names(priority_path, source)
    selected = select_matches(
        matches,
        sample_size,
        seed=seed,
        priority_names=priorities,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_selected(archive_path, selected, output_dir / "extracted")
    rows = audit_selected(
        selected,
        extracted,
        decode=decode,
        silence=analyze_audio_silence,
        frames_dir=output_dir / "frames" if extract_frame_images else None,
        frame_fps=frame_fps,
        ffprobe=ffprobe,
        ffmpeg=ffmpeg,
    )
    summary = {
        **archive_summary,
        "annotation_video_name_count": len(annotations),
        "selected_count": len(selected),
        "probe_status": dict(Counter(row["probe_status"] for row in rows)),
        "decode_status": dict(Counter(row["decode_status"] for row in rows)),
        "frame_status": dict(Counter(row["frame_status"] for row in rows)),
        "silence_status": dict(Counter(row["silence_status"] for row in rows)),
        "media_usable_count": sum(bool(row["media_usable"]) for row in rows),
        "audio_usable_count": sum(bool(row["audio_usable"]) for row in rows),
        "important_note": (
            "audio_usable only proves that an audio stream exists and is probeable; "
            "it does not prove speech, event semantics, or audio-video alignment."
        ),
    }
    _write_jsonl(output_dir / "media_audit.jsonl", rows)
    _write_csv(output_dir / "media_audit.csv", rows)
    (output_dir / "media_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "media_audit_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    converted = []
    for row in rows:
        item = dict(row)
        item["frames"] = json.dumps(item["frames"], ensure_ascii=False)
        converted.append(item)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 媒体归档抽样审计",
            "",
            f"- 归档：`{summary['archive']}`",
            f"- 归档大小：{summary['archive_size_bytes']} 字节",
            f"- 媒体成员：{summary['media_member_count']}",
            f"- 匹配 JoyAI 名称：{summary['matched_unique_video_names']}",
            f"- 抽取审计：{summary['selected_count']}",
            f"- 媒体可用：{summary['media_usable_count']}",
            f"- 包含可探测音轨：{summary['audio_usable_count']}",
            "",
            "“包含可探测音轨”不代表音频内容与画面或标注匹配，仍需人工试听和抽查。",
            "",
        ]
    )


def _first_float(*values: Any) -> float:
    for value in values:
        try:
            if value not in (None, "", "N/A"):
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _tool_available(command: str) -> bool:
    path = Path(command)
    return (
        path.is_file()
        if path.parent != Path(".")
        else shutil.which(command) is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="CharadesEgo")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", default="20260730")
    parser.add_argument("--priority-jsonl", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--analyze-silence", action="store_true")
    parser.add_argument("--extract-frames", action="store_true")
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--expected-size", type=int)
    args = parser.parse_args()
    summary = run(
        args.archive,
        args.annotations,
        args.output_dir,
        source=args.source,
        sample_size=args.sample_size,
        seed=args.seed,
        priority_path=args.priority_jsonl,
        decode=args.decode,
        analyze_audio_silence=args.analyze_silence,
        extract_frame_images=args.extract_frames,
        frame_fps=args.frame_fps,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        expected_size=args.expected_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
