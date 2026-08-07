"""Convert official SpokenWOZ user turns into single-channel audio clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tarfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _load_dialogues(data_json: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("data.json must contain a dialogue mapping")
    return value


def _load_dev_ids(val_list: Path) -> set[str]:
    ids = {line.strip() for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"empty validation split: {val_list}")
    return ids


def _parse_wav(source: bytes) -> tuple[int, int, int, int, bytes, int]:
    """Read PCM or IEEE float RIFF/WAVE without relying on wave's format support."""
    if len(source) < 12 or source[:4] != b"RIFF" or source[8:12] != b"WAVE":
        raise ValueError("source is not a RIFF/WAVE file")
    fmt: bytes | None = None
    data: bytes | None = None
    offset = 12
    while offset + 8 <= len(source):
        chunk_id = source[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", source, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        if end > len(source):
            raise ValueError("WAV chunk exceeds file bounds")
        if chunk_id == b"fmt ":
            fmt = source[start:end]
        elif chunk_id == b"data":
            data = source[start:end]
        offset = end + (chunk_size & 1)
    if fmt is None or data is None or len(fmt) < 16:
        raise ValueError("WAV is missing fmt or data chunk")
    audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    if audio_format not in {1, 3}:
        raise ValueError(f"unsupported WAV format: {audio_format}")
    if channels < 1 or sample_rate < 1 or bits < 8 or bits % 8:
        raise ValueError("invalid WAV format parameters")
    sample_width = bits // 8
    if block_align != channels * sample_width or len(data) % block_align:
        raise ValueError("invalid WAV block alignment")
    return audio_format, channels, sample_width, sample_rate, data, len(data) // block_align


def _write_wav(audio_format: int, channels: int, sample_width: int, sample_rate: int, data: bytes) -> bytes:
    if audio_format == 1:
        output = BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            writer.writeframes(data)
        return output.getvalue()
    if audio_format != 3:
        raise ValueError(f"unsupported WAV format: {audio_format}")
    fmt = struct.pack("<HHIIHH", audio_format, channels, sample_rate, sample_rate * channels * sample_width,
                      channels * sample_width, sample_width * 8)
    riff_size = 4 + 8 + len(fmt) + 8 + len(data)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data


def _clip_wav(source: bytes, words: list[dict[str, Any]], dialogue_id: str, turn_id: int) -> tuple[bytes, int, int]:
    if not words:
        raise ValueError(f"{dialogue_id} turn {turn_id}: words is empty")
    audio_format, channels, sample_width, sample_rate, frames, frame_count = _parse_wav(source)
    channel_ids: set[int] = set()
    starts: list[int] = []
    ends: list[int] = []
    for word in words:
        try:
            begin_ms = float(word["BeginTime"])
            end_ms = float(word["EndTime"])
            channel = int(word["ChannelId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{dialogue_id} turn {turn_id}: invalid word timing/channel") from exc
        if not math.isfinite(begin_ms) or not math.isfinite(end_ms):
            raise ValueError(f"{dialogue_id} turn {turn_id}: non-finite word timing")
        if channel < 0 or channel >= channels:
            raise ValueError(f"{dialogue_id} turn {turn_id}: channel {channel} is out of range")
        if begin_ms < 0 or end_ms <= begin_ms:
            raise ValueError(f"{dialogue_id} turn {turn_id}: invalid word interval")
        if end_ms > frame_count * 1000 / sample_rate:
            raise ValueError(f"{dialogue_id} turn {turn_id}: word interval exceeds WAV duration")
        channel_ids.add(channel)
        starts.append(max(0, math.floor(begin_ms * sample_rate / 1000)))
        ends.append(min(frame_count, math.ceil(end_ms * sample_rate / 1000)))
    if len(channel_ids) != 1:
        raise ValueError(f"{dialogue_id} turn {turn_id}: words cross audio channels")
    start_frame, end_frame = min(starts), max(ends)
    if not 0 <= start_frame < end_frame <= frame_count:
        raise ValueError(f"{dialogue_id} turn {turn_id}: clip is out of bounds")
    selected_channel = next(iter(channel_ids))
    block_width = channels * sample_width
    clip = b"".join(
        frame[selected_channel * sample_width : (selected_channel + 1) * sample_width]
        for frame in (
            frames[offset : offset + block_width]
            for offset in range(start_frame * block_width, end_frame * block_width, block_width)
        )
    )
    return _write_wav(audio_format, 1, sample_width, sample_rate, clip), selected_channel, round((end_frame - start_frame) * 1000 / sample_rate, 6)


def _audit(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train, dev = rows["train"], rows["dev"]
    fields = ("dialogue_id", "source_audio_sha256", "clip_sha256")
    result: dict[str, Any] = {}
    for field in fields:
        left = {row[field] for row in train}
        right = {row[field] for row in dev}
        overlap = sorted(left & right)
        result[field] = {"intersection_count": len(overlap), "examples": overlap[:10]}
    result["leakage"] = any(item["intersection_count"] for item in result.values())
    if result["leakage"]:
        raise ValueError("train/dev audio or dialogue leakage detected")
    return result


def _row_for_turn(
    source: bytes,
    member_name: str,
    source_hash: str,
    dialogue_id: str,
    split: str,
    turns: list[dict[str, Any]],
    turn_id: int,
    history: list[dict[str, str]],
    output_root: Path | None,
    data_json: Path,
    val_list: Path,
    audio_archive: Path,
    dataset: str,
    version: str,
) -> dict[str, Any]:
    turn = turns[turn_id]
    if turn_id + 1 >= len(turns):
        raise ValueError(f"{dialogue_id} turn {turn_id}: user turn has no adjacent assistant")
    response_turn = turns[turn_id + 1]
    response_tag = _text(response_turn.get("tag", response_turn.get("speaker"))).lower()
    if response_tag not in {"system", "assistant"}:
        raise ValueError(f"{dialogue_id} turn {turn_id}: assistant is not adjacent")
    response = _text(response_turn.get("text"))
    if not response:
        raise ValueError(f"{dialogue_id} turn {turn_id}: assistant response is empty")
    clip, channel, duration_ms = _clip_wav(source, turn.get("words", []), dialogue_id, turn_id)
    clip_name = f"{dialogue_id}_turn{turn_id:04d}.wav"
    clip_path = Path("audio") / split / clip_name
    if output_root is not None:
        destination = output_root / clip_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(clip)
    clip_hash = _sha256(clip)
    return {
        "sample_id": f"{split}:{dialogue_id}:{turn_id}",
        "dialogue_id": dialogue_id,
        "turn_id": str(turn_id),
        "split": split,
        "channel_id": channel,
        "audio_path": clip_path.as_posix(),
        "clip_sha256": clip_hash,
        "clip_duration_ms": duration_ms,
        "source_audio_sha256": source_hash,
        "user_text": _text(turn.get("text")),
        "assistant_response": response,
        "dialogue_history": list(history),
        "provenance": {
            "dataset": dataset,
            "version": version,
            "split": split,
            "data_json": str(data_json),
            "val_list": str(val_list),
            "audio_archive": str(audio_archive),
            "audio_member": member_name,
        },
    }


def _scan_archive(
    data_json: Path,
    val_list: Path,
    audio_archive: Path,
    output_root: Path | None,
    *,
    dataset: str,
    version: str,
) -> dict[str, Any]:
    dialogues = _load_dialogues(data_json)
    dev_ids = _load_dev_ids(val_list)
    unknown_dev = dev_ids - dialogues.keys()
    if unknown_dev:
        raise ValueError(f"validation split contains unknown dialogue IDs: {sorted(unknown_dev)[:5]}")
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    seen: set[str] = set()
    with tarfile.open(audio_archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            dialogue_id = Path(member.name).stem
            if dialogue_id in seen:
                raise ValueError(f"duplicate audio stem in archive: {dialogue_id}")
            seen.add(dialogue_id)
            if dialogue_id not in dialogues:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            source_bytes = source.read()
            source_hash = _sha256(source_bytes)
            dialogue = dialogues[dialogue_id]
            turns = dialogue.get("log") if isinstance(dialogue, dict) else None
            if not isinstance(turns, list):
                raise TypeError(f"dialogue {dialogue_id} has no list log")
            split = "dev" if dialogue_id in dev_ids else "train"
            history: list[dict[str, str]] = []
            for turn_id, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise TypeError(f"dialogue {dialogue_id} turn {turn_id} is not an object")
                tag = _text(turn.get("tag", turn.get("speaker"))).lower()
                if tag in {"user", "human"}:
                    rows[split].append(
                        _row_for_turn(
                            source_bytes,
                            member.name,
                            source_hash,
                            dialogue_id,
                            split,
                            turns,
                            turn_id,
                            history,
                            output_root,
                            data_json,
                            val_list,
                            audio_archive,
                            dataset,
                            version,
                        )
                    )
                current_text = _text(turn.get("text"))
                if tag in {"user", "human"}:
                    history.append({"role": "user", "text": current_text})
                elif tag in {"system", "assistant"}:
                    history.append({"role": "assistant", "text": current_text})
                else:
                    raise ValueError(f"{dialogue_id} turn {turn_id}: unsupported speaker tag {tag!r}")
    missing_audio = sorted(set(dialogues) - seen)
    extra_audio = sorted(seen - set(dialogues))
    if missing_audio or extra_audio:
        raise ValueError(f"audio/dialogue ID mismatch: missing={missing_audio[:5]}, extra={extra_audio[:5]}")
    audit = _audit(rows)
    for split_rows in rows.values():
        split_rows.sort(key=lambda row: (row["dialogue_id"], int(row["turn_id"])))
    return {"rows": rows, "audit": audit, "audio_members": len(seen)}


def preflight_spokenwoz_turns(
    data_json: Path,
    val_list: Path,
    audio_archive: Path,
    *,
    dataset: str = "SpokenWOZ",
    version: str = "official-main",
) -> dict[str, Any]:
    """Validate every archive WAV and user turn without creating output files."""
    result = _scan_archive(
        data_json.resolve(), val_list.resolve(), audio_archive.resolve(), None,
        dataset=dataset, version=version,
    )
    return {
        "audio_members": result["audio_members"],
        "leakage_audit": result["audit"],
        "splits": {
            split: {
                "samples": len(rows),
                "dialogues": len({row["dialogue_id"] for row in rows}),
            }
            for split, rows in result["rows"].items()
        },
    }


def convert_spokenwoz_turns(
    data_json: Path,
    val_list: Path,
    audio_archive: Path,
    output_root: Path,
    *,
    dataset: str = "SpokenWOZ",
    version: str = "official-main",
) -> dict[str, Any]:
    """Write one single-channel WAV clip and manifest row for every user turn."""
    data_json = data_json.resolve()
    val_list = val_list.resolve()
    audio_archive = audio_archive.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_root}")
    preflight = preflight_spokenwoz_turns(
        data_json, val_list, audio_archive, dataset=dataset, version=version
    )
    output_root.mkdir(parents=True)
    result = _scan_archive(
        data_json, val_list, audio_archive, output_root,
        dataset=dataset, version=version,
    )
    rows = result["rows"]
    audit = result["audit"]
    manifest_hashes: dict[str, str] = {}
    for split, split_rows in rows.items():
        split_rows.sort(key=lambda row: (row["dialogue_id"], int(row["turn_id"])))
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split_rows).encode()
        (output_root / f"{split}.jsonl").write_bytes(data)
        manifest_hashes[split] = _sha256(data)
    provenance = {
        "schema_version": 1,
        "dataset": dataset,
        "version": version,
        "data_json": str(data_json),
        "val_list": str(val_list),
        "audio_archive": str(audio_archive),
        "preflight": preflight,
        "splits": {
            split: {
                "samples": len(split_rows),
                "dialogues": len({row["dialogue_id"] for row in split_rows}),
                "manifest": f"{split}.jsonl",
                "manifest_sha256": manifest_hashes[split],
            }
            for split, split_rows in rows.items()
        },
        "leakage_audit": audit,
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert official SpokenWOZ user turns to WAV clips.")
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true", help="validate without writing output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight:
        result = preflight_spokenwoz_turns(args.data_json, args.val_list, args.audio_archive)
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required unless --preflight is used")
        result = convert_spokenwoz_turns(
            args.data_json, args.val_list, args.audio_archive, args.output_root
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
