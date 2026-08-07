"""Convert official SpokenWOZ user turns into single-channel audio clips."""

from __future__ import annotations

import hashlib
import json
import math
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


def _audio_members(archive: Path) -> dict[str, tuple[str, tarfile.TarInfo]]:
    index: dict[str, tuple[str, tarfile.TarInfo]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            stem = Path(member.name).stem
            if stem in index:
                raise ValueError(f"duplicate audio stem in archive: {stem}")
            index[stem] = (member.name, member)
    return index


def _clip_wav(source: bytes, words: list[dict[str, Any]], dialogue_id: str, turn_id: int) -> tuple[bytes, int, int]:
    if not words:
        raise ValueError(f"{dialogue_id} turn {turn_id}: words is empty")
    with wave.open(BytesIO(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        if channels < 1 or sample_width < 1 or sample_rate < 1:
            raise ValueError(f"{dialogue_id} turn {turn_id}: invalid WAV format")
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
        reader.rewind()
        frames = reader.readframes(frame_count)
        selected_channel = next(iter(channel_ids))
        block_width = channels * sample_width
        clip = b"".join(
            frame[selected_channel * sample_width : (selected_channel + 1) * sample_width]
            for frame in (
                frames[offset : offset + block_width]
                for offset in range(start_frame * block_width, end_frame * block_width, block_width)
            )
        )
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(clip)
    return output.getvalue(), selected_channel, round((end_frame - start_frame) * 1000 / sample_rate, 6)


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
    dialogues = _load_dialogues(data_json)
    dev_ids = _load_dev_ids(val_list)
    unknown_dev = dev_ids - dialogues.keys()
    if unknown_dev:
        raise ValueError(f"validation split contains unknown dialogue IDs: {sorted(unknown_dev)[:5]}")
    members = _audio_members(audio_archive)
    if set(dialogues) != set(members):
        missing_audio = sorted(set(dialogues) - set(members))
        extra_audio = sorted(set(members) - set(dialogues))
        raise ValueError(f"audio/dialogue ID mismatch: missing={missing_audio[:5]}, extra={extra_audio[:5]}")

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    output_root.mkdir(parents=True)
    archive = tarfile.open(audio_archive, "r:gz")  # noqa: SIM115
    for dialogue_id in sorted(dialogues):
        dialogue = dialogues[dialogue_id]
        turns = dialogue.get("log") if isinstance(dialogue, dict) else None
        if not isinstance(turns, list):
            raise TypeError(f"dialogue {dialogue_id} has no list log")
        split = "dev" if dialogue_id in dev_ids else "train"
        member_name, member = members[dialogue_id]
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member_name}")
        source_bytes = source.read()
        source_hash = _sha256(source_bytes)
        history: list[dict[str, str]] = []
        for turn_id, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise TypeError(f"dialogue {dialogue_id} turn {turn_id} is not an object")
            tag = _text(turn.get("tag", turn.get("speaker"))).lower()
            current_text = _text(turn.get("text"))
            if tag in {"user", "human"}:
                if turn_id + 1 >= len(turns):
                    raise ValueError(f"{dialogue_id} turn {turn_id}: user turn has no adjacent assistant")
                response_turn = turns[turn_id + 1]
                response_tag = _text(response_turn.get("tag", response_turn.get("speaker"))).lower()
                if response_tag not in {"system", "assistant"}:
                    raise ValueError(f"{dialogue_id} turn {turn_id}: assistant is not adjacent")
                response = _text(response_turn.get("text"))
                if not response:
                    raise ValueError(f"{dialogue_id} turn {turn_id}: assistant response is empty")
                clip, channel, duration_ms = _clip_wav(source_bytes, turn.get("words", []), dialogue_id, turn_id)
                clip_name = f"{dialogue_id}_turn{turn_id:04d}.wav"
                clip_path = output_root / "audio" / split / clip_name
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                clip_path.write_bytes(clip)
                clip_hash = _sha256(clip)
                row = {
                    "sample_id": f"{split}:{dialogue_id}:{turn_id}",
                    "dialogue_id": dialogue_id,
                    "turn_id": str(turn_id),
                    "split": split,
                    "channel_id": channel,
                    "audio_path": clip_path.relative_to(output_root).as_posix(),
                    "clip_sha256": clip_hash,
                    "clip_duration_ms": duration_ms,
                    "source_audio_sha256": source_hash,
                    "user_text": current_text,
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
                rows[split].append(row)
            if tag in {"user", "human"}:
                history.append({"role": "user", "text": current_text})
            elif tag in {"system", "assistant"}:
                history.append({"role": "assistant", "text": current_text})
            else:
                raise ValueError(f"{dialogue_id} turn {turn_id}: unsupported speaker tag {tag!r}")

    archive.close()

    audit = _audit(rows)
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
