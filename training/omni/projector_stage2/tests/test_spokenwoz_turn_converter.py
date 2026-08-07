import hashlib
import io
import json
import struct
import tarfile
import wave
from pathlib import Path

import pytest

from training.omni.projector_stage2.spokenwoz_turn_converter import (
    convert_spokenwoz_turns,
    preflight_spokenwoz_turns,
)


def _wav_bytes(seed: int, frame_count: int = 10) -> bytes:
    payload = b"".join(
        struct.pack(
            "<hh",
            (seed + frame) % 30000,
            1000 + (((seed >> (8 * max(frame - 2, 0))) & 255) + frame if frame < 6 else frame),
        )
        for frame in range(frame_count)
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(1000)
        writer.writeframes(payload)
    return output.getvalue() + seed.to_bytes(4, "little")


def _dialogue(dialogue_id: str) -> dict:
    return {
        "goal": {},
        "log": [
            {
                "tag": "user",
                "text": "first request",
                "words": [
                    {"Word": "first", "BeginTime": 2, "EndTime": 4, "ChannelId": 1},
                    {"Word": "request", "BeginTime": 4, "EndTime": 6, "ChannelId": 1},
                ],
            },
            {"tag": "system", "text": "first answer", "words": []},
            {
                "tag": "user",
                "text": "follow up",
                "words": [{"Word": "follow", "BeginTime": 7, "EndTime": 9, "ChannelId": 0}],
            },
            {"tag": "system", "text": "second answer", "words": []},
        ],
    }


def _split_dialogue() -> dict:
    return {
        "goal": {},
        "log": [
            {
                "tag": "user",
                "text": "request",
                "words": [{"Word": "request", "BeginTime": 2, "EndTime": 6, "ChannelId": 1}],
            },
            {"tag": "system", "text": "answer", "words": []},
        ],
    }


def _fixture(
    tmp_path: Path,
    dialogues: dict[str, dict] | None = None,
    archive_order: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    dialogues = dialogues or {"MUL0001": _dialogue("MUL0001"), "SNG0002": _dialogue("SNG0002")}
    data_json = tmp_path / "data.json"
    val_list = tmp_path / "valListFile.json"
    archive_path = tmp_path / "audio_5700_train_dev.tar.gz"
    data_json.write_text(json.dumps(dialogues), encoding="utf-8")
    val_list.write_text("SNG0002\n", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        for index, dialogue_id in enumerate(archive_order or list(dialogues)):
            source = tmp_path / f"{dialogue_id}.wav"
            source.write_bytes(_wav_bytes(index * 10))
            archive.add(source, arcname=f"audio_5700_train_dev/{dialogue_id}.wav")
    return data_json, val_list, archive_path


def test_converts_channel_slice_history_hashes_and_split(tmp_path: Path):
    data_json, val_list, archive = _fixture(tmp_path)
    provenance = convert_spokenwoz_turns(data_json, val_list, archive, tmp_path / "out")

    assert provenance["splits"]["train"]["dialogues"] == 1
    assert provenance["splits"]["dev"]["dialogues"] == 1
    train_rows = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    first = train_rows[0]
    assert first["channel_id"] == 1
    assert first["clip_duration_ms"] == 4.0
    assert first["assistant_response"] == "first answer"
    assert first["dialogue_history"] == []
    second = train_rows[1]
    assert second["dialogue_history"] == [
        {"role": "user", "text": "first request"},
        {"role": "assistant", "text": "first answer"},
    ]
    clip = (tmp_path / "out" / first["audio_path"]).read_bytes()
    with wave.open(io.BytesIO(clip), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getframerate() == 1000
        assert reader.readframes(reader.getnframes()) == struct.pack("<hhhh", 1002, 1003, 1004, 1005)
    assert first["clip_sha256"] == hashlib.sha256(clip).hexdigest()
    assert provenance["leakage_audit"]["leakage"] is False


def test_preflight_is_read_only_and_乱序_archive_is_supported(tmp_path: Path):
    dialogues = {"MUL0001": _dialogue("MUL0001"), "SNG0002": _dialogue("SNG0002")}
    data_json, val_list, archive = _fixture(
        tmp_path, dialogues, archive_order=["SNG0002", "MUL0001"]
    )
    output = tmp_path / "out"
    preflight = preflight_spokenwoz_turns(data_json, val_list, archive)
    assert preflight["audio_members"] == 2
    assert preflight["splits"]["train"]["dialogues"] == 1
    assert not output.exists()
    convert_spokenwoz_turns(data_json, val_list, archive, output)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert {row["dialogue_id"] for row in rows} == {"MUL0001"}


def test_realistic_4200_500_dialogue_split(tmp_path: Path):
    dialogues = {f"MUL{i:04d}": _split_dialogue() for i in range(4200)}
    dialogues.update({f"SNG{i:04d}": _split_dialogue() for i in range(500)})
    data_json, val_list, archive = _fixture(tmp_path, dialogues)
    val_list.write_text("".join(f"SNG{i:04d}\n" for i in range(500)), encoding="utf-8")
    provenance = convert_spokenwoz_turns(data_json, val_list, archive, tmp_path / "out")
    assert provenance["splits"]["train"]["dialogues"] == 4200
    assert provenance["splits"]["dev"]["dialogues"] == 500


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda turns: turns[0].update(words=[]), "words is empty"),
        (
            lambda turns: turns[0].update(
                words=[{"BeginTime": 1, "EndTime": 2, "ChannelId": 0}, {"BeginTime": 2, "EndTime": 3, "ChannelId": 1}]
            ),
            "cross audio channels",
        ),
        (lambda turns: turns[1].update(tag="user"), "assistant is not adjacent"),
        (lambda turns: turns[0].update(words=[{"BeginTime": 9, "EndTime": 11, "ChannelId": 0}]), "exceeds WAV duration"),
    ],
)
def test_rejects_invalid_turns(tmp_path: Path, change, message):
    data = {"MUL0001": _dialogue("MUL0001")}
    change(data["MUL0001"]["log"])
    data_json, val_list, archive = _fixture(tmp_path, data)
    val_list.write_text("MUL0001\n", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        convert_spokenwoz_turns(data_json, val_list, archive, output)
    assert not output.exists()
