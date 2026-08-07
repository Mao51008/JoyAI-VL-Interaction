import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from training.omni.projector_stage2 import spokenwoz_manifest
from training.omni.projector_stage2.spokenwoz_manifest import (
    audit_manifests,
    build_parser,
    prepare_official_manifests,
    prepare_manifests,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "audio"
    text = tmp_path / "text"
    for split, dialogue, speaker in (
        ("train", "MUL0001", "spk-train"),
        ("dev", "MUL0002", "spk-dev"),
    ):
        (audio / split).mkdir(parents=True)
        (text / split).mkdir(parents=True)
        (audio / split / f"{dialogue}.wav").write_bytes(f"audio-{split}".encode())
        row = {
            "dialogue_id": dialogue,
            "turn_id": 0,
            "speaker_id": speaker,
            "response": f"response {split}",
        }
        (text / split / "dialogs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return audio, text


def test_cli_help_and_deterministic_manifests(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--help"])
    assert error.value.code == 0
    audio, text = _fixture(tmp_path)
    first = prepare_manifests(audio, text, tmp_path / "out1")
    second = prepare_manifests(audio, text, tmp_path / "out2")
    assert first["splits"] == second["splits"]
    assert (tmp_path / "out1" / "train.jsonl").read_bytes() == (
        tmp_path / "out2" / "train.jsonl"
    ).read_bytes()
    row = json.loads((tmp_path / "out1" / "train.jsonl").read_text(encoding="utf-8"))
    expected = hashlib.sha256(b"audio-train").hexdigest()
    assert row["audio_sha256"] == expected
    assert first["leakage_audit"]["leakage"] is False


def test_missing_split_and_existing_output_are_rejected(tmp_path):
    audio, text = _fixture(tmp_path)
    incomplete_audio = tmp_path / "incomplete-audio"
    (incomplete_audio / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="audio/dev"):
        prepare_manifests(incomplete_audio, text, tmp_path / "out")
    audio, text = _fixture(tmp_path / "again")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_manifests(audio, text, output)


def test_overlap_audit_reports_all_supported_layers():
    common = {
        "sample_id": "same-sample",
        "dialogue_id": "same-dialogue",
        "speaker_id": "same-speaker",
        "audio_path": "same.wav",
        "text_audio_sha256": "same-hash",
        "provenance": {"dataset": "SpokenWOZ", "version": "v1", "source_file": "same.json"},
    }
    with pytest.raises(ValueError, match="sample_id.*dialogue_id"):
        audit_manifests([common], [common])
    clean_dev = {
        **common,
        "sample_id": "dev-sample",
        "dialogue_id": "dev-dialogue",
        "speaker_id": None,
        "audio_path": "dev.wav",
        "text_audio_sha256": "dev-hash",
        "provenance": {"dataset": "SpokenWOZ", "version": "v1", "source_file": "dev.json"},
    }
    assert audit_manifests([common], [clean_dev])["leakage"] is False


def _official_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = {
        "MUL0001": {
            "goal": {},
            "log": [
                {"tag": "user", "text": "book a train", "words": []},
                {"tag": "system", "text": "sure", "words": []},
            ],
        },
        "MUL0002": {
            "goal": {},
            "log": [
                {"tag": "user", "text": "find a hotel", "words": []},
                {"tag": "assistant", "text": "okay", "words": []},
            ],
        },
    }
    data_json = tmp_path / "data.json"
    data_json.write_text(json.dumps(data), encoding="utf-8")
    dev_list = tmp_path / "valListFile.json"
    dev_list.write_text("MUL0002\n", encoding="utf-8")
    archive = tmp_path / "audio_5700_train_dev.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for dialogue in data:
            source = tmp_path / f"{dialogue}.wav"
            source.write_bytes(dialogue.encode())
            handle.add(source, arcname=f"audio_5700_train_dev/{dialogue}.wav")
    return data_json, dev_list, archive


def test_official_format_delegates_to_turn_converter(tmp_path, monkeypatch):
    data_json, dev_list, archive = _official_fixture(tmp_path)
    expected = {"splits": {"train": {"samples": 1}}, "leakage_audit": {"leakage": False}}
    captured = {}

    def fake_converter(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(spokenwoz_manifest, "convert_spokenwoz_turns", fake_converter)
    provenance = prepare_official_manifests(
        data_json, dev_list, archive, tmp_path / "official-out"
    )
    assert provenance is expected
    assert captured["args"] == (data_json, dev_list, archive, tmp_path / "official-out")
    assert captured["kwargs"] == {"dataset": "SpokenWOZ", "version": "official-main"}

