import json
from pathlib import Path

from training.omni.projector_stage2 import prepare_audio_pilot_manifests as manifests


def _write_jsonl(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_prepare_manifests_preserves_task_roles_and_source_provenance(tmp_path, monkeypatch):
    clotho = tmp_path / "clotho.jsonl"
    voice = tmp_path / "voice.jsonl"
    asr_train = tmp_path / "asr_train.jsonl"
    asr_dev = tmp_path / "asr_dev.jsonl"
    (tmp_path / "birds.wav").write_bytes(b"fixture")
    (tmp_path / "user.wav").write_bytes(b"fixture")
    _write_jsonl(
        clotho,
        {
            "sample_id": "clotho:1", "split": "train", "task": "environment_audio_qa",
            "source_audio_path": "birds.wav", "question": "Are birds audible?",
            "assistant_response": "yes", "provenance": {"dataset": "Clotho-AQA"},
        },
    )
    _write_jsonl(
        voice,
        {
            "sample_id": "voice:1", "split": "dev", "task": "audio_dialogue_response",
            "source_audio_path": "user.wav", "assistant_response": "Sure.",
            "provenance": {"dataset": "VoiceAssistant"},
        },
    )
    source = {
        "sample_id": "libri-1", "audio": [{"path": str(tmp_path / "audio.flac")}],
        "metadata": {"assistant_target_text": "HELLO"}, "provenance": {"dataset": "LibriSpeech"},
    }
    _write_jsonl(asr_train, source)
    _write_jsonl(asr_dev, {**source, "sample_id": "libri-2"})

    def fake_formal_row(source, **kwargs):
        return {
            "sample_id": source["sample_id"], "dialogue_id": source["sample_id"],
            "turn_id": "0", "split": kwargs["split"], "audio_path": str(kwargs["audio_path"]),
            "clip_duration_ms": 1.0, "clip_sha256": source["sample_id"],
            "source_audio_sha256": source["sample_id"], "user_text": kwargs["user_text"],
            "assistant_response": source["assistant_response"],
            "dialogue_history": kwargs["dialogue_history"],
            "training_task": kwargs["training_task"],
            "provenance": {**source.get("provenance", {}), "source_manifest": str(kwargs["source_manifest"])},
        }

    monkeypatch.setattr(manifests, "_formal_row", fake_formal_row)
    result = manifests.prepare_manifests(
        clotho_manifest=clotho, voice_manifest=voice, asr_train_manifest=asr_train,
        asr_dev_manifest=asr_dev, clotho_audio_root=tmp_path, voice_audio_root=tmp_path,
        output_dir=tmp_path / "out", dev_limit=2, no_progress=True,
    )
    assert result == {"train": 2, "dev": 2}
    train = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    dev = [json.loads(line) for line in (tmp_path / "out" / "dev.jsonl").read_text().splitlines()]
    assert train[0]["dialogue_history"] == [{"role": "user", "text": "Are birds audible?"}]
    assert train[1]["training_task"] == "asr_transcription"
    assert dev[0]["provenance"]["source_manifest"] == str(voice)
    assert dev[1]["user_text"] == "HELLO"


def test_clotho_audio_paths_resolves_unique_punctuation_variant(tmp_path):
    expected = tmp_path / "souffle_me.tallique.wav"
    expected.write_bytes(b"fixture")
    paths = manifests._clotho_audio_paths(tmp_path)
    assert paths[manifests._normalized_audio_name("souffle_me_tallique.wav")] == expected
