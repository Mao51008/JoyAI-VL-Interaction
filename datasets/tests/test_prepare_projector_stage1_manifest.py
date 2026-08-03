import json
import wave
from pathlib import Path

from datasets.prepare_projector_stage1_manifest import convert


def _write_wav(path: Path, samples: int = 1600) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * samples)


def _provenance(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "version": "1",
                "source_uri": "https://example.invalid/fixture",
                "license_name": "CC0",
                "license_tier": "redistributable",
                "allows_training": True,
                "allows_modification": True,
                "allows_redistribution": True,
                "allows_commercial_use": True,
            }
        ),
        encoding="utf-8",
    )


def test_convert_valid_audio_text(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    metadata = tmp_path / "metadata.jsonl"
    provenance = tmp_path / "provenance.json"
    output = tmp_path / "samples.jsonl"
    _write_wav(audio)
    metadata.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "text": "hello",
                "duration_ms": 100,
                "sample_rate": 16000,
                "num_samples": 1600,
                "source_record": "fixture-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _provenance(provenance)

    summary = convert(metadata, output, provenance)

    assert summary["converted_samples"] == 1
    assert summary["skipped"] == {}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_convert_skips_missing_audio(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    provenance = tmp_path / "provenance.json"
    output = tmp_path / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "audio_path": str(tmp_path / "missing.wav"),
                "text": "hello",
                "duration_ms": 100,
                "sample_rate": 16000,
                "num_samples": 1600,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _provenance(provenance)

    summary = convert(metadata, output, provenance)

    assert summary["converted_samples"] == 0
    assert summary["skipped"] == {"audio_missing": 1}
