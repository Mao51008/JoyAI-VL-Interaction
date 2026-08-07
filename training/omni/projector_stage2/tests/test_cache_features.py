import json
from pathlib import Path

import pytest

from training.omni.projector_stage2.cache_features import (
    _select_shard_rows,
    merge_feature_cache_parts,
    validate_feature_cache,
)


def _row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.wav",
        "clip_sha256": f"clip-{sample_id}",
        "source_audio_sha256": f"source-{sample_id}",
    }


def test_select_shard_rows_is_deterministic_and_disjoint():
    rows = [_row(f"sample-{index}") for index in range(9)]
    parts = [_select_shard_rows(rows, 4, index) for index in range(4)]
    assert [[row["sample_id"] for row in part] for part in parts] == [
        ["sample-0", "sample-4", "sample-8"],
        ["sample-1", "sample-5"],
        ["sample-2", "sample-6"],
        ["sample-3", "sample-7"],
    ]
    assert {row["sample_id"] for part in parts for row in part} == {
        row["sample_id"] for row in rows
    }
    with pytest.raises(ValueError, match="shard_index"):
        _select_shard_rows(rows, 4, 4)


def test_merge_cache_parts_validates_and_rewrites_shard_paths(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [_row("train:1"), _row("dev:1")]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    parts = []
    for index, row in enumerate(rows):
        part = tmp_path / "parts" / f"worker-{index}"
        part.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "audio_model": "/models/asr",
            "audio_model_revision": "local",
            "audio_model_config_sha256": "config",
            "target_sample_rate": 16000,
        }
        record = {
            **row,
            "shard": "shard-00000.pt",
            "offset": 0,
            "tokens": 2,
            "feature_dim": 2048,
            "audio_model_config_sha256": "config",
        }
        (part / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (part / "index.json").write_text(json.dumps([record]), encoding="utf-8")
        parts.append(part)
    merged = tmp_path / "merged"
    metadata = merge_feature_cache_parts(
        manifests=[manifest], part_dirs=parts, output_dir=merged, limit=None
    )
    assert metadata["samples"] == 2
    assert validate_feature_cache(merged, rows)["samples"] == 2
    index = json.loads((merged / "index.json").read_text(encoding="utf-8"))
    assert (merged / index[0]["shard"]).resolve() == (parts[0] / "shard-00000.pt").resolve()
