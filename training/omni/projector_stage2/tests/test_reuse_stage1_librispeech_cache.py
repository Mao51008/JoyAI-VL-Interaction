import json
from pathlib import Path

from training.omni.projector_stage2.reuse_stage1_librispeech_cache import reuse_cache


def _manifest_row(sample_id: str, split: str, sha: str) -> dict:
    return {
        "sample_id": sample_id, "split": split, "training_task": "asr_transcription",
        "clip_sha256": sha, "source_audio_sha256": sha,
    }


def test_reuse_stage1_cache_creates_relative_shard_index(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    caches = []
    for split, sample, sha in (("train", "libri-train", "train-sha"), ("dev", "libri-dev", "dev-sha")):
        cache = tmp_path / f"{split}-cache"
        cache.mkdir()
        (cache / "shard-00000.pt").write_bytes(b"fixture")
        (cache / "index.json").write_text(
            json.dumps([{"sample_id": sample, "shard": "shard-00000.pt", "offset": 0, "tokens": 2, "media_sha256": sha}]),
            encoding="utf-8",
        )
        caches.append(cache)
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps(_manifest_row("libri-train", "train", "train-sha")) + "\n")
    dev.write_text(json.dumps(_manifest_row("libri-dev", "dev", "dev-sha")) + "\n")
    result = reuse_cache(
        train_manifest=train, dev_manifest=dev, stage1_train_cache=caches[0],
        stage1_dev_cache=caches[1], audio_model=model, output_dir=tmp_path / "out",
    )
    index = json.loads((tmp_path / "out" / "index.json").read_text())
    assert result == {"reused_samples": 2, "source_shards": 2}
    assert index[0]["shard"].startswith("..")
