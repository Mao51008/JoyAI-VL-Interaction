"""Expose selected Stage1 LibriSpeech feature shards through the Stage2 cache contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .cache_features import CACHE_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    contents = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(contents)
    except json.JSONDecodeError:
        return [json.loads(line) for line in contents.splitlines() if line.strip()]
    if not isinstance(loaded, list):
        raise ValueError(f"cache index must be a JSON array or JSONL: {path}")
    return loaded


def reuse_cache(
    *,
    train_manifest: Path,
    dev_manifest: Path,
    stage1_train_cache: Path,
    stage1_dev_cache: Path,
    audio_model: Path,
    output_dir: Path,
    model_revision: str = "local",
) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    config = audio_model / "config.json"
    cache_by_split = {"train": stage1_train_cache, "dev": stage1_dev_cache}
    if not config.is_file() or any(not (cache / "index.json").is_file() for cache in cache_by_split.values()):
        raise FileNotFoundError("audio model config and Stage1 cache index are required")
    formal = [*_rows(train_manifest), *_rows(dev_manifest)]
    selected = [row for row in formal if row.get("training_task") == "asr_transcription"]
    if not selected:
        raise ValueError("no asr_transcription rows to reuse")
    source_indexes = {
        split: {str(row["sample_id"]): row for row in _rows(cache / "index.json")}
        for split, cache in cache_by_split.items()
    }
    migrated: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        cache = cache_by_split[str(row["split"])]
        cached = source_indexes[str(row["split"])].get(sample_id)
        if cached is None:
            raise ValueError(f"Stage1 cache lacks replay sample: {sample_id}")
        if cached.get("media_sha256") != row.get("clip_sha256"):
            raise ValueError(f"audio hash mismatch for replay sample: {sample_id}")
        source_path = cache / str(cached.get("shard", cached.get("path", "")))
        if not source_path.is_file():
            raise FileNotFoundError(f"Stage1 feature file is missing: {source_path}")
        migrated_row = {
            "sample_id": sample_id,
            "tokens": int(cached["tokens"]),
            "clip_sha256": row["clip_sha256"],
            "source_audio_sha256": row["source_audio_sha256"],
            "audio_model_config_sha256": _sha256(config),
        }
        if "shard" in cached:
            migrated_row.update(
                shard=os.path.relpath(source_path, output_dir), offset=int(cached["offset"])
            )
        else:
            migrated_row["path"] = os.path.relpath(source_path, output_dir)
        migrated.append(migrated_row)
    if len({row["sample_id"] for row in migrated}) != len(migrated):
        raise ValueError("duplicate ASR replay sample_id")
    output_dir.mkdir(parents=True)
    (output_dir / "index.json").write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "audio_model": str(audio_model.resolve()),
                "audio_model_revision": model_revision,
                "audio_model_config_sha256": _sha256(config),
                "source_stage1_train_cache": str(stage1_train_cache.resolve()),
                "source_stage1_dev_cache": str(stage1_dev_cache.resolve()),
                "total_samples": len(migrated),
                "samples": len(migrated),
                "shards": len({row.get("shard", row.get("path")) for row in migrated}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "reused_samples": len(migrated),
        "source_shards": len({row.get("shard", row.get("path")) for row in migrated}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--stage1-train-cache", type=Path, required=True)
    parser.add_argument("--stage1-dev-cache", type=Path, required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", default="local")
    args = parser.parse_args()
    print(json.dumps(reuse_cache(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
