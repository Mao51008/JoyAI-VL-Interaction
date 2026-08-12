"""Build a fingerprinted frozen-Qwen3-ASR cache for formal SpokenWOZ manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    required = {"sample_id", "audio_path", "clip_sha256", "source_audio_sha256"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"manifest row {index} lacks cache fields: {sorted(missing)}"
            )
    return rows


def _load_rows(
    manifests: list[Path],
    limit: int | None,
    training_task: str | None = None,
    provenance_dataset: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for manifest in manifests:
        resolved = manifest.resolve()
        for row in _load_manifest(resolved):
            if training_task is not None and row.get("training_task") != training_task:
                continue
            if provenance_dataset is not None:
                provenance = row.get("provenance", {})
                if not isinstance(provenance, dict) or provenance.get("dataset") != provenance_dataset:
                    continue
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id across manifests: {sample_id}")
            seen_ids.add(sample_id)
            rows.append(dict(row, _manifest_root=str(resolved.parent)))
        fingerprints.append({"path": str(resolved), "sha256": _sha256_file(resolved)})
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    return rows, fingerprints


def _validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def _select_shard_rows(
    rows: list[dict[str, Any]], num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    _validate_shard_args(num_shards, shard_index)
    return rows[shard_index::num_shards]


def _audio_model_config_hash(audio_model: Path) -> str:
    config = audio_model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"audio model config does not exist: {config}")
    return _sha256_file(config)


def validate_feature_cache(
    directory: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify cached features match the exact clips selected by a manifest subset."""
    metadata_path = directory / "metadata.json"
    index_path = directory / "index.json"
    if not metadata_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("feature cache requires metadata.json and index.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported feature cache schema")
    index = {
        row["sample_id"]: row
        for row in json.loads(index_path.read_text(encoding="utf-8"))
    }
    missing = []
    mismatched = []
    for row in rows:
        cached = index.get(str(row["sample_id"]))
        if cached is None:
            missing.append(str(row["sample_id"]))
            continue
        if (
            cached.get("clip_sha256") != row["clip_sha256"]
            or cached.get("source_audio_sha256") != row["source_audio_sha256"]
        ):
            mismatched.append(str(row["sample_id"]))
    if missing or mismatched:
        raise ValueError(
            f"feature cache manifest mismatch: missing={missing[:5]}, mismatched={mismatched[:5]}"
        )
    return metadata


def _load_mono_audio(path: Path, target_sample_rate: int) -> Any:
    try:
        import soundfile as sf
        import soxr
    except ImportError as exc:
        raise RuntimeError(
            "soundfile and soxr are required for feature caching"
        ) from exc
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not waveform.size:
        raise ValueError(f"audio clip is empty: {path}")
    source_rate = int(source_rate)
    if source_rate <= 0:
        raise ValueError(f"invalid sample rate {source_rate}: {path}")
    if source_rate != target_sample_rate:
        waveform = soxr.resample(waveform, source_rate, target_sample_rate)
    return waveform


def cache_features(
    *,
    manifests: list[Path],
    audio_model: Path,
    output_dir: Path,
    device: str,
    shard_size: int,
    limit: int | None,
    model_revision: str,
    num_shards: int = 1,
    shard_index: int = 0,
    progress_file: Path | None = None,
    progress_every: int = 100,
    no_progress: bool = False,
    training_task: str | None = None,
    provenance_dataset: str | None = None,
) -> dict[str, Any]:
    """Extract frozen ASR features once; refuse to reuse an output directory."""
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse feature cache directory: {output_dir}"
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    all_rows, manifest_fingerprints = _load_rows(
        manifests, limit, training_task, provenance_dataset
    )
    rows = _select_shard_rows(all_rows, num_shards, shard_index)
    cache_part_index = shard_index

    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoProcessor

    output_dir.mkdir(parents=True)

    def write_progress(completed: int, status: str) -> None:
        if progress_file is None:
            return
        progress_file.write_text(
            json.dumps(
                {
                    "completed": completed,
                    "total": len(rows),
                    "shard_index": cache_part_index,
                    "status": status,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    write_progress(0, "loading")
    wrapper = Qwen3ASRModel.from_pretrained(
        str(audio_model),
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=1,
    )
    model = wrapper.model.eval()
    processor = AutoProcessor.from_pretrained(str(audio_model), fix_mistral_regex=True)
    target_sample_rate = int(
        getattr(processor.feature_extractor, "sampling_rate", 16000)
    )
    config_hash = _audio_model_config_hash(audio_model)
    records: list[dict[str, Any]] = []
    shard_samples: list[dict[str, Any]] = []
    file_shard_index = 0

    def flush() -> None:
        nonlocal file_shard_index, shard_samples
        if not shard_samples:
            return
        name = f"shard-{file_shard_index:05d}.pt"
        torch.save({"samples": shard_samples}, output_dir / name)
        file_shard_index += 1
        shard_samples = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    progress = (
        tqdm(rows, total=len(rows), desc="cache audio features", unit="sample")
        if tqdm is not None and not no_progress
        else rows
    )
    for completed, row in enumerate(progress, start=1):
        clip_path = Path(row["_manifest_root"]) / row["audio_path"]
        if not clip_path.is_file():
            raise FileNotFoundError(f"manifest audio clip does not exist: {clip_path}")
        waveform = _load_mono_audio(clip_path, target_sample_rate)
        batch = (
            processor(
                text=processor.audio_token,
                audio=waveform,
                sampling_rate=target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            .to(model.device)
            .to(model.dtype)
        )
        with torch.inference_mode():
            features = model.thinker.get_audio_features(
                input_features=batch["input_features"],
                feature_attention_mask=batch["feature_attention_mask"],
            ).cpu()
        if features.ndim != 2 or not features.shape[0]:
            raise ValueError(
                f"invalid cached feature shape {tuple(features.shape)} for {row['sample_id']}"
            )
        name = f"shard-{file_shard_index:05d}.pt"
        shard_samples.append({"sample_id": row["sample_id"], "features": features})
        records.append(
            {
                "sample_id": row["sample_id"],
                "shard": name,
                "offset": len(shard_samples) - 1,
                "tokens": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "clip_sha256": row["clip_sha256"],
                "source_audio_sha256": row["source_audio_sha256"],
                "audio_model_config_sha256": config_hash,
            }
        )
        if len(shard_samples) == shard_size:
            flush()
        if completed % progress_every == 0 or completed == len(rows):
            write_progress(completed, "running")
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(samples=completed, shards=shard_index + 1)
    flush()
    write_progress(len(rows), "complete")
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "audio_model": str(audio_model.resolve()),
        "audio_model_revision": model_revision,
        "audio_model_config_sha256": config_hash,
        "target_sample_rate": target_sample_rate,
        "manifests": manifest_fingerprints,
        "total_samples": len(all_rows),
        "samples": len(records),
        "shards": file_shard_index,
        "num_shards": num_shards,
        "shard_index": cache_part_index,
    }
    (output_dir / "index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def merge_feature_cache_parts(
    *,
    manifests: list[Path],
    part_dirs: list[Path],
    output_dir: Path,
    limit: int | None,
    allow_unselected_cache_records: bool = False,
) -> dict[str, Any]:
    """Merge deterministic cache parts without copying their feature shard files."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature cache directory: {output_dir}")
    if not part_dirs:
        raise ValueError("at least one cache part is required")
    expected_rows, manifest_fingerprints = _load_rows(manifests, limit)
    expected = {str(row["sample_id"]): row for row in expected_rows}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    reference: dict[str, Any] | None = None
    for part_dir in part_dirs:
        metadata_path, index_path = part_dir / "metadata.json", part_dir / "index.json"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"cache part is incomplete: {part_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema in {part_dir}")
        identity = {
            key: metadata.get(key)
            for key in ("audio_model", "audio_model_revision", "audio_model_config_sha256")
        }
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("cache parts use different audio model metadata")
        for record in json.loads(index_path.read_text(encoding="utf-8")):
            sample_id = str(record["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across cache parts: {sample_id}")
            source = expected.get(sample_id)
            if source is None:
                if allow_unselected_cache_records:
                    continue
                raise ValueError(f"cache part contains unknown sample_id: {sample_id}")
            if (
                record.get("clip_sha256") != source["clip_sha256"]
                or record.get("source_audio_sha256") != source["source_audio_sha256"]
            ):
                raise ValueError(f"cache part fingerprint mismatch: {sample_id}")
            seen.add(sample_id)
            if "shard" in record:
                merged.append(
                    {
                        **record,
                        "shard": os.path.relpath(part_dir / record["shard"], output_dir),
                    }
                )
            elif "path" in record:
                merged.append(
                    {
                        **record,
                        "path": os.path.relpath(part_dir / record["path"], output_dir),
                    }
                )
            else:
                raise ValueError(f"cache record lacks shard/path: {sample_id}")
    missing = [sample_id for sample_id in expected if sample_id not in seen]
    if missing:
        raise ValueError(f"cache parts are incomplete: missing={missing[:5]}")
    position = {str(row["sample_id"]): index for index, row in enumerate(expected_rows)}
    merged.sort(key=lambda row: position[str(row["sample_id"])])
    output_dir.mkdir(parents=True)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **(reference or {}),
        "manifests": manifest_fingerprints,
        "total_samples": len(expected_rows),
        "samples": len(merged),
        "parts": [str(directory.resolve()) for directory in part_dirs],
        "shards": len({record.get("shard", record.get("path")) for record in merged}),
    }
    (output_dir / "index.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--audio-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-part", type=Path, action="append")
    parser.add_argument("--allow-unselected-cache-records", action="store_true")
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--training-task")
    parser.add_argument("--provenance-dataset")
    args = parser.parse_args()
    if args.merge_part:
        result = merge_feature_cache_parts(
            manifests=args.manifest,
            part_dirs=args.merge_part,
            output_dir=args.output_dir,
            limit=args.limit,
            allow_unselected_cache_records=args.allow_unselected_cache_records,
        )
    else:
        if args.audio_model is None:
            parser.error("--audio-model is required unless --merge-part is used")
        result = cache_features(
            manifests=args.manifest,
            audio_model=args.audio_model,
            output_dir=args.output_dir,
            device=args.device,
            shard_size=args.shard_size,
            limit=args.limit,
            model_revision=args.model_revision,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            progress_file=args.progress_file,
            progress_every=args.progress_every,
            no_progress=args.no_progress,
            training_task=args.training_task,
            provenance_dataset=args.provenance_dataset,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
