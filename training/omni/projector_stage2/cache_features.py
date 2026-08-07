"""Build a fingerprinted frozen-Qwen3-ASR cache for formal SpokenWOZ manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
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
) -> dict[str, Any]:
    """Extract frozen ASR features once; refuse to reuse an output directory."""
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse feature cache directory: {output_dir}"
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    rows: list[dict[str, Any]] = []
    manifest_fingerprints: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for manifest in manifests:
        resolved = manifest.resolve()
        manifest_rows = _load_manifest(resolved)
        for row in manifest_rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id across manifests: {sample_id}")
            seen_ids.add(sample_id)
            rows.append(dict(row, _manifest_root=str(resolved.parent)))
        manifest_fingerprints.append(
            {"path": str(resolved), "sha256": _sha256_file(resolved)}
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]

    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoProcessor

    output_dir.mkdir(parents=True)
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
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, shard_samples
        if not shard_samples:
            return
        name = f"shard-{shard_index:05d}.pt"
        torch.save({"samples": shard_samples}, output_dir / name)
        shard_index += 1
        shard_samples = []

    for row in rows:
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
        name = f"shard-{shard_index:05d}.pt"
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
    flush()
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "audio_model": str(audio_model.resolve()),
        "audio_model_revision": model_revision,
        "audio_model_config_sha256": config_hash,
        "target_sample_rate": target_sample_rate,
        "manifests": manifest_fingerprints,
        "samples": len(records),
        "shards": shard_index,
    }
    (output_dir / "index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-revision", default="local")
    args = parser.parse_args()
    result = cache_features(
        manifests=args.manifest,
        audio_model=args.audio_model,
        output_dir=args.output_dir,
        device=args.device,
        shard_size=args.shard_size,
        limit=args.limit,
        model_revision=args.model_revision,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
