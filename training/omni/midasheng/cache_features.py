"""Cache raw frozen MiDashengLM audio-encoder features for phase one."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.omni.projector_stage2.cache_features import (
    CACHE_SCHEMA_VERSION,
    _load_mono_audio,
    _load_rows,
    _select_shard_rows,
)


EXPECTED_MODEL_TYPE = "midashenglm"
EXPECTED_FEATURE_DIM = 1280
EXPECTED_SAMPLE_RATE = 16_000


def checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    """Validate that a local final MiDashengLM checkpoint exposes the expected encoder."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiDashengLM config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    encoder = config.get("audio_encoder_config", {})
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("audio model is not a MiDashengLM checkpoint")
    if int(encoder.get("embed_dim", 0)) != EXPECTED_FEATURE_DIM:
        raise ValueError(f"expected MiDasheng encoder dimension {EXPECTED_FEATURE_DIM}")
    if int(encoder.get("sample_rate", 0)) != EXPECTED_SAMPLE_RATE:
        raise ValueError(f"expected MiDasheng sample rate {EXPECTED_SAMPLE_RATE}")
    return {
        "model_type": config["model_type"],
        "audio_encoder_dim": int(encoder["embed_dim"]),
        "sample_rate": int(encoder["sample_rate"]),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def _load_encoder(model_dir: Path, device: str) -> Any:
    import gc
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    encoder = model.audio_encoder
    del model
    gc.collect()
    return encoder.to(device=device, dtype=torch.bfloat16).eval()


def cache_features(
    *, manifests: list[Path], model_dir: Path, output_dir: Path, device: str, shard_size: int,
    model_revision: str, limit: int | None = None, num_shards: int = 1, shard_index: int = 0,
) -> dict[str, Any]:
    """Cache only raw 1280-dim encoder outputs; never use MiDasheng's projector."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature cache directory: {output_dir}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    identity = checkpoint_identity(model_dir)
    all_rows, manifest_fingerprints = _load_rows(manifests, limit)
    rows = _select_shard_rows(all_rows, num_shards, shard_index)
    part_index = shard_index

    import torch

    encoder = _load_encoder(model_dir, device)
    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    shard_samples: list[dict[str, Any]] = []
    file_shard_index = 0

    def flush() -> None:
        nonlocal file_shard_index, shard_samples
        if shard_samples:
            name = f"shard-{file_shard_index:05d}.pt"
            torch.save({"samples": shard_samples}, output_dir / name)
            shard_samples = []
            file_shard_index += 1

    with torch.inference_mode():
        for row in rows:
            clip_path = Path(row["_manifest_root"]) / row["audio_path"]
            if not clip_path.is_file():
                raise FileNotFoundError(f"manifest audio clip does not exist: {clip_path}")
            waveform = _load_mono_audio(clip_path, EXPECTED_SAMPLE_RATE)
            values = torch.from_numpy(waveform).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            lengths = torch.tensor([values.shape[-1]], device=device)
            encoded, feature_mask = encoder(values, x_length=lengths)
            valid = encoded[0, feature_mask[0].bool()].cpu()
            if valid.ndim != 2 or not valid.shape[0] or valid.shape[1] != EXPECTED_FEATURE_DIM:
                raise ValueError(f"{row['sample_id']}: invalid MiDasheng feature shape {tuple(valid.shape)}")
            name = f"shard-{file_shard_index:05d}.pt"
            shard_samples.append({"sample_id": row["sample_id"], "features": valid})
            records.append({"sample_id": row["sample_id"], "shard": name, "offset": len(shard_samples) - 1,
                            "tokens": int(valid.shape[0]), "feature_dim": int(valid.shape[1]),
                            "clip_sha256": row["clip_sha256"],
                            "source_audio_sha256": row["source_audio_sha256"],
                            "audio_model_config_sha256": identity["config_sha256"]})
            if len(shard_samples) == shard_size:
                flush()
    flush()
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "feature_source": "midashenglm.audio_encoder.raw",
        "audio_model": str(model_dir),
        "audio_model_revision": model_revision,
        "manifests": manifest_fingerprints,
        "total_samples": len(all_rows),
        "samples": len(records),
        "shards": file_shard_index,
        "num_shards": num_shards,
        "shard_index": part_index,
        **identity,
    }
    (output_dir / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def merge_feature_cache_parts(
    *, manifests: list[Path], part_dirs: list[Path], output_dir: Path, limit: int | None = None
) -> dict[str, Any]:
    """Merge train/dev indices without copying any cached feature shard."""
    from training.omni.projector_stage2.cache_features import merge_feature_cache_parts as merge_base

    part_metadata = [json.loads((directory / "metadata.json").read_text(encoding="utf-8")) for directory in part_dirs]
    if not part_metadata:
        raise ValueError("at least one cache part is required")
    keys = ("feature_source", "audio_model", "audio_model_revision", "audio_encoder_dim", "sample_rate", "config_sha256")
    identity = {key: part_metadata[0].get(key) for key in keys}
    if any({key: metadata.get(key) for key in keys} != identity for metadata in part_metadata[1:]):
        raise ValueError("cache parts do not use the same MiDasheng encoder identity")
    metadata = merge_base(manifests=manifests, part_dirs=part_dirs, output_dir=output_dir, limit=limit)
    metadata.update(identity)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--model-revision", default="76c3019")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge-part", type=Path, action="append")
    args = parser.parse_args()
    arguments = vars(args)
    arguments["manifests"] = arguments.pop("manifest")
    merge_parts = arguments.pop("merge_part")
    if merge_parts:
        if arguments["model_dir"] is not None:
            parser.error("--model-dir cannot be used with --merge-part")
        result = merge_feature_cache_parts(
            manifests=arguments["manifests"], part_dirs=merge_parts,
            output_dir=arguments["output_dir"], limit=arguments["limit"],
        )
    else:
        if arguments["model_dir"] is None:
            parser.error("--model-dir is required unless --merge-part is used")
        result = cache_features(**arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
