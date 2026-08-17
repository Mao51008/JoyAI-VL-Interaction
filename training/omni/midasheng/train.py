"""Phase-one MiDasheng projector-only training; GPU execution requires ``--run``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.train import (
    CachedConversationBatchSource,
    Stage2Config,
    load_manifest,
    train_model,
    validate_manifests,
)

from .model import build_phase1_model


def preflight(
    *, train_manifest: Path, dev_manifest: Path, feature_dir: Path, output_dir: Path,
    create_output: bool = True,
) -> dict[str, Any]:
    """Validate the task-aware data/cache contract without loading any model."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    manifest_info = validate_manifests(train_manifest, dev_manifest)
    rows = [*load_manifest(train_manifest), *load_manifest(dev_manifest)]
    cache_metadata = validate_feature_cache(feature_dir, rows)
    if cache_metadata.get("feature_source") != "midashenglm.audio_encoder.raw":
        raise ValueError("feature cache is not raw MiDasheng audio_encoder output")
    if cache_metadata.get("audio_encoder_dim") != 1280:
        raise ValueError("MiDasheng feature cache must use 1280-dimensional encoder output")
    result = {
        "schema_version": 1,
        "status": "preflight-only",
        "training": {"trainable": ["audio_projector"], "lora": False},
        "audio_encoder": {"frozen": True, "feature_source": cache_metadata["feature_source"]},
        "manifests": manifest_info,
        "feature_cache": cache_metadata,
    }
    if create_output:
        output_dir.mkdir(parents=True)
        (output_dir / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def run_training(
    *, train_manifest: Path, dev_manifest: Path, feature_dir: Path, output_dir: Path,
    llm_model: str, batch_size: int, max_cached_feature_shards: int, steps: int,
    learning_rate: float, weight_decay: float, gradient_accumulation_steps: int,
    validation_every: int, warmup_steps: int, max_grad_norm: float, seed: int,
    device: str, no_progress: bool,
) -> dict[str, Any]:
    """Load JoyAI only after explicit authorization and train exactly the projector."""
    preflight_result = preflight(
        train_manifest=train_manifest, dev_manifest=dev_manifest,
        feature_dir=feature_dir, output_dir=output_dir, create_output=False,
    )
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
    model = build_phase1_model(
        AutoModelForImageTextToText.from_pretrained(llm_model, dtype=torch.bfloat16),
    ).to(device=device, dtype=torch.bfloat16)
    train_rows, dev_rows = load_manifest(train_manifest), load_manifest(dev_manifest)
    train_batches = CachedConversationBatchSource(
        train_rows, tokenizer, feature_dir, int(placeholder_id), batch_size,
        max_cached_feature_shards, shuffle=True, seed=seed,
    )
    dev_batches = CachedConversationBatchSource(
        dev_rows, tokenizer, feature_dir, int(placeholder_id), batch_size,
        max_cached_feature_shards,
    )
    config = Stage2Config(
        train_manifest=train_manifest, dev_manifest=dev_manifest, output_dir=output_dir,
        lora_rank=8, lora_alpha=16.0, projector_learning_rate=learning_rate,
        lora_learning_rate=learning_rate, weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm,
        shuffle_seed=seed, asr_replay_ratio=0.0, steps=steps,
        validation_every=validation_every, warmup_steps=warmup_steps,
        no_progress=no_progress,
    )
    result = train_model(model, train_batches, dev_batches, config)
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"preflight": preflight_result, "training": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-feature-shards", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.run:
        if not args.llm_model:
            parser.error("--run requires --llm-model")
        arguments = vars(args)
        arguments.pop("run")
        result = run_training(**arguments)
    else:
        result = preflight(
            train_manifest=args.train_manifest, dev_manifest=args.dev_manifest,
            feature_dir=args.feature_dir, output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
