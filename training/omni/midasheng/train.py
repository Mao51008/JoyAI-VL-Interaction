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
    _build_supervised_sequence,
    load_manifest,
    train_model,
    validate_manifests,
    WeightedAudioBatchSource,
)

from .model import build_phase1_model
from .official_projector import build_frozen_projector_adapter, subsampled_token_count


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


def filter_rows_by_input_tokens(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    feature_dir: Path,
    audio_placeholder_id: int,
    max_input_tokens: int,
    max_audio_tokens: int,
    max_cached_feature_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discard rows that exceed either the total or audio-token hard limit."""
    if min(max_input_tokens, max_audio_tokens) <= 0:
        raise ValueError("max_input_tokens and max_audio_tokens must be positive")
    from training.omni.projector_stage1.feature_cache import FeatureCache

    cache = FeatureCache(feature_dir, max_loaded_shards=max_cached_feature_shards)
    kept: list[dict[str, Any]] = []
    dropped_input = 0
    dropped_audio = 0
    dropped_too_short = 0
    longest_kept = 0
    longest_dropped = 0
    for row in rows:
        feature = cache.get(str(row["sample_id"]))["features"]
        try:
            audio_tokens = subsampled_token_count(int(feature.shape[0]))
        except ValueError:
            dropped_too_short += 1
            continue
        sequence = _build_supervised_sequence(
            row,
            tokenizer,
            audio_placeholder_id,
            audio_tokens,
            max_length=2**31 - 1,
        )
        token_count = len(sequence["input_ids"])
        if audio_tokens > max_audio_tokens:
            dropped_audio += 1
            longest_dropped = max(longest_dropped, token_count)
        elif token_count > max_input_tokens:
            dropped_input += 1
            longest_dropped = max(longest_dropped, token_count)
        else:
            kept.append(row)
            longest_kept = max(longest_kept, token_count)
    if not kept:
        raise ValueError("all samples exceed the configured token limits")
    return kept, {
        "max_input_tokens": max_input_tokens,
        "max_audio_tokens": max_audio_tokens,
        "input_samples": len(rows),
        "kept_samples": len(kept),
        "dropped_samples": dropped_input + dropped_audio,
        "dropped_for_input_tokens": dropped_input,
        "dropped_for_audio_tokens": dropped_audio,
        "dropped_for_projector_subsampling": dropped_too_short,
        "longest_kept_tokens": longest_kept,
        "longest_dropped_tokens": longest_dropped,
    }


def run_training(
    *, train_manifest: Path, dev_manifest: Path, feature_dir: Path, output_dir: Path,
    llm_model: str, audio_model: str, batch_size: int, max_cached_feature_shards: int, steps: int,
    learning_rate: float, weight_decay: float, gradient_accumulation_steps: int,
    validation_every: int, warmup_steps: int, max_grad_norm: float, seed: int,
    device: str, no_progress: bool, max_input_tokens: int = 1536,
    max_audio_tokens: int = 512,
    distributed: bool = False,
) -> dict[str, Any]:
    """Load JoyAI only after explicit authorization and train exactly the projector."""
    preflight_result = preflight(
        train_manifest=train_manifest, dev_manifest=dev_manifest,
        feature_dir=feature_dir, output_dir=output_dir, create_output=False,
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if distributed != torch.distributed.is_initialized():
        raise RuntimeError("distributed flag and process-group state differ")
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1

    tokenizer = AutoTokenizer.from_pretrained(llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
    train_rows, dev_rows = load_manifest(train_manifest), load_manifest(dev_manifest)
    if rank == 0:
        train_rows, train_filter = filter_rows_by_input_tokens(
            train_rows, tokenizer, feature_dir, int(placeholder_id), max_input_tokens,
            max_audio_tokens,
            max_cached_feature_shards,
        )
        dev_rows, dev_filter = filter_rows_by_input_tokens(
            dev_rows, tokenizer, feature_dir, int(placeholder_id), max_input_tokens,
            max_audio_tokens,
            max_cached_feature_shards,
        )
        filtered_rows: list[Any] = [train_rows, dev_rows, train_filter, dev_filter]
    else:
        filtered_rows = [None, None, None, None]
    if distributed:
        torch.distributed.broadcast_object_list(filtered_rows, src=0)
    train_rows, dev_rows, train_filter, dev_filter = filtered_rows
    preflight_result["input_token_filter"] = {"train": train_filter, "dev": dev_filter}
    audio_model_instance = AutoModelForCausalLM.from_pretrained(
        audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    projector = build_frozen_projector_adapter(audio_model_instance.audio_projector)
    del audio_model_instance
    model = build_phase1_model(
        AutoModelForImageTextToText.from_pretrained(llm_model, dtype=torch.bfloat16),
        audio_projector=projector,
    ).to(device=device, dtype=torch.bfloat16)
    train_rows, dev_rows = train_rows[rank::world_size], dev_rows[rank::world_size]
    if not train_rows or not dev_rows:
        raise ValueError("every DDP rank must receive train and dev samples")
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    train_batches = WeightedAudioBatchSource(
        train_rows, tokenizer, feature_dir, int(placeholder_id), batch_size,
        max_cached_feature_shards, voiceassistant_weight=0.40,
        librispeech_weight=0.40, clotho_weight=0.20, seed=seed, audio_token_factor=5,
    )
    dev_batches = CachedConversationBatchSource(
        dev_rows, tokenizer, feature_dir, int(placeholder_id), batch_size,
        max_cached_feature_shards, audio_token_factor=5,
    )
    validation_sources = {
        name: CachedConversationBatchSource(
            [row for row in dev_rows if row.get("provenance", {}).get("dataset") == dataset],
            tokenizer, feature_dir, int(placeholder_id), batch_size,
            max_cached_feature_shards, audio_token_factor=5,
        )
        for name, dataset in {
            "voiceassistant": "shenyunhang/VoiceAssistant-400K",
            "librispeech": "LibriSpeech",
            "clotho_aqa": "Clotho-AQA",
        }.items()
    }
    if any(len(source) == 0 for source in validation_sources.values()):
        raise ValueError("every task requires at least one fixed validation batch")
    config = Stage2Config(
        train_manifest=train_manifest, dev_manifest=dev_manifest, output_dir=output_dir,
        lora_rank=8, lora_alpha=16.0, projector_learning_rate=learning_rate,
        lora_learning_rate=learning_rate, weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm,
        shuffle_seed=seed, asr_replay_ratio=0.0, steps=steps,
        validation_every=validation_every, warmup_steps=warmup_steps,
        no_progress=no_progress,
    )
    result = train_model(model, train_batches, dev_batches, config, validation_sources)
    if rank == 0:
        (output_dir / "preflight.json").write_text(
            json.dumps(preflight_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if distributed:
        torch.distributed.barrier()
    return {"preflight": preflight_result, "training": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-model")
    parser.add_argument("--audio-model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-feature-shards", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-audio-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.distributed:
        import torch

        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = f"cuda:{local_rank}"
    if args.run:
        if not args.llm_model or not args.audio_model:
            parser.error("--run requires --llm-model and --audio-model")
        arguments = vars(args)
        arguments.pop("run")
        result = run_training(**arguments)
    else:
        result = preflight(
            train_manifest=args.train_manifest, dev_manifest=args.dev_manifest,
            feature_dir=args.feature_dir, output_dir=args.output_dir,
        )
    if not args.distributed or __import__("torch").distributed.get_rank() == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.distributed:
        __import__("torch").distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
