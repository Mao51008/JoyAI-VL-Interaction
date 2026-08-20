"""Stage 2: train MiDasheng's official projector, Stage 1 adapter, and JoyAI LoRA.

The MiDasheng encoder remains frozen through the validated raw-feature cache.  This
keeps the Stage 1 audio frontend and prompt contract unchanged while allowing the
official projector to receive gradients for the first time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.train import (
    CachedConversationBatchSource,
    Stage2Config,
    collate_cached_audio_conversations,
    inject_lora,
    load_manifest,
    train_model,
    validate_manifests,
)

from .model import build_official_projector_lora_model
from .official_projector import build_official_projector_adapter
from .sampling import PHASE1_TASK_WEIGHTS, task_draw_counts


TASK_DATASETS = {
    "voiceassistant": "shenyunhang/VoiceAssistant-400K",
    "librispeech": "LibriSpeech",
    "clotho_aqa": "Clotho-AQA",
}


def language_all_linear_targets(llm: Any) -> list[str]:
    """Return JoyAI language-attention and MLP Linear leaves, excluding vision."""
    language = getattr(llm, "language_model", llm)
    leaf_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        ("language_model." if language is not llm else "") + name
        for name, module in language.named_modules()
        if name.rsplit(".", 1)[-1] in leaf_names and module.__class__.__name__ == "Linear"
    ]
    missing = leaf_names - {name.rsplit(".", 1)[-1] for name in targets}
    if missing:
        raise ValueError(f"JoyAI language model lacks all-linear targets: {sorted(missing)}")
    return targets


class ExplicitTaskBatchSource:
    """Reiterable 40/40/20 sampler independent of manifest source sizes."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        feature_dir: Path,
        placeholder_id: int,
        *,
        batch_size: int,
        samples_per_epoch: int,
        max_cached_shards: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or samples_per_epoch <= 0:
            raise ValueError("batch_size and samples_per_epoch must be positive")
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir
        self.placeholder_id = placeholder_id
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.max_cached_shards = max_cached_shards
        self.seed = seed
        self.epoch = 0
        self.groups = {
            task: [dict(row) for row in rows if row.get("provenance", {}).get("dataset") == dataset]
            for task, dataset in TASK_DATASETS.items()
        }
        if missing := [task for task, group in self.groups.items() if not group]:
            raise ValueError(f"missing sampling task rows: {missing}")

    def __len__(self) -> int:
        return math.ceil(self.samples_per_epoch / self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        from training.omni.projector_stage1.feature_cache import FeatureCache

        rng = random.Random(self.seed + self.epoch)
        rows: list[dict[str, Any]] = []
        for task, count in task_draw_counts(self.samples_per_epoch, PHASE1_TASK_WEIGHTS).items():
            for _ in range(count):
                row = dict(rng.choice(self.groups[task]))
                row["_sampling_task"] = task
                rows.append(row)
        rng.shuffle(rows)
        self.epoch += 1
        cache = FeatureCache(self.feature_dir, max_loaded_shards=self.max_cached_shards)
        for index in range(0, len(rows), self.batch_size):
            yield collate_cached_audio_conversations(
                rows[index : index + self.batch_size],
                self.tokenizer,
                cache,
                self.placeholder_id,
                max_length=1536,
                audio_token_factor=5,
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1_adapter(projector: Any, checkpoint: Path) -> dict[str, Any]:
    """Load only the verified Stage 1 3584→4096 adapter weights."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError("Stage 1 checkpoint has an unsupported format")
    tensors = state.get("trainable_state", {})
    adapter_state = {
        name.rsplit("joyai_adapter.", 1)[1]: value
        for name, value in tensors.items()
        if "joyai_adapter." in name
    }
    if set(adapter_state) != {"weight", "bias"}:
        raise ValueError("Stage 1 checkpoint must contain exactly joyai_adapter weight and bias")
    projector.joyai_adapter.load_state_dict(adapter_state, strict=True)
    return {"checkpoint": str(checkpoint), "sha256": _sha256(checkpoint)}


def deepspeed_bf16_config() -> dict[str, Any]:
    """Two-GPU BF16 ZeRO-3 configuration; external loop owns accumulation."""
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if rank == 0 and args.output_dir.exists() and args.resume_from is None:
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
    train_rows = load_manifest(args.train_manifest)
    dev_rows = load_manifest(args.dev_manifest)
    from .train import filter_rows_by_input_tokens

    train_rows, train_filter = filter_rows_by_input_tokens(
        train_rows, tokenizer, args.feature_dir, int(placeholder_id), 1536, 512,
        args.max_cached_feature_shards,
    )
    dev_rows, dev_filter = filter_rows_by_input_tokens(
        dev_rows, tokenizer, args.feature_dir, int(placeholder_id), 1536, 512,
        args.max_cached_feature_shards,
    )
    cache_metadata = validate_feature_cache(args.feature_dir, [*train_rows, *dev_rows])
    train_rows = train_rows[rank::world_size]
    dev_rows = dev_rows[rank::world_size]
    audio_model = AutoModelForCausalLM.from_pretrained(
        args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    projector = build_official_projector_adapter(audio_model.audio_projector, freeze_official=False)
    del audio_model
    adapter_initialization = load_stage1_adapter(projector, args.stage1_checkpoint)
    llm = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16)
    if not hasattr(llm, "gradient_checkpointing_enable") or not hasattr(llm, "enable_input_require_grads"):
        raise TypeError("JoyAI does not support LoRA gradient checkpointing")
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    targets = language_all_linear_targets(llm)
    inject_lora(llm, targets, args.lora_rank, args.lora_alpha)
    model = build_official_projector_lora_model(llm, audio_projector=projector).to(
        args.device, dtype=torch.bfloat16
    )
    if args.distributed and not args.deepspeed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    base_model = getattr(model, "module", model)
    trainable = {name for name, parameter in base_model.named_parameters() if parameter.requires_grad}
    expected_prefixes = ("core.audio_projector.official_projector.", "core.audio_projector.joyai_adapter.")
    if not any(name.startswith(expected_prefixes[0]) for name in trainable):
        raise AssertionError("official projector is not trainable")
    if not any(name.startswith(expected_prefixes[1]) for name in trainable):
        raise AssertionError("Stage 1 adapter is not trainable")
    if not any(".lora_" in name for name in trainable):
        raise AssertionError("JoyAI LoRA is not trainable")
    if any(parameter.grad is not None for parameter in base_model.audio_encoder.parameters()):
        raise AssertionError("frozen encoder unexpectedly has gradients before training")
    smoke_gradient_squares = {"official_projector": 0.0, "adapter": 0.0, "lora": 0.0}
    if args.smoke:
        def gradient_group(name: str) -> str:
            if ".official_projector." in name:
                return "official_projector"
            if ".joyai_adapter." in name:
                return "adapter"
            if ".lora_" in name:
                return "lora"
            raise AssertionError(f"unexpected smoke trainable: {name}")

        for name, parameter in base_model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = gradient_group(name)

            def record_gradient(gradient: Any, *, group_name: str = group) -> Any:
                smoke_gradient_squares[group_name] += float(gradient.detach().float().square().sum())
                return gradient

            parameter.register_hook(record_gradient)
    train_source = ExplicitTaskBatchSource(
        train_rows, tokenizer, args.feature_dir, int(placeholder_id), batch_size=args.batch_size,
        samples_per_epoch=math.ceil(args.samples_per_epoch / world_size),
        max_cached_shards=args.max_cached_feature_shards,
        seed=args.seed,
    )
    validation_sources = {
        task: CachedConversationBatchSource(
            [row for row in dev_rows if row.get("provenance", {}).get("dataset") == dataset],
            tokenizer, args.feature_dir, int(placeholder_id), args.batch_size,
            args.max_cached_feature_shards, audio_token_factor=5,
        )
        for task, dataset in TASK_DATASETS.items()
    }
    if any(len(source) == 0 for source in validation_sources.values()):
        raise ValueError("fixed validation must retain every task")
    steps_per_epoch = math.ceil(len(train_source) / args.gradient_accumulation_steps)
    config = Stage2Config(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        projector_learning_rate=args.projector_lr,
        official_projector_learning_rate=args.projector_lr,
        adapter_learning_rate=args.adapter_lr,
        lora_learning_rate=args.lora_lr,
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        shuffle_seed=args.seed,
        steps=steps_per_epoch * args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_every=args.validation_every,
        warmup_steps=args.warmup_steps,
        no_progress=args.no_progress,
        resume_from=args.resume_from,
        deepspeed_config=deepspeed_bf16_config() if args.deepspeed else None,
        sampling_task_names=tuple(PHASE1_TASK_WEIGHTS),
        checkpoint_metadata={
            "stage": "midasheng_official_projector_lora_stage2",
            "adapter_initialization": adapter_initialization,
            "official_projector": "MiDasheng 5x + Linear(6400,3584) + GELU + Linear(3584,3584)",
            "encoder": {"frozen": True, "feature_cache": cache_metadata},
            "sampling": {"weights": PHASE1_TASK_WEIGHTS, "samples_per_epoch": args.samples_per_epoch},
            "input_limits": {"max_input_tokens": 1536, "max_audio_tokens": 512},
        },
        require_nonzero_grad_groups=False,
    )
    result = train_model(model, train_source, next(iter(validation_sources.values())), config, validation_sources)
    if args.smoke:
        gradient_tensor = torch.tensor(
            [smoke_gradient_squares[name] for name in ("official_projector", "adapter", "lora")],
            device=args.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gradient_tensor)
        smoke_gradients = {
            name: float(value.sqrt())
            for name, value in zip(("official_projector", "adapter", "lora"), gradient_tensor.cpu(), strict=True)
        }
        if any(value == 0.0 for value in smoke_gradients.values()):
            raise AssertionError(f"zero Stage 2 smoke gradient group: {smoke_gradients}")
        result["smoke_gradient_norm_by_group"] = smoke_gradients
    local_memory = {
        "rank": rank,
        "device": args.device,
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }
    memory_by_rank: list[dict[str, Any] | None] = [None] * world_size
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(memory_by_rank, local_memory)
    else:
        memory_by_rank = [local_memory]
    result["memory_by_rank"] = memory_by_rank
    result["runtime"] = {
        "adapter_initialization": adapter_initialization,
        "trainable_parameter_names": sorted(trainable),
        "input_filter": {"train": train_filter, "dev": dev_filter},
        "validation_sizes": {name: len(source) for name, source in validation_sources.items()},
    }
    if args.smoke and rank == 0:
        state = torch.load(args.output_dir / "latest.pt", map_location="cpu", weights_only=True)
        saved_names = set(state["trainable_state"])
        required = ("official_projector", "joyai_adapter", ".lora_")
        if not all(any(marker in name for name in saved_names) for marker in required):
            raise AssertionError("latest checkpoint lacks one Stage 2 trainable component")
        result["smoke_checkpoint_restored"] = True
    if rank == 0:
        (args.output_dir / "stage2_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-epoch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--projector-lr", type=float, default=1e-6)
    parser.add_argument("--adapter-lr", type=float, default=1e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--validation-every", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-cached-feature-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.samples_per_epoch <= 0:
        parser.error("--epochs and --samples-per-epoch must be positive")
    if args.deepspeed and not args.distributed:
        parser.error("--deepspeed requires --distributed")
    if not args.smoke and not args.deepspeed:
        parser.error("formal Stage 2 training requires --deepspeed; use --smoke for the audited DDP run")
    if args.distributed:
        import os
        import torch

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
        )
        args.device = f"cuda:{torch.cuda.current_device()}"
    manifests = validate_manifests(args.train_manifest, args.dev_manifest)
    if not args.distributed or __import__("torch").distributed.get_rank() == 0:
        print("STAGE2_OFFICIAL_CONFIG=" + json.dumps({
            "manifests": manifests, "sampling": PHASE1_TASK_WEIGHTS,
            "learning_rates": {"official_projector": args.projector_lr, "adapter": args.adapter_lr, "lora": args.lora_lr},
            "validation_every": args.validation_every, "stage1_checkpoint": str(args.stage1_checkpoint),
        }, ensure_ascii=False), flush=True)
    result = run(args)
    if not args.distributed or __import__("torch").distributed.get_rank() == 0:
        print("STAGE2_OFFICIAL_RESULT=" + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    if args.distributed:
        __import__("torch").distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
