"""Joint MiDasheng high-layer, projector and language-only LoRA training.

Unlike :mod:`training.omni.midasheng.train`, this entry intentionally reads raw
audio at training time.  Cached encoder features cannot be used when encoder
blocks are trainable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from training.omni.projector_stage1.model import replace_audio_placeholders
from training.omni.projector_stage1.projector import AudioProjector, AudioProjectorConfig
from training.omni.projector_stage2.cache_features import _load_mono_audio
from training.omni.projector_stage2.train import (
    Stage2Config,
    Stage2ConversationBatch,
    _build_supervised_sequence,
    _optimizer_parameter_groups as _stage2_optimizer_groups,
    inject_lora,
    load_manifest,
    train_model,
    validate_manifests,
)
from .hybrid_trainer import HybridConfig, HybridTrainer, assert_hybrid_partition, build_audio_encoder_branch, build_projector_lora_core

TASK_WEIGHTS = {"voiceassistant": 0.50, "clotho_aqa": 0.30, "librispeech": 0.20}
TASK_DATASETS = {
    "voiceassistant": "shenyunhang/VoiceAssistant-400K",
    "clotho_aqa": "Clotho-AQA",
    "librispeech": "LibriSpeech",
}
ENCODER_LR = 3e-7
PROJECTOR_LR = 3e-6
LORA_LR = 1e-5


def deepspeed_zero3_config(*, steps: int, warmup_steps: int) -> dict[str, Any]:
    """ZeRO-3 BF16 configuration for the two-group Projector/LoRA core.

    HybridTrainer owns the eight-microbatch accumulation boundary. DeepSpeed
    therefore sees one externally accumulated update at a time.
    """
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True,
                              "reduce_scatter": True, "allgather_partitions": True,
                              "offload_optimizer": {"device": "none"}, "offload_param": {"device": "none"}},
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "scheduler": {"type": "WarmupDecayLR", "params": {"warmup_min_lr": [0.0, 0.0],
                      "warmup_max_lr": [PROJECTOR_LR, LORA_LR],
                      "warmup_num_steps": warmup_steps, "total_num_steps": steps}},
    }


def write_loss_curve(points: list[dict[str, float]], path: Path) -> None:
    """Write a dependency-free SVG curve for the assistant-token training loss."""
    width, height, margin = 960, 480, 52
    losses = [point["loss"] for point in points]
    low, high = min(losses), max(losses)
    span = max(high - low, 1e-8)
    coordinates = []
    for index, loss in enumerate(losses):
        x = margin + (width - 2 * margin) * index / max(1, len(losses) - 1)
        y = height - margin - (height - 2 * margin) * (loss - low) / span
        coordinates.append(f"{x:.2f},{y:.2f}")
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{margin}" y="28" font-size="18">Joint training loss</text>'
        f'<text x="{margin}" y="{height - 14}" font-size="12">step 1–{len(points)}</text>'
        f'<text x="{width - 210}" y="{height - 14}" font-size="12">loss {low:.5f}–{high:.5f}</text>'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(coordinates)}"/>'
        "</svg>\n", encoding="utf-8"
    )


def _task_counts(total: int, weights: dict[str, float] = TASK_WEIGHTS) -> dict[str, int]:
    if total <= 0:
        raise ValueError("samples_per_epoch must be positive")
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(weights, key=lambda x: (raw[x] - counts[x], x), reverse=True)[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def classify_task(row: dict[str, Any]) -> str:
    dataset = row.get("provenance", {}).get("dataset")
    for task, expected in TASK_DATASETS.items():
        if dataset == expected:
            return task
    raise ValueError(f"unsupported joint-training dataset: {dataset!r}")


def encoder_block_layout(audio_encoder: Any) -> tuple[str, Sequence[Any]]:
    """Return the only supported MiDasheng Transformer block container."""
    blocks = getattr(audio_encoder, "blocks", None)
    if blocks is None or not hasattr(blocks, "__len__"):
        raise ValueError("MiDasheng audio encoder must expose Transformer blocks as .blocks")
    if not len(blocks):
        raise ValueError("MiDasheng audio encoder has no Transformer blocks")
    return "blocks", blocks


def configure_high_encoder_blocks(audio_encoder: Any, fraction: float = 0.25) -> dict[str, Any]:
    """Freeze MiDasheng except the closest complete final fraction of blocks."""
    if not 0 < fraction <= 1:
        raise ValueError("encoder_unfreeze_fraction must be in (0, 1]")
    block_name, blocks = encoder_block_layout(audio_encoder)
    total_blocks = len(blocks)
    unfreeze_count = max(1, round(total_blocks * fraction))
    start = total_blocks - unfreeze_count
    for parameter in audio_encoder.parameters():
        parameter.requires_grad_(False)
    for block in blocks[start:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    names = [f"{block_name}.{index}" for index in range(start, total_blocks)]
    return {
        "total_blocks": total_blocks,
        "unfrozen_block_indices": list(range(start, total_blocks)),
        "unfrozen_block_names": names,
        "total_parameters": sum(p.numel() for p in audio_encoder.parameters()),
        "unfrozen_parameters": sum(p.numel() for p in audio_encoder.parameters() if p.requires_grad),
    }


def language_all_linear_targets(llm: Any) -> list[str]:
    """Find q/k/v/o and MLP linears under the language model, never vision modules."""
    language = getattr(llm, "language_model", llm)
    required = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = []
    for name, module in language.named_modules():
        if name.rsplit(".", 1)[-1] not in required:
            continue
        if module.__class__.__name__ != "Linear":
            continue
        targets.append(("language_model." if language is not llm else "") + name)
    found = {name.rsplit(".", 1)[-1] for name in targets}
    missing = required - found
    if missing:
        raise ValueError(f"JoyAI language Transformer lacks required all-linear modules: {sorted(missing)}")
    return targets


def _feature_tokens(waveform_samples: int, max_audio_tokens: int) -> int:
    # DashengAudioTransformer uses floor(x_length / (hop_length * 4)), hop=160.
    tokens = waveform_samples // 640
    if tokens <= 0:
        raise ValueError("audio is shorter than one MiDasheng encoder token")
    if tokens > max_audio_tokens:
        raise ValueError(
            f"MiDasheng audio token count {tokens} exceeds max_audio_tokens={max_audio_tokens}"
        )
    return tokens


def filter_rows_by_audio_limit(
    rows: Sequence[dict[str, Any]], max_audio_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use manifest duration to exclude clips that cannot fit 512 MiDasheng tokens."""
    max_duration_ms = max_audio_tokens * 40  # 640 samples / 16 kHz = 40 ms per token.
    kept = [row for row in rows if int(row["clip_duration_ms"]) <= max_duration_ms]
    if not kept:
        raise ValueError("all rows exceed the configured MiDasheng audio-token limit")
    return kept, {"input_samples": len(rows), "kept_samples": len(kept),
                  "dropped_for_audio_tokens": len(rows) - len(kept),
                  "max_audio_tokens": max_audio_tokens, "max_duration_ms": max_duration_ms}


def load_joint_manifest(path: Path, max_audio_tokens: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load a formal manifest and retain its root for raw audio resolution."""
    rows, report = filter_rows_by_audio_limit(load_manifest(path), max_audio_tokens)
    for row in rows:
        row["_manifest_root"] = str(path.parent)
    return rows, report


def load_midasheng_projector_initialization(
    projector: Any, checkpoint: Path, expected_sha256: str, source_preflight: Path,
    audio_model: str,
) -> dict[str, Any]:
    """Load only a verified MiDasheng projector-only checkpoint; never any LoRA."""
    if not checkpoint.is_file() or not source_preflight.is_file():
        raise FileNotFoundError("projector checkpoint and its MiDasheng preflight.json must exist")
    actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("projector checkpoint SHA256 mismatch")
    provenance = json.loads(source_preflight.read_text(encoding="utf-8"))
    feature_cache = provenance.get("feature_cache", {})
    if (
        provenance.get("audio_encoder", {}).get("feature_source") != "midashenglm.audio_encoder.raw"
        or feature_cache.get("feature_source") != "midashenglm.audio_encoder.raw"
        or feature_cache.get("audio_model") != audio_model
        or feature_cache.get("audio_encoder_dim") != 1280
    ):
        raise ValueError("projector provenance is not the requested raw MiDashengLM Audio Encoder")
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") == "projector-stage1-v4":
        weights = state.get("projector")
        if state.get("config", {}).get("projector") != projector.export_config():
            raise ValueError("MiDasheng projector checkpoint config does not match 1280→4096 projector")
    elif state.get("format") == "projector-stage2-v2":
        weights = {
            name.removeprefix("audio_projector.").removeprefix("core.audio_projector."): value
            for name, value in state.get("trainable_state", {}).items()
            if name.startswith(("audio_projector.", "core.audio_projector."))
        }
    else:
        raise ValueError("unsupported projector-only checkpoint format")
    if set(weights) != set(projector.state_dict()):
        raise ValueError("checkpoint does not contain exactly one compatible audio_projector state")
    projector.load_state_dict(weights, strict=True)
    return {"checkpoint": str(checkpoint.resolve()), "sha256": actual_sha256,
            "source_preflight": str(source_preflight.resolve()),
            "source_preflight_sha256": hashlib.sha256(source_preflight.read_bytes()).hexdigest(),
            "format": state["format"], "audio_model": audio_model}


class JointRawAudioBatchSource:
    """Deterministic task-level replacement sampler plus raw waveform collation."""

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, placeholder_id: int, *,
                 batch_size: int, samples_per_epoch: int, seed: int, max_length: int = 2048,
                 task_weights: dict[str, float] = TASK_WEIGHTS,
                 sample_with_replacement: bool = True, max_audio_tokens: int = 512) -> None:
        if max_length <= 0 or max_audio_tokens <= 0:
            raise ValueError("max_length and max_audio_tokens must be positive")
        self.tokenizer, self.placeholder_id = tokenizer, placeholder_id
        self.batch_size, self.samples_per_epoch, self.seed, self.max_length = batch_size, samples_per_epoch, seed, max_length
        self.task_weights = task_weights
        self.sample_with_replacement = sample_with_replacement
        self.max_audio_tokens = max_audio_tokens
        self.groups = {task: [] for task in task_weights}
        for row in rows:
            task = classify_task(row)
            if task in self.groups:
                self.groups[task].append(dict(row))
        missing = [task for task, rows in self.groups.items() if not rows]
        if missing:
            raise ValueError("joint sampler requires all task datasets: " + ", ".join(missing))
        self.epoch = 0
        self.last_sampling_report: dict[str, Any] = {}

    def __len__(self) -> int:
        return math.ceil(self.samples_per_epoch / self.batch_size)

    def _rows(self) -> list[dict[str, Any]]:
        counts = _task_counts(self.samples_per_epoch, self.task_weights)
        rng = random.Random(self.seed + self.epoch)
        sampled = []
        for task, count in counts.items():
            if not self.sample_with_replacement and count != len(self.groups[task]):
                raise ValueError("fixed validation source must include every task row exactly once")
            selected = rng.choices(self.groups[task], k=count) if self.sample_with_replacement else list(self.groups[task])
            for row in selected:
                value = dict(row)
                value["training_task"] = "asr_transcription" if task == "librispeech" else "dialogue_response"
                value["_joint_task"] = task
                sampled.append(value)
        rng.shuffle(sampled)
        self.last_sampling_report = {"samples": len(sampled), "counts": counts,
            "ratios": {task: counts[task] / len(sampled) for task in counts}}
        self.epoch += 1
        return sampled

    def __iter__(self) -> Iterator[Stage2ConversationBatch]:
        import torch
        from torch.nn.utils.rnn import pad_sequence
        rows = self._rows()
        for offset in range(0, len(rows), self.batch_size):
            current = rows[offset: offset + self.batch_size]
            waveforms, sequences = [], []
            for row in current:
                path = Path(row["_manifest_root"]) / row["audio_path"]
                waveform = torch.from_numpy(_load_mono_audio(path, 16_000))
                waveforms.append(waveform)
                sequences.append(_build_supervised_sequence(
                    row, self.tokenizer, self.placeholder_id,
                    _feature_tokens(waveform.numel(), self.max_audio_tokens), self.max_length,
                ))
            audio = pad_sequence(waveforms, batch_first=True)
            lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
            def pad(key: str, value: int | bool):
                return pad_sequence([torch.tensor(item[key]) for item in sequences], batch_first=True, padding_value=value)
            yield Stage2ConversationBatch(
                input_ids=pad("input_ids", getattr(self.tokenizer, "pad_token_id", 0) or 0), labels=pad("labels", -100),
                attention_mask=pad("attention_mask", False).bool(), audio_features=audio,
                audio_attention_mask=lengths, audio_placeholder_mask=pad("audio_placeholder_mask", False).bool(),
                sample_ids=[str(row["sample_id"]) for row in current], dialogue_ids=[str(row["dialogue_id"]) for row in current],
                task_types=[row["_joint_task"] for row in current],
            )


def build_joint_model(audio_encoder: Any, llm: Any, *, projector_checkpoint: Path,
                      projector_sha256: str, projector_preflight: Path, audio_model: str,
                      dropout: float = 0.0) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn
    targets = language_all_linear_targets(llm)
    inject_lora(llm, targets, 8, 16.0)
    report = configure_high_encoder_blocks(audio_encoder)
    projector = AudioProjector(AudioProjectorConfig(1280, 4096, 4096, dropout))
    projector_initialization = load_midasheng_projector_initialization(
        projector, projector_checkpoint, projector_sha256, projector_preflight, audio_model
    )
    for name, parameter in llm.named_parameters():
        parameter.requires_grad_(".lora_" in name)

    class JointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.audio_encoder = audio_encoder; self.language_model = llm; self.audio_projector = projector
        def forward(self, batch: Any) -> Any:
            device = next(self.parameters()).device
            waveform = batch.audio_features.to(device=device, dtype=torch.float32)
            lengths = batch.audio_attention_mask.to(device)
            # DeepSpeed BF16 autocast must not enter MiDasheng's FP32 encoder graph.
            with torch.autocast(device_type="cuda", enabled=False):
                encoded, mask = self.audio_encoder(waveform, x_length=lengths)
            if [int(value) for value in mask.sum(dim=1)] != [int(value) for value in batch.audio_placeholder_mask.sum(dim=1)]:
                raise ValueError("MiDasheng encoder token count differs from audio placeholders")
            projector_dtype = next(self.audio_projector.parameters()).dtype
            projected = self.audio_projector(encoded.to(dtype=projector_dtype))
            ids = batch.input_ids.to(device); text = self.language_model.get_input_embeddings()(ids)
            embeds = replace_audio_placeholders(text, projected, batch.audio_placeholder_mask.to(device), mask.bool())
            return self.language_model(inputs_embeds=embeds, attention_mask=batch.attention_mask.to(device), labels=batch.labels.to(device))
    model = JointModel()
    for parameter in model.audio_projector.parameters(): parameter.requires_grad_(True)
    parameter_counts = {
        "midasheng_high_blocks": sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad),
        "audio_projector": sum(p.numel() for p in model.audio_projector.parameters()),
        "joyai_lora": sum(p.numel() for n, p in model.language_model.named_parameters() if ".lora_" in n),
    }
    parameter_counts["total_trainable"] = sum(parameter_counts.values())
    return model, {**report, "lora_targets": targets, "projector_initialization": projector_initialization,
                   "trainable_parameter_counts": parameter_counts}


def replace_midasheng_frontend_batch_norm(audio_encoder: Any) -> Any:
    """Install an explicit FP32 BatchNorm boundary; no forward hooks are used."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    batch_norm = getattr(audio_encoder, "init_bn", None)
    if batch_norm is None:
        raise TypeError("MiDasheng audio encoder does not expose init_bn")

    class Fp32FrozenBatchNorm2d(nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.register_buffer("weight", source.weight.detach().float().clone())
            self.register_buffer("bias", source.bias.detach().float().clone())
            self.register_buffer("running_mean", source.running_mean.detach().float().clone())
            self.register_buffer("running_var", source.running_var.detach().float().clone())
            self.eps = float(source.eps)

        def _apply(self, fn: Any) -> "Fp32FrozenBatchNorm2d":
            # DeepSpeed BF16 conversion must not change this frozen FP32 boundary.
            return self

        def forward(self, inputs: Any) -> Any:
            normalized = functional.batch_norm(
                inputs.float(), self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps,
            )
            return normalized.to(dtype=torch.bfloat16)

        def assert_fp32(self) -> None:
            if any(t.dtype != torch.float32 for t in (self.weight, self.bias, self.running_mean, self.running_var)):
                raise AssertionError("MiDasheng init_bn FP32 boundary was converted")

    replacement = Fp32FrozenBatchNorm2d(batch_norm)
    audio_encoder.init_bn = replacement
    return replacement


def optimizer_parameter_groups(model: Any, config: Stage2Config) -> list[dict[str, Any]]:
    return _stage2_optimizer_groups(model, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True); parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--audio-model", required=True); parser.add_argument("--llm-model", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-epoch", type=int, required=True); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-audio-tokens", type=int, default=512)
    parser.add_argument("--init-projector-checkpoint", type=Path, required=True); parser.add_argument("--init-projector-sha256", required=True)
    parser.add_argument("--init-projector-preflight", type=Path, required=True)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    manifests = validate_manifests(args.train_manifest, args.dev_manifest)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if args.deepspeed and world_size != 3:
        parser.error("--deepspeed joint training requires exactly 3 ranks")
    if args.deepspeed and (args.batch_size, args.gradient_accumulation_steps) != (1, 8):
        parser.error("ZeRO-2 baseline requires micro batch 1 and gradient accumulation 8")
    steps_per_epoch = math.ceil(args.samples_per_epoch / (args.batch_size * args.gradient_accumulation_steps * world_size))
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    preflight = {"status": "preflight-only", "manifests": manifests, "audio_model": {"path": args.audio_model, "checkpoint_kind": "MiDashengLM final checkpoint Audio Encoder"},
                 "projector_initialization": {"checkpoint": str(args.init_projector_checkpoint), "sha256": args.init_projector_sha256,
                 "source_preflight": str(args.init_projector_preflight), "required_source": "midashenglm.audio_encoder.raw"},
                 "lora_initialization": "new random LoRA A; zero LoRA B; no previous LoRA loaded", "encoder": {"total_blocks": 32, "unfrozen_block_indices": list(range(24, 32))},
                 "learning_rates": {"midasheng_high_blocks": ENCODER_LR, "audio_projector": PROJECTOR_LR, "joyai_lora": LORA_LR},
                 "sampling": {"weights": TASK_WEIGHTS, "samples_per_epoch": args.samples_per_epoch, "counts": _task_counts(args.samples_per_epoch)},
                 "sequence_limits": {"max_input_tokens": args.max_input_tokens, "max_midasheng_audio_tokens": args.max_audio_tokens},
                 "training_schedule": {"epochs": args.epochs, "optimizer_steps_per_epoch": steps_per_epoch, "optimizer_steps": steps_per_epoch * args.epochs,
                 "per_device_batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "world_size": world_size, "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size, "warmup_steps": args.warmup_steps,
                 "scheduler": "linear decay to 10% of each group LR", "max_grad_norm": args.max_grad_norm,
                 "max_validation_batches": args.max_validation_batches, "log_interval": args.log_interval}}
    if not args.run:
        if args.output_dir.exists(): raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True); (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8"); print(json.dumps(preflight, indent=2)); return 0
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    if args.deepspeed:
        import deepspeed

        deepspeed.init_distributed()
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
    audio_model = AutoModelForCausalLM.from_pretrained(args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    audio_encoder = audio_model.audio_encoder
    del audio_model; gc.collect()
    llm = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16)
    if not hasattr(audio_encoder, "gradient_checkpointing_enable"):
        raise TypeError("MiDasheng audio encoder does not support gradient checkpointing")
    audio_encoder.gradient_checkpointing_enable()
    if not hasattr(llm, "gradient_checkpointing_enable") or not hasattr(llm, "enable_input_require_grads"):
        raise TypeError("JoyAI language model does not support LoRA gradient checkpointing")
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    model, runtime_report = build_joint_model(audio_encoder, llm, projector_checkpoint=args.init_projector_checkpoint,
        projector_sha256=args.init_projector_sha256, projector_preflight=args.init_projector_preflight,
        audio_model=args.audio_model)
    encoder_branch = build_audio_encoder_branch(model.audio_encoder, args.device, args.deepspeed)
    core = build_projector_lora_core(model.audio_projector, model.language_model).to(device=args.device, dtype=torch.bfloat16)
    assert_hybrid_partition(encoder_branch, core)
    train_rows, train_filter = load_joint_manifest(args.train_manifest, args.max_audio_tokens)
    dev_rows, dev_filter = load_joint_manifest(args.dev_manifest, args.max_audio_tokens)
    rank = torch.distributed.get_rank() if args.deepspeed else 0
    local_epoch_samples = math.ceil(args.samples_per_epoch / world_size)
    train_batches = JointRawAudioBatchSource(train_rows, tokenizer, int(placeholder_id), batch_size=args.batch_size,
        samples_per_epoch=local_epoch_samples, seed=3407 + rank, max_length=args.max_input_tokens,
        max_audio_tokens=args.max_audio_tokens)
    validation_sources = {
        task: JointRawAudioBatchSource([row for row in dev_rows if classify_task(row) == task], tokenizer, int(placeholder_id),
            batch_size=args.batch_size, samples_per_epoch=sum(classify_task(row) == task for row in dev_rows), seed=3407,
            task_weights={task: 1.0}, sample_with_replacement=False, max_length=args.max_input_tokens,
            max_audio_tokens=args.max_audio_tokens)
        for task in TASK_WEIGHTS
    }
    preflight["encoder"] = runtime_report
    preflight["input_token_filter"] = {"train": train_filter, "dev": dev_filter}
    preflight["validation"] = {"sampling": "disabled", "sources": {
        task: {"samples": len(source.groups[task]), "fixed_complete_dev_set": True}
        for task, source in validation_sources.items()
    }}
    print("JOINT_TRAINING_CONFIG=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not args.deepspeed:
        raise RuntimeError("real joint training requires the hybrid DeepSpeed/DDP launcher")
    projector_parameters = [p for n, p in core.named_parameters() if p.requires_grad and "audio_projector" in n]
    lora_parameters = [p for n, p in core.named_parameters() if p.requires_grad and ".lora_" in n]
    core_optimizer = torch.optim.AdamW([
        {"params": projector_parameters, "lr": PROJECTOR_LR, "weight_decay": 0.01},
        {"params": lora_parameters, "lr": LORA_LR, "weight_decay": 0.01},
    ])
    core_engine, _, _, _ = deepspeed.initialize(model=core, optimizer=core_optimizer,
        config=deepspeed_zero3_config(steps=steps_per_epoch * args.epochs, warmup_steps=args.warmup_steps))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_optimizer = torch.optim.AdamW([p for p in encoder_branch.parameters() if p.requires_grad], lr=ENCODER_LR, weight_decay=0.01)
    encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        encoder_optimizer,
        lr_lambda=lambda step: (step + 1) / max(1, args.warmup_steps)
        if step < args.warmup_steps else max(0.1, 1 - 0.9 * (step - args.warmup_steps) / max(1, steps_per_epoch * args.epochs - args.warmup_steps)),
    )
    trainer = HybridTrainer(encoder_branch, core_engine, encoder_optimizer,
        HybridConfig(args.gradient_accumulation_steps, args.max_grad_norm), encoder_scheduler)
    iterator = iter(train_batches)
    losses: list[float] = []
    loss_history: list[dict[str, float]] = []
    started_at = time.monotonic()
    for _step in range(steps_per_epoch * args.epochs):
        micros = []
        for _ in range(args.gradient_accumulation_steps):
            try: micros.append(next(iterator))
            except StopIteration: iterator = iter(train_batches); micros.append(next(iterator))
        local_loss = trainer.run_update(micros)
        loss_tensor = torch.tensor(local_loss, device=args.device)
        torch.distributed.all_reduce(loss_tensor)
        loss = float(loss_tensor / world_size)
        losses.append(loss)
        if trainer.global_step % args.log_interval == 0:
            elapsed = time.monotonic() - started_at
            record = {"step": trainer.global_step, "loss": loss, "elapsed_seconds": elapsed,
                      "eta_seconds": elapsed * (steps_per_epoch * args.epochs - trainer.global_step) / trainer.global_step}
            if rank == 0:
                loss_history.append(record)
                with (args.output_dir / "train_loss.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("TRAIN_PROGRESS=" + json.dumps(record), flush=True)
    validation = {
        name: trainer.validate(source if args.max_validation_batches is None else islice(source, args.max_validation_batches))
        for name, source in validation_sources.items()
    }
    trainer.save_checkpoint(args.output_dir / "hybrid.pt")
    trainer.load_checkpoint(args.output_dir / "hybrid.pt")
    memory_by_rank: list[dict[str, int] | None] = [None] * world_size
    torch.distributed.all_gather_object(memory_by_rank, {
        "rank": rank,
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
    })
    result = {"steps": trainer.global_step, "train_loss": losses[-1], "validation": validation,
              "checkpoint_restored": True, "memory_by_rank": memory_by_rank, "gradient_audit": trainer.last_gradient_audit}
    if rank == 0:
        (args.output_dir / "train_loss_curve.json").write_text(json.dumps(loss_history, indent=2) + "\n", encoding="utf-8")
        write_loss_curve(loss_history, args.output_dir / "train_loss_curve.svg")
        (args.output_dir / "joint_training_report.json").write_text(
            json.dumps({"preflight": preflight, "runtime": runtime_report, "training": result}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
    print("HYBRID_SMOKE_RESULT=" + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
