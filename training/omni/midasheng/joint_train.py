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
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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

TASK_WEIGHTS = {"voiceassistant": 0.50, "clotho_aqa": 0.30, "librispeech": 0.20}
TASK_DATASETS = {
    "voiceassistant": "shenyunhang/VoiceAssistant-400K",
    "clotho_aqa": "Clotho-AQA",
    "librispeech": "LibriSpeech",
}
ENCODER_LR = 3e-7
PROJECTOR_LR = 3e-6
LORA_LR = 1e-5


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


def _feature_tokens(waveform_samples: int) -> int:
    # DashengAudioTransformer uses floor(x_length / (hop_length * 4)), hop=160.
    tokens = waveform_samples // 640
    if tokens <= 0:
        raise ValueError("audio is shorter than one MiDasheng encoder token")
    return tokens


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
                 sample_with_replacement: bool = True) -> None:
        self.tokenizer, self.placeholder_id = tokenizer, placeholder_id
        self.batch_size, self.samples_per_epoch, self.seed, self.max_length = batch_size, samples_per_epoch, seed, max_length
        self.task_weights = task_weights
        self.sample_with_replacement = sample_with_replacement
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
                sequences.append(_build_supervised_sequence(row, self.tokenizer, self.placeholder_id, _feature_tokens(waveform.numel()), self.max_length))
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
            waveform, lengths = batch.audio_features.to(device), batch.audio_attention_mask.to(device)
            encoded, mask = self.audio_encoder(waveform, x_length=lengths)
            if [int(value) for value in mask.sum(dim=1)] != [int(value) for value in batch.audio_placeholder_mask.sum(dim=1)]:
                raise ValueError("MiDasheng encoder token count differs from audio placeholders")
            projected = self.audio_projector(encoded)
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


def optimizer_parameter_groups(model: Any, config: Stage2Config) -> list[dict[str, Any]]:
    return _stage2_optimizer_groups(model, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True); parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--audio-model", required=True); parser.add_argument("--llm-model", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-epoch", type=int, required=True); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-projector-checkpoint", type=Path, required=True); parser.add_argument("--init-projector-sha256", required=True)
    parser.add_argument("--init-projector-preflight", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    manifests = validate_manifests(args.train_manifest, args.dev_manifest)
    steps_per_epoch = math.ceil(math.ceil(args.samples_per_epoch / args.batch_size) / args.gradient_accumulation_steps)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    preflight = {"status": "preflight-only", "manifests": manifests, "audio_model": {"path": args.audio_model, "checkpoint_kind": "MiDashengLM final checkpoint Audio Encoder"},
                 "projector_initialization": {"checkpoint": str(args.init_projector_checkpoint), "sha256": args.init_projector_sha256,
                 "source_preflight": str(args.init_projector_preflight), "required_source": "midashenglm.audio_encoder.raw"},
                 "lora_initialization": "new random LoRA A; zero LoRA B; no previous LoRA loaded", "encoder": {"total_blocks": 32, "unfrozen_block_indices": list(range(24, 32))},
                 "learning_rates": {"midasheng_high_blocks": ENCODER_LR, "audio_projector": PROJECTOR_LR, "joyai_lora": LORA_LR},
                 "sampling": {"weights": TASK_WEIGHTS, "samples_per_epoch": args.samples_per_epoch, "counts": _task_counts(args.samples_per_epoch)},
                 "training_schedule": {"epochs": args.epochs, "optimizer_steps_per_epoch": steps_per_epoch, "optimizer_steps": steps_per_epoch * args.epochs,
                 "per_device_batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "effective_batch_size": args.batch_size * args.gradient_accumulation_steps, "warmup_steps": args.warmup_steps,
                 "scheduler": "linear decay to 10% of each group LR", "max_grad_norm": args.max_grad_norm}}
    if not args.run:
        if args.output_dir.exists(): raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True); (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8"); print(json.dumps(preflight, indent=2)); return 0
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
    audio_model = AutoModelForCausalLM.from_pretrained(args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    audio_encoder = audio_model.audio_encoder
    del audio_model; gc.collect()
    llm = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16)
    model, runtime_report = build_joint_model(audio_encoder, llm, projector_checkpoint=args.init_projector_checkpoint,
        projector_sha256=args.init_projector_sha256, projector_preflight=args.init_projector_preflight,
        audio_model=args.audio_model)
    model = model.to(device=args.device, dtype=torch.bfloat16)
    train_rows, dev_rows = load_manifest(args.train_manifest), load_manifest(args.dev_manifest)
    train_batches = JointRawAudioBatchSource(train_rows, tokenizer, int(placeholder_id), batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch, seed=3407)
    validation_sources = {
        task: JointRawAudioBatchSource([row for row in dev_rows if classify_task(row) == task], tokenizer, int(placeholder_id),
            batch_size=args.batch_size, samples_per_epoch=sum(classify_task(row) == task for row in dev_rows), seed=3407,
            task_weights={task: 1.0}, sample_with_replacement=False)
        for task in TASK_WEIGHTS
    }
    config = Stage2Config(args.train_manifest, args.dev_manifest, args.output_dir, projector_learning_rate=PROJECTOR_LR,
        lora_learning_rate=LORA_LR, encoder_learning_rate=ENCODER_LR, steps=steps_per_epoch * args.epochs,
        validation_every=steps_per_epoch, gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps, max_grad_norm=args.max_grad_norm)
    preflight["encoder"] = runtime_report
    preflight["validation"] = {"sampling": "disabled", "sources": {
        task: {"samples": len(source.groups[task]), "fixed_complete_dev_set": True}
        for task, source in validation_sources.items()
    }}
    print("JOINT_TRAINING_CONFIG=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    result = train_model(model, train_batches, validation_sources["voiceassistant"], config, validation_sources=validation_sources)
    (args.output_dir / "joint_training_report.json").write_text(
        json.dumps({"preflight": preflight, "runtime": runtime_report, "training": result}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"preflight": preflight, "training": result}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
