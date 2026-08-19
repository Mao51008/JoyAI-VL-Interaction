"""Distributed greedy free generation from a hybrid MiDasheng/ZeRO-3 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training.omni.projector_stage1.model import replace_audio_placeholders
from training.omni.projector_stage2.cache_features import _load_mono_audio
from training.omni.projector_stage2.train import Stage2ConversationBatch, _build_supervised_sequence

from .hybrid_trainer import build_projector_lora_core
from .joint_train import (
    TASK_WEIGHTS,
    _feature_tokens,
    build_joint_model,
    classify_task,
    load_joint_manifest,
)


def select_rows(rows: list[dict[str, Any]], task: str, count: int, seed: int) -> list[dict[str, Any]]:
    candidates = sorted((row for row in rows if classify_task(row) == task), key=lambda row: str(row["sample_id"]))
    if len(candidates) < count:
        raise ValueError(f"{task} has only {len(candidates)} eligible rows, need {count}")
    return random.Random(seed).sample(candidates, count)


def make_batch(row: dict[str, Any], tokenizer: Any, placeholder_id: int, max_length: int, max_audio_tokens: int) -> Stage2ConversationBatch:
    import torch

    waveform = torch.from_numpy(_load_mono_audio(Path(row["_manifest_root"]) / row["audio_path"], 16_000))
    value = dict(row)
    value["training_task"] = "asr_transcription" if classify_task(row) == "librispeech" else "dialogue_response"
    sequence = _build_supervised_sequence(value, tokenizer, placeholder_id, _feature_tokens(waveform.numel(), max_audio_tokens), max_length)
    answer_start = next(index for index, label in enumerate(sequence["labels"]) if label != -100)
    return Stage2ConversationBatch(
        input_ids=torch.tensor(sequence["input_ids"][:answer_start]).unsqueeze(0),
        labels=torch.empty((1, 0), dtype=torch.long),
        attention_mask=torch.tensor(sequence["attention_mask"][:answer_start]).unsqueeze(0).bool(),
        audio_features=waveform.unsqueeze(0), audio_attention_mask=torch.tensor([waveform.numel()]),
        audio_placeholder_mask=torch.tensor(sequence["audio_placeholder_mask"][:answer_start]).unsqueeze(0).bool(),
        sample_ids=[str(row["sample_id"])], dialogue_ids=[str(row["dialogue_id"])], task_types=[classify_task(row)],
    )


def generate(core: Any, encoder: Any, batch: Stage2ConversationBatch, tokenizer: Any, max_new_tokens: int) -> tuple[str, int, bool]:
    import torch

    device = next(core.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        encoded, mask = encoder(batch.audio_features.to(device=device, dtype=torch.float32), batch.audio_attention_mask.to(device))
    if int(mask.sum()) != int(batch.audio_placeholder_mask.sum()):
        raise ValueError("MiDasheng output tokens do not match audio placeholders")
    with torch.inference_mode():
        projector = core.audio_projector
        projected = projector(encoded.to(device=device, dtype=next(projector.parameters()).dtype))
        language = core.language_model
        embeddings = replace_audio_placeholders(
            language.get_input_embeddings()(batch.input_ids.to(device)), projected,
            batch.audio_placeholder_mask.to(device), mask.bool(),
        )
        generated = language.generate(inputs_embeds=embeddings, attention_mask=batch.attention_mask.to(device),
                                      do_sample=False, max_new_tokens=max_new_tokens)[0]
    eos = tokenizer.eos_token_id
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), int(generated.numel()), bool(eos is not None and eos in generated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--audio-model", required=True); parser.add_argument("--llm-model", required=True)
    parser.add_argument("--init-projector-checkpoint", type=Path, required=True); parser.add_argument("--init-projector-sha256", required=True)
    parser.add_argument("--init-projector-preflight", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=20); parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=128); parser.add_argument("--max-input-tokens", type=int, default=1536); parser.add_argument("--max-audio-tokens", type=int, default=512)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.samples_per_task <= 0: parser.error("--samples-per-task must be positive")
    import torch
    import deepspeed
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    deepspeed.init_distributed(); torch.cuda.set_device(args.local_rank)
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    rows, _ = load_joint_manifest(args.manifest, args.max_audio_tokens)
    selected = [row for task_index, task in enumerate(TASK_WEIGHTS) for row in select_rows(rows, task, args.samples_per_task, args.seed + task_index)]
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    audio = AutoModelForCausalLM.from_pretrained(args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    encoder = audio.audio_encoder; del audio
    llm = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16)
    llm.config.use_cache = True
    model, _ = build_joint_model(encoder, llm, projector_checkpoint=args.init_projector_checkpoint,
        projector_sha256=args.init_projector_sha256, projector_preflight=args.init_projector_preflight, audio_model=args.audio_model)
    encoder = model.audio_encoder.to(f"cuda:{args.local_rank}", dtype=torch.float32).eval()
    encoder_state = torch.load(args.checkpoint_dir / "hybrid.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(encoder_state["encoder_trainable"], strict=False)
    core = build_projector_lora_core(model.audio_projector, model.language_model).to(f"cuda:{args.local_rank}", dtype=torch.bfloat16).eval()
    trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
    engine, _, _, _ = deepspeed.initialize(model=core, optimizer=torch.optim.AdamW(trainable, lr=1e-5), config={
        "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3, "offload_optimizer": {"device": "none"}, "offload_param": {"device": "none"}},
    })
    loaded, _ = engine.load_checkpoint(str(args.checkpoint_dir), tag="hybrid.pt.core",
                                       load_optimizer_states=False, load_lr_scheduler_states=False)
    if loaded is None: raise RuntimeError("failed to load ZeRO-3 Core checkpoint")
    records = []
    for index, row in enumerate(selected):
        if index % world != rank: continue
        batch = make_batch(row, tokenizer, int(placeholder_id), args.max_input_tokens, args.max_audio_tokens)
        text, tokens, eos = generate(engine.module, encoder, batch, tokenizer, args.max_new_tokens)
        records.append({"sample_id": str(row["sample_id"]), "task": classify_task(row), "reference": row["user_text"] if classify_task(row) == "librispeech" else row["assistant_response"], "generation": text, "generated_tokens": tokens, "generated_eos": eos})
    gathered: list[list[dict[str, Any]] | None] = [None] * world
    torch.distributed.all_gather_object(gathered, records)
    if rank == 0:
        output = [item for partition in gathered for item in partition]
        output.sort(key=lambda item: (item["task"], item["sample_id"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checkpoint_dir": str(args.checkpoint_dir), "manifest": str(args.manifest), "samples_per_task": args.samples_per_task, "seed": args.seed, "records": output}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("JOINT_GENERATION_COMPLETE=" + json.dumps({"records": len(output), "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
