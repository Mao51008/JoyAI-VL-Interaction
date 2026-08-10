"""Evaluate a Stage2 checkpoint on ASR or dialogue-generation overfit gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .cache_features import validate_feature_cache
from .train import (
    FROZEN_STAGE1_PROJECTOR_SHA256,
    CachedConversationBatchSource,
    Stage2ConversationBatch,
    build_model_from_pretrained,
    load_manifest,
)


def _normalise(text: str) -> str:
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in text.upper()).split())


def summarise_records(records: Sequence[dict[str, Any]], task: str) -> dict[str, Any]:
    """Aggregate task-specific generation gates without hiding invalid outputs."""
    if not records:
        raise ValueError("cannot summarise empty records")
    summary: dict[str, Any] = {
        "samples": len(records),
        "exact_match_rate": sum(record["exact_match"] for record in records) / len(records),
        "eos_rate": sum(record["generated_eos"] for record in records) / len(records),
        "empty_rate": sum(not record["hypothesis"] for record in records) / len(records),
        "bare_user_rate": sum(record["bare_user"] for record in records) / len(records),
        "user_prefix_rate": sum(record["user_prefix"] for record in records) / len(records),
        "wrapper_rate": sum(record["wrapper"] for record in records) / len(records),
    }
    if task == "asr_transcription":
        word_errors = sum(int(record["word_errors"]) for record in records)
        words = sum(int(record["reference_words"]) for record in records)
        char_errors = sum(int(record["char_errors"]) for record in records)
        chars = sum(int(record["reference_chars"]) for record in records)
        summary.update(
            {
                "word_errors": word_errors,
                "reference_words": words,
                "wer": word_errors / max(1, words),
                "char_errors": char_errors,
                "reference_chars": chars,
                "cer": char_errors / max(1, chars),
            }
        )
    return summary


def _checkpoint_lora_layout(state: dict[str, Any]) -> tuple[list[str], int]:
    lora_targets = list(state.get("lora_targets") or [])
    lora_tensors = {
        name: tensor
        for name, tensor in state["trainable_state"].items()
        if ".lora_" in name
    }
    if not lora_tensors:
        return [], 8
    if not lora_targets:
        lora_targets = sorted(
            {
                name.removeprefix("core.language_model.").rsplit(".", 1)[0]
                for name in lora_tensors
            }
        )
    ranks = {
        int(tensor.shape[0] if name.endswith(".lora_A") else tensor.shape[1])
        for name, tensor in lora_tensors.items()
    }
    if len(ranks) != 1:
        raise ValueError(f"inconsistent checkpoint LoRA ranks: {sorted(ranks)}")
    return lora_targets, ranks.pop()


def _prompt_batch(batch: Stage2ConversationBatch) -> Stage2ConversationBatch:
    target_positions = batch.labels[0].ne(-100).nonzero(as_tuple=False)
    if not len(target_positions):
        raise ValueError(f"{batch.sample_ids}: no supervised target")
    target_start = int(target_positions[0].item())
    return Stage2ConversationBatch(
        input_ids=batch.input_ids[:, :target_start],
        labels=batch.labels[:, :target_start],
        attention_mask=batch.attention_mask[:, :target_start],
        sample_ids=batch.sample_ids,
        dialogue_ids=batch.dialogue_ids,
        audio_features=batch.audio_features,
        audio_attention_mask=batch.audio_attention_mask,
        audio_placeholder_mask=batch.audio_placeholder_mask[:, :target_start],
        task_types=batch.task_types,
    )


def _generate(model: Any, batch: Stage2ConversationBatch, tokenizer: Any, max_new_tokens: int) -> tuple[str, int, bool]:
    from ..projector_stage1.model import replace_audio_placeholders

    core_model = getattr(model, "module", model)
    language_model = core_model.core.language_model
    device = next(core_model.parameters()).device
    input_ids = batch.input_ids.to(device)
    audio = core_model.core.audio_projector(
        batch.audio_features.to(device=device, dtype=next(core_model.parameters()).dtype)
    )
    text_embeddings = language_model.get_input_embeddings()(input_ids)
    embeddings = replace_audio_placeholders(
        text_embeddings,
        audio,
        input_ids.eq(tokenizer.convert_tokens_to_ids("<|vision_pad|>")),
        batch.audio_attention_mask.to(device),
    )
    generated = language_model.generate(
        inputs_embeds=embeddings,
        attention_mask=batch.attention_mask.to(device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )[0]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    eos_token_id = tokenizer.eos_token_id
    return text, int(generated.numel()), bool(eos_token_id is not None and eos_token_id in generated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-sha256", default=FROZEN_STAGE1_PROJECTOR_SHA256)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--projector-in-features", type=int, required=True)
    parser.add_argument("--projector-out-features", type=int, required=True)
    parser.add_argument("--task", choices=("asr_transcription", "dialogue_response"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    import torch
    import transformers

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError(f"unsupported checkpoint format: {state.get('format')!r}")
    lora_targets, lora_rank = _checkpoint_lora_layout(state)
    rows = [dict(row, training_task=args.task) for row in load_manifest(args.manifest)]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    validate_feature_cache(args.feature_dir, rows)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    model = build_model_from_pretrained(
        args.llm_model,
        args.projector_in_features,
        args.projector_out_features,
        lora_targets,
        lora_rank,
        args.lora_alpha,
        args.stage1_checkpoint,
        args.stage1_sha256,
        args.device,
        args.dtype,
    ).eval()
    incompatible = model.load_state_dict(state["trainable_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint tensors: {incompatible.unexpected_keys[:5]}")

    from training.omni.projector_stage1.evaluate_wer import score_transcript

    records = []
    source = CachedConversationBatchSource(
        rows, tokenizer, args.feature_dir, int(placeholder_id), batch_size=1, max_cached_shards=8
    )
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    progress = (
        tqdm(source, total=len(source), desc=args.task, unit="sample")
        if tqdm is not None and not args.no_progress
        else source
    )
    with torch.inference_mode():
        for batch in progress:
            prompt = _prompt_batch(batch)
            raw_text, token_count, eos = _generate(model, prompt, tokenizer, args.max_new_tokens)
            hypothesis = _normalise(raw_text)
            row = rows[len(records)]
            reference = _normalise(
                row["user_text"] if args.task == "asr_transcription" else row["assistant_response"]
            )
            lowered = raw_text.strip().lower()
            record = {
                "sample_id": batch.sample_ids[0],
                "dialogue_id": batch.dialogue_ids[0],
                "task": args.task,
                "reference": reference,
                "hypothesis": hypothesis,
                "generated_tokens": token_count,
                "generated_eos": eos,
                "exact_match": hypothesis == reference,
                "bare_user": lowered == "user",
                "user_prefix": lowered.startswith(("user\n", "user ")),
                "wrapper": "<answer>" in lowered or "<function=" in lowered,
            }
            if args.task == "asr_transcription":
                record.update(score_transcript(reference, hypothesis))
            records.append(record)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(samples=len(records), eos=int(eos))
    result = {
        "format": "projector-stage2-overfit-gate-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "task": args.task,
        "summary": summarise_records(records, args.task),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
