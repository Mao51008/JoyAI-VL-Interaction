"""Greedy free generation for cached MiDasheng projector-only checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.evaluate_overfit import _generate, _prompt_batch
from training.omni.projector_stage2.train import CachedConversationBatchSource, load_manifest

from .model import build_phase1_model
from .official_projector import build_frozen_projector_adapter


_TASK_DATASETS = {
    "voiceassistant": "shenyunhang/VoiceAssistant-400K",
    "librispeech": "LibriSpeech",
    "clotho_aqa": "Clotho-AQA",
}


def select_fixed_task_rows(
    rows: list[dict[str, Any]], samples_per_task: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically select the same number of held-out rows per source."""
    selected: list[dict[str, Any]] = []
    for offset, (task, dataset) in enumerate(_TASK_DATASETS.items()):
        candidates = [row for row in rows if row.get("provenance", {}).get("dataset") == dataset]
        if len(candidates) < samples_per_task:
            raise ValueError(
                f"{task} has {len(candidates)} rows, fewer than {samples_per_task} requested"
            )
        selected.extend(random.Random(seed + offset).sample(candidates, samples_per_task))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    rows = load_manifest(args.manifest)
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    rows = select_fixed_task_rows(rows, args.samples_per_task, args.seed)
    validate_feature_cache(args.feature_dir, rows)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    audio_model = AutoModelForCausalLM.from_pretrained(
        args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    audio_projector = build_frozen_projector_adapter(audio_model.audio_projector)
    del audio_model
    model = build_phase1_model(
        AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16),
        audio_projector=audio_projector,
    ).to(args.device, dtype=torch.bfloat16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(state["trainable_state"], strict=False)
    source = CachedConversationBatchSource(
        rows, tokenizer, args.feature_dir, int(placeholder_id), 1, 8, audio_token_factor=5
    )
    records = []
    with torch.inference_mode():
        for row, batch in zip(rows, source, strict=True):
            text, tokens, eos = _generate(model, _prompt_batch(batch), tokenizer, args.max_new_tokens)
            records.append({"sample_id": row["sample_id"], "task": row["training_task"],
                            "reference": row["user_text"] if row["training_task"] == "asr_transcription" else row["assistant_response"],
                            "generation": text, "generated_tokens": tokens, "generated_eos": eos})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "audio_model": args.audio_model,
        "samples_per_task": args.samples_per_task,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
