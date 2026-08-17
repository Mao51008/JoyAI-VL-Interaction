"""Greedy free generation for cached MiDasheng projector-only checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.evaluate_overfit import _generate, _prompt_batch
from training.omni.projector_stage2.train import CachedConversationBatchSource, load_manifest

from .model import build_phase1_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    rows = load_manifest(args.manifest)
    validate_feature_cache(args.feature_dir, rows)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    model = build_phase1_model(AutoModelForImageTextToText.from_pretrained(
        args.llm_model, dtype=torch.bfloat16
    )).to(args.device, dtype=torch.bfloat16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(state["trainable_state"], strict=False)
    source = CachedConversationBatchSource(rows, tokenizer, args.feature_dir, int(placeholder_id), 1, 8)
    records = []
    with torch.inference_mode():
        for row, batch in zip(rows, source, strict=True):
            text, tokens, eos = _generate(model, _prompt_batch(batch), tokenizer, args.max_new_tokens)
            records.append({"sample_id": row["sample_id"], "task": row["training_task"],
                            "reference": row["user_text"] if row["training_task"] == "asr_transcription" else row["assistant_response"],
                            "generation": text, "generated_tokens": tokens, "generated_eos": eos})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
