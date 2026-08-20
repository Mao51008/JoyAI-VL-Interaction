"""Print token-level teacher-forcing predictions for one saved adapter checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.train import CachedConversationBatchSource, load_manifest

from .model import build_phase1_model
from .official_projector import build_frozen_projector_adapter
from .overfit_one import _select_row


def _token_text(tokenizer: object, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    row = _select_row(load_manifest(args.manifest), args.sample_id)
    validate_feature_cache(args.feature_dir, [row])
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    source = CachedConversationBatchSource(
        [row], tokenizer, args.feature_dir, int(placeholder_id), 1, 8, audio_token_factor=5
    )
    batch = next(iter(source))
    audio_model = AutoModelForCausalLM.from_pretrained(
        args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    projector = build_frozen_projector_adapter(audio_model.audio_projector)
    del audio_model
    model = build_phase1_model(
        AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16),
        audio_projector=projector,
    ).to(args.device, dtype=torch.bfloat16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError("unsupported checkpoint format")
    incompatible = model.load_state_dict(state["trainable_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint tensors: {incompatible.unexpected_keys}")
    with torch.inference_mode():
        output = model(batch)
    labels = batch.labels[0]
    positions = labels.ne(-100).nonzero(as_tuple=False).flatten().tolist()
    records = []
    for ordinal, label_position in enumerate(positions):
        if label_position == 0:
            raise ValueError("assistant target cannot occupy position zero")
        prediction_position = label_position - 1
        logits = output.logits[0, prediction_position].float()
        reference_id = int(labels[label_position])
        argmax_id = int(logits.argmax())
        reference_probability = float(torch.softmax(logits, dim=-1)[reference_id])
        record = {
            "assistant_token_ordinal": ordinal,
            "label_position": label_position,
            "prediction_position": prediction_position,
            "reference_token_id": reference_id,
            "reference_token": _token_text(tokenizer, reference_id),
            "argmax_token_id": argmax_id,
            "argmax_token": _token_text(tokenizer, argmax_id),
            "reference_probability": reference_probability,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "sample_id": row["sample_id"],
        "reference": row["user_text"],
        "assistant_positions": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
