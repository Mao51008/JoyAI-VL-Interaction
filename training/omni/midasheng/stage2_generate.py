"""Free-generate and score a completed official-projector Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training.omni.projector_stage1.evaluate_wer import _normalize as normalize_transcript
from training.omni.projector_stage1.evaluate_wer import score_transcript
from training.omni.projector_stage2.evaluate_overfit import _generate, _prompt_batch
from training.omni.projector_stage2.prepare_clotho_high_confidence import normalize_answer
from training.omni.projector_stage2.train import (
    CachedConversationBatchSource,
    inject_lora,
    load_manifest,
)

from .model import build_official_projector_lora_model
from .official_projector import build_official_projector_adapter
from .stage2_train import TASK_DATASETS, language_all_linear_targets


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, generation: str) -> float:
    reference_tokens, generation_tokens = reference.lower().split(), generation.lower().split()
    if not reference_tokens or not generation_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, generation_tokens)
    precision, recall = lcs / len(generation_tokens), lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _task_rows(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("provenance", {}).get("dataset") == TASK_DATASETS[task]]
    if not selected:
        raise ValueError(f"manifest contains no {task} rows")
    return selected


def _summary(task: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    eos_rate = sum(record["generated_eos"] for record in records) / len(records)
    average_tokens = sum(record["generated_tokens"] for record in records) / len(records)
    if task == "librispeech":
        return {
            "samples": len(records), "wer": sum(record["word_errors"] for record in records) / max(1, sum(record["reference_words"] for record in records)),
            "cer": sum(record["char_errors"] for record in records) / max(1, sum(record["reference_chars"] for record in records)),
            "exact_match": sum(record["exact_match"] for record in records) / len(records),
            "eos_rate": eos_rate, "average_generated_tokens": average_tokens,
        }
    if task == "voiceassistant":
        return {
            "samples": len(records), "macro_rouge_l_f1": sum(record["rouge_l_f1"] for record in records) / len(records),
            "eos_rate": eos_rate, "average_generated_tokens": average_tokens,
        }
    return {
        "samples": len(records), "normalized_accuracy": sum(record["normalized_exact"] for record in records) / len(records),
        "strict_exact": sum(record["strict_exact"] for record in records) / len(records),
        "eos_rate": eos_rate, "average_generated_tokens": average_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--task", choices=tuple(TASK_DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") not in {"projector-stage2-v3", "projector-stage2-inference-v1"}:
        raise ValueError(f"unsupported checkpoint format: {state.get('format')!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    audio = AutoModelForCausalLM.from_pretrained(args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    projector = build_official_projector_adapter(audio.audio_projector, freeze_official=False)
    del audio
    llm = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16)
    inject_lora(llm, language_all_linear_targets(llm), 8, 16.0)
    model = build_official_projector_lora_model(llm, audio_projector=projector).to(args.device, dtype=torch.bfloat16).eval()
    zero3 = state.get("zero3")
    if isinstance(zero3, dict):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tag = zero3.get("tag")
        if not isinstance(tag, str) or zero3.get("public_checkpoint") != args.checkpoint.name:
            raise ValueError("checkpoint has an invalid ZeRO-3 association")
        trainable_state = get_fp32_state_dict_from_zero_checkpoint(str(args.checkpoint.parent), tag=tag)
    else:
        trainable_state = state["trainable_state"]
    incompatible = model.load_state_dict(trainable_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"checkpoint tensor mismatch: {incompatible}")

    rows = _task_rows(load_manifest(args.manifest), args.task)
    source = CachedConversationBatchSource(rows, tokenizer, args.feature_dir, int(placeholder_id), 1, 8, audio_token_factor=5)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row, batch in zip(rows, source, strict=True):
            generation, tokens, eos = _generate(model, _prompt_batch(batch), tokenizer, args.max_new_tokens)
            reference = str(row["user_text"] if args.task == "librispeech" else row["assistant_response"])
            record: dict[str, Any] = {
                "sample_id": str(row["sample_id"]), "task": args.task, "reference": reference,
                "generation": generation, "generated_tokens": tokens, "generated_eos": eos,
            }
            if args.task == "librispeech":
                record.update(score_transcript(normalize_transcript(reference), normalize_transcript(generation)))
            elif args.task == "voiceassistant":
                record["rouge_l_f1"] = rouge_l_f1(reference, generation)
            else:
                record["normalized_exact"] = normalize_answer(reference) == normalize_answer(generation)
                record["strict_exact"] = reference.strip() == generation.strip()
            records.append(record)
            print(json.dumps({"sample_id": record["sample_id"], "count": len(records)}, ensure_ascii=False), flush=True)
    result = {"format": "midasheng-stage2-generation-v1", "checkpoint": str(args.checkpoint), "task": args.task,
              "max_new_tokens": args.max_new_tokens, "summary": _summary(args.task, records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("STAGE2_GENERATION_SUMMARY=" + json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
