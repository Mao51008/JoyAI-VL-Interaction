"""Decode projector-stage-one checkpoints and report transcript WER/CER."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..schema import load_samples
from .collator import JoyAIStage1TokenLayout, build_chat_parts
from .data import target_text
from .feature_cache import FeatureCache


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, value in enumerate(reference, start=1):
        current = [index]
        for column, candidate in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (value != candidate)))
        previous = current
    return previous[-1]


def _normalize(value: str) -> str:
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in value.upper()).split())


def score_transcript(reference: str, hypothesis: str) -> dict[str, int | float | bool]:
    """Return normalized transcript metrics used in per-sample evaluation output."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    reference_chars = list(reference.replace(" ", ""))
    hypothesis_chars = list(hypothesis.replace(" ", ""))
    word_errors = _distance(reference_words, hypothesis_words)
    char_errors = _distance(reference_chars, hypothesis_chars)
    return {
        "wer": word_errors / max(1, len(reference_words)),
        "cer": char_errors / max(1, len(reference_chars)),
        "word_errors": word_errors,
        "char_errors": char_errors,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "exact_match": reference == hypothesis,
    }


def has_eos(generated_tokens, eos_token_id: int | None) -> bool:
    return eos_token_id is not None and eos_token_id in generated_tokens.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-ablation", choices=("none", "zero", "projected-zero", "shuffle"), default="none")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0: raise ValueError("--max-samples must be positive")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from .model import replace_audio_placeholders
    from .projector import AudioProjector, AudioProjectorConfig
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.joyai_model, fix_mistral_regex=True)
    layout = JoyAIStage1TokenLayout.from_tokenizer(tokenizer)
    llm = AutoModelForImageTextToText.from_pretrained(args.joyai_model, dtype=torch.bfloat16).to(args.device).eval()
    projector = AudioProjector(AudioProjectorConfig(**state["config"]["projector"])).to(args.device, dtype=torch.bfloat16).eval()
    projector.load_state_dict(state["projector"])
    samples = load_samples(args.manifest)[:args.max_samples]
    cache = FeatureCache(args.feature_dir); context, assistant_prefix = build_chat_parts(tokenizer)
    rows = []; randomizer = random.Random(args.seed)
    with torch.inference_mode():
        for sample in samples:
            feature = cache.get(sample.sample_id)["features"].unsqueeze(0).to(args.device, dtype=torch.bfloat16)
            if args.audio_ablation == "zero": feature.zero_()
            elif args.audio_ablation == "shuffle": feature = feature[:, randomizer.sample(range(feature.shape[1]), feature.shape[1])]
            input_ids = context + [layout.audio_start_id] + [layout.audio_placeholder_id] * feature.shape[1] + [layout.audio_end_id] + assistant_prefix
            ids = torch.tensor([input_ids], device=args.device)
            audio = projector(feature)
            if args.audio_ablation == "projected-zero": audio.zero_()
            text = llm.get_input_embeddings()(ids)
            mask = ids.eq(layout.audio_placeholder_id)
            embeds = replace_audio_placeholders(text, audio, mask, torch.ones(audio.shape[:2], device=args.device, dtype=torch.bool))
            generated = llm.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens, do_sample=False)
            generated_tokens = generated[0]
            hypothesis = _normalize(tokenizer.decode(generated_tokens, skip_special_tokens=True))
            reference = _normalize(target_text(sample))
            row = {"sample_id": sample.sample_id, "reference": reference, "hypothesis": hypothesis,
                   "generated_tokens": int(generated_tokens.numel()),
                   "generated_eos": has_eos(generated_tokens, tokenizer.eos_token_id)}
            row.update(score_transcript(reference, hypothesis))
            rows.append(row)
    word_errors = sum(_distance(row["reference"].split(), row["hypothesis"].split()) for row in rows)
    word_total = sum(len(row["reference"].split()) for row in rows)
    char_errors = sum(_distance(list(row["reference"].replace(" ", "")), list(row["hypothesis"].replace(" ", ""))) for row in rows)
    char_total = sum(len(row["reference"].replace(" ", "")) for row in rows)
    result = {"checkpoint": str(args.checkpoint), "audio_ablation": args.audio_ablation, "samples": len(rows), "wer": word_errors / max(1, word_total), "cer": char_errors / max(1, char_total), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
