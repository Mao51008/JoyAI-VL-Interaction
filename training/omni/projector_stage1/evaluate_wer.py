"""Decode projector-stage-one checkpoints and report transcript WER/CER."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..schema import load_samples
from .ablation import apply_feature_ablation, apply_temporal_shuffle
from .checkpoint_loading import load_trusted_checkpoint
from .collator import JoyAIStage1TokenLayout, build_chat_parts
from .data import target_text
from .feature_cache import FeatureCache
from .sharding import (
    global_exchange_order,
    load_sharded_samples,
    result_metadata,
    temporal_seed,
)


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return insertion, deletion, substitution counts from a stable alignment."""
    # Each cell stores (total errors, insertions, deletions, substitutions).
    rows = [[(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)]
            for _ in range(len(reference) + 1)]
    for column in range(1, len(hypothesis) + 1):
        rows[0][column] = (column, column, 0, 0)
    for index in range(1, len(reference) + 1):
        rows[index][0] = (index, 0, index, 0)
    for index, value in enumerate(reference, start=1):
        for column, candidate in enumerate(hypothesis, start=1):
            insertion = rows[index][column - 1]
            deletion = rows[index - 1][column]
            substitution = rows[index - 1][column - 1]
            candidates = [
                (insertion[0] + 1, insertion[1] + 1, insertion[2], insertion[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (substitution[0] + (value != candidate), substitution[1], substitution[2],
                 substitution[3] + (value != candidate)),
            ]
            rows[index][column] = min(candidates, key=lambda item: (item[0], item[3], item[2], item[1]))
    _total, insertions, deletions, substitutions = rows[-1][-1]
    return insertions, deletions, substitutions


def _normalize(value: str) -> str:
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in value.upper()).split())


def score_transcript(reference: str, hypothesis: str) -> dict[str, int | float | bool]:
    """Return normalized transcript metrics used in per-sample evaluation output."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    reference_chars = list(reference.replace(" ", ""))
    hypothesis_chars = list(hypothesis.replace(" ", ""))
    insertions, deletions, substitutions = _edit_counts(reference_words, hypothesis_words)
    char_insertions, char_deletions, char_substitutions = _edit_counts(reference_chars, hypothesis_chars)
    word_errors = insertions + deletions + substitutions
    char_errors = char_insertions + char_deletions + char_substitutions
    return {
        "wer": word_errors / max(1, len(reference_words)),
        "cer": char_errors / max(1, len(reference_chars)),
        "word_errors": word_errors,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "char_errors": char_errors,
        "char_insertions": char_insertions,
        "char_deletions": char_deletions,
        "char_substitutions": char_substitutions,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "exact_match": reference == hypothesis,
    }


def has_eos(generated_tokens, eos_token_id: int | None) -> bool:
    return eos_token_id is not None and eos_token_id in generated_tokens.tolist()


def canonical_ablation(mode: str) -> str:
    return {"none": "none", "zero": "waveform-zero", "feature-zero": "feature-zero",
            "projected-zero": "projected-zero", "shuffle": "cross-sample-shuffle",
            "temporal-shuffle": "within-sample-temporal-shuffle"}[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-ablation", choices=("none", "zero", "waveform-zero", "feature-zero", "projected-zero", "shuffle", "temporal-shuffle"), default="none")
    parser.add_argument(
        "--zero-feature-dir",
        type=Path,
        help="feature cache extracted from zero waveforms; required for --audio-ablation zero",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0: raise ValueError("--max-samples must be positive")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from .model import replace_audio_placeholders
    from .projector import AudioProjector, AudioProjectorConfig
    state = load_trusted_checkpoint(args.checkpoint, torch)
    tokenizer = AutoTokenizer.from_pretrained(args.joyai_model, fix_mistral_regex=True)
    layout = JoyAIStage1TokenLayout.from_tokenizer(tokenizer)
    llm = AutoModelForImageTextToText.from_pretrained(args.joyai_model, dtype=torch.bfloat16).to(args.device).eval()
    projector = AudioProjector(AudioProjectorConfig(**state["config"]["projector"])).to(args.device, dtype=torch.bfloat16).eval()
    projector.load_state_dict(state["projector"])
    manifest_samples, samples, global_indices = load_sharded_samples(
        args.manifest, args.max_samples, args.num_shards, args.shard_index
    )
    cache = FeatureCache(args.feature_dir)
    zero_cache = FeatureCache(args.zero_feature_dir) if args.zero_feature_dir is not None else None
    if args.audio_ablation in {"zero", "waveform-zero"} and zero_cache is None:
        raise ValueError("--zero-feature-dir is required for waveform-zero evaluation")
    context, assistant_prefix = build_chat_parts(tokenizer)
    all_features = [cache.get(sample.sample_id) for sample in manifest_samples]
    sample_indices = {sample.sample_id: index for index, sample in enumerate(manifest_samples)}
    if any(item["media_sha256"] != sample.metadata["media_sha256"] for item, sample in zip(all_features, manifest_samples)):
        raise ValueError("cached feature fingerprint mismatch")
    exchange_order = None
    if args.audio_ablation == "shuffle":
        exchange_order = global_exchange_order(len(manifest_samples), args.seed)
    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            global_index = global_indices[index]
            if args.audio_ablation in {"zero", "waveform-zero"}:
                cached = zero_cache.get(sample.sample_id)
                if not zero_cache.index[sample.sample_id].get("waveform_zero", False):
                    raise ValueError("zero feature cache was not extracted from zero waveforms")
                if cached["media_sha256"] != sample.metadata["media_sha256"]:
                    raise ValueError("zero feature cache fingerprint mismatch")
            elif args.audio_ablation == "shuffle":
                cached = all_features[exchange_order[global_index]]
            else:
                cached = all_features[global_index]
            feature = cached["features"]
            if args.audio_ablation == "feature-zero":
                feature = apply_feature_ablation(feature, "feature-zero")
            if args.audio_ablation == "temporal-shuffle":
                feature = apply_temporal_shuffle(feature, random.Random(temporal_seed(args.seed, global_index)))
            feature = feature.unsqueeze(0).to(args.device, dtype=torch.bfloat16)
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
            speaker = sample.metadata.get("speaker") or sample.sample_id.split("-")[1]
            row = {"sample_id": sample.sample_id, "global_index": global_index,
                   "speaker": str(speaker),
                   "duration_ms": sample.duration_ms, "reference": reference, "hypothesis": hypothesis,
                   "generated_tokens": int(generated_tokens.numel()),
                   "generated_eos": has_eos(generated_tokens, tokenizer.eos_token_id)}
            if args.audio_ablation == "shuffle":
                row["feature_donor_sample_id"] = manifest_samples[exchange_order[global_index]].sample_id
            row.update(score_transcript(reference, hypothesis))
            rows.append(row)
    word_errors = sum(int(row["word_errors"]) for row in rows)
    word_total = sum(len(row["reference"].split()) for row in rows)
    char_errors = sum(int(row["char_errors"]) for row in rows)
    char_total = sum(len(row["reference"].replace(" ", "")) for row in rows)
    result = result_metadata(
        manifest=args.manifest, manifest_sample_count=len(manifest_samples),
        evaluation_sample_count=(len(manifest_samples) if args.max_samples is None
                                 else min(args.max_samples, len(manifest_samples))), indices=global_indices,
        num_shards=args.num_shards, shard_index=args.shard_index,
        checkpoint=args.checkpoint, ablation=canonical_ablation(args.audio_ablation), seed=args.seed,
    )
    result.update({"samples": len(rows), "word_errors": word_errors, "reference_words": word_total,
                   "char_errors": char_errors, "reference_chars": char_total,
                   "wer": word_errors / max(1, word_total), "cer": char_errors / max(1, char_total), "rows": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
