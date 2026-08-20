"""Compare teacher forcing, cached decoding, and ``generate`` for one adapter sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.omni.projector_stage1.model import replace_audio_placeholders
from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.evaluate_overfit import _prompt_batch
from training.omni.projector_stage2.train import CachedConversationBatchSource, load_manifest

from .model import build_phase1_model
from .official_projector import build_frozen_projector_adapter
from .overfit_one import _select_row


def _top_k(logits: Any, tokenizer: Any, count: int = 5) -> list[dict[str, Any]]:
    import torch

    probabilities = torch.softmax(logits.float(), dim=-1)
    values, ids = probabilities.topk(count)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "probability": float(value),
        }
        for value, token_id in zip(values.tolist(), ids.tolist(), strict=True)
    ]


def _prepare_embeddings(model: Any, batch: Any, tokenizer: Any) -> tuple[Any, Any, Any, Any]:
    """Mirror the adapter-only ``_generate`` prompt construction exactly."""
    core = getattr(model, "module", model)
    language_model = core.core.language_model
    device = next(core.parameters()).device
    input_ids = batch.input_ids.to(device)
    projector = core.core.audio_projector
    features = batch.audio_features.to(device=device, dtype=next(core.parameters()).dtype)
    raw_mask = batch.audio_attention_mask.to(device)
    audio, projected_mask = projector(features, raw_mask)
    placeholders = input_ids.eq(tokenizer.convert_tokens_to_ids("<|vision_pad|>"))
    embeddings = replace_audio_placeholders(
        language_model.get_input_embeddings()(input_ids), audio, placeholders, projected_mask
    )
    return input_ids, embeddings, placeholders, projected_mask


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
    full_batch = next(iter(source))
    prompt_batch = _prompt_batch(full_batch)
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
    model.load_state_dict(state["trainable_state"], strict=False)
    core = model.core
    language_model = core.language_model
    device = next(model.parameters()).device
    full_ids, full_embeddings, full_placeholders, full_audio_mask = _prepare_embeddings(
        model, full_batch, tokenizer
    )
    prompt_ids, prompt_embeddings, prompt_placeholders, prompt_audio_mask = _prepare_embeddings(
        model, prompt_batch, tokenizer
    )
    target_start = int(full_batch.labels[0].ne(-100).nonzero(as_tuple=False)[0].item())
    if not torch.equal(prompt_ids, full_ids[:, :target_start]):
        raise AssertionError("prompt input_ids differ from teacher-forcing prefix")
    if not torch.equal(prompt_embeddings, full_embeddings[:, :target_start]):
        raise AssertionError("prompt embeddings differ from teacher-forcing prefix")
    with torch.inference_mode():
        full_output = model(full_batch)
        direct_full_output = language_model(
            inputs_embeds=full_embeddings,
            attention_mask=full_batch.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
        direct_labeled_output = language_model(
            inputs_embeds=full_embeddings,
            attention_mask=full_batch.attention_mask.to(device),
            labels=full_batch.labels.to(device),
            return_dict=True,
        )
        direct_labeled_no_cache_output = language_model(
            inputs_embeds=full_embeddings,
            attention_mask=full_batch.attention_mask.to(device),
            labels=full_batch.labels.to(device),
            use_cache=False,
            return_dict=True,
        )
        prompt_output = language_model(
            inputs_embeds=prompt_embeddings,
            attention_mask=prompt_batch.attention_mask.to(device),
            use_cache=True,
            return_dict=True,
        )
        first_logits = prompt_output.logits[0, -1]
        first_token = int(first_logits.argmax())
        extended_attention = torch.cat(
            [prompt_batch.attention_mask.to(device), torch.ones((1, 1), device=device, dtype=prompt_batch.attention_mask.dtype)],
            dim=1,
        )
        cached_output = language_model(
            input_ids=torch.tensor([[first_token]], device=device),
            attention_mask=extended_attention,
            past_key_values=prompt_output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        second_logits = cached_output.logits[0, -1]
        generated = language_model.generate(
            inputs_embeds=prompt_embeddings,
            attention_mask=prompt_batch.attention_mask.to(device),
            do_sample=False,
            max_new_tokens=2,
            return_dict_in_generate=True,
            output_scores=True,
        )
    labels = full_batch.labels[0]
    reference_first = int(labels[target_start])
    reference_second = int(labels[target_start + 1])
    eos_id = tokenizer.eos_token_id
    generated_tokens = generated.sequences[0].tolist()
    result = {
        "sample_id": row["sample_id"],
        "reference": row["user_text"],
        "chatml": {
            "full_length": int(full_ids.shape[1]),
            "prompt_length": int(prompt_ids.shape[1]),
            "target_start": target_start,
            "full_prefix_matches_prompt": True,
            "prompt_last_input_token": tokenizer.decode([int(prompt_ids[0, -1])], skip_special_tokens=False),
            "reference_first": tokenizer.decode([reference_first], skip_special_tokens=False),
            "reference_second": tokenizer.decode([reference_second], skip_special_tokens=False),
        },
        "audio": {
            "raw_audio_tokens": int(full_batch.audio_attention_mask[0].sum()),
            "projected_audio_tokens": int(full_audio_mask[0].sum()),
            "full_placeholder_tokens": int(full_placeholders[0].sum()),
            "prompt_placeholder_tokens": int(prompt_placeholders[0].sum()),
            "full_projected_mask_equals_prompt": bool(torch.equal(full_audio_mask, prompt_audio_mask)),
        },
        "teacher_forcing": {
            "first_top_k": _top_k(full_output.logits[0, target_start - 1], tokenizer),
            "second_top_k": _top_k(full_output.logits[0, target_start], tokenizer),
            "direct_full_first_top_k": _top_k(
                direct_full_output.logits[0, target_start - 1], tokenizer
            ),
            "direct_labeled_first_top_k": _top_k(
                direct_labeled_output.logits[0, target_start - 1], tokenizer
            ),
            "direct_labeled_no_cache_first_top_k": _top_k(
                direct_labeled_no_cache_output.logits[0, target_start - 1], tokenizer
            ),
            "full_vs_prompt_first_logit_max_abs": float(
                (full_output.logits[0, target_start - 1].float() - first_logits.float()).abs().max()
            ),
            "model_vs_direct_full_first_logit_max_abs": float(
                (
                    full_output.logits[0, target_start - 1].float()
                    - direct_full_output.logits[0, target_start - 1].float()
                ).abs().max()
            ),
            "model_vs_direct_labeled_first_logit_max_abs": float(
                (
                    full_output.logits[0, target_start - 1].float()
                    - direct_labeled_output.logits[0, target_start - 1].float()
                ).abs().max()
            ),
            "direct_labeled_vs_no_cache_first_logit_max_abs": float(
                (
                    direct_labeled_output.logits[0, target_start - 1].float()
                    - direct_labeled_no_cache_output.logits[0, target_start - 1].float()
                ).abs().max()
            ),
        },
        "language_model": {
            "class": type(language_model).__name__,
            "is_encoder_decoder": bool(getattr(language_model.config, "is_encoder_decoder", False)),
            "is_decoder": bool(getattr(language_model.config, "is_decoder", False)),
            "attn_implementation": getattr(language_model.config, "_attn_implementation", None),
        },
        "manual_cache": {
            "first_top_k": _top_k(first_logits, tokenizer),
            "first_token": tokenizer.decode([first_token], skip_special_tokens=False),
            "second_top_k": _top_k(second_logits, tokenizer),
            "second_token": tokenizer.decode([int(second_logits.argmax())], skip_special_tokens=False),
            "attention_lengths": [int(prompt_batch.attention_mask.sum()), int(extended_attention.sum())],
            "past_key_values_present": prompt_output.past_key_values is not None,
        },
        "hf_generate": {
            "sequence_token_ids": generated_tokens,
            "sequence_tokens": [tokenizer.decode([token], skip_special_tokens=False) for token in generated_tokens],
            "first_top_k": _top_k(generated.scores[0][0], tokenizer),
            "second_top_k": _top_k(generated.scores[1][0], tokenizer) if len(generated.scores) > 1 else [],
            "eos_probability_first": float(torch.softmax(generated.scores[0][0].float(), dim=-1)[eos_id]),
            "eos_probability_second": (
                float(torch.softmax(generated.scores[1][0].float(), dim=-1)[eos_id])
                if len(generated.scores) > 1 else None
            ),
            "score_steps": len(generated.scores),
            "stopped_on_eos": bool(eos_id in generated_tokens),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
