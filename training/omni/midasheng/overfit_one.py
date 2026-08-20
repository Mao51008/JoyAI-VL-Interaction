"""Overfit one LibriSpeech sample with only the MiDasheng-to-JoyAI adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.evaluate_overfit import _generate, _prompt_batch
from training.omni.projector_stage2.train import CachedConversationBatchSource, load_manifest

from .model import build_phase1_model
from .official_projector import build_frozen_projector_adapter


LIBRISPEECH_DATASET = "LibriSpeech"


def _select_row(rows: list[dict[str, Any]], sample_id: str | None) -> dict[str, Any]:
    candidates = [
        row for row in rows if row.get("provenance", {}).get("dataset") == LIBRISPEECH_DATASET
    ]
    if sample_id is not None:
        candidates = [row for row in candidates if row["sample_id"] == sample_id]
    if not candidates:
        raise ValueError("no matching LibriSpeech sample in manifest")
    return sorted(candidates, key=lambda row: str(row["sample_id"]))[0]


def _greedy_generation(model: Any, batch: Any, tokenizer: Any, max_new_tokens: int) -> tuple[str, int, bool]:
    import torch

    model.eval()
    with torch.inference_mode():
        return _generate(model, _prompt_batch(batch), tokenizer, max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.steps <= 0 or args.check_every <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--steps, --check-every, and --max-new-tokens must be positive")
    if args.learning_rate != 1e-4:
        raise ValueError("this overfit gate requires --learning-rate 1e-4")

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    row = _select_row(load_manifest(args.manifest), args.sample_id)
    validate_feature_cache(args.feature_dir, [row])
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("JoyAI tokenizer has no <|vision_pad|> placeholder")
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
    ).to(args.device, dtype=torch.bfloat16)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = {"core.audio_projector.joyai_adapter.weight", "core.audio_projector.joyai_adapter.bias"}
    if {name for name, _ in trainable} != expected:
        raise AssertionError(f"unexpected trainable parameters: {[name for name, _ in trainable]}")
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.learning_rate)

    reference = str(row["user_text"])
    initial_text, initial_tokens, initial_eos = _greedy_generation(
        model, batch, tokenizer, args.max_new_tokens
    )
    records: list[dict[str, Any]] = [{
        "step": 0,
        "train_ce": None,
        "generation": initial_text,
        "generated_tokens": initial_tokens,
        "generated_eos": initial_eos,
        "exact_match": initial_text == reference,
    }]
    first_exact_step: int | None = 0 if initial_text == reference else None
    final_loss: float | None = None
    for step in range(1, args.steps + 1):
        if first_exact_step is not None:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if step % args.check_every != 0 and step != args.steps:
            continue
        text, tokens, eos = _greedy_generation(model, batch, tokenizer, args.max_new_tokens)
        exact_match = text == reference
        records.append({
            "step": step,
            "train_ce": final_loss,
            "generation": text,
            "generated_tokens": tokens,
            "generated_eos": eos,
            "exact_match": exact_match,
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        if exact_match:
            first_exact_step = step
            break
    final = records[-1]
    report = {
        "architecture": "frozen_encoder_official_projector_trainable_3584_to_4096_adapter_frozen_joyai",
        "sample_id": row["sample_id"],
        "reference": reference,
        "learning_rate": args.learning_rate,
        "max_steps": args.steps,
        "check_every": args.check_every,
        "initial_generation": initial_text,
        "first_exact_match_step": first_exact_step,
        "final_loss": final_loss,
        "final_generation": final["generation"],
        "final_exact_match": final["exact_match"],
        "records": records,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
