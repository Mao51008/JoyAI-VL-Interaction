"""Compare stage-one encoder-projector-VLM transcription with Qwen3-ASR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..schema import load_samples
from .checkpoint_loading import load_trusted_checkpoint
from .collator import JoyAIStage1TokenLayout, _load_mono_audio, build_chat_parts
from .data import target_text
from .evaluate_wer import _normalize, has_eos, score_transcript
from .progress import progress_iter


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """Aggregate paired transcript rows using corpus-level WER/CER."""
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    summary: dict[str, int | float] = {"samples": len(rows)}
    for name in ("projector_vlm", "qwen_asr"):
        prefix = f"{name}_"
        word_errors = sum(int(row[prefix + "word_errors"]) for row in rows)
        words = sum(int(row[prefix + "reference_words"]) for row in rows)
        char_errors = sum(int(row[prefix + "char_errors"]) for row in rows)
        chars = sum(int(row[prefix + "reference_chars"]) for row in rows)
        latency = [float(row[prefix + "latency_ms"]) for row in rows]
        summary.update(
            {
                prefix + "word_errors": word_errors,
                prefix + "reference_words": words,
                prefix + "wer": word_errors / max(1, words),
                prefix + "char_errors": char_errors,
                prefix + "reference_chars": chars,
                prefix + "cer": char_errors / max(1, chars),
                prefix + "latency_ms_mean": sum(latency) / len(latency),
            }
        )
    summary["projector_vlm_minus_qwen_asr_wer"] = float(
        summary["projector_vlm_wer"]
    ) - float(summary["qwen_asr_wer"])
    summary["projector_vlm_minus_qwen_asr_cer"] = float(
        summary["projector_vlm_cer"]
    ) - float(summary["qwen_asr_cer"])
    return summary


def _projector_transcribe(
    *,
    waveform: Any,
    audio_model: Any,
    audio_processor: Any,
    projector: Any,
    llm: Any,
    tokenizer: Any,
    layout: JoyAIStage1TokenLayout,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int, bool]:
    import torch

    from .model import replace_audio_placeholders

    batch = (
        audio_processor(
            text=audio_processor.audio_token,
            audio=waveform,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        .to(audio_model.device)
        .to(audio_model.dtype)
    )
    features = (
        audio_model.thinker.get_audio_features(
            input_features=batch["input_features"],
            feature_attention_mask=batch["feature_attention_mask"],
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.bfloat16)
    )
    context, assistant_prefix = build_chat_parts(tokenizer)
    input_ids = (
        context
        + [layout.audio_start_id]
        + [layout.audio_placeholder_id] * features.shape[1]
        + [layout.audio_end_id]
        + assistant_prefix
    )
    ids = torch.tensor([input_ids], device=device)
    audio = projector(features)
    text_embeddings = llm.get_input_embeddings()(ids)
    embeds = replace_audio_placeholders(
        text_embeddings,
        audio,
        ids.eq(layout.audio_placeholder_id),
        torch.ones(audio.shape[:2], device=device, dtype=torch.bool),
    )
    generated = llm.generate(
        inputs_embeds=embeds,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )[0]
    return (
        _normalize(tokenizer.decode(generated, skip_special_tokens=True)),
        int(generated.numel()),
        has_eos(generated, tokenizer.eos_token_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qwen-language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    checkpoint = load_trusted_checkpoint(args.checkpoint, torch)
    audio_wrapper = Qwen3ASRModel.from_pretrained(
        args.audio_model,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    audio_model = audio_wrapper.model.eval()
    audio_processor = AutoProcessor.from_pretrained(
        args.audio_model, fix_mistral_regex=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.joyai_model, fix_mistral_regex=True)
    layout = JoyAIStage1TokenLayout.from_tokenizer(tokenizer)
    llm = (
        AutoModelForImageTextToText.from_pretrained(
            args.joyai_model, dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    from .projector import AudioProjector, AudioProjectorConfig

    projector = (
        AudioProjector(AudioProjectorConfig(**checkpoint["config"]["projector"]))
        .to(args.device, dtype=torch.bfloat16)
        .eval()
    )
    projector.load_state_dict(checkpoint["projector"])

    rows = []
    samples = load_samples(args.manifest)[: args.max_samples]
    with torch.inference_mode():
        progress = progress_iter(
            samples,
            total=len(samples),
            description="paired transcription",
            unit="sample",
            no_progress=args.no_progress,
        )
        for sample in progress:
            if len(sample.audio) != 1:
                raise ValueError(
                    f"{sample.sample_id}: expected exactly one audio segment"
                )
            audio_path = sample.audio[0].path
            waveform, rate = _load_mono_audio(audio_path)
            if rate != 16_000:
                raise ValueError(
                    f"{sample.sample_id}: expected 16 kHz audio, got {rate}"
                )
            started = time.perf_counter()
            projector_text, generated_tokens, generated_eos = _projector_transcribe(
                waveform=waveform,
                audio_model=audio_model,
                audio_processor=audio_processor,
                projector=projector,
                llm=llm,
                tokenizer=tokenizer,
                layout=layout,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            projector_latency_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            qwen_result = audio_wrapper.transcribe(
                audio=audio_path, language=args.qwen_language
            )[0]
            qwen_text = _normalize(qwen_result.text)
            qwen_latency_ms = (time.perf_counter() - started) * 1000
            reference = _normalize(target_text(sample))
            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "duration_ms": sample.duration_ms,
                "reference": reference,
                "projector_vlm_hypothesis": projector_text,
                "projector_vlm_latency_ms": projector_latency_ms,
                "projector_vlm_generated_tokens": generated_tokens,
                "projector_vlm_generated_eos": generated_eos,
                "qwen_asr_hypothesis": qwen_text,
                "qwen_asr_latency_ms": qwen_latency_ms,
                "qwen_asr_language": str(qwen_result.language),
            }
            row.update(
                {
                    "projector_vlm_" + key: value
                    for key, value in score_transcript(
                        reference, projector_text
                    ).items()
                }
            )
            row.update(
                {
                    "qwen_asr_" + key: value
                    for key, value in score_transcript(reference, qwen_text).items()
                }
            )
            rows.append(row)
    result = {
        "format": "projector-stage1-paired-transcription-v1",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "audio_model": args.audio_model,
        "joyai_model": args.joyai_model,
        "qwen_language": args.qwen_language,
        "max_new_tokens": args.max_new_tokens,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
