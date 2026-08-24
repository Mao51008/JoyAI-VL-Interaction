"""Prepare VoiceAssistant pseudo-transcripts and frozen text-teacher targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.omni.midasheng.evaluate_official_asr import OFFICIAL_ASR_PROMPT
from training.omni.midasheng.stage2_generate import rouge_l_f1
from training.omni.projector_stage2.train import CANONICAL_SYSTEM_PROMPT


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _shard(rows: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    return rows[index::count]


def _quality_flags(text: str, tokens: int, eos: bool) -> list[str]:
    compact = "".join(text.split())
    flags: list[str] = []
    if not compact:
        flags.append("empty")
    if not eos:
        flags.append("missing_eos")
    if tokens >= 128:
        flags.append("max_token_limit")
    if compact and sum(character.isascii() and (character.isalpha() or character.isspace()) for character in text) / len(compact) < 0.7:
        flags.append("non_english_or_symbol_heavy")
    words = text.lower().split()
    if len(words) >= 12 and len(set(words[-8:])) <= 2:
        flags.append("repetitive_tail")
    return flags


def transcribe(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.audio_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    model.audio_encoder.float()
    model.audio_projector.float()
    tokenizer = AutoTokenizer.from_pretrained(args.audio_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.audio_model, trust_remote_code=True)
    with args.output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for position, row in enumerate(rows, start=1):
            audio_path = args.audio_root / str(row["source_audio_path"])
            messages = [{"role": "user", "content": [
                {"type": "text", "text": OFFICIAL_ASR_PROMPT},
                {"type": "audio", "path": str(audio_path)},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, add_special_tokens=True, return_dict=True
            )
            inputs = {name: value.to(args.device) if hasattr(value, "to") else value for name, value in inputs.items()}
            generated = model.generate(**inputs)
            transcript = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            token_count = int(generated.shape[-1])
            eos_id = tokenizer.eos_token_id
            eos = bool(eos_id is not None and eos_id in generated[0])
            record = {
                "sample_id": row["sample_id"], "audio_path": str(audio_path), "audio_id": row["source_audio_path"],
                "transcript": transcript, "transcript_source": "midasheng_pseudo", "generated_tokens": token_count,
                "eos": eos, "generation": {"prompt": OFFICIAL_ASR_PROMPT, "do_sample": False, "official_chat_template": True},
                "quality_flags": _quality_flags(transcript, token_count, eos),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"TRANSCRIPT {position}/{len(rows)} {record['sample_id']}", flush=True)


def teach(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    transcripts = {record["sample_id"]: record for record in _read_jsonl(args.transcripts)}
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in transcripts]
    if missing:
        raise ValueError(f"missing transcripts for {len(missing)} rows")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, fix_mistral_regex=True)
    model = AutoModelForImageTextToText.from_pretrained(args.llm_model, dtype=torch.bfloat16).to(args.device).eval()
    with args.output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for position, row in enumerate(rows, start=1):
            transcript = transcripts[row["sample_id"]]
            messages = [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": transcript["transcript"]},
            ]
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            input_ids = torch.tensor([prompt_ids], device=args.device)
            generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False, max_new_tokens=args.max_new_tokens)[0]
            continuation = generated[len(prompt_ids):]
            teacher = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            eos_id = tokenizer.eos_token_id
            eos = bool(eos_id is not None and eos_id in continuation)
            record = {
                "sample_id": row["sample_id"], "transcript": transcript["transcript"],
                "transcript_source": transcript["transcript_source"], "original_reference_answer": row["assistant_response"],
                "teacher_generation": teacher, "teacher_generated_tokens": int(continuation.numel()), "teacher_eos": eos,
                "chat_messages": messages, "text_prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "rouge_l_f1_vs_reference": rouge_l_f1(row["assistant_response"], teacher),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"TEACHER {position}/{len(rows)} {record['sample_id']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("transcribe", "teach"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--audio-model")
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--llm-model")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    rows = _shard(_read_jsonl(args.manifest), args.shard_index, args.shard_count)
    if args.mode == "transcribe":
        if args.audio_root is None or not args.audio_model:
            parser.error("transcribe requires --audio-root and --audio-model")
        transcribe(args, rows)
    else:
        if args.transcripts is None or not args.llm_model:
            parser.error("teach requires --transcripts and --llm-model")
        teach(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
