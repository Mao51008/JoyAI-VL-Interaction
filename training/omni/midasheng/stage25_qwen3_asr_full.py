"""Transcribe the full VoiceAssistant manifest with the validated Qwen3-ASR path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    completed: set[str] = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        completed = {str(row["sample_id"]) for row in read_jsonl(args.output)}
        expected = {str(row["sample_id"]) for row in rows}
        if not completed.issubset(expected):
            raise ValueError("resume output contains IDs outside the manifest")

    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        str(args.model), device_map=args.device, dtype=torch.bfloat16,
        max_inference_batch_size=1, max_new_tokens=512,
    )
    captured: list[dict[str, object]] = []
    original_generate = model.model.generate
    configured_eos = model.model.generation_config.eos_token_id
    eos_ids = {configured_eos} if isinstance(configured_eos, int) else set(configured_eos or [])

    def capture_generate(*call_args, **call_kwargs):
        result = original_generate(*call_args, **call_kwargs)
        input_ids = call_kwargs["input_ids"]
        continuation = result.sequences[:, input_ids.shape[1] :]
        captured.append({
            "tokens": int(continuation.shape[1]),
            "eos": bool(eos_ids and any(int(token) in eos_ids for token in continuation[0])),
        })
        return result

    model.model.generate = capture_generate
    pending = [row for row in rows if row["sample_id"] not in completed]
    with args.output.open("a" if completed else "x", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            audio_path = args.audio_root / str(row["source_audio_path"])
            before = len(captured)
            try:
                result = model.transcribe(str(audio_path), language="English")[0]
                parts = captured[before:]
                tokens = sum(int(part["tokens"]) for part in parts)
                eos = bool(parts and all(bool(part["eos"]) for part in parts))
                error = None
                text = result.text.strip()
            except Exception as exc:
                text, tokens, eos = "", None, False
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "sample_id": row["sample_id"], "audio_path": str(audio_path),
                "audio_id": row["source_audio_path"], "transcript": text,
                "transcript_source": "qwen3_asr_1.7b", "generated_tokens": tokens,
                "eos": eos, "max_token_limit": tokens is not None and tokens >= 512,
                "generation": {"language": "English", "max_new_tokens": 512, "backend": "transformers"},
            }
            if error:
                record["error"] = error
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"QWEN_FULL {index}/{len(pending)} {row['sample_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
