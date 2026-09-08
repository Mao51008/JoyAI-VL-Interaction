"""Run a curated VoiceAssistant Qwen3-ASR comparison pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TARGET_IDS = {
    "voiceassistant:0039314", "voiceassistant:0250062", "voiceassistant:0204465",
    "voiceassistant:0191573", "voiceassistant:0032429", "voiceassistant:0219348",
}
KEYWORDS = ("disney", "carmine", "target", "spell", "paraphrase", "vaccination", "date", "number")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midasheng-transcripts", nargs="+", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_jsonl(args.manifest)
    old = {row["sample_id"]: row for path in args.midasheng_transcripts for row in read_jsonl(path)}
    ranked = sorted(
        manifest,
        key=lambda row: (
            0 if row["sample_id"] in TARGET_IDS else 1,
            0 if any(word in (old.get(row["sample_id"], {}).get("transcript", "").lower()) for word in KEYWORDS) else 1,
            random.Random(f"stage25-qwen-pilot:{row['sample_id']}").random(),
        ),
    )
    selected = ranked[:args.count]
    if len(selected) != args.count:
        raise ValueError("pilot count exceeds manifest")
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        str(args.model), device_map=args.device, dtype=torch.bfloat16, max_inference_batch_size=8,
        max_new_tokens=512,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(selected, 1):
            audio = args.audio_root / row["source_audio_path"]
            try:
                result = model.transcribe(str(audio), language="English")[0]
                transcript, error = result.text.strip(), None
            except Exception as exc:
                transcript, error = "", f"{type(exc).__name__}: {exc}"
            record = {
                "sample_id": row["sample_id"], "audio_path": str(audio),
                "midasheng_transcript": old.get(row["sample_id"], {}).get("transcript"),
                "qwen3_asr_transcript": transcript, "qwen3_asr_error": error,
                "reference": row["assistant_response"], "pilot_target": row["sample_id"] in TARGET_IDS,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"QWEN_ASR {index}/{len(selected)} {row['sample_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
