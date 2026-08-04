"""Build speaker-disjoint LibriSpeech source JSONL for stage-one projector data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _split(speaker: str, validation_percent: int) -> str:
    value = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 100
    return "validation" if value < validation_percent else "train"


def build(root: Path, output: Path, validation_percent: int) -> dict[str, int]:
    import soundfile as sf
    if not 1 <= validation_percent < 50:
        raise ValueError("validation percent must be between 1 and 49")
    rows = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(" ", 1)
            audio = transcript.with_name(f"{utterance_id}.flac")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            info = sf.info(audio)
            speaker = utterance_id.split("-", 1)[0]
            rows.append({"sample_id": f"librispeech-{utterance_id}", "audio_path": str(audio),
                         "target_text": text, "duration_ms": round(info.frames * 1000 / info.samplerate),
                         "sample_rate": info.samplerate, "num_samples": info.frames,
                         "split": _split(speaker, validation_percent),
                         "source_record": utterance_id, "speaker_id": speaker})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"samples": len(rows), "train": sum(r["split"] == "train" for r in rows),
            "validation": sum(r["split"] == "validation" for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.validation_percent), indent=2))


if __name__ == "__main__":
    main()
