"""Build bounded training manifests for the audio-understanding pilot.

    The script does not download media.  VoiceAssistant rows retain only user-audio
    to assistant-text turns; Clotho-AQA rows retain only unanimous three-annotator
    answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATA_ROOT = Path("/data/maoyy/datasets/audio_understanding_pilot")


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/data/maoyy").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"output directory must be under {allowed}: {resolved}")
    return resolved


def _split(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "dev" if bucket < 95 else "test"


def _voiceassistant_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    messages = row.get("messages")
    audios = row.get("audios")
    if not isinstance(messages, list) or not isinstance(audios, list) or len(messages) != 3:
        return None
    roles = [str(message.get("role", "")) for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"] or len(audios) != 2:
        return None
    user_content = str(messages[1].get("content", "")).strip()
    assistant_content = str(messages[2].get("content", ""))
    if user_content != "<|audio|>" or "<|audio|>" not in assistant_content:
        return None
    answer = assistant_content.replace("<|audio|>", "").strip()
    user_audio = str(audios[0]).strip()
    if not answer or not user_audio.startswith("user/"):
        return None
    sample_id = f"voiceassistant:{index:07d}"
    return {
        "sample_id": sample_id,
        "split": _split(sample_id),
        "task": "audio_dialogue_response",
        "source_audio_path": user_audio,
        "assistant_response": answer,
        "provenance": {
            "dataset": "shenyunhang/VoiceAssistant-400K",
            "original_row": index,
            "source_audio_role": "user",
            "excluded_audio_role": "assistant",
        },
    }


def iter_voiceassistant(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                parsed = _voiceassistant_row(json.loads(line), index)
                if parsed is not None:
                    yield parsed


def load_clotho_consensus(path: Path, split: str) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_name", "QuestionText", "answer"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"unexpected Clotho-AQA columns in {path}: {reader.fieldnames}")
        for row in reader:
            filename = str(row["file_name"]).strip()
            question = str(row["QuestionText"]).strip()
            answer = str(row["answer"]).strip()
            if filename and question and answer:
                grouped[(filename, question)].append(answer)
    retained: list[dict[str, Any]] = []
    rejected = 0
    for (filename, question), answers in grouped.items():
        counts = Counter(answers)
        answer, votes = max(counts.items(), key=lambda item: (item[1], item[0]))
        if len(answers) != 3 or votes != 3:
            rejected += 1
            continue
        sample_id = "clotho_aqa:" + hashlib.sha256(
            f"{split}\0{filename}\0{question}".encode("utf-8")
        ).hexdigest()[:16]
        retained.append(
            {
                "sample_id": sample_id,
                "split": split,
                "task": "environment_audio_qa",
                "source_audio_path": filename,
                "question": question,
                "assistant_response": answer,
                "reference_answers": answers,
                "provenance": {
                    "dataset": "Clotho-AQA",
                    "answer_votes": votes,
                    "annotation_count": len(answers),
                    "consensus_rule": "all_three_exact_answers",
                },
            }
        )
    return retained, rejected


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create single-turn VoiceAssistant and consensus Clotho-AQA manifests.")
    parser.add_argument("--voiceassistant-jsonl", type=Path, default=DATA_ROOT / "voiceassistant_400k/raw/data.jsonl")
    parser.add_argument("--clotho-train-csv", type=Path, default=DATA_ROOT / "clotho_aqa/raw/clotho_aqa_train.csv")
    parser.add_argument("--clotho-dev-csv", type=Path, default=DATA_ROOT / "clotho_aqa/raw/clotho_aqa_val.csv")
    parser.add_argument("--clotho-test-csv", type=Path, default=DATA_ROOT / "clotho_aqa/raw/clotho_aqa_test.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "manifests")
    parser.add_argument("--voice-sample-count", type=int, default=20_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true", help="Write manifests; omitted means count and validate only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _require_data_root(args.output_dir)
    for path in (args.voiceassistant_jsonl, args.clotho_train_csv, args.clotho_dev_csv, args.clotho_test_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda rows, **_: rows
    voice_candidates = list(
        tqdm(
            iter_voiceassistant(args.voiceassistant_jsonl),
            desc="VoiceAssistant",
            unit="sample",
            disable=args.no_progress,
        )
    )
    if not 0 < args.voice_sample_count <= len(voice_candidates):
        raise ValueError(
            f"--voice-sample-count must be in [1, {len(voice_candidates)}]"
        )
    voice_rows = sorted(
        voice_candidates,
        key=lambda row: hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest(),
    )[: args.voice_sample_count]
    clotho_rows: list[dict[str, Any]] = []
    clotho_rejected = 0
    for split, path in (("train", args.clotho_train_csv), ("dev", args.clotho_dev_csv), ("test", args.clotho_test_csv)):
        rows, rejected = load_clotho_consensus(path, split)
        clotho_rows.extend(rows)
        clotho_rejected += rejected
    summary = {
        "voiceassistant_eligible": len(voice_candidates),
        "voiceassistant_selected": len(voice_rows),
        "clotho_consensus": len(clotho_rows),
        "clotho_rejected_no_consensus": clotho_rejected,
    }
    if args.run:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["written"] = {
            "voiceassistant": _write_jsonl(output_dir / "voiceassistant_single_turn.jsonl", voice_rows),
            "clotho_aqa": _write_jsonl(output_dir / "clotho_aqa_consensus.jsonl", clotho_rows),
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
