"""Build auditable three-annotator-consensus Clotho-AQA manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}
ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """Normalize harmless answer-format variations before vote comparison."""
    words = re.sub(r"[^a-z0-9\s]", " ", answer.lower()).split()
    return " ".join(NUMBER_WORDS.get(word, word) for word in words if word not in ARTICLES)


def question_type(question: str) -> str:
    """Assign a transparent coarse type for balance auditing, not supervision."""
    normalized = normalize_answer(question)
    if re.match(r"^(is|are|do|does|did|can|could|has|have|was|were|will|would|should)\b", normalized):
        return "yes_no"
    if re.match(r"^(how many|what number|how much)\b", normalized):
        return "quantity"
    if re.search(
        r"\b(what|which).*(sound|noise|animal|instrument|vehicle|object|thing)"
        r"|\bsource\b|\bwhat can be heard\b",
        normalized,
    ):
        return "source_object"
    if re.search(
        r"\b(what (is|are|was|were) happening|what (is|are|does|do).*(doing|happen)"
        r"|what action|what event)\b",
        normalized,
    ):
        return "action_event"
    return "other"


def load_strict_consensus(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only exactly three normalized-equal answers for one source split."""
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

    all_types: Counter[str] = Counter()
    retained_types: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    rejected_by_reason: Counter[str] = Counter()
    for (filename, question), answers in grouped.items():
        task_type = question_type(question)
        all_types[task_type] += 1
        normalized_answers = [normalize_answer(answer) for answer in answers]
        if len(answers) != 3:
            rejected_by_reason["annotation_count_not_three"] += 1
            continue
        if len(set(normalized_answers)) != 1:
            rejected_by_reason["answers_not_all_equal_after_normalization"] += 1
            continue
        canonical_answer = normalized_answers[0]
        sample_id = "clotho_aqa:" + hashlib.sha256(
            f"{split}\0{filename}\0{question}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "task": "environment_audio_qa",
                "source_audio_path": filename,
                "question": question,
                "assistant_response": canonical_answer,
                "reference_answers": answers,
                "normalized_reference_answers": normalized_answers,
                "question_type": task_type,
                "provenance": {
                    "dataset": "Clotho-AQA",
                    "annotation_count": 3,
                    "answer_votes": 3,
                    "consensus_rule": "three_normalized_answers_equal_v1",
                    "answer_normalization": "lowercase_punctuation_articles_number_words_v1",
                },
            }
        )
        retained_types[task_type] += 1
    summary = {
        "split": split,
        "unique_audio_question_pairs": len(grouped),
        "strict_consensus_rows": len(rows),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "all_by_question_type": dict(sorted(all_types.items())),
        "strict_by_question_type": dict(sorted(retained_types.items())),
        "strict_rate_by_question_type": {
            key: retained_types[key] / value for key, value in sorted(all_types.items())
        },
    }
    return rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create strict high-confidence Clotho-AQA manifests.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Write outputs; omitted means audit only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_rows, train_summary = load_strict_consensus(args.train_csv, "train")
    dev_rows, dev_summary = load_strict_consensus(args.dev_csv, "dev")
    overlap = {row["source_audio_path"] for row in train_rows} & {
        row["source_audio_path"] for row in dev_rows
    }
    if overlap:
        raise ValueError(f"audio clip train/dev leakage detected: {sorted(overlap)[:10]}")
    summary = {"train": train_summary, "dev": dev_summary, "audio_clip_overlap": 0}
    if args.run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(args.output_dir / "clotho_aqa_high_confidence_train.jsonl", train_rows)
        _write_jsonl(args.output_dir / "clotho_aqa_high_confidence_dev.jsonl", dev_rows)
        summary["written"] = {"train": len(train_rows), "dev": len(dev_rows)}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
