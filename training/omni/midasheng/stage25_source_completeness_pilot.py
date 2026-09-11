"""Precision-oriented single-utterance source-completeness pilot."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


REQUIRED_IDS = {"voiceassistant:0191573", "voiceassistant:0032429"}
REFERENTIAL = re.compile(
    r"\b(the following|these words|this sentence|the sentence|the passage|the text below|"
    r"this text|rewrite this|translate this|summarize this|spell the following|"
    r"this headline|the headline|this article|the article|this paragraph|the paragraph|"
    r"this conversation|the conversation)\b",
    re.IGNORECASE,
)
QUESTION = re.compile(r"^(what|who|when|where|why|how|which|is|are|can|could|do|does|did|will|would)\b", re.I)
INSTRUCTION = re.compile(r"\b(edit|rewrite|translate|paraphrase|summarize|spell|correct|classify)\b", re.I)
SAFETY = re.compile(r"\b(safe|safety|weapon|harm|illegal|vaccine|medical|suicide)\b", re.I)
NUMBER_OR_DATE = re.compile(r"\b\d{1,4}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)


def classify(text: str) -> tuple[str, str, str]:
    normalized = " ".join(text.split())
    if not normalized:
        return "UNCERTAIN", "empty_transcript", "rule"
    match = REFERENTIAL.search(normalized)
    if match:
        tail = normalized[match.end() :].strip(" :,-")
        # A following payload must be materially longer than a bare noun phrase.
        if len(tail.split()) < 4:
            return "INCOMPLETE", f"referential_phrase_without_payload:{match.group(0)}", "rule"
    if QUESTION.match(normalized) and normalized.endswith("?") and not match:
        return "COMPLETE", "self_contained_interrogative", "rule"
    if normalized.endswith("?") and not match:
        return "COMPLETE", "self_contained_question", "rule"
    return "UNCERTAIN", "rule_not_high_confidence", "rule"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.transcripts.read_text(encoding="utf-8").splitlines() if line]
    by_id = {str(row["sample_id"]): row for row in rows}
    missing = REQUIRED_IDS - by_id.keys()
    if missing:
        raise ValueError(f"required IDs missing: {sorted(missing)}")
    ordered_rows = sorted(
        rows,
        key=lambda row: random.Random(f"source-completeness:{row['sample_id']}").random(),
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    def take(predicate: object, limit: int) -> None:
        for row in ordered_rows:
            sample_id = str(row["sample_id"])
            if len(selected) >= args.count or sum(
                1 for prior in selected if predicate(prior)
            ) >= limit:
                return
            if sample_id not in selected_ids and predicate(row):
                selected.append(row)
                selected_ids.add(sample_id)

    # Deliberately cap each group: the first version let referential prompts fill all 400 slots.
    take(lambda row: str(row["sample_id"]) in REQUIRED_IDS, len(REQUIRED_IDS))
    take(lambda row: bool(REFERENTIAL.search(str(row.get("transcript", "")))), 90)
    take(lambda row: bool(INSTRUCTION.search(str(row.get("transcript", "")))), 90)
    take(lambda row: bool(QUESTION.match(str(row.get("transcript", "")))), 110)
    take(lambda row: bool(SAFETY.search(str(row.get("transcript", "")))), 45)
    take(lambda row: bool(NUMBER_OR_DATE.search(str(row.get("transcript", "")))), 45)
    take(lambda row: len(str(row.get("transcript", "")).split()) >= 25, 45)
    take(lambda row: len(str(row.get("transcript", "")).split()) <= 8, 45)
    for row in ordered_rows:
        if len(selected) >= args.count:
            break
        if str(row["sample_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["sample_id"]))
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            label, reason, method = classify(str(row.get("transcript", "")))
            record = {
                "sample_id": row["sample_id"], "audio_path": row["audio_path"], "audio_id": row["audio_id"],
                "transcript": row.get("transcript", ""), "source_completeness": label,
                "source_completeness_reason": reason, "source_completeness_method": method,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
