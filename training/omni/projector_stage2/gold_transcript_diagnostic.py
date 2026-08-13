"""Create and validate a non-training gold-transcript diagnostic manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"sample_id", "category", "reference_answer", "gold_transcript", "gold_status", "fixed_asr_transcript", "decision"}
SEED_SAMPLE_IDS = {
    "voiceassistant:0091552": "safety_request",
    "voiceassistant:0051215": "fact_entity",
    "voiceassistant:0204465": "fact_entity",
    "voiceassistant:0139758": "general_semantics",
    "voiceassistant:0228280": "general_semantics",
}


def validate_rows(rows: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"row {index} missing {sorted(missing)}")
        sample_id = str(row["sample_id"])
        if sample_id in ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        if row["gold_status"] not in {"pending_human", "verified"}:
            raise ValueError("gold_status must be pending_human or verified")
        if row["decision"] not in {"include", "exclude", "review"}:
            raise ValueError("invalid decision")
        if row["gold_status"] == "verified" and not str(row["gold_transcript"]).strip():
            raise ValueError("verified gold transcript cannot be empty")
        if not str(row["fixed_asr_transcript"]).strip():
            raise ValueError("fixed-ASR transcript is required and distinct from gold")


def seed_rows() -> list[dict[str, Any]]:
    return [
        {"sample_id": sample_id, "category": category, "reference_answer": "", "gold_transcript": "", "gold_status": "pending_human", "fixed_asr_transcript": "PENDING_FIXED_ASR", "decision": "review", "rules": "Human transcript is independently verified; never copy fixed-ASR text."}
        for sample_id, category in SEED_SAMPLE_IDS.items()
    ]


def write_seed(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    rows = seed_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def load_and_validate(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_rows(rows)
    return rows
