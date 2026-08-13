from __future__ import annotations

import csv

from training.omni.projector_stage2.prepare_clotho_high_confidence import (
    load_strict_consensus,
    normalize_answer,
    question_type,
)


def _write_csv(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "QuestionText", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_answer_handles_articles_punctuation_and_number_words() -> None:
    assert normalize_answer("An umbrella!") == "umbrella"
    assert normalize_answer("Two") == "2"


def test_question_type_covers_audit_categories() -> None:
    assert question_type("Are birds audible?") == "yes_no"
    assert question_type("How many people speak?") == "quantity"
    assert question_type("What sound can be heard?") == "source_object"
    assert question_type("What is happening in the audio?") == "action_event"


def test_strict_consensus_rejects_two_to_one_and_keeps_normalized_three_way_match(tmp_path) -> None:
    path = tmp_path / "clotho.csv"
    _write_csv(
        path,
        [
            {"file_name": "one.wav", "QuestionText": "What is heard?", "answer": "An umbrella"},
            {"file_name": "one.wav", "QuestionText": "What is heard?", "answer": "umbrella!"},
            {"file_name": "one.wav", "QuestionText": "What is heard?", "answer": "the umbrella"},
            {"file_name": "two.wav", "QuestionText": "Are birds audible?", "answer": "yes"},
            {"file_name": "two.wav", "QuestionText": "Are birds audible?", "answer": "yes"},
            {"file_name": "two.wav", "QuestionText": "Are birds audible?", "answer": "no"},
        ],
    )

    rows, summary = load_strict_consensus(path, "train")

    assert len(rows) == 1
    assert rows[0]["assistant_response"] == "umbrella"
    assert rows[0]["question_type"] == "source_object"
    assert summary["rejected_by_reason"]["answers_not_all_equal_after_normalization"] == 1
