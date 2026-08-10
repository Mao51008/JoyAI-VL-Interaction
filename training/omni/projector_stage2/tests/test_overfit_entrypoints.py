import pytest

from training.omni.projector_stage2.evaluate_overfit import select_task_rows, summarise_records
from training.omni.projector_stage2.prepare_overfit_manifests import (
    assign_mixed_tasks,
    select_one_turn_per_dialogue,
)


def _row(dialogue_id: str, sample_id: str) -> dict[str, str]:
    return {"dialogue_id": dialogue_id, "sample_id": sample_id}


def test_select_one_turn_per_dialogue_is_deterministic_and_unique():
    rows = [
        _row("d1", "d1-0"),
        _row("d1", "d1-1"),
        _row("d2", "d2-0"),
        _row("d3", "d3-0"),
    ]
    first = select_one_turn_per_dialogue(rows, sample_count=3, seed=7)
    second = select_one_turn_per_dialogue(rows, sample_count=3, seed=7)
    assert first == second
    assert len({row["dialogue_id"] for row in first}) == 3
    with pytest.raises(ValueError, match="only 3 dialogues"):
        select_one_turn_per_dialogue(rows, sample_count=4, seed=7)


def test_assign_mixed_tasks_is_fixed_and_has_requested_balance():
    rows = [_row(f"d{index}", f"s{index}") for index in range(32)]
    first = assign_mixed_tasks(rows, asr_samples=16, seed=9)
    second = assign_mixed_tasks(rows, asr_samples=16, seed=9)
    assert first == second
    assert sum(row["training_task"] == "asr_transcription" for row in first) == 16
    assert sum(row["training_task"] == "dialogue_response" for row in first) == 16
    with pytest.raises(ValueError, match=r"\[0, 32\]"):
        assign_mixed_tasks(rows, asr_samples=33, seed=9)


def test_select_task_rows_preserves_only_the_original_supervision_prompt():
    rows = [
        {"sample_id": "asr", "training_task": "asr_transcription"},
        {"sample_id": "dialogue", "training_task": "dialogue_response"},
    ]
    assert [row["sample_id"] for row in select_task_rows(rows, "asr_transcription")] == ["asr"]
    assert [row["sample_id"] for row in select_task_rows(rows, "dialogue_response")] == ["dialogue"]
    with pytest.raises(ValueError, match="no rows"):
        select_task_rows(rows, "missing")


def test_asr_gate_summary_reports_error_rates_and_format_failures():
    records = [
        {
            "exact_match": True,
            "generated_eos": True,
            "hypothesis": "ONE TWO",
            "bare_user": False,
            "user_prefix": False,
            "wrapper": False,
            "word_errors": 0,
            "reference_words": 2,
            "char_errors": 0,
            "reference_chars": 6,
        },
        {
            "exact_match": False,
            "generated_eos": False,
            "hypothesis": "",
            "bare_user": True,
            "user_prefix": True,
            "wrapper": True,
            "word_errors": 2,
            "reference_words": 4,
            "char_errors": 3,
            "reference_chars": 8,
        },
    ]
    summary = summarise_records(records, "asr_transcription")
    assert summary["exact_match_rate"] == 0.5
    assert summary["wer"] == pytest.approx(2 / 6)
    assert summary["cer"] == pytest.approx(3 / 14)
    assert summary["bare_user_rate"] == 0.5
    assert summary["wrapper_rate"] == 0.5
