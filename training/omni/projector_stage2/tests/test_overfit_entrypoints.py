import pytest

from training.omni.projector_stage2.evaluate_overfit import summarise_records
from training.omni.projector_stage2.prepare_overfit_manifests import (
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
