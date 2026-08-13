import json

import pytest

from training.omni.projector_stage2.gold_transcript_diagnostic import seed_rows, validate_rows
from training.omni.projector_stage2.voiceassistant_quality_gate import gate_rows


def test_gate_quarantines_known_audio_label_mismatch_and_preserves_review_status():
    candidates = [
        {"sample_id": "voiceassistant:0204465", "question": "What competitors does Multi offer price matches for?", "assistant_response": "Target", "fact_or_safety": True},
        {"sample_id": "voice:ok", "question": "How far is Moon from Earth?", "assistant_response": "384400 km"},
    ]
    asr = [
        {"sample_id": "voiceassistant:0204465", "fixed_asr_transcript": "What competitors does Multi offer price matches for?"},
        {"sample_id": "voice:ok", "fixed_asr_transcript": "How far is Moon from Earth?"},
    ]
    ready, quarantine = gate_rows(candidates, asr)
    assert [row["sample_id"] for row in ready] == ["voice:ok"]
    assert quarantine[0]["quarantine_reason"] == "known_audio_label_mismatch"
    assert quarantine[0]["audio_question_consistency"]["asr_mismatch_is_not_audio_error"]
    assert quarantine[0]["answer_quality_review"]["required_for_training"]


def test_gold_seed_requires_independent_gold_and_fixed_asr_fields():
    rows = seed_rows()
    validate_rows(rows)
    rows[0].pop("fixed_asr_transcript")
    with pytest.raises(ValueError, match="fixed_asr_transcript"):
        validate_rows(rows)
