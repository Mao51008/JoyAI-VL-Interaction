import json

from training.omni.projector_stage2.voiceassistant_quality_gate import gate_rows


def test_gate_quarantines_known_unresolved_asr_metadata_conflict():
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
    assert quarantine[0]["quarantine_reason"] == "known_unresolved_asr_metadata_conflict"
    assert quarantine[0]["audio_question_consistency"]["asr_mismatch_is_not_audio_error"]
    assert quarantine[0]["answer_quality_review"]["required_for_training"]


def test_gate_quarantines_ambiguous_asr_without_claiming_audio_is_wrong():
    ready, quarantine = gate_rows(
        [{"sample_id": "voice:ambiguous", "question": "How far is Moon from Earth?"}],
        [{"sample_id": "voice:ambiguous", "fixed_asr_transcript": "How far Moon"}],
    )
    assert not ready
    assert quarantine[0]["quarantine_reason"] == "asr_question_low_confidence"
    assert quarantine[0]["audio_question_consistency"]["confidence"] == "low"
