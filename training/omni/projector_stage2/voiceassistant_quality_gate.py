"""Build auditable VoiceAssistant candidates after a fixed-ASR consistency gate.

This CPU-only command never transcribes audio.  It consumes a separately generated,
fixed Qwen3-ASR JSONL result so an ASR mismatch is recorded as a review risk, never
as proof that the original audio is wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

KNOWN_AUDIO_LABEL_MISMATCHES = {"voiceassistant:0204465"}


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def lexical_consistency(question: str, transcript: str) -> tuple[bool, float]:
    """A conservative, explainable screening score; human review remains authoritative."""
    question_tokens = set(normalize_text(question).split())
    transcript_tokens = set(normalize_text(transcript).split())
    if not question_tokens or not transcript_tokens:
        return False, 0.0
    overlap = len(question_tokens & transcript_tokens) / len(question_tokens)
    return overlap >= 0.6, overlap


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_rows(
    candidates: list[dict[str, Any]], asr_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transcripts: dict[str, str] = {}
    for row in asr_rows:
        sample_id = str(row.get("sample_id", ""))
        transcript = str(row.get("fixed_asr_transcript", row.get("transcript", ""))).strip()
        if not sample_id or not transcript or sample_id in transcripts:
            raise ValueError("ASR rows require unique sample_id and non-empty fixed_asr_transcript")
        transcripts[sample_id] = transcript
    ready, quarantine = [], []
    for source in candidates:
        row = dict(source)
        sample_id = str(row.get("sample_id", ""))
        question = str(row.get("question", row.get("metadata", {}).get("question", ""))).strip()
        if not sample_id or not question:
            raise ValueError("candidate requires sample_id and metadata question")
        transcript = transcripts.get(sample_id)
        if transcript is None:
            raise ValueError(f"candidate has no fixed-ASR result: {sample_id}")
        consistent, score = lexical_consistency(question, transcript)
        known_bad = sample_id in KNOWN_AUDIO_LABEL_MISMATCHES
        row["fixed_asr_transcript"] = transcript
        row["audio_question_consistency"] = {
            "method": "normalized_lexical_overlap_v1",
            "score": score,
            "status": "consistent" if consistent and not known_bad else "needs_human_review",
            "asr_mismatch_is_not_audio_error": True,
        }
        row["answer_quality_review"] = {
            "status": "not_reviewed",
            "required_for_training": True,
            "fact_or_safety": bool(row.get("fact_or_safety", False)),
        }
        row.setdefault("provenance", {})["voiceassistant_quality_gate"] = "v1"
        if consistent and not known_bad:
            ready.append(row)
        else:
            row["quarantine_reason"] = "known_audio_label_mismatch" if known_bad else "asr_question_mismatch"
            quarantine.append(row)
    return ready, quarantine


def write_gate_output(candidate_manifest: Path, asr_manifest: Path, output_dir: Path) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    ready, quarantine = gate_rows(_read_jsonl(candidate_manifest), _read_jsonl(asr_manifest))
    output_dir.mkdir(parents=True)
    for name, rows in (("answer_review_candidates.jsonl", ready), ("quarantine.jsonl", quarantine)):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "format": "voiceassistant-quality-gate-v1",
        "candidate_manifest": str(candidate_manifest.resolve()),
        "candidate_sha256": _sha256(candidate_manifest),
        "fixed_asr_manifest": str(asr_manifest.resolve()),
        "fixed_asr_sha256": _sha256(asr_manifest),
        "answer_review_candidates": len(ready),
        "quarantine": len(quarantine),
        "known_audio_label_mismatches": sorted(KNOWN_AUDIO_LABEL_MISMATCHES),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {"answer_review_candidates": len(ready), "quarantine": len(quarantine)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--fixed-asr-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_gate_output(args.candidate_manifest, args.fixed_asr_manifest, args.output_dir)))


if __name__ == "__main__":
    main()
