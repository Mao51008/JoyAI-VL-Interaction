"""Select additional VoiceAssistant user-audio → assistant-text examples for Stage2.1.

The source contract deliberately contains no user transcript.  This command preserves
that contract and performs only structural checks; audio-feature sequence length is
validated later when frozen features are cached.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_audio_understanding_pilot import _split, _voiceassistant_row

KNOWN_CONFLICT = "voiceassistant:0204465"

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def select(raw: Path, existing: Path, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0: raise ValueError("count must be positive")
    existing_ids = {str(row["sample_id"]) for row in _read_jsonl(existing)}
    candidates, rejected = [], Counter()
    seen: set[str] = set()
    with raw.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip(): continue
            row = _voiceassistant_row(json.loads(line), index)
            if row is None:
                rejected["invalid_three_message_two_audio_contract"] += 1; continue
            sample_id = str(row["sample_id"])
            if sample_id in seen: raise ValueError(f"duplicate source ID: {sample_id}")
            seen.add(sample_id)
            if sample_id == KNOWN_CONFLICT:
                rejected["known_unresolved_conflict"] += 1; continue
            if row["split"] != "train":
                rejected["fixed_holdout_split"] += 1; continue
            if sample_id in existing_ids:
                rejected["already_in_existing_20000"] += 1; continue
            row["provenance"]["selection"] = {
                "name": "stage21_voiceassistant_remaining_train_hash_v1", "raw_index": index,
                "sequence_length_status": "requires_frozen_feature_cache_validation",
            }
            candidates.append(row)
    candidates.sort(key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest())
    selected = candidates[:count]
    if len(selected) < count: raise ValueError(f"only {len(selected)} eligible candidates for requested {count}")
    return selected, {"raw_structure_eligible": len(candidates) + rejected["already_in_existing_20000"] + rejected["fixed_holdout_split"], "eligible_remaining_train": len(candidates), "selected": len(selected), "rejected": dict(rejected)}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",type=Path,required=True);p.add_argument("--existing",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--count",type=int,default=20000);a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    rows,audit=select(a.raw,a.existing,a.count);a.output_dir.mkdir(parents=True)
    manifest=a.output_dir/"voiceassistant_stage21_train.jsonl"
    manifest.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    audit.update({"format":"voiceassistant-stage21-selection-v1","raw":str(a.raw.resolve()),"raw_sha256":_sha256(a.raw),"existing":str(a.existing.resolve()),"existing_sha256":_sha256(a.existing),"manifest":str(manifest.resolve()),"manifest_sha256":_sha256(manifest),"user_transcript_injected":False,"audio_download_status":"planned"})
    (a.output_dir/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))
if __name__ == "__main__": main()
