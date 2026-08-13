"""CPU-only acceptance audit for completed sharded visual teacher generation."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

def _rows(paths):
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line]

def audit(source_paths: list[Path], output_paths: list[Path], rejected_paths: list[Path]) -> dict:
    source = _rows(source_paths); kept = _rows(output_paths); rejected = _rows(rejected_paths)
    ids = [str(x["sample_id"]) for x in kept + rejected]
    if len(ids) != len(set(ids)) or set(ids) != {str(x["sample_id"]) for x in source}:
        raise ValueError("source/output sample IDs do not match exactly")
    all_rows = kept + rejected
    return {"format":"vision-teacher-acceptance-v1", "attempted":len(all_rows), "kept":len(kept), "rejected":len(rejected),
      "rejected_by_reason":dict(Counter(x.get("rejection_reason", "none") for x in rejected)),
      "natural_eos":sum(bool(x["provenance"]["generation"]["finished_by_eos"]) for x in all_rows),
      "empty_answers":sum(not str(x.get("teacher_response", "")).strip() for x in all_rows),
      "at_256":sum(x["provenance"]["generation"]["generated_tokens"] >= 256 for x in all_rows),
      "full_sequence_over_2048":sum((x["provenance"]["sequence"].get("full_tokens") or 0)>2048 for x in all_rows),
      "sentence_distribution":dict(Counter(min(3, len([s for s in re.split(r"[.!?]+", str(x.get("teacher_response", ""))) if s.strip()])) for x in kept))}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,action="append",required=True);p.add_argument("--output",type=Path,action="append",required=True);p.add_argument("--rejected",type=Path,action="append",required=True);p.add_argument("--audit",type=Path,required=True);a=p.parse_args();
 if a.audit.exists(): raise FileExistsError(a.audit)
 result=audit(a.source,a.output,a.rejected);a.audit.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result))
if __name__ == "__main__": main()
