"""Compare normal, zero, and shuffled-audio transcript evaluation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows_by_id(result: dict, name: str) -> dict[str, dict]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{name} does not contain rows")
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{name} contains duplicate sample_id values")
    return indexed


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def compare_results(normal: dict, zero: dict, shuffle: dict, typical_count: int = 10) -> dict:
    """Build a per-sample ablation comparison without loading models or tensors."""
    named = {"normal": _rows_by_id(normal, "normal"), "zero": _rows_by_id(zero, "zero"), "shuffle": _rows_by_id(shuffle, "shuffle")}
    sample_ids = set(named["normal"])
    if any(set(rows) != sample_ids for rows in named.values()):
        raise ValueError("normal, zero, and shuffle must contain the same sample_id values")
    ordered_ids = sorted(sample_ids)
    rows = []
    for sample_id in ordered_ids:
        current = {name: indexed[sample_id] for name, indexed in named.items()}
        rows.append({
            "sample_id": sample_id,
            "normal_wer": current["normal"]["wer"], "zero_wer": current["zero"]["wer"], "shuffle_wer": current["shuffle"]["wer"],
            "normal_cer": current["normal"]["cer"], "zero_cer": current["zero"]["cer"], "shuffle_cer": current["shuffle"]["cer"],
            "zero_minus_normal_wer": current["zero"]["wer"] - current["normal"]["wer"],
            "shuffle_minus_normal_wer": current["shuffle"]["wer"] - current["normal"]["wer"],
            "zero_minus_normal_cer": current["zero"]["cer"] - current["normal"]["cer"],
            "shuffle_minus_normal_cer": current["shuffle"]["cer"] - current["normal"]["cer"],
        })
    same = sum(named["normal"][sample_id]["hypothesis"] == named["shuffle"][sample_id]["hypothesis"] for sample_id in ordered_ids)
    summaries = {}
    for name, indexed in named.items():
        values = list(indexed.values())
        summaries[name] = {"average_generated_tokens": _mean(values, "generated_tokens"),
                           "eos_rate": _mean(values, "generated_eos")}
    failures = sorted(rows, key=lambda row: (row["shuffle_minus_normal_wer"], row["zero_minus_normal_wer"], row["shuffle_minus_normal_cer"]), reverse=True)[:typical_count]
    return {"samples": len(rows), "normal_shuffle_identical_rate": same / max(1, len(rows)),
            "generation_summary": summaries, "rows": rows, "typical_failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--shuffle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--typical-count", type=int, default=10)
    args = parser.parse_args()
    if args.typical_count <= 0:
        raise ValueError("--typical-count must be positive")
    result = compare_results(*(json.loads(path.read_text(encoding="utf-8")) for path in (args.normal, args.zero, args.shuffle)), args.typical_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "typical_failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
