"""Compare named normal and audio-ablation transcript evaluation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows_by_id(result: dict, name: str) -> dict[str, dict]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{name} does not contain rows")
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{name} contains duplicate sample_id values")
    return indexed


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def compare_results(
    normal: dict,
    zero: dict,
    shuffle: dict,
    typical_count: int = 10,
    projected_zero: dict | None = None,
    temporal_shuffle: dict | None = None,
) -> dict:
    """Build a per-sample ablation comparison without loading models or tensors."""
    named = {"normal": _rows_by_id(normal, "normal"), "waveform_zero": _rows_by_id(zero, "waveform-zero"),
             "cross_sample_shuffle": _rows_by_id(shuffle, "cross-sample-shuffle")}
    if projected_zero is not None:
        named["projected_zero"] = _rows_by_id(projected_zero, "projected_zero")
    if temporal_shuffle is not None:
        named["within_sample_temporal_shuffle"] = _rows_by_id(temporal_shuffle, "within_sample_temporal_shuffle")
    sample_ids = set(named["normal"])
    if any(set(rows) != sample_ids for rows in named.values()):
        raise ValueError("normal, zero, and shuffle must contain the same sample_id values")
    ordered_ids = sorted(sample_ids)
    rows = []
    for sample_id in ordered_ids:
        current = {name: indexed[sample_id] for name, indexed in named.items()}
        row = {
            "sample_id": sample_id,
            "normal_wer": current["normal"]["wer"], "waveform_zero_wer": current["waveform_zero"]["wer"], "cross_sample_shuffle_wer": current["cross_sample_shuffle"]["wer"],
            "normal_cer": current["normal"]["cer"], "waveform_zero_cer": current["waveform_zero"]["cer"], "cross_sample_shuffle_cer": current["cross_sample_shuffle"]["cer"],
            "waveform_zero_minus_normal_wer": current["waveform_zero"]["wer"] - current["normal"]["wer"],
            "cross_sample_shuffle_minus_normal_wer": current["cross_sample_shuffle"]["wer"] - current["normal"]["wer"],
            "waveform_zero_minus_normal_cer": current["waveform_zero"]["cer"] - current["normal"]["cer"],
            "cross_sample_shuffle_minus_normal_cer": current["cross_sample_shuffle"]["cer"] - current["normal"]["cer"],
        }
        if projected_zero is not None:
            row.update({
                "projected_zero_wer": current["projected_zero"]["wer"],
                "projected_zero_cer": current["projected_zero"]["cer"],
                "projected_zero_minus_normal_wer": current["projected_zero"]["wer"] - current["normal"]["wer"],
                "projected_zero_minus_normal_cer": current["projected_zero"]["cer"] - current["normal"]["cer"],
            })
        if temporal_shuffle is not None:
            row.update({
                "within_sample_temporal_shuffle_wer": current["within_sample_temporal_shuffle"]["wer"],
                "within_sample_temporal_shuffle_cer": current["within_sample_temporal_shuffle"]["cer"],
                "within_sample_temporal_shuffle_minus_normal_wer": current["within_sample_temporal_shuffle"]["wer"] - current["normal"]["wer"],
                "within_sample_temporal_shuffle_minus_normal_cer": current["within_sample_temporal_shuffle"]["cer"] - current["normal"]["cer"],
            })
        rows.append(row)
    same = sum(named["normal"][sample_id]["hypothesis"] == named["cross_sample_shuffle"][sample_id]["hypothesis"] for sample_id in ordered_ids)
    summaries = {}
    for name, indexed in named.items():
        values = list(indexed.values())
        summaries[name] = {"average_generated_tokens": _mean(values, "generated_tokens"),
                           "eos_rate": _mean(values, "generated_eos")}
    failures = sorted(rows, key=lambda row: (row["cross_sample_shuffle_minus_normal_wer"], row["waveform_zero_minus_normal_wer"], row["cross_sample_shuffle_minus_normal_cer"]), reverse=True)[:typical_count]
    return {"samples": len(rows), "normal_shuffle_identical_rate": same / max(1, len(rows)),
            "generation_summary": summaries, "rows": rows, "typical_failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--shuffle", type=Path, required=True)
    parser.add_argument("--projected-zero", type=Path)
    parser.add_argument("--temporal-shuffle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--typical-count", type=int, default=10)
    args = parser.parse_args()
    if args.typical_count <= 0:
        raise ValueError("--typical-count must be positive")
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.normal, args.zero, args.shuffle)]
    projected_zero = json.loads(args.projected_zero.read_text(encoding="utf-8")) if args.projected_zero else None
    temporal_shuffle = json.loads(args.temporal_shuffle.read_text(encoding="utf-8")) if args.temporal_shuffle else None
    result = compare_results(*inputs, args.typical_count, projected_zero, temporal_shuffle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "typical_failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
