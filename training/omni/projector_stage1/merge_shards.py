"""Strictly merge stage-one teacher-forced or WER shard JSON results."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence


_COMMON_KEYS = (
    "manifest", "manifest_sha256", "manifest_sample_count", "total_samples", "num_shards",
    "checkpoint", "audio_ablation", "seed",
)


def _validate_inputs(results: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    if not results:
        raise ValueError("at least one shard result is required")
    first = results[0]
    for key in _COMMON_KEYS:
        if key not in first:
            raise ValueError(f"missing shard metadata: {key}")
    shard_count = int(first["num_shards"])
    total_samples = int(first["total_samples"])
    if shard_count <= 0 or total_samples < 0:
        raise ValueError("invalid shard metadata")
    expected_indices = set(range(shard_count))
    seen_shards: set[int] = set()
    seen_global: set[int] = set()
    seen_samples: set[str] = set()
    rows: list[dict] = []
    for result in results:
        for key in _COMMON_KEYS:
            if result.get(key) != first.get(key):
                raise ValueError(f"shard metadata mismatch: {key}")
        index = result.get("shard_index")
        if not isinstance(index, int) or index in seen_shards or index not in expected_indices:
            raise ValueError("duplicate or invalid shard index")
        if result.get("selected_samples") != len(result.get("global_indices", [])):
            raise ValueError("selected_samples does not match global_indices")
        if len(result.get("rows", [])) != result.get("selected_samples"):
            raise ValueError("row count does not match selected_samples")
        shard_globals = list(result["global_indices"])
        if len(set(shard_globals)) != len(shard_globals):
            raise ValueError("duplicate global index within shard")
        row_globals = [row.get("global_index") for row in result["rows"]]
        if row_globals != shard_globals:
            raise ValueError("rows are not aligned with global_indices")
        if seen_global.intersection(shard_globals):
            raise ValueError("duplicate sample across shards")
        sample_ids = [row.get("sample_id") for row in result["rows"]]
        if any(sample_id is None for sample_id in sample_ids) or seen_samples.intersection(sample_ids):
            raise ValueError("duplicate or missing sample_id across shards")
        seen_shards.add(index)
        seen_global.update(shard_globals)
        seen_samples.update(sample_ids)
        rows.extend(result["rows"])
    if seen_shards != expected_indices:
        raise ValueError("missing shard result")
    if seen_global != set(range(total_samples)):
        raise ValueError("shards do not cover every global sample")
    return first, sorted(rows, key=lambda row: row["global_index"])


def _base_result(first: dict, rows: list[dict]) -> dict:
    return {
        "manifest": first.get("manifest"),
        "manifest_sha256": first["manifest_sha256"],
        "manifest_sample_count": first["manifest_sample_count"],
        "total_samples": first["total_samples"],
        "num_shards": first["num_shards"],
        "shards": list(range(first["num_shards"])),
        "checkpoint": first["checkpoint"],
        "audio_ablation": first["audio_ablation"],
        "seed": first["seed"],
        "samples": len(rows),
        "rows": rows,
    }


def _merge_tf(first: dict, rows: list[dict]) -> dict:
    result = _base_result(first, rows)
    total_tokens = sum(int(row["supervised_tokens"]) for row in rows)
    if total_tokens <= 0:
        raise ValueError("TF shards contain no supervised tokens")
    result["supervised_tokens"] = total_tokens
    result["loss"] = sum(float(row["loss"]) * int(row["supervised_tokens"]) for row in rows) / total_tokens
    result["perplexity"] = math.exp(min(result["loss"], 20.0))
    eos_rows = [row for row in rows if row.get("eos_tokens", 0) and row.get("eos_loss") is not None]
    eos_tokens = sum(int(row.get("eos_tokens", 0)) for row in eos_rows)
    result["eos_tokens"] = eos_tokens
    result["eos_loss"] = (
        sum(float(row["eos_loss"]) * int(row["eos_tokens"]) for row in eos_rows) / eos_tokens
        if eos_tokens else None
    )
    return result


def _duration_bucket(duration_ms: int) -> str:
    if duration_ms < 10_000:
        return "<10s"
    if duration_ms < 15_000:
        return "10-15s"
    return ">=15s"


def _group_summary(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    summary = {}
    for name, group in sorted(groups.items()):
        words = sum(int(row["reference_words"]) for row in group)
        chars = sum(int(row["reference_chars"]) for row in group)
        summary[name] = {
            "samples": len(group),
            "word_errors": sum(int(row["word_errors"]) for row in group),
            "reference_words": words,
            "wer": sum(int(row["word_errors"]) for row in group) / max(1, words),
            "char_errors": sum(int(row["char_errors"]) for row in group),
            "reference_chars": chars,
            "cer": sum(int(row["char_errors"]) for row in group) / max(1, chars),
        }
    return summary


def _merge_wer(first: dict, rows: list[dict]) -> dict:
    result = _base_result(first, rows)
    word_errors = sum(int(row["word_errors"]) for row in rows)
    reference_words = sum(int(row["reference_words"]) for row in rows)
    char_errors = sum(int(row["char_errors"]) for row in rows)
    reference_chars = sum(int(row["reference_chars"]) for row in rows)
    result.update({
        "word_errors": word_errors,
        "reference_words": reference_words,
        "char_errors": char_errors,
        "reference_chars": reference_chars,
        "wer": word_errors / max(1, reference_words),
        "cer": char_errors / max(1, reference_chars),
        "eos_count": sum(bool(row.get("generated_eos", False)) for row in rows),
        "eos_rate": sum(bool(row.get("generated_eos", False)) for row in rows) / max(1, len(rows)),
        "exact_match_count": sum(bool(row.get("exact_match", False)) for row in rows),
        "exact_match_rate": sum(bool(row.get("exact_match", False)) for row in rows) / max(1, len(rows)),
        "average_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / max(1, len(rows)),
        "speaker_summary": _group_summary(rows, lambda row: row["speaker"]),
        "duration_summary": _group_summary(rows, lambda row: _duration_bucket(int(row["duration_ms"]))),
        "typical_failures": sorted(rows, key=lambda row: (float(row["wer"]), float(row["cer"])), reverse=True)[:10],
    })
    return result


def merge_shard_results(results: Sequence[dict]) -> dict:
    first, rows = _validate_inputs(results)
    if rows and "supervised_tokens" in rows[0]:
        return _merge_tf(first, rows)
    if rows and "word_errors" in rows[0]:
        return _merge_wer(first, rows)
    raise ValueError("unrecognized shard result kind")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    merged = merge_shard_results(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in merged.items() if key not in {"rows", "typical_failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
