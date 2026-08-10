"""Download mapped JoyAI Stage3 media with per-source storage quotas.

The input manifest is JSONL.  Each record must contain ``source``,
``video_name``, ``download_url`` and ``relative_path``.  Optional
``size_bytes`` and ``sha256`` make planning and post-download verification
deterministic.  The command is dry-run by default; network I/O requires
``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("source", "video_name", "download_url", "relative_path"):
                if not str(row.get(field, "")).strip():
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            rows.append(row)
    return rows


def load_quotas(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quota config must be a JSON object of source -> bytes")
    quotas: dict[str, int] = {}
    for source, limit in value.items():
        if not isinstance(limit, int) or limit < 0:
            raise ValueError(f"invalid quota for {source!r}")
        quotas[str(source)] = limit
    return quotas


def safe_destination(output_root: Path, relative_path: str) -> Path:
    candidate = PurePosixPath(relative_path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"unsafe relative_path: {relative_path!r}")
    destination = (output_root / Path(*candidate.parts)).resolve()
    root = output_root.resolve()
    if root not in destination.parents:
        raise ValueError(f"destination escapes output root: {relative_path!r}")
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_usage(output_root: Path, rows: list[dict[str, Any]]) -> Counter[str]:
    usage: Counter[str] = Counter()
    for row in rows:
        destination = safe_destination(output_root, str(row["relative_path"]))
        if destination.is_file():
            usage[str(row["source"])] += destination.stat().st_size
    return usage


def build_plan(
    rows: list[dict[str, Any]],
    output_root: Path,
    quotas: dict[str, int],
    *,
    total_limit_bytes: int,
    max_file_bytes: int | None,
) -> list[dict[str, Any]]:
    if total_limit_bytes <= 0:
        raise ValueError("total_limit_bytes must be positive")
    usage = existing_usage(output_root, rows)
    total_usage = sum(usage.values())
    planned: list[dict[str, Any]] = []
    seen_destinations: set[Path] = set()
    for row in rows:
        source = str(row["source"])
        destination = safe_destination(output_root, str(row["relative_path"]))
        if destination in seen_destinations:
            raise ValueError(f"duplicate destination in manifest: {destination}")
        seen_destinations.add(destination)
        size = int(row.get("size_bytes", 0) or 0)
        if size < 0:
            raise ValueError(f"negative size_bytes for {source}/{row['video_name']}")
        status = "ready"
        reason = ""
        if destination.is_file():
            expected_hash = str(row.get("sha256", "")).strip().lower()
            status = "already_present"
            if expected_hash and sha256(destination) != expected_hash:
                status, reason = "hash_mismatch", "existing_file_hash_mismatch"
        elif not size:
            status, reason = "blocked", "missing_size_bytes"
        elif max_file_bytes is not None and size > max_file_bytes:
            status, reason = "blocked", "file_exceeds_max_file_bytes"
        elif source not in quotas:
            status, reason = "blocked", "source_has_no_quota"
        elif usage[source] + size > quotas[source]:
            status, reason = "blocked", "source_quota_exceeded"
        elif total_usage + size > total_limit_bytes:
            status, reason = "blocked", "total_quota_exceeded"
        if status == "ready":
            usage[source] += size
            total_usage += size
        planned.append(
            {
                **row,
                "destination": str(destination),
                "planned_status": status,
                "planned_reason": reason,
            }
        )
    return planned


def execute_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in plan:
        receipt = dict(item)
        if item["planned_status"] != "ready":
            receipts.append(receipt)
            continue
        destination = Path(str(item["destination"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(str(item["download_url"])) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            expected_size = int(item.get("size_bytes", 0) or 0)
            expected_hash = str(item.get("sha256", "")).strip().lower()
            if expected_size and temporary.stat().st_size != expected_size:
                raise ValueError("downloaded_size_mismatch")
            if expected_hash and sha256(temporary) != expected_hash:
                raise ValueError("downloaded_hash_mismatch")
            temporary.replace(destination)
            receipt["planned_status"] = "downloaded"
        except Exception as exc:  # preserve the partial file for explicit inspection
            receipt["planned_status"] = "failed"
            receipt["planned_reason"] = f"{type(exc).__name__}: {exc}"
        receipts.append(receipt)
    return receipts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--quota-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--total-limit-gb", type=int, default=475)
    parser.add_argument("--max-file-bytes", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    plan = build_plan(
        rows,
        args.output_root,
        load_quotas(args.quota_config),
        total_limit_bytes=args.total_limit_gb * 1024**3,
        max_file_bytes=args.max_file_bytes,
    )
    receipts = execute_plan(plan) if args.execute else plan
    write_jsonl(args.receipt, receipts)
    summary = Counter(item["planned_status"] for item in receipts)
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
