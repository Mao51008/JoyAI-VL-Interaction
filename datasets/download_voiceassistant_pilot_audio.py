"""Download the selected VoiceAssistant user-audio pilot with safe resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote


DATA_ROOT = Path("/data/maoyy/datasets/audio_understanding_pilot")
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/data/maoyy").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"output directory must be under {allowed}: {resolved}")
    return resolved


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    paths: set[str] = set()
    for index, row in enumerate(rows):
        source = str(row.get("source_audio_path", ""))
        if not source.startswith("user/") or source in paths:
            raise ValueError(f"manifest row {index} has invalid or duplicate user audio path")
        paths.add(source)
    return rows


def _audio_url(endpoint: str, source_audio_path: str) -> str:
    encoded = quote(source_audio_path, safe="/")
    return (
        f"{endpoint.rstrip('/')}/datasets/shenyunhang/VoiceAssistant-400K/resolve/main/"
        f"audio/{encoded}?download=true"
    )


def _download_one(
    row: dict[str, Any], output_dir: Path, endpoint: str, proxy: str | None
) -> str:
    source = str(row["source_audio_path"])
    target = output_dir / source
    partial = target.with_name(target.name + ".partial")
    if target.is_file():
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl"]
    if proxy:
        command.extend(["--proxy", proxy])
    command.extend(["--fail", "--location", "--retry", "5", "--retry-all-errors"])
    if partial.exists():
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(partial), _audio_url(endpoint, source)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace existing audio: {target}")
    partial.replace(target)
    return "downloaded"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download selected VoiceAssistant user audio in parallel.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_ROOT / "manifests/voiceassistant_single_turn.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "voiceassistant_400k/audio")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--proxy", help="Optional explicit HTTP proxy passed to curl.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true", help="Download audio; omitted means validate only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output_dir = _require_data_root(args.output_dir)
    rows = load_manifest(args.manifest)
    if not args.run:
        print(json.dumps({"run": False, "samples": len(rows)}, ensure_ascii=False))
        return 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    outcomes = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_one, row, output_dir, args.endpoint, args.proxy): row
            for row in rows
        }
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="VoiceAssistant audio", unit="file", disable=args.no_progress)
        for future in iterator:
            row = futures[future]
            try:
                outcomes[future.result()] += 1
            except Exception as error:
                outcomes["failed"] += 1
                failures.append(f"{row['sample_id']}: {error}")
    print(json.dumps({"samples": len(rows), **outcomes}, ensure_ascii=False, sort_keys=True))
    if failures:
        raise RuntimeError("VoiceAssistant audio failures: " + "\n".join(failures[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
