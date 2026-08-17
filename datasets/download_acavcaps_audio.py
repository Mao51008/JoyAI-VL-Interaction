"""Download ACAVCaps manifest entries as 16 kHz mono WAV clips with yt-dlp."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
from pathlib import Path
from typing import Any


def destination_for(row: dict[str, Any], audio_root: Path) -> Path:
    return audio_root / row["category"] / f"{row['sample_key']}.wav"


def yt_dlp_command(row: dict[str, Any], audio_root: Path, yt_dlp: str, proxy: str | None) -> list[str]:
    destination = destination_for(row, audio_root)
    command = [
        yt_dlp,
        "--no-playlist",
        "--continue",
        "--no-overwrites",
        "--format",
        "bestaudio/best",
        "--download-sections",
        f"*{row['start_seconds']}-{row['end_seconds']}",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "ffmpeg:-ac 1 -ar 16000",
        "--output",
        str(destination.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={row['video_id']}",
    ]
    if proxy:
        command[1:1] = ["--proxy", proxy]
    return command


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) == limit:
                break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--yt-dlp", default="yt-dlp")
    parser.add_argument("--proxy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    rows = read_manifest(args.manifest, args.limit)
    args.audio_root.mkdir(parents=True, exist_ok=True)
    args.status_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def download(row: dict[str, Any]) -> dict[str, Any]:
        destination = destination_for(row, args.audio_root)
        if destination.is_file() and destination.stat().st_size > 0:
            return {"sample_key": row["sample_key"], "status": "existing", "path": str(destination)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(yt_dlp_command(row, args.audio_root, args.yt_dlp, args.proxy), capture_output=True, text=True)
        return {
            "sample_key": row["sample_key"],
            "status": "downloaded" if result.returncode == 0 and destination.is_file() else "failed",
            "returncode": result.returncode,
            "path": str(destination),
            "stderr": result.stderr[-1000:],
        }

    with args.status_jsonl.open("a", encoding="utf-8") as status_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(download, rows):
                with lock:
                    status_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    status_handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
