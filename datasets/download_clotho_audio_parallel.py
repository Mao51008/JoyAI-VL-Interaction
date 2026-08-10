"""Reliably download the Clotho-AQA audio archive with parallel HTTP ranges."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler


DATA_ROOT = Path("/data/maoyy/datasets/audio_understanding_pilot/clotho_aqa/raw")
DEFAULT_URL = "https://zenodo.org/records/6473207/files/audio_files.zip?download=1"
DEFAULT_SIZE = 3_140_677_494


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/data/maoyy").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"output path must be under {allowed}: {resolved}")
    return resolved


def _load_completed(path: Path) -> set[int]:
    completed: set[int] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                completed.add(int(json.loads(line)["offset"]))
    return completed


def _download_range(
    url: str,
    proxy: str | None,
    output_fd: int,
    offset: int,
    end: int,
    retries: int,
) -> int:
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}) if proxy else ProxyHandler({}))
    expected = end - offset + 1
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
            with opener.open(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"range {offset}-{end} returned HTTP {response.status}")
                position = offset
                while (chunk := response.read(1024 * 1024)):
                    os.pwrite(output_fd, chunk, position)
                    position += len(chunk)
            if position != offset + expected:
                raise RuntimeError(f"range {offset}-{end} ended at {position}")
            return offset
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(30, attempt * 2))
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "audio_files.zip")
    parser.add_argument("--state", type=Path, default=DATA_ROOT / "audio_files.ranges.jsonl")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--proxy", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_mib < 1 or args.workers < 1 or args.size < 1:
        raise ValueError("size, chunk-mib, and workers must be positive")
    output = _require_data_root(args.output)
    state = _require_data_root(args.state)
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_size = output.stat().st_size if output.exists() else 0
    chunk_size = args.chunk_mib * 1024 * 1024
    resume_offset = initial_size // chunk_size * chunk_size
    completed = _load_completed(state)
    offsets = list(range(resume_offset, args.size, chunk_size))
    pending = [offset for offset in offsets if offset not in completed]
    print(
        f"resume_offset={resume_offset} completed={len(completed)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )
    output_fd = os.open(output, os.O_RDWR | os.O_CREAT)
    lock = threading.Lock()
    try:
        with state.open("a", encoding="utf-8") as state_file:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        _download_range,
                        args.url,
                        args.proxy,
                        output_fd,
                        offset,
                        min(args.size - 1, offset + chunk_size - 1),
                        args.retries,
                    ): offset
                    for offset in pending
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    offset = future.result()
                    with lock:
                        state_file.write(json.dumps({"offset": offset}) + "\n")
                        state_file.flush()
                        os.fsync(state_file.fileno())
                    print(f"completed={index}/{len(pending)} offset={offset}", flush=True)
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    if output.stat().st_size != args.size:
        raise RuntimeError(f"archive size mismatch: {output.stat().st_size} != {args.size}")
    print("download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
