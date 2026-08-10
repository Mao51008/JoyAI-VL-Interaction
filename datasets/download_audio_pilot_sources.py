"""Download bounded source artifacts for the audio-understanding pilot.

The script is intentionally manifest-first.  It never downloads media unless
``--run`` is present, and Clotho-AQA audio additionally needs an explicit flag
because Zenodo publishes it as one 3.1 GB archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

VOICEASSISTANT_DATASET = "datasets/shenyunhang/VoiceAssistant-400K"
DEFAULT_HUGGINGFACE_ENDPOINT = "https://huggingface.co"
CLOTHO_AQA_RECORD = "https://zenodo.org/records/6473207/files"
CLOTHO_AQA_FILES = {
    "train": "clotho_aqa_train.csv",
    "dev": "clotho_aqa_val.csv",
    "test": "clotho_aqa_test.csv",
    "metadata": "clotho_aqa_metadata.csv",
    "license": "LICENSE.txt",
}
CLOTHO_AQA_AUDIO = "audio_files.zip"


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/data/maoyy").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"output root must be under {allowed}: {resolved}")
    return resolved


def _curl_args(proxy: str | None) -> list[str]:
    return ["curl", "--proxy", proxy] if proxy else ["curl"]


def _content_length(url: str, transport: str, proxy: str | None) -> int | None:
    if transport == "curl":
        result = subprocess.run(
            [*_curl_args(proxy), "--fail", "--location", "--silent", "--show-error", "--head", url],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.lower().startswith("content-length:")
        ]
        return int(values[-1]) if values else None
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def _download(
    url: str,
    destination: Path,
    max_bytes: int,
    run: bool,
    check_remote: bool,
    transport: str,
    proxy: str | None,
    resume: bool,
) -> dict[str, Any]:
    size = _content_length(url, transport, proxy) if run or check_remote else None
    existing_bytes = destination.stat().st_size if destination.exists() else 0
    if destination.exists() and not (run and resume and transport == "curl"):
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    if size is not None and existing_bytes > size:
        raise ValueError(f"existing file exceeds remote size: {destination}")
    if size is not None and size > max_bytes:
        raise ValueError(f"refusing {url}: {size} bytes exceeds limit {max_bytes}")
    result = {
        "url": url,
        "destination": str(destination),
        "bytes": size,
        "existing_bytes": existing_bytes,
        "downloaded": False,
    }
    if not run:
        return result
    if size is not None and existing_bytes == size:
        result["sha256"] = _sha256(destination)
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    if transport == "curl":
        command = [*_curl_args(proxy), "--fail", "--location", "--retry", "3"]
        if resume and destination.exists():
            command.extend(["--continue-at", "-"])
        command.extend(["--output", str(destination), url])
        subprocess.run(command, check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("xb") as handle:
            shutil.copyfileobj(response, handle)
    actual = destination.stat().st_size
    if actual > max_bytes:
        raise ValueError(f"download exceeded limit: {destination} is {actual} bytes")
    result.update({"bytes": actual, "sha256": _sha256(destination), "downloaded": True})
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_downloads(
    output_root: Path,
    include_clotho_audio: bool,
    run: bool,
    check_remote: bool = False,
    transport: str = "urllib",
    proxy: str | None = None,
    resume: bool = False,
    huggingface_endpoint: str = DEFAULT_HUGGINGFACE_ENDPOINT,
) -> list[dict[str, Any]]:
    root = _require_data_root(output_root)
    voiceassistant_url = (
        f"{huggingface_endpoint.rstrip('/')}/{VOICEASSISTANT_DATASET}/resolve/main/data.jsonl?download=true"
    )
    plans = [
        _download(
            voiceassistant_url,
            root / "voiceassistant_400k" / "raw" / "data.jsonl",
            512 * 1024**2,
            run,
            check_remote,
            transport,
            proxy,
            resume,
        )
    ]
    for name, filename in CLOTHO_AQA_FILES.items():
        plans.append(
            _download(
                f"{CLOTHO_AQA_RECORD}/{filename}?download=1",
                root / "clotho_aqa" / "raw" / filename,
                16 * 1024**2,
                run,
                check_remote,
                transport,
                proxy,
                resume,
            )
        )
    if include_clotho_audio:
        plans.append(
            _download(
                f"{CLOTHO_AQA_RECORD}/{CLOTHO_AQA_AUDIO}?download=1",
                root / "clotho_aqa" / "raw" / CLOTHO_AQA_AUDIO,
                4 * 1024**3,
                run,
                check_remote,
                transport,
                proxy,
                resume,
            )
        )
    return plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded manifest-first audio pilot downloader.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/maoyy/datasets/audio_understanding_pilot"),
    )
    parser.add_argument("--proxy", help="Explicit HTTP proxy address passed to curl transport.")
    parser.add_argument(
        "--huggingface-endpoint",
        default=DEFAULT_HUGGINGFACE_ENDPOINT,
        help="Hugging Face endpoint, e.g. https://hf-mirror.com for the regional mirror.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing partial file; supported only with --run --transport curl.",
    )
    parser.add_argument(
        "--transport",
        choices=("urllib", "curl"),
        default="urllib",
        help="Use curl for proxy-compatible server transfers when Python urllib fails.",
    )
    parser.add_argument(
        "--download-clotho-aqa-audio",
        action="store_true",
        help="Also download Zenodo's 3.1 GB audio archive after manifest review.",
    )
    parser.add_argument("--run", action="store_true", help="Perform downloads; omitted means dry-run.")
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Read remote Content-Length values during dry-run; requires server proxy access.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = plan_downloads(
        args.output_root,
        args.download_clotho_aqa_audio,
        args.run,
        args.check_remote,
        args.transport,
        args.proxy,
        args.resume,
        args.huggingface_endpoint,
    )
    print(json.dumps({"run": args.run, "downloads": plans}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
