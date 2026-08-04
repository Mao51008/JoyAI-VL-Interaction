"""Run CPU-only checks before loading real stage-one models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..schema import load_samples


def run(config_path: Path, manifest_path: Path, *, check_media: bool = True) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "projector_only_audio_alignment":
        raise ValueError("config stage is not projector_only_audio_alignment")
    if config.get("dimensions_require_probe", True):
        raise ValueError("config still requires a real model shape probe")

    samples = load_samples(manifest_path)
    missing_media = []
    missing_targets = []
    for sample in samples:
        if not str(sample.metadata.get("assistant_target_text", "")).strip():
            missing_targets.append(sample.sample_id)
        if check_media:
            for segment in sample.audio:
                if not Path(segment.path).is_file():
                    missing_media.append(segment.path)
    if missing_targets:
        raise ValueError(f"samples missing assistant targets: {missing_targets[:3]}")
    if missing_media:
        raise ValueError(f"missing audio files: {missing_media[:3]}")
    return {
        "config": str(config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "samples": len(samples),
        "assistant_targets": len(samples),
        "media_checked": check_media,
        "dimensions_require_probe": False,
        "gpu_touched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--no-media-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.manifest, check_media=not args.no_media_check), indent=2))


if __name__ == "__main__":
    main()
