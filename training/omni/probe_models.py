"""Inspect base model configs before wiring the real stage-one adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTERESTING_FIELDS = (
    "hidden_size",
    "audio_hidden_size",
    "text_config",
    "audio_config",
    "projector_hidden_size",
    "sampling_rate",
    "model_type",
    "architectures",
)


def inspect_config(model_path: str, *, trust_remote_code: bool) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to inspect real model configs") from exc

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    raw = config.to_dict()
    return {
        "model_path": model_path,
        "config_class": type(config).__name__,
        "fields": {name: raw[name] for name in INTERESTING_FIELDS if name in raw},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    result = {
        "audio_model": inspect_config(
            args.audio_model, trust_remote_code=args.trust_remote_code
        ),
        "joyai_model": inspect_config(
            args.joyai_model, trust_remote_code=args.trust_remote_code
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
