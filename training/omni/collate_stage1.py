"""CPU-only smoke test for the real projector stage-one collator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import load_samples
from .stage1_collator import ProjectorStage1Collator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    import qwen_asr  # Registers Qwen3-ASR's processor with Transformers.
    from transformers import AutoProcessor, AutoTokenizer

    del qwen_asr
    samples = load_samples(args.manifest)[: args.limit]
    batch = ProjectorStage1Collator(
        asr_processor=AutoProcessor.from_pretrained(args.audio_model),
        joyai_tokenizer=AutoTokenizer.from_pretrained(args.joyai_model),
    )(samples)
    print(json.dumps({
        "sample_ids": batch["sample_ids"],
        "input_ids_shape": list(batch["input_ids"].shape),
        "input_features_shape": list(batch["input_features"].shape),
        "audio_placeholder_counts": batch["audio_placeholder_mask"].sum(dim=1).tolist(),
        "supervised_token_counts": (batch["labels"] != -100).sum(dim=1).tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
