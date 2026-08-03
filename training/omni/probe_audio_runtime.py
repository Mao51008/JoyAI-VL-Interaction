"""Measure Qwen3-ASR encoder and native projector shapes with synthetic audio features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames <= 0 or args.frames % 100:
        raise ValueError("--frames must be a positive multiple of 100 for Qwen3-ASR")

    import torch
    from transformers import Qwen3ASRForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the runtime shape probe")

    device = torch.device(args.device)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.audio_model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    audio_config = model.config.audio_config
    features = torch.zeros(
        (1, audio_config.num_mel_bins, args.frames), dtype=torch.bfloat16, device=device
    )
    feature_mask = torch.ones((1, args.frames), dtype=torch.long, device=device)
    with torch.inference_mode():
        encoder_output = model.model.audio_tower(
            input_features=features, input_features_mask=feature_mask
        )
        projected_output = model.get_audio_features(
            input_features=features, input_features_mask=feature_mask, return_dict=True
        )
    result = {
        "device": str(device),
        "input_features_shape": list(features.shape),
        "encoder_last_hidden_state_shape": list(encoder_output.last_hidden_state.shape),
        "native_projector_shape": list(projected_output.pooler_output.shape),
        "encoder_hidden_size": int(encoder_output.last_hidden_state.shape[-1]),
        "native_projector_output_size": int(projected_output.pooler_output.shape[-1]),
        "audio_tokens": int(projected_output.pooler_output.shape[0]),
        "input_frames_per_audio_token": args.frames / int(projected_output.pooler_output.shape[0]),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
