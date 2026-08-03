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
    from qwen_asr import Qwen3ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the runtime shape probe")

    device = torch.device(args.device)
    wrapper = Qwen3ASRModel.from_pretrained(
        args.audio_model,
        dtype=torch.bfloat16,
        device_map=str(device),
        max_inference_batch_size=1,
    )
    model = wrapper.model
    model.eval()
    thinker = model.thinker
    audio_config = thinker.config.audio_config
    features = torch.zeros(
        (1, audio_config.num_mel_bins, args.frames), dtype=torch.bfloat16, device=device
    )
    feature_mask = torch.ones((1, args.frames), dtype=torch.long, device=device)
    with torch.inference_mode():
        projected_output = thinker.get_audio_features(
            input_features=features, feature_attention_mask=feature_mask
        )
    result = {
        "device": str(device),
        "input_features_shape": list(features.shape),
        "native_projector_shape": list(projected_output.shape),
        "encoder_hidden_size": int(audio_config.d_model),
        "native_projector_output_size": int(projected_output.shape[-1]),
        "audio_tokens": int(projected_output.shape[0]),
        "input_frames_per_audio_token": args.frames / int(projected_output.shape[0]),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
