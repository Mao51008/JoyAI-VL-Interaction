"""Cache frozen Qwen3-ASR features for projector stage one."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..schema import load_samples
from .collator import _load_mono_audio


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument(
        "--waveform-zero",
        action="store_true",
        help="replace each waveform with zeros before Qwen3-ASR feature extraction",
    )
    a = p.parse_args()
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(a.audio_model, dtype=torch.bfloat16, device_map=a.device,
                                          max_inference_batch_size=1).model.eval()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.audio_model, fix_mistral_regex=True)
    samples = load_samples(a.manifest)[:a.limit]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.shard_size <= 0: raise ValueError("--shard-size must be positive")
    records = []; shard_samples = []; shard_index = 0
    def flush() -> None:
        nonlocal shard_samples, shard_index
        if not shard_samples: return
        name = f"shard-{shard_index:05d}.pt"; torch.save({"samples": shard_samples}, a.output_dir / name)
        shard_samples = []; shard_index += 1
    for sample in samples:
        waveform, rate = _load_mono_audio(sample.audio[0].path)
        if rate != 16000: raise ValueError(f"{sample.sample_id}: expected 16 kHz")
        if a.waveform_zero:
            waveform = waveform * 0.0
        batch = processor(text=processor.audio_token, audio=waveform, sampling_rate=16000,
                          return_tensors="pt", padding=True).to(model.device).to(model.dtype)
        with torch.inference_mode():
            features = model.thinker.get_audio_features(
                input_features=batch["input_features"],
                feature_attention_mask=batch["feature_attention_mask"],
            ).cpu()
        shard_samples.append({"sample_id": sample.sample_id, "features": features,
                              "media_sha256": sample.metadata["media_sha256"]})
        records.append({"sample_id": sample.sample_id, "shard": f"shard-{shard_index:05d}.pt",
                        "offset": len(shard_samples) - 1, "tokens": int(features.shape[0]),
                        "media_sha256": sample.metadata["media_sha256"],
                        "waveform_zero": a.waveform_zero})
        if len(shard_samples) == a.shard_size: flush()
    flush()
    (a.output_dir / "index.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cached_samples": len(records), "output_dir": str(a.output_dir),
                      "waveform_zero": a.waveform_zero}, indent=2))
if __name__ == "__main__": main()
