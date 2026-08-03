"""Cache frozen Qwen3-ASR features for projector stage one."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from .schema import load_samples
from .stage1_collator import _load_mono_audio

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(a.audio_model, dtype=torch.bfloat16, device_map=a.device,
                                          max_inference_batch_size=1).model.eval()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.audio_model, fix_mistral_regex=True)
    samples = load_samples(a.manifest)[:a.limit]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        waveform, rate = _load_mono_audio(sample.audio[0].path)
        if rate != 16000: raise ValueError(f"{sample.sample_id}: expected 16 kHz")
        batch = processor(text=processor.audio_token, audio=waveform, sampling_rate=16000,
                          return_tensors="pt", padding=True).to(model.device).to(model.dtype)
        with torch.inference_mode():
            features = model.thinker.get_audio_features(
                input_features=batch["input_features"],
                feature_attention_mask=batch["feature_attention_mask"],
            ).cpu()
        path = a.output_dir / f"{sample.sample_id}.pt"
        torch.save({"sample_id": sample.sample_id, "features": features,
                    "media_sha256": sample.metadata["media_sha256"]}, path)
        records.append({"sample_id": sample.sample_id, "path": path.name,
                        "tokens": int(features.shape[0]), "media_sha256": sample.metadata["media_sha256"]})
    (a.output_dir / "index.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cached_samples": len(records), "output_dir": str(a.output_dir)}, indent=2))
if __name__ == "__main__": main()
