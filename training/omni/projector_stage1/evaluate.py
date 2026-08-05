"""Evaluate a projector checkpoint on cached, speaker-disjoint stage-one data."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from ..schema import load_samples
from .ablation import apply_feature_ablation, sample_exchange_order, validate_exchange
from .collator import JoyAIStage1TokenLayout, build_sample_sequence
from .data import pad_sequences
from .feature_cache import FeatureCache
from .model import replace_audio_placeholders
from .projector import AudioProjector, AudioProjectorConfig


def apply_audio_ablation(feature, mode: str):
    """Apply a feature-space ablation; waveform-zero is supplied by a separate cache."""
    return apply_feature_ablation(feature, mode)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--joyai-model", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--audio-ablation", choices=("none", "zero", "feature-zero", "projected-zero", "shuffle"), default="none")
    p.add_argument("--zero-feature-dir", type=Path,
                   help="feature cache extracted from zero waveforms; required for --audio-ablation zero")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no-progress", action="store_true")
    a = p.parse_args()
    if a.batch_size <= 0: raise ValueError("--batch-size must be positive")
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if state.get("joyai_model") != a.joyai_model: raise ValueError("checkpoint JoyAI model mismatch")
    tok = AutoTokenizer.from_pretrained(a.joyai_model, fix_mistral_regex=True); layout = JoyAIStage1TokenLayout.from_tokenizer(tok)
    llm = AutoModelForImageTextToText.from_pretrained(a.joyai_model, dtype=torch.bfloat16).to(a.device)
    projector = AudioProjector(AudioProjectorConfig(**state["config"]["projector"])).to(a.device, dtype=torch.bfloat16)
    projector.load_state_dict(state["projector"])
    samples = load_samples(a.manifest); cache = FeatureCache(a.feature_dir)
    zero_cache = FeatureCache(a.zero_feature_dir) if a.zero_feature_dir is not None else None
    if a.audio_ablation == "zero" and zero_cache is None:
        raise ValueError("--zero-feature-dir is required for waveform-zero evaluation")
    normal_features = [cache.get(sample.sample_id) for sample in samples]
    sample_indices = {sample.sample_id: index for index, sample in enumerate(samples)}
    if any(item["media_sha256"] != sample.metadata["media_sha256"] for item, sample in zip(normal_features, samples)):
        raise ValueError("cached feature fingerprint mismatch")
    exchange_order = None
    if a.audio_ablation == "shuffle":
        exchange_order = sample_exchange_order(len(samples), random.Random(a.seed))
        validate_exchange(normal_features, exchange_order)
    total_loss = 0.0; total_tokens = 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    offsets = range(0, len(samples), a.batch_size)
    progress = None if a.no_progress or tqdm is None else tqdm(offsets, total=math.ceil(len(samples) / a.batch_size), unit="batch")
    with torch.inference_mode():
        for offset in offsets if progress is None else progress:
            batch_samples = samples[offset:offset + a.batch_size]; features = []; sequences = []
            for sample in batch_samples:
                sample_index = sample_indices[sample.sample_id]
                if a.audio_ablation == "zero":
                    cached = zero_cache.get(sample.sample_id)
                    if not zero_cache.index[sample.sample_id].get("waveform_zero", False):
                        raise ValueError("zero feature cache was not extracted from zero waveforms")
                    if cached["media_sha256"] != sample.metadata["media_sha256"]:
                        raise ValueError("zero feature cache fingerprint mismatch")
                elif a.audio_ablation == "shuffle":
                    cached = normal_features[exchange_order[sample_index]]
                else:
                    cached = normal_features[sample_index]
                feature_mode = "feature-zero" if a.audio_ablation == "feature-zero" else "none"
                feature = apply_audio_ablation(cached["features"], feature_mode); features.append(feature)
                sequences.append(build_sample_sequence(sample, tokenizer=tok, layout=layout, audio_token_count=feature.shape[0]))
            padded = pad_sequences(sequences, pad_token_id=layout.pad_token_id)
            audio = pad_sequence(features, batch_first=True).to(a.device, dtype=torch.bfloat16)
            mask = torch.arange(audio.shape[1], device=a.device)[None, :] < torch.tensor([x.shape[0] for x in features], device=a.device)[:, None]
            ids = torch.tensor(padded["input_ids"], device=a.device); labels = torch.tensor(padded["labels"], device=a.device)
            projected = projector(audio)
            if a.audio_ablation == "projected-zero":
                projected = torch.zeros_like(projected)
            embeds = replace_audio_placeholders(llm.get_input_embeddings()(ids), projected, torch.tensor(padded["audio_placeholder_mask"], device=a.device), mask)
            out = llm(inputs_embeds=embeds, attention_mask=torch.tensor(padded["attention_mask"], device=a.device), labels=labels)
            tokens = int((labels != -100).sum()); total_tokens += tokens; total_loss += float(out.loss) * tokens
            if progress is not None:
                progress.set_postfix(samples=min(offset + len(batch_samples), len(samples)), loss=f"{total_loss / total_tokens:.4f}")
    loss = total_loss / total_tokens
    print(json.dumps({"checkpoint": str(a.checkpoint), "audio_ablation": a.audio_ablation, "samples": len(samples), "supervised_tokens": total_tokens, "loss": loss, "perplexity": math.exp(min(loss, 20.0))}, indent=2))
if __name__ == "__main__": main()
