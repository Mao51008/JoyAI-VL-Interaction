"""Evaluate a projector checkpoint on cached, speaker-disjoint stage-one data."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from .projector import AudioProjector, AudioProjectorConfig
from .schema import load_samples
from .stage1_collator import JoyAIStage1TokenLayout, build_sample_sequence
from .stage1_data import pad_sequences
from .stage1_model import CachedProjectorStage1Model

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--joyai-model", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=2)
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
    projector.load_state_dict(state["projector"]); model = CachedProjectorStage1Model(llm, projector).eval()
    samples = load_samples(a.manifest); total_loss = 0.0; total_tokens = 0
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
                cached = torch.load(a.feature_dir / f"{sample.sample_id}.pt", map_location="cpu", weights_only=True)
                if cached["media_sha256"] != sample.metadata["media_sha256"]: raise ValueError("cached feature fingerprint mismatch")
                feature = cached["features"]; features.append(feature)
                sequences.append(build_sample_sequence(sample, tokenizer=tok, layout=layout, audio_token_count=feature.shape[0]))
            padded = pad_sequences(sequences, pad_token_id=layout.pad_token_id)
            audio = pad_sequence(features, batch_first=True).to(a.device, dtype=torch.bfloat16)
            mask = torch.arange(audio.shape[1], device=a.device)[None, :] < torch.tensor([x.shape[0] for x in features], device=a.device)[:, None]
            ids = torch.tensor(padded["input_ids"], device=a.device); labels = torch.tensor(padded["labels"], device=a.device)
            out = model(audio_features=audio, audio_attention_mask=mask, text_embeddings=llm.get_input_embeddings()(ids), audio_placeholder_mask=torch.tensor(padded["audio_placeholder_mask"], device=a.device), attention_mask=torch.tensor(padded["attention_mask"], device=a.device), labels=labels)
            tokens = int((labels != -100).sum()); total_tokens += tokens; total_loss += float(out.loss) * tokens
            if progress is not None:
                progress.set_postfix(samples=min(offset + len(batch_samples), len(samples)), loss=f"{total_loss / total_tokens:.4f}")
    loss = total_loss / total_tokens
    print(json.dumps({"checkpoint": str(a.checkpoint), "samples": len(samples), "supervised_tokens": total_tokens, "loss": loss, "perplexity": math.exp(min(loss, 20.0))}, indent=2))
if __name__ == "__main__": main()
