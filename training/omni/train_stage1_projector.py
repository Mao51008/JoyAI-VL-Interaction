"""Train only audio_projector from cached ASR features; never saves base weights."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .projector import AudioProjector, AudioProjectorConfig, assert_projector_gradients, trainable_parameters
from .schema import load_samples
from .stage1_collator import JoyAIStage1TokenLayout, build_sample_sequence
from .stage1_data import pad_sequences
from .stage1_model import CachedProjectorStage1Model
from .stage1_feature_cache import FeatureCache

def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--joyai-model", required=True); p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, required=True); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resume", action="store_true"); p.add_argument("--no-progress", action="store_true")
    a = p.parse_args()
    if a.steps <= 0 or a.batch_size <= 0: raise ValueError("--steps and --batch-size must be positive")
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    config = json.loads(a.config.read_text(encoding="utf-8")); config_hash = _hash(a.config); manifest_hash = _hash(a.manifest)
    tok = AutoTokenizer.from_pretrained(a.joyai_model, fix_mistral_regex=True); layout = JoyAIStage1TokenLayout.from_tokenizer(tok)
    llm = AutoModelForImageTextToText.from_pretrained(a.joyai_model, dtype=torch.bfloat16).to(a.device)
    model = CachedProjectorStage1Model(llm, AudioProjector(AudioProjectorConfig(**config["projector"])).to(device=a.device, dtype=torch.bfloat16))
    opt = torch.optim.AdamW(trainable_parameters(model), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=a.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    samples = load_samples(a.manifest); cache = FeatureCache(a.feature_dir); a.output_dir.mkdir(parents=True, exist_ok=True); latest = a.output_dir / "latest.pt"; start = 0
    if a.resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash or state["config_sha256"] != config_hash or state["joyai_model"] != a.joyai_model: raise ValueError("checkpoint provenance mismatch")
        model.audio_projector.load_state_dict(state["projector"]); opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"]); start = state["step"]
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    steps = range(start, a.steps)
    progress = None if a.no_progress or tqdm is None else tqdm(steps, total=a.steps, initial=start, unit="step")
    for step in steps if progress is None else progress:
        batch_samples = [samples[(step * a.batch_size + i) % len(samples)] for i in range(a.batch_size)]
        features = []; sequences = []
        for sample in batch_samples:
            cached = cache.get(sample.sample_id)
            if cached["media_sha256"] != sample.metadata["media_sha256"]: raise ValueError("cached feature fingerprint mismatch")
            feature = cached["features"]; features.append(feature)
            sequences.append(build_sample_sequence(sample, tokenizer=tok, layout=layout, audio_token_count=feature.shape[0]))
        padded = pad_sequences(sequences, pad_token_id=layout.pad_token_id)
        audio = pad_sequence(features, batch_first=True).to(a.device, dtype=torch.bfloat16)
        audio_mask = torch.arange(audio.shape[1], device=a.device)[None, :] < torch.tensor([x.shape[0] for x in features], device=a.device)[:, None]
        ids = torch.tensor(padded["input_ids"], device=a.device); text = llm.get_input_embeddings()(ids)
        out = model(audio_features=audio, audio_attention_mask=audio_mask, text_embeddings=text, audio_placeholder_mask=torch.tensor(padded["audio_placeholder_mask"], device=a.device), attention_mask=torch.tensor(padded["attention_mask"], device=a.device), labels=torch.tensor(padded["labels"], device=a.device))
        opt.zero_grad(); out.loss.backward(); assert_projector_gradients(model, model.audio_projector); opt.step(); scheduler.step()
        torch.save({"format":"projector-stage1-v2", "step":step + 1, "projector":model.audio_projector.state_dict(), "optimizer":opt.state_dict(), "scheduler":scheduler.state_dict(), "scaler":scaler.state_dict(), "manifest_sha256":manifest_hash, "config_sha256":config_hash, "config":config, "joyai_model":a.joyai_model}, latest)
        record = {"step":step + 1, "loss":float(out.loss.detach()), "batch_size":a.batch_size,
                  "learning_rate": scheduler.get_last_lr()[0]}
        if progress is not None:
            progress.set_postfix(loss=f"{record['loss']:.4f}", lr=f"{record['learning_rate']:.2e}")
        print(json.dumps(record))
if __name__ == "__main__": main()
