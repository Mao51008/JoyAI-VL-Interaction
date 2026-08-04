"""Train only audio_projector from cached ASR features; never saves base weights."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from .projector import AudioProjector, AudioProjectorConfig, assert_projector_gradients, trainable_parameters
from .schema import load_samples
from .stage1_batching import distribute_batches, plan_length_aware_batches
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
    p.add_argument("--max-attention-cost", type=int, default=1_000_000,
                   help="maximum padded audio attention cost per local batch; long samples remain single-item batches")
    p.add_argument("--resume", action="store_true"); p.add_argument("--no-progress", action="store_true")
    p.add_argument("--ddp", action="store_true")
    a = p.parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.max_attention_cost <= 0:
        raise ValueError("--steps, --batch-size and --max-attention-cost must be positive")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    world_size = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if a.ddp:
        if world_size < 2: raise ValueError("--ddp requires torchrun with at least two processes")
        local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank); dist.init_process_group("nccl")
        device = f"cuda:{local_rank}"
    else: device = a.device
    config = json.loads(a.config.read_text(encoding="utf-8")); config_hash = _hash(a.config); manifest_hash = _hash(a.manifest)
    tok = AutoTokenizer.from_pretrained(a.joyai_model, fix_mistral_regex=True); layout = JoyAIStage1TokenLayout.from_tokenizer(tok)
    llm = AutoModelForImageTextToText.from_pretrained(a.joyai_model, dtype=torch.bfloat16).to(device)
    core_model = CachedProjectorStage1Model(llm, AudioProjector(AudioProjectorConfig(**config["projector"])).to(device=device, dtype=torch.bfloat16))
    model = DistributedDataParallel(core_model, device_ids=[local_rank]) if a.ddp else core_model
    opt = torch.optim.AdamW(trainable_parameters(core_model), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=a.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    samples = load_samples(a.manifest); cache = FeatureCache(a.feature_dir)
    token_counts = {sample.sample_id: cache.index[sample.sample_id]["tokens"] for sample in samples}
    global_batches = plan_length_aware_batches(
        [sample.sample_id for sample in samples], token_counts,
        max_batch_size=a.batch_size, max_attention_cost=a.max_attention_cost,
    )
    rank_batches = distribute_batches(global_batches, world_size)[rank]
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if rank == 0: a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.ddp: dist.barrier()
    latest = a.output_dir / "latest.pt"; start = 0
    if a.resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash or state["config_sha256"] != config_hash or state["joyai_model"] != a.joyai_model: raise ValueError("checkpoint provenance mismatch")
        core_model.audio_projector.load_state_dict(state["projector"]); opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"]); start = state["step"]
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    steps = range(start, a.steps)
    progress = None if rank or a.no_progress or tqdm is None else tqdm(steps, total=a.steps, initial=start, unit="step")
    for step in steps if progress is None else progress:
        batch_samples = [sample_by_id[sample_id] for sample_id in rank_batches[step % len(rank_batches)]]
        features = []; sequences = []
        for sample in batch_samples:
            cached = cache.get(sample.sample_id)
            if cached["media_sha256"] != sample.metadata["media_sha256"]: raise ValueError("cached feature fingerprint mismatch")
            feature = cached["features"]; features.append(feature)
            sequences.append(build_sample_sequence(sample, tokenizer=tok, layout=layout, audio_token_count=feature.shape[0]))
        padded = pad_sequences(sequences, pad_token_id=layout.pad_token_id)
        audio = pad_sequence(features, batch_first=True).to(device, dtype=torch.bfloat16)
        audio_mask = torch.arange(audio.shape[1], device=device)[None, :] < torch.tensor([x.shape[0] for x in features], device=device)[:, None]
        ids = torch.tensor(padded["input_ids"], device=device); text = llm.get_input_embeddings()(ids)
        out = model(audio_features=audio, audio_attention_mask=audio_mask, text_embeddings=text, audio_placeholder_mask=torch.tensor(padded["audio_placeholder_mask"], device=device), attention_mask=torch.tensor(padded["attention_mask"], device=device), labels=torch.tensor(padded["labels"], device=device))
        opt.zero_grad(); out.loss.backward(); assert_projector_gradients(core_model, core_model.audio_projector); opt.step(); scheduler.step()
        if rank == 0: torch.save({"format":"projector-stage1-v2", "step":step + 1, "projector":core_model.audio_projector.state_dict(), "optimizer":opt.state_dict(), "scheduler":scheduler.state_dict(), "scaler":scaler.state_dict(), "manifest_sha256":manifest_hash, "config_sha256":config_hash, "config":config, "joyai_model":a.joyai_model}, latest)
        loss = out.loss.detach()
        if a.ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        record = {"step":step + 1, "loss":float(loss), "batch_size":len(batch_samples),
                  "max_attention_cost": a.max_attention_cost, "planned_batches_per_rank": len(rank_batches),
                  "learning_rate": scheduler.get_last_lr()[0]}
        if progress is not None:
            progress.set_postfix(loss=f"{record['loss']:.4f}", lr=f"{record['learning_rate']:.2e}")
        if rank == 0: print(json.dumps(record))
    if a.ddp: dist.destroy_process_group()
if __name__ == "__main__": main()
