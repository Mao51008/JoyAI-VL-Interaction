"""Train only audio_projector from cached ASR features; never saves base weights."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .projector import AudioProjector, AudioProjectorConfig, assert_projector_gradients, trainable_parameters
from .schema import load_samples
from .stage1_collator import JoyAIStage1TokenLayout, build_sample_sequence
from .stage1_data import pad_sequences
from .stage1_model import CachedProjectorStage1Model

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--feature-dir",type=Path,required=True)
    p.add_argument("--joyai-model",required=True); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,required=True); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    config=json.loads(a.config.read_text(encoding="utf-8")); pc=AudioProjectorConfig(**config["projector"])
    tok=AutoTokenizer.from_pretrained(a.joyai_model,fix_mistral_regex=True); layout=JoyAIStage1TokenLayout.from_tokenizer(tok)
    llm=AutoModelForImageTextToText.from_pretrained(a.joyai_model,torch_dtype=torch.bfloat16).to(a.device)
    model=CachedProjectorStage1Model(llm,AudioProjector(pc).to(a.device)); opt=torch.optim.AdamW(trainable_parameters(model),lr=config["optimizer"]["learning_rate"],weight_decay=config["optimizer"]["weight_decay"])
    samples=load_samples(a.manifest); a.output_dir.mkdir(parents=True,exist_ok=True); start=0
    latest=a.output_dir/"latest.pt"
    if a.resume:
        state=torch.load(latest,map_location="cpu",weights_only=True); model.audio_projector.load_state_dict(state["projector"]); opt.load_state_dict(state["optimizer"]); start=state["step"]
    for step in range(start,a.steps):
        sample=samples[step%len(samples)]; cached=torch.load(a.feature_dir/f"{sample.sample_id}.pt",map_location="cpu",weights_only=True)
        if cached["media_sha256"]!=sample.metadata["media_sha256"]: raise ValueError("cached feature fingerprint mismatch")
        feature=cached["features"]; seq=build_sample_sequence(sample,tokenizer=tok,layout=layout,audio_token_count=feature.shape[0]); padded=pad_sequences([seq],pad_token_id=layout.pad_token_id)
        ids=torch.tensor(padded["input_ids"],device=a.device); text=llm.get_input_embeddings()(ids)
        out=model(audio_features=feature.unsqueeze(0).to(a.device,dtype=torch.bfloat16),audio_attention_mask=torch.ones(1,feature.shape[0],device=a.device,dtype=torch.bool),text_embeddings=text,audio_placeholder_mask=torch.tensor(padded["audio_placeholder_mask"],device=a.device),attention_mask=torch.tensor(padded["attention_mask"],device=a.device),labels=torch.tensor(padded["labels"],device=a.device))
        opt.zero_grad(); out.loss.backward(); assert_projector_gradients(model,model.audio_projector); opt.step()
        torch.save({"format":"projector-stage1-v1","step":step+1,"projector":model.audio_projector.state_dict(),"optimizer":opt.state_dict(),"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"config":config,"joyai_model":a.joyai_model},latest)
        print(json.dumps({"step":step+1,"loss":float(out.loss)}))
if __name__ == "__main__": main()
