"""Independent FP32-encoder DDP plus BF16-DeepSpeed-core trainer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from training.omni.projector_stage1.model import replace_audio_placeholders


@dataclass(frozen=True)
class HybridConfig:
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0


def encoder_to_core_features(features: Any) -> Any:
    """The sole precision boundary: FP32 Encoder output to BF16 Core input."""
    import torch

    if features.dtype != torch.float32:
        raise AssertionError("AudioEncoderBranch must return FP32 features")
    return features.to(torch.bfloat16)


class HybridTrainer:
    """Coordinate exactly one Encoder-DDP and one DeepSpeed-Core update boundary."""

    def __init__(self, encoder: Any, core_engine: Any, encoder_optimizer: Any,
                 config: HybridConfig = HybridConfig()) -> None:
        self.encoder, self.core_engine, self.encoder_optimizer, self.config = (
            encoder, core_engine, encoder_optimizer, config
        )
        self.global_step = 0
        self.epoch = 0

    def run_update(self, microbatches: Iterable[Any]) -> float:
        import torch
        from contextlib import nullcontext

        batches = list(microbatches)
        if len(batches) != self.config.gradient_accumulation_steps:
            raise ValueError("one update requires exactly gradient_accumulation_steps microbatches")
        self.encoder_optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=next(self.encoder.parameters()).device)
        for index, batch in enumerate(batches):
            boundary = index + 1 == len(batches)
            sync = nullcontext() if boundary or not hasattr(self.encoder, "no_sync") else self.encoder.no_sync()
            with sync:
                features, mask = self.encoder(batch.audio_features.float(), batch.audio_attention_mask)
                output = self.core_engine(batch, encoder_to_core_features(features), mask)
                loss = output["loss"] if isinstance(output, dict) else getattr(output, "loss", output)
                self.core_engine.backward(loss / len(batches))
            if boundary:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.max_grad_norm)
                self.core_engine.step()
                self.encoder_optimizer.step()
            total += loss.detach()
        self.global_step += 1
        return float(total / len(batches))

    def validate(self, batches: Iterable[Any]) -> float:
        import torch

        total = 0.0; count = 0
        self.encoder.eval(); self.core_engine.eval()
        with torch.no_grad():
            for batch in batches:
                features, mask = self.encoder(batch.audio_features.float(), batch.audio_attention_mask)
                output = self.core_engine(batch, encoder_to_core_features(features), mask)
                loss = output["loss"] if isinstance(output, dict) else getattr(output, "loss", output)
                total += float(loss); count += 1
        if not count: raise ValueError("validation source is empty")
        return total / count

    def save_checkpoint(self, path: Path) -> None:
        import torch

        module = getattr(self.encoder, "module", self.encoder)
        torch.save({"global_step": self.global_step, "epoch": self.epoch,
                    "encoder_trainable": {n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad},
                    "encoder_optimizer": self.encoder_optimizer.state_dict()}, path)
        self.core_engine.save_checkpoint(str(path.parent), tag=path.name + ".core")

    def load_checkpoint(self, path: Path) -> None:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        module = getattr(self.encoder, "module", self.encoder)
        module.load_state_dict(state["encoder_trainable"], strict=False)
        self.encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        self.global_step, self.epoch = int(state["global_step"]), int(state["epoch"])
        self.core_engine.load_checkpoint(str(path.parent), tag=path.name + ".core")


def build_audio_encoder_branch(audio_encoder: Any, device: str, distributed: bool) -> Any:
    """Place the entire MiDasheng branch in FP32 DDP, outside DeepSpeed."""
    import torch

    audio_encoder = audio_encoder.to(device=device, dtype=torch.float32)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        audio_encoder = DistributedDataParallel(audio_encoder, device_ids=[torch.cuda.current_device()])
    return audio_encoder


def assert_hybrid_partition(encoder: Any, core: Any) -> None:
    """Prove the FP32 Encoder is outside the BF16 DeepSpeed Core module tree."""
    encoder_ids = {id(module) for module in getattr(encoder, "module", encoder).modules()}
    core_names = {name for name, _module in core.named_modules()}
    if any("audio_encoder" in name for name in core_names):
        raise AssertionError("ProjectorLoraCore must not contain MiDasheng audio_encoder")
    if any(id(module) in encoder_ids for _name, module in core.named_modules()):
        raise AssertionError("Encoder module was attached to ProjectorLoraCore")


def build_projector_lora_core(projector: Any, language_model: Any) -> Any:
    """Build the DeepSpeed-owned BF16 half of the joint graph only."""
    from torch import nn

    class ProjectorLoraCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_projector = projector
            self.language_model = language_model

        def forward(self, batch: Any, encoder_features: Any, encoder_mask: Any) -> Any:
            device = next(self.parameters()).device
            projected = self.audio_projector(encoder_features.to(device=device, dtype=next(self.audio_projector.parameters()).dtype))
            ids = batch.input_ids.to(device)
            text_embeddings = self.language_model.get_input_embeddings()(ids)
            embeddings = replace_audio_placeholders(
                text_embeddings, projected, batch.audio_placeholder_mask.to(device), encoder_mask.to(device).bool()
            )
            return self.language_model(
                inputs_embeds=embeddings, attention_mask=batch.attention_mask.to(device), labels=batch.labels.to(device)
            )

    core = ProjectorLoraCore()
    return core
