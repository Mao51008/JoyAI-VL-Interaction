"""Independent FP32-encoder DDP plus BF16-DeepSpeed-core trainer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
                features, mask = self.encoder(batch.waveforms.float(), batch.lengths)
                loss = self.core_engine(batch, encoder_to_core_features(features), mask)
                self.core_engine.backward(loss / len(batches))
            if boundary:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.max_grad_norm)
                self.core_engine.step()
                self.encoder_optimizer.step()
            total += loss.detach()
        self.global_step += 1
        return float(total / len(batches))

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
