"""Projector-only building blocks for the first native Omni training stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AudioProjectorConfig:
    input_size: int
    output_size: int
    hidden_size: int | None = None
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.input_size, self.output_size) <= 0:
            raise ValueError("projector input_size and output_size must be positive")
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ValueError("projector hidden_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("projector dropout must be in [0, 1)")


class AudioProjector(nn.Module):
    """Map frozen ASR features into the JoyAI token embedding space."""

    def __init__(self, config: AudioProjectorConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden_size = config.hidden_size or config.output_size
        self.layers = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.output_size),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[-1] != self.config.input_size:
            raise ValueError(
                f"expected audio hidden size {self.config.input_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        return self.layers(hidden_states)

    def export_config(self) -> dict[str, Any]:
        return asdict(self.config)


@dataclass(frozen=True)
class TrainableParameterReport:
    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]

    @property
    def trainable_ratio(self) -> float:
        if not self.total_parameters:
            return 0.0
        return self.trainable_parameters / self.total_parameters


def freeze_for_projector_training(
    model: nn.Module, projector: nn.Module
) -> TrainableParameterReport:
    """Freeze the full graph and re-enable exactly the selected projector."""
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    if not projector_ids:
        raise ValueError("projector has no parameters")

    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in projector_ids)

    report = parameter_report(model)
    if not report.trainable_names:
        raise RuntimeError("no trainable projector parameters found in model")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in projector_ids
    ]
    if unexpected:
        raise RuntimeError(f"non-projector parameters are trainable: {unexpected}")
    return report


def parameter_report(model: nn.Module) -> TrainableParameterReport:
    named_parameters = tuple(model.named_parameters())
    return TrainableParameterReport(
        total_parameters=sum(parameter.numel() for _name, parameter in named_parameters),
        trainable_parameters=sum(
            parameter.numel() for _name, parameter in named_parameters if parameter.requires_grad
        ),
        trainable_names=tuple(
            name for name, parameter in named_parameters if parameter.requires_grad
        ),
    )


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    return parameters


def assert_projector_gradients(model: nn.Module, projector: nn.Module) -> None:
    """Fail fast when gradients are missing, non-finite, or leak into frozen weights."""
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    missing = []
    leaking = []
    non_finite = []
    zero = []
    for name, parameter in model.named_parameters():
        if id(parameter) in projector_ids:
            if parameter.grad is None:
                missing.append(name)
            elif not torch.isfinite(parameter.grad).all():
                non_finite.append(name)
            elif not torch.count_nonzero(parameter.grad):
                zero.append(name)
        elif parameter.grad is not None:
            leaking.append(name)
    if missing or leaking or non_finite or zero:
        raise RuntimeError(
            "invalid projector-only gradients: "
            f"missing={missing}, leaking={leaking}, non_finite={non_finite}, zero={zero}"
        )
