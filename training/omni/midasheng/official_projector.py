"""Frozen official MiDasheng projector plus the trainable JoyAI adapter."""

from __future__ import annotations

from typing import Any


def subsampled_token_count(raw_tokens: int, factor: int = 5) -> int:
    if raw_tokens < factor:
        raise ValueError("audio must contain at least five MiDasheng encoder tokens")
    return raw_tokens // factor


def build_frozen_projector_adapter(source: Any) -> Any:
    """Preserve the complete official projector and append Linear(3584, 4096)."""
    import copy

    from torch import nn

    official = copy.deepcopy(source)
    if getattr(official, "k", None) != 5:
        raise ValueError("official MiDasheng projector must use 5x subsampling")
    net = getattr(official, "net", None)
    if not isinstance(net, nn.Sequential) or len(net) != 3:
        raise ValueError("unexpected official MiDasheng projector layout")
    first, activation, final = net
    if not isinstance(first, nn.Linear) or (first.in_features, first.out_features) != (6400, 3584):
        raise ValueError("official projector first layer must be Linear(6400, 3584)")
    if not isinstance(activation, nn.GELU) or not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (3584, 3584):
        raise ValueError("official projector must retain GELU and Linear(3584, 3584)")
    for parameter in official.parameters():
        parameter.requires_grad_(False)

    class OfficialProjectorJoyAIAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.official_projector = official
            self.joyai_adapter = nn.Linear(3584, 4096, dtype=final.weight.dtype)

        def forward(self, features: Any, mask: Any) -> tuple[Any, Any]:
            with __import__("torch").no_grad():
                projected, projected_mask = self.official_projector(features, mask)
            return self.joyai_adapter(projected), projected_mask

    return OfficialProjectorJoyAIAdapter()
