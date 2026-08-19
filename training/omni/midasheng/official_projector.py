"""MiDashengLM's official audio projector adapted to JoyAI's embedding width."""

from __future__ import annotations

from typing import Any


def adapt_official_projector(source: Any, output_size: int = 4096) -> Any:
    """Keep official 5x subsampling and first projection, replacing only its head.

    ``source`` must be the ``audio_projector`` from the unmodified official
    MiDashengLM checkpoint.  Its first Linear(6400, 3584) and GELU retain their
    checkpoint weights; the incompatible final Linear(3584, 3584) is replaced
    with a newly initialized Linear(3584, 4096).
    """
    import copy

    from torch import nn

    projector = copy.deepcopy(source)
    if getattr(projector, "k", None) != 5:
        raise ValueError("official MiDasheng projector must use 5x subsampling")
    net = getattr(projector, "net", None)
    if not isinstance(net, nn.Sequential) or len(net) != 3:
        raise ValueError("unexpected official MiDasheng projector layout")
    first, activation, final = net
    if not isinstance(first, nn.Linear) or (first.in_features, first.out_features) != (6400, 3584):
        raise ValueError("official projector first layer must be Linear(6400, 3584)")
    if not isinstance(activation, nn.GELU):
        raise ValueError("official projector must retain GELU")
    if not isinstance(final, nn.Linear) or (final.in_features, final.out_features) != (3584, 3584):
        raise ValueError("official projector final layer must be Linear(3584, 3584)")
    net[2] = nn.Linear(3584, output_size, bias=final.bias is not None, dtype=final.weight.dtype)
    return projector


def subsampled_token_count(raw_tokens: int, factor: int = 5) -> int:
    """Mirror the official projector's trailing-frame discard rule."""
    if raw_tokens < factor:
        raise ValueError("audio must contain at least five MiDasheng encoder tokens")
    return raw_tokens // factor
