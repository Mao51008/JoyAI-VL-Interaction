"""Safe loading helpers for trusted local stage-one checkpoints."""
from __future__ import annotations

from pathlib import Path


def load_trusted_checkpoint(path: Path, torch):
    """Load a user-specified local checkpoint with a narrow PyTorch 2.6 allowlist."""
    from pathlib import PosixPath

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        return torch.load(path, map_location="cpu", weights_only=False)
    with safe_globals([PosixPath]):
        return torch.load(path, map_location="cpu", weights_only=True)
