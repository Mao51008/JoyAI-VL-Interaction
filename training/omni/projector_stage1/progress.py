"""Small optional tqdm adapter used by long-running stage-one commands."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int,
    description: str,
    unit: str,
    no_progress: bool,
    tqdm_factory: Callable | None = None,
) -> Iterable[T]:
    """Wrap an iterable in tqdm when available, preserving a testable fallback."""
    if no_progress:
        return iterable
    if tqdm_factory is None:
        try:
            from tqdm import tqdm as tqdm_factory
        except ImportError:
            return iterable
    return tqdm_factory(
        iterable,
        total=total,
        desc=description,
        unit=unit,
        mininterval=1.0,
        dynamic_ncols=False,
    )
