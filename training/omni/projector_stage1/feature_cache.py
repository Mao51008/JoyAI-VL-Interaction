"""Sharded frozen-ASR feature cache with lazy shard loading."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class FeatureCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = {row["sample_id"]: row for row in json.loads((directory / "index.json").read_text())}
        self._shards: dict[str, list[dict[str, Any]]] = {}

    def get(self, sample_id: str) -> dict[str, Any]:
        row = self.index[sample_id]
        shard = row.get("shard")
        if shard is None:  # Backward compatibility with pilot per-sample cache.
            import torch
            return torch.load(self.directory / row["path"], map_location="cpu", weights_only=True)
        if shard not in self._shards:
            import torch
            self._shards[shard] = torch.load(self.directory / shard, map_location="cpu", weights_only=True)["samples"]
        return self._shards[shard][row["offset"]]
