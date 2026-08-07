"""Sharded frozen-ASR feature cache with lazy shard loading."""
from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

class FeatureCache:
    def __init__(self, directory: Path, max_loaded_shards: int | None = None) -> None:
        if max_loaded_shards is not None and max_loaded_shards <= 0:
            raise ValueError("max_loaded_shards must be positive")
        self.directory = directory
        self.index = {row["sample_id"]: row for row in json.loads((directory / "index.json").read_text())}
        self.max_loaded_shards = max_loaded_shards
        self._shards: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def get(self, sample_id: str) -> dict[str, Any]:
        row = self.index[sample_id]
        shard = row.get("shard")
        if shard is None:  # Backward compatibility with pilot per-sample cache.
            import torch
            return torch.load(self.directory / row["path"], map_location="cpu", weights_only=True)
        if shard not in self._shards:
            import torch
            self._shards[shard] = torch.load(self.directory / shard, map_location="cpu", weights_only=True)["samples"]
            if self.max_loaded_shards is not None and len(self._shards) > self.max_loaded_shards:
                self._shards.popitem(last=False)
        else:
            self._shards.move_to_end(shard)
        return self._shards[shard][row["offset"]]
