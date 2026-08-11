"""Extract selected COCO images from HF Parquet shards without using a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _load_targets(paths: list[Path]) -> dict[int, Path]:
    targets: dict[int, Path] = {}
    for manifest in paths:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(str(row["image_path"]))
            name = image_path.name
            if not name.startswith("COCO_") or not name.endswith(".jpg"):
                raise ValueError(f"unexpected COCO image path: {image_path}")
            image_id = int(name.rsplit("_", 1)[1][:-4])
            previous = targets.get(image_id)
            if previous is not None and previous != image_path:
                raise ValueError(f"image id maps to multiple paths: {image_id}")
            targets[image_id] = image_path
    if not targets:
        raise ValueError("manifests contain no target images")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    targets = _load_targets(args.manifest)
    shards = sorted(args.parquet_root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {args.parquet_root}")
    found: set[int] = set()
    if args.run:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(columns=["image_id", "image"], batch_size=512):
                for row in batch.to_pylist():
                    image_id = int(row["image_id"])
                    if image_id not in targets or image_id in found:
                        continue
                    image = row["image"]
                    data = image.get("bytes") if isinstance(image, dict) else None
                    if not data:
                        raise ValueError(f"image {image_id} has no embedded bytes")
                    output = targets[image_id]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists():
                        if output.read_bytes() != data:
                            raise FileExistsError(f"refusing to replace existing image: {output}")
                    else:
                        output.write_bytes(data)
                    found.add(image_id)
            if len(found) == len(targets):
                break
    else:
        found = set()
    missing = sorted(set(targets) - found) if args.run else []
    if missing:
        raise RuntimeError(f"missing {len(missing)} target images; examples={missing[:5]}")
    print(json.dumps({"target_images": len(targets), "extracted_images": len(found), "run": args.run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
