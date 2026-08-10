"""Selectively prepare LLaVA/COCO image prompts for offline VLM distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data/maoyy/datasets/audio_understanding_pilot")
LLAVA_FILE = "llava_instruct_150k.json"
LLAVA_URL = (
    "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/"
    f"{LLAVA_FILE}?download=true"
)


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/data/maoyy").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"output directory must be under {allowed}: {resolved}")
    return resolved


def _curl(proxy: str | None, *args: str) -> list[str]:
    return ["curl", *( ["--proxy", proxy] if proxy else [] ), *args]


def _download(url: str, destination: Path, proxy: str | None) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        _curl(proxy, "--fail", "--location", "--retry", "3", "--output", str(destination), url),
        check=True,
    )


def _normalise_prompt(value: str) -> str:
    return value.replace("<image>", "").strip()


def _coco_url(image: str) -> tuple[str, str] | None:
    path = Path(image)
    filename = path.name
    if not filename.lower().endswith((".jpg", ".jpeg")):
        return None
    split = next((part for part in path.parts if part in {"train2014", "val2014", "train2017", "val2017"}), None)
    if split is None:
        for candidate in ("train2014", "val2014"):
            if candidate in filename:
                split = candidate
                break
    if split is None:
        return None
    return f"https://images.cocodataset.org/{split}/{filename}", f"{split}/{filename}"


def select_single_turn_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    selected: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            continue
        human, assistant = conversations
        if not isinstance(human, dict) or not isinstance(assistant, dict):
            continue
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            continue
        prompt = _normalise_prompt(str(human.get("value", "")))
        if not prompt or not str(assistant.get("value", "")).strip():
            continue
        image = _coco_url(str(row.get("image", "")))
        if image is None:
            continue
        sample_id = "llava:" + str(row.get("id", ""))
        if sample_id == "llava:":
            continue
        source_url, relative_path = image
        selected.append(
            (
                hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
                {
                    "sample_id": sample_id,
                    "source_image_url": source_url,
                    "relative_image_path": relative_path,
                    "prompt": prompt,
                    "provenance": {
                        "dataset": "liuhaotian/LLaVA-Instruct-150K",
                        "source_image": str(row["image"]),
                        "selection": "single_turn_human_to_gpt",
                    },
                },
            )
        )
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} eligible LLaVA/COCO rows; requested {count}")
    return [row for _, row in sorted(selected)[:count]]


def _split(sample_id: str) -> str:
    return "dev" if int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a bounded LLaVA/COCO visual distillation pilot.")
    parser.add_argument("--annotation", type=Path, default=DATA_ROOT / "llava/raw" / LLAVA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "llava/manifests")
    parser.add_argument("--image-dir", type=Path, default=DATA_ROOT / "llava/images")
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--proxy", help="Explicit HTTP proxy forwarded to curl.")
    parser.add_argument("--download-annotation", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true", help="Write manifests and requested images.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _require_data_root(args.output_dir)
    image_dir = _require_data_root(args.image_dir)
    if args.download_annotation and not args.run:
        raise ValueError("--download-annotation requires --run")
    if args.download_annotation:
        _download(LLAVA_URL, args.annotation, args.proxy)
    if not args.annotation.is_file():
        raise FileNotFoundError(args.annotation)
    rows = json.loads(args.annotation.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("LLaVA annotation must be a JSON array")
    selected = select_single_turn_rows(rows, args.sample_count)
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in selected:
        split = _split(row["sample_id"])
        image_path = image_dir / row["relative_image_path"]
        manifest_row = {
            "sample_id": row["sample_id"],
            "split": split,
            "image_path": str(image_path),
            "prompt": row["prompt"],
            "provenance": row["provenance"],
        }
        (train_rows if split == "train" else dev_rows).append(manifest_row)
    if not train_rows or not dev_rows:
        raise ValueError("deterministic split produced an empty train or dev set")
    if args.run:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "train": _write_jsonl(output_dir / "llava_coco_source_train.jsonl", train_rows),
            "dev": _write_jsonl(output_dir / "llava_coco_source_dev.jsonl", dev_rows),
        }
        if args.download_images:
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda items, **_: items
            for row in tqdm(selected, desc="COCO images", unit="image", disable=args.no_progress):
                destination = image_dir / row["relative_image_path"]
                _download(row["source_image_url"], destination, args.proxy)
            summary["images"] = len(selected)
    else:
        summary = {"train": len(train_rows), "dev": len(dev_rows), "images": len(selected)}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
