"""Generate fixed image-conditioned teacher responses for Stage 2 retention data.

This is deliberately a separate offline step: the Stage 2 trainer consumes only
the JSONL produced here and never loads the teacher for an image batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_FIELDS = {"sample_id", "split", "image_path", "prompt", "provenance"}


def load_source_manifest(path: Path) -> list[dict[str, Any]]:
    """Validate source rows and resolve their image paths before using a GPU."""
    if not path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"source manifest is empty: {path}")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"source row {index} is not an object")
        missing = SOURCE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"source row {index} lacks fields {sorted(missing)}")
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"source row {index} has missing or duplicate sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        if not str(row["prompt"]).strip():
            raise ValueError(f"source row {index} has an empty prompt")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"source row {index} image does not exist: {image_path}")
        row["_image_path"] = str(image_path.resolve())
    return rows


def _model_inputs(processor: Any, image_path: str, prompt: str, device: str) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def generate_teacher_response(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """Run deterministic image-conditioned generation and return only new tokens."""
    inputs = _model_inputs(processor, image_path, prompt, device)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    input_length = int(inputs["input_ids"].shape[1])
    answer = processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    if not answer:
        raise ValueError("teacher generated an empty response")
    return answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline image-to-text distillation labels.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true", help="Generate labels; omitted means manifest preflight.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {args.output_manifest}")
    rows = load_source_manifest(args.source_manifest)
    if not args.run:
        print(json.dumps({"run": False, "validated_samples": len(rows)}, ensure_ascii=False))
        return 0

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.llm_model, revision=args.teacher_revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.llm_model, revision=args.teacher_revision, torch_dtype=dtype
    ).to(args.device)
    model.eval()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for row in tqdm(rows, desc="vision distill", unit="image", disable=args.no_progress):
            answer = generate_teacher_response(
                model, processor, row["_image_path"], str(row["prompt"]), args.device, args.max_new_tokens
            )
            output = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "image_path": row["_image_path"],
                "prompt": row["prompt"],
                "teacher_response": answer,
                "provenance": {
                    **dict(row["provenance"]),
                    "teacher_model": args.llm_model,
                    "teacher_revision": args.teacher_revision,
                    "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                },
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(json.dumps({"run": True, "written_samples": len(rows), "output": str(args.output_manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
