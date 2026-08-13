"""Generate fixed image-conditioned teacher responses for Stage 2 retention data.

This is deliberately a separate offline step: the Stage 2 trainer consumes only
the JSONL produced here and never loads the teacher for an image batch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_FIELDS = {"sample_id", "split", "image_path", "prompt", "provenance"}
SHORT_ANSWER_INSTRUCTION = (
    "Answer the question about the image directly and concisely.\n"
    "Use one or two sentences only.\n"
    "Do not add unnecessary explanation."
)
PROMPT_TEMPLATE_NAME = "vision_short_answer_v1"


@dataclass(frozen=True)
class TeacherGeneration:
    """Decoded teacher output plus the stopping information needed for filtering."""

    answer: str
    prompt_tokens: int
    generated_tokens: int
    answer_tokens: int
    eos_token_id: int | None
    finished_by_eos: bool


def build_teacher_prompt(original_prompt: str) -> str:
    """Apply the fixed short-answer instruction without altering the source question."""
    return f"{SHORT_ANSWER_INSTRUCTION}\n\nQuestion: {original_prompt.strip()}"


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
) -> TeacherGeneration:
    """Run deterministic generation and retain the exact stopping condition."""
    inputs = _model_inputs(processor, image_path, prompt, device)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    input_length = int(inputs["input_ids"].shape[1])
    generated_ids = generated[0, input_length:].tolist()
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        eos_token_id = None
    eos_index = generated_ids.index(eos_token_id) if eos_token_id in generated_ids else None
    answer_ids = generated_ids[:eos_index] if eos_index is not None else generated_ids
    answer = processor.batch_decode(
        [answer_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return TeacherGeneration(
        answer=answer,
        prompt_tokens=input_length,
        generated_tokens=len(generated_ids),
        answer_tokens=len(answer_ids),
        eos_token_id=eos_token_id,
        finished_by_eos=eos_index is not None,
    )


def full_sequence_tokens(processor: Any, image_path: str, prompt: str, answer: str) -> int:
    """Measure the exact sequence that the Stage 2 vision collator will train on."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    return int(encoded["input_ids"].shape[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline image-to-text distillation labels.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-full-tokens", type=int, default=2048)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--run", action="store_true", help="Generate labels; omitted means manifest preflight.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_full_tokens <= 0:
        raise ValueError("--max-full-tokens must be positive")
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {args.output_manifest}")
    if args.rejected_manifest.exists():
        raise FileExistsError(f"refusing to overwrite rejected manifest: {args.rejected_manifest}")
    if args.output_manifest.resolve() == args.rejected_manifest.resolve():
        raise ValueError("output and rejected manifests must be different paths")
    rows = load_source_manifest(args.source_manifest)
    if not args.run:
        print(json.dumps({"run": False, "validated_samples": len(rows)}, ensure_ascii=False))
        return 0

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(
        args.llm_model, revision=args.teacher_revision, fix_mistral_regex=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.llm_model, revision=args.teacher_revision, torch_dtype=dtype
    ).to(args.device)
    model.eval()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_manifest.parent.mkdir(parents=True, exist_ok=True)
    written_samples = 0
    rejected_samples = 0
    with args.output_manifest.open("x", encoding="utf-8") as output_handle, args.rejected_manifest.open(
        "x", encoding="utf-8"
    ) as rejected_handle, torch.inference_mode():
        for row in tqdm(rows, desc="vision distill", unit="image", disable=args.no_progress):
            original_prompt = str(row["prompt"])
            teacher_prompt = build_teacher_prompt(original_prompt)
            generation = generate_teacher_response(
                model, processor, row["_image_path"], teacher_prompt, args.device, args.max_new_tokens
            )
            rejection_reason: str | None = None
            full_tokens: int | None = None
            if not generation.finished_by_eos:
                rejection_reason = "no_eos_at_max_new_tokens"
            elif not generation.answer:
                rejection_reason = "empty_answer"
            else:
                full_tokens = full_sequence_tokens(
                    processor, row["_image_path"], teacher_prompt, generation.answer
                )
                if full_tokens > args.max_full_tokens:
                    rejection_reason = "full_sequence_exceeds_max_tokens"
            generation_info = {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "generated_tokens": generation.generated_tokens,
                "answer_tokens": generation.answer_tokens,
                "eos_token_id": generation.eos_token_id,
                "finished_by_eos": generation.finished_by_eos,
            }
            output = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "image_path": row["_image_path"],
                "prompt": teacher_prompt,
                "original_prompt": original_prompt,
                "teacher_response": generation.answer,
                "provenance": {
                    **dict(row["provenance"]),
                    "teacher_model": args.llm_model,
                    "teacher_revision": args.teacher_revision,
                    "prompt_template": {
                        "name": PROMPT_TEMPLATE_NAME,
                        "instruction": SHORT_ANSWER_INSTRUCTION,
                    },
                    "generation": generation_info,
                    "sequence": {
                        "prompt_tokens": generation.prompt_tokens,
                        "full_tokens": full_tokens,
                        "max_full_tokens": args.max_full_tokens,
                    },
                },
            }
            if rejection_reason is not None:
                output["rejection_reason"] = rejection_reason
                rejected_handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                rejected_samples += 1
            else:
                output_handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                written_samples += 1
    print(
        json.dumps(
            {
                "run": True,
                "written_samples": written_samples,
                "rejected_samples": rejected_samples,
                "output": str(args.output_manifest),
                "rejected": str(args.rejected_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
