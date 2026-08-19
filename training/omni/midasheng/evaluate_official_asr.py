"""Run the untouched MiDashengLM checkpoint with its official Transformers flow.

This evaluator deliberately does not import JoyAI code or any project projector.  It
loads the full MiDashengLM checkpoint, uses ``apply_chat_template`` exactly as in the
official README, and writes per-sample transcripts plus corpus WER/CER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


OFFICIAL_ASR_PROMPT = "Transcribe the speech into text <|en|>"


def normalize(text: str) -> str:
    """Use the established English ASR scoring convention: case/punctuation insensitive."""
    return " ".join(re.sub(r"[^A-Z0-9']+", " ", text.upper()).split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int, int]:
    """Return Levenshtein total, insertions, deletions, and substitutions."""
    previous = [(index, 0, index, 0) for index in range(len(reference) + 1)]
    for hyp_index, hyp_token in enumerate(hypothesis, start=1):
        current = [(hyp_index, hyp_index, 0, 0)]
        for ref_index, ref_token in enumerate(reference, start=1):
            deletion = previous[ref_index][0] + 1, previous[ref_index][1], previous[ref_index][2] + 1, previous[ref_index][3]
            insertion = current[ref_index - 1][0] + 1, current[ref_index - 1][1] + 1, current[ref_index - 1][2], current[ref_index - 1][3]
            substitution = previous[ref_index - 1][0] + (ref_token != hyp_token), previous[ref_index - 1][1], previous[ref_index - 1][2], previous[ref_index - 1][3] + (ref_token != hyp_token)
            current.append(min(deletion, insertion, substitution, key=lambda item: item[0]))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict[str, int | float]:
    reference_words, hypothesis_words = normalize(reference).split(), normalize(hypothesis).split()
    reference_chars = list("".join(reference_words))
    hypothesis_chars = list("".join(hypothesis_words))
    word_errors, insertions, deletions, substitutions = edit_counts(reference_words, hypothesis_words)
    char_errors, char_insertions, char_deletions, char_substitutions = edit_counts(reference_chars, hypothesis_chars)
    return {
        "word_errors": word_errors,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "reference_words": len(reference_words),
        "char_errors": char_errors,
        "char_insertions": char_insertions,
        "char_deletions": char_deletions,
        "char_substitutions": char_substitutions,
        "reference_chars": len(reference_chars),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    config = {
        "checkpoint": str(args.model_dir),
        "checkpoint_config_sha256": sha256(args.model_dir / "config.json"),
        "checkpoint_index_sha256": sha256(args.model_dir / "model.safetensors.index.json"),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "samples": len(records),
        "device": args.device,
        "prompt": OFFICIAL_ASR_PROMPT,
        "audio_preprocessing": "Official AutoProcessor.apply_chat_template audio path handling; no project preprocessing. Audio tensors stay float32.",
        "model_dtype": "BF16 decoder with official audio encoder/projector kept FP32 for their BatchNorm frontend",
        "template_kwargs": {"tokenize": True, "add_generation_prompt": True, "add_special_tokens": True, "return_dict": True},
        "generation_kwargs": {},
        "generation_note": "Official README call: model.generate(**model_inputs); checkpoint generation_config is used unchanged.",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    # Dasheng's frontend includes BatchNorm and expects FP32 activations.  The
    # official model casts projected audio to the decoder embedding dtype later.
    # This only changes runtime tensor dtypes; it never writes checkpoint files.
    model.audio_encoder.float()
    model.audio_projector.float()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    totals = {key: 0 for key in ("word_errors", "insertions", "deletions", "substitutions", "reference_words", "char_errors", "char_insertions", "char_deletions", "char_substitutions", "reference_chars")}
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, record in enumerate(records, start=1):
            audio_path = record["audio"][0]["path"]
            reference = record["metadata"]["assistant_target_text"]
            messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": OFFICIAL_ASR_PROMPT}, {"type": "audio", "path": audio_path}]}]
            with torch.no_grad():
                model_inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    add_special_tokens=True,
                    return_dict=True,
                )
                model_inputs = {
                    key: value.to(args.device) if hasattr(value, "to") else value
                    for key, value in model_inputs.items()
                }
                generation = model.generate(**model_inputs)
            hypothesis = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            counts = score(reference, hypothesis)
            for key in totals:
                totals[key] += int(counts[key])
            output.write(json.dumps({"sample_id": record["sample_id"], "audio": audio_path, "reference": reference, "hypothesis": hypothesis, **counts}) + "\n")
            output.flush()
            print(f"[{index}/{len(records)}] {record['sample_id']}", flush=True)
    summary: dict[str, Any] = {**config, **totals}
    summary["wer"] = totals["word_errors"] / max(1, totals["reference_words"])
    summary["cer"] = totals["char_errors"] / max(1, totals["reference_chars"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
