"""Two-stage, transcript-only source-completeness pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABELS = {"COMPLETE", "INCOMPLETE", "UNCERTAIN"}
MISSING_REFERENCE = re.compile(
    r"\b(rewrite what i just said|continue the previous|question above|"
    r"this article|this headline|the headline|this attachment|this image|this file)\b",
    re.IGNORECASE,
)
MISSING_TASK = re.compile(
    r"\b(paraphrase|edit|summarize|translate|rewrite|classify|spell|correct|"
    r"identify|describe|assign|sort|judge|determine)\b.*\b(the following|this|the)\b",
    re.IGNORECASE,
)
WHICH_FOLLOWING = re.compile(r"^which of the following\b", re.IGNORECASE)
QUESTION = re.compile(r"^(what|who|when|where|why|how|which|is|are|can|could|do|does|did|will|would)\b", re.I)
PLACEHOLDER = re.compile(r"^(person (one|two)|option [a-z]|sentence|text|article|\d+|[a-z])\.?$", re.I)


def payload_after_colon(text: str) -> str | None:
    if ":" not in text:
        return None
    payload = text.rsplit(":", 1)[1].strip()
    if len(payload.split()) < 2 or PLACEHOLDER.fullmatch(payload):
        return None
    return payload


def rule_label(transcript: str) -> tuple[str, str]:
    text = " ".join(transcript.split())
    if not text:
        return "UNCERTAIN", "empty_transcript"
    payload = payload_after_colon(text)
    if payload:
        return "COMPLETE", "explicit_payload_after_colon"
    if WHICH_FOLLOWING.match(text):
        if re.search(r",\s*[^,?]+\s+(or|and)\s+[^,?]+\?*$", text, re.I):
            return "COMPLETE", "explicit_options_in_question"
        return "INCOMPLETE", "which_of_following_without_options"
    if MISSING_REFERENCE.search(text):
        return "INCOMPLETE", "missing_referenced_context"
    if re.search(r"\b(translate|rewrite|paraphrase) the sentence .+\binto\b", text, re.I):
        return "COMPLETE", "inline_payload_in_instruction"
    if MISSING_TASK.search(text) and text.endswith((".", "?", "!")):
        return "INCOMPLETE", "referenced_payload_not_present"
    if QUESTION.match(text) and text.endswith("?"):
        return "COMPLETE", "self_contained_question"
    return "UNCERTAIN", "rule_not_decisive"


PROMPT = """You are checking whether a user's single utterance is self-contained.

Decide whether the request can be understood and completed using only the information explicitly present in this utterance.

COMPLETE: The request contains all required input information. The answer itself does NOT need to appear in the utterance. Ordinary knowledge questions are COMPLETE.
INCOMPLETE: The request depends on missing content or context, such as missing words, sentence/text, options, attachment, article, or previous conversation.
UNCERTAIN: Use only if genuinely ambiguous.

Return JSON only: {{"label":"COMPLETE|INCOMPLETE|UNCERTAIN","reason":"brief reason","confidence":0.0}}

Utterance:
{utterance}"""


def parse_response(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
        if result.get("label") not in LABELS:
            return None
        result["confidence"] = float(result.get("confidence", 0.0))
        return result
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    pending = []
    for row in rows:
        label, reason = rule_label(str(row["transcript"]))
        row["source_completeness"] = label
        row["source_completeness_reason"] = reason
        row["source_completeness_method"] = "enhanced_rule"
        if label == "UNCERTAIN":
            pending.append(row)
    if pending:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": args.device}
        ).eval()
        for row in pending:
            prompt = PROMPT.format(utterance=row["transcript"])
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            for attempt in range(2):
                output = model.generate(**inputs, do_sample=False, max_new_tokens=96)
                generated = processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
                verdict = parse_response(generated)
                if verdict is not None:
                    break
            if verdict and verdict["confidence"] >= 0.90:
                row["source_completeness"] = verdict["label"]
                row["source_completeness_reason"] = str(verdict["reason"])
                row["source_completeness_method"] = "qwen35_judge"
            else:
                row["source_completeness_reason"] = "judge_low_confidence_or_format_error"
                row["source_completeness_method"] = "qwen35_judge_unaccepted"
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
