"""Compare stage-one and stage-two projector/LoRA combinations on SpokenWOZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cache_features import validate_feature_cache
from .train import (
    CachedConversationBatchSource,
    FROZEN_STAGE1_PROJECTOR_SHA256,
    build_model_from_pretrained,
    load_manifest,
)


CONFIGURATIONS = (
    "stage1_no_lora",
    "stage2_projector_no_lora",
    "stage1_projector_stage2_lora",
    "stage2_full",
)


@dataclass(frozen=True)
class CheckpointParts:
    projector: dict[str, Any]
    lora: dict[str, Any]
    lora_targets: tuple[str, ...]
    lora_rank: int
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_checkpoint_parts(
    path: Path, expected_stage1_sha256: str
) -> CheckpointParts:
    """Load and validate the independently switchable Stage2 trainable weights."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "projector-stage2-v2":
        raise ValueError(f"unsupported Stage2 checkpoint format: {state.get('format')!r}")
    initialization = state.get("projector_initialization") or {}
    if initialization.get("sha256") != expected_stage1_sha256:
        raise ValueError("Stage2 checkpoint does not derive from the requested Stage1 checkpoint")
    trainable_state = state.get("trainable_state")
    if not isinstance(trainable_state, dict) or not trainable_state:
        raise ValueError("Stage2 checkpoint has no trainable_state")
    projector = {
        name: tensor
        for name, tensor in trainable_state.items()
        if name.startswith("core.audio_projector.")
    }
    lora = {name: tensor for name, tensor in trainable_state.items() if ".lora_" in name}
    unexpected = set(trainable_state) - set(projector) - set(lora)
    if unexpected:
        raise ValueError(f"unexpected Stage2 trainable tensors: {sorted(unexpected)[:5]}")
    if not projector or not lora:
        raise ValueError("Stage2 checkpoint must contain projector and LoRA tensors")
    targets = {
        name.removeprefix("core.language_model.").rsplit(".", 1)[0]
        for name in lora
    }
    ranks = {
        int(tensor.shape[0] if name.endswith(".lora_A") else tensor.shape[1])
        for name, tensor in lora.items()
    }
    if len(ranks) != 1:
        raise ValueError(f"inconsistent LoRA ranks in checkpoint: {sorted(ranks)}")
    return CheckpointParts(
        projector=projector,
        lora=lora,
        lora_targets=tuple(sorted(targets)),
        lora_rank=ranks.pop(),
        metadata={
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "step": int(state["step"]),
            "best_validation_loss": float(state["best_validation_loss"]),
            "projector_initialization": initialization,
            "feature_cache": state.get("feature_cache"),
        },
    )


def select_dialogue_balanced_rows(
    rows: Sequence[dict[str, Any]], max_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Select at most one deterministic random turn per dialogue."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dialogue_id"]), []).append(row)
    rng = random.Random(seed)
    dialogue_ids = sorted(grouped)
    rng.shuffle(dialogue_ids)
    selected = [rng.choice(grouped[dialogue_id]) for dialogue_id in dialogue_ids[:max_samples]]
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def capture_stage1_projector(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("core.audio_projector.")
    }


def apply_configuration(
    model: Any,
    configuration: str,
    stage1_projector: dict[str, Any],
    stage2: CheckpointParts,
) -> None:
    """Apply one cell of the projector x LoRA factorial comparison in place."""
    import torch

    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown evaluation configuration: {configuration}")
    parameters = dict(model.named_parameters())
    projector = stage2.projector if configuration in {
        "stage2_projector_no_lora",
        "stage2_full",
    } else stage1_projector
    lora = stage2.lora if configuration in {
        "stage1_projector_stage2_lora",
        "stage2_full",
    } else None
    with torch.no_grad():
        for name, tensor in projector.items():
            if name not in parameters:
                raise KeyError(f"model lacks projector tensor {name}")
            parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))
        for name, parameter in parameters.items():
            if ".lora_" in name:
                parameter.zero_()
        if lora is not None:
            for name, tensor in lora.items():
                if name not in parameters:
                    raise KeyError(f"model lacks LoRA tensor {name}")
                parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty evaluation records")
    token_count = sum(int(record["supervised_tokens"]) for record in records)
    if token_count <= 0:
        raise ValueError("evaluation records have no supervised tokens")
    token_nll = sum(
        float(record["nll"]) * int(record["supervised_tokens"])
        for record in records
    ) / token_count
    sample_nll = sum(float(record["nll"]) for record in records) / len(records)
    return {
        "samples": len(records),
        "supervised_tokens": token_count,
        "token_weighted_nll": token_nll,
        "token_weighted_perplexity": math.exp(min(token_nll, 50.0)),
        "sample_mean_nll": sample_nll,
    }


def paired_comparison(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    baseline_by_id = {record["sample_id"]: record for record in baseline}
    candidate_by_id = {record["sample_id"]: record for record in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired configurations do not contain the same samples")
    deltas = [
        float(candidate_by_id[sample_id]["nll"])
        - float(baseline_by_id[sample_id]["nll"])
        for sample_id in baseline_by_id
    ]
    return {
        "samples": len(deltas),
        "mean_nll_delta": sum(deltas) / len(deltas),
        "fraction_candidate_lower": sum(delta < 0 for delta in deltas) / len(deltas),
    }


def evaluate_configuration(
    model: Any,
    batches: Iterable[Any],
    configuration: str,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    records = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches, start=1):
            output = model(batch)
            loss = output["loss"] if isinstance(output, dict) else output.loss
            supervised_tokens = int(batch.labels[:, 1:].ne(-100).sum().item())
            if supervised_tokens <= 0:
                raise ValueError(f"batch {batch.sample_ids} has no shifted supervised tokens")
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"non-finite loss for batch {batch.sample_ids}")
            for sample_id, dialogue_id in zip(batch.sample_ids, batch.dialogue_ids, strict=True):
                records.append(
                    {
                        "configuration": configuration,
                        "sample_id": sample_id,
                        "dialogue_id": dialogue_id,
                        "supervised_tokens": supervised_tokens,
                        "nll": float(loss),
                    }
                )
            if progress_every and batch_index % progress_every == 0:
                print(
                    f"{configuration}: evaluated {batch_index} samples",
                    file=sys.stderr,
                    flush=True,
                )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-sha256", default=FROZEN_STAGE1_PROJECTOR_SHA256)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-feature-shards", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.batch_size != 1:
        raise ValueError("the current evaluator requires --batch-size 1 for per-sample records")
    if args.progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    import torch
    import transformers

    stage2 = load_stage2_checkpoint_parts(args.stage2_checkpoint, args.stage1_sha256)
    projector_config = stage2.metadata["projector_initialization"]["config"]
    rows = select_dialogue_balanced_rows(
        load_manifest(args.dev_manifest), args.max_samples, args.seed
    )
    cache_metadata = validate_feature_cache(args.feature_dir, rows)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm_model, fix_mistral_regex=True
    )
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    model = build_model_from_pretrained(
        args.llm_model,
        int(projector_config["input_size"]),
        int(projector_config["output_size"]),
        stage2.lora_targets,
        stage2.lora_rank,
        args.lora_alpha,
        args.stage1_checkpoint,
        args.stage1_sha256,
        args.device,
        args.dtype,
    )
    stage1_projector = capture_stage1_projector(model)
    all_records: dict[str, list[dict[str, Any]]] = {}
    for configuration in CONFIGURATIONS:
        apply_configuration(model, configuration, stage1_projector, stage2)
        batches = CachedConversationBatchSource(
            rows,
            tokenizer,
            args.feature_dir,
            int(placeholder_id),
            args.batch_size,
            args.max_cached_feature_shards,
        )
        all_records[configuration] = evaluate_configuration(
            model, batches, configuration, args.progress_every
        )
    summary = {
        "schema_version": 1,
        "stage1_checkpoint": {
            "path": str(args.stage1_checkpoint.resolve()),
            "sha256": args.stage1_sha256,
        },
        "stage2_checkpoint": stage2.metadata,
        "selection": {
            "manifest": str(args.dev_manifest.resolve()),
            "manifest_sha256": _sha256(args.dev_manifest),
            "strategy": "one_deterministic_random_turn_per_dialogue",
            "seed": args.seed,
            "samples": len(rows),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in rows).encode()
            ).hexdigest(),
        },
        "feature_cache": cache_metadata,
        "lora": {"rank": stage2.lora_rank, "alpha": args.lora_alpha},
        "metrics": {
            name: aggregate_records(records) for name, records in all_records.items()
        },
        "paired": {
            "stage2_projector_effect_without_lora": paired_comparison(
                all_records["stage1_no_lora"],
                all_records["stage2_projector_no_lora"],
            ),
            "stage2_lora_effect_with_stage1_projector": paired_comparison(
                all_records["stage1_no_lora"],
                all_records["stage1_projector_stage2_lora"],
            ),
            "stage2_projector_effect_with_stage2_lora": paired_comparison(
                all_records["stage1_projector_stage2_lora"],
                all_records["stage2_full"],
            ),
            "stage2_full_effect": paired_comparison(
                all_records["stage1_no_lora"], all_records["stage2_full"]
            ),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for configuration in CONFIGURATIONS
            for record in all_records[configuration]
        ),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
