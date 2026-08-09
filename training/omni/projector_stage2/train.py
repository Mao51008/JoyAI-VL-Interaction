"""Stage 2 projector plus LLM-LoRA training entry and CPU preflight utilities.

The command-line entry deliberately performs preflight only.  Loading a real ASR/LLM
checkpoint and launching device training is a separate, explicitly authorized step.
The small ``train_model`` API is model-agnostic so CPU tests can use a mock model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import islice
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .cache_features import validate_feature_cache

CANONICAL_SYSTEM_PROMPT = (
    "You are a helpful dialogue assistant. Use the conversation history and the current "
    "user's speech to provide an appropriate response."
)
SYSTEM_PROMPT_VARIANTS = (
    CANONICAL_SYSTEM_PROMPT,
    "Listen to the current user audio and respond appropriately using the conversation history.",
    "Respond to the user's spoken request while considering the preceding dialogue.",
)
AUDIO_START_TOKEN = "<|vision_start|>"
AUDIO_PLACEHOLDER_TOKEN = "<|vision_pad|>"
AUDIO_END_TOKEN = "<|vision_end|>"
FORMAL_MANIFEST_FIELDS = {
    "sample_id",
    "dialogue_id",
    "turn_id",
    "split",
    "audio_path",
    "clip_duration_ms",
    "clip_sha256",
    "source_audio_sha256",
    "user_text",
    "assistant_response",
    "dialogue_history",
    "provenance",
}
FROZEN_STAGE1_PROJECTOR_SHA256 = (
    "4e1573a3091d7ed438af16cee130d28e11b9701f5c22eb830e179e2ef315a22c"
)


@dataclass(frozen=True)
class Stage2Config:
    train_manifest: Path
    dev_manifest: Path
    output_dir: Path
    asr_encoder_prefix: str = "audio_encoder"
    projector_prefix: str = "audio_projector"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 3e-5
    steps: int = 1000
    validation_every: int = 100
    early_stopping_patience: int = 0
    no_progress: bool = False
    resume_from: Path | None = None
    max_validation_batches: int | None = None


@dataclass
class Stage2ConversationBatch:
    """Tokenized multi-turn batch consumed by the future authorized runtime."""

    input_ids: Any
    labels: Any
    attention_mask: Any
    sample_ids: list[str]
    dialogue_ids: list[str]
    audio_features: Any = None
    audio_attention_mask: Any = None
    audio_placeholder_mask: Any = None


class CachedConversationBatchSource:
    """Reiterable, bounded-memory batches backed by on-disk frozen ASR features."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        feature_dir: Path,
        audio_placeholder_id: int,
        batch_size: int,
        max_cached_shards: int,
        cache_factory: Callable[[Path, int], Any] | None = None,
    ) -> None:
        if batch_size <= 0 or max_cached_shards <= 0:
            raise ValueError("batch_size and max_cached_shards must be positive")
        self.rows = rows
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir
        self.audio_placeholder_id = audio_placeholder_id
        self.batch_size = batch_size
        self.max_cached_shards = max_cached_shards
        self.cache_factory = cache_factory

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[Stage2ConversationBatch]:
        if self.cache_factory is None:
            from training.omni.projector_stage1.feature_cache import FeatureCache

            feature_cache = FeatureCache(
                self.feature_dir, max_loaded_shards=self.max_cached_shards
            )
        else:
            feature_cache = self.cache_factory(self.feature_dir, self.max_cached_shards)
        for index in range(0, len(self.rows), self.batch_size):
            yield collate_cached_audio_conversations(
                self.rows[index : index + self.batch_size],
                self.tokenizer,
                feature_cache,
                self.audio_placeholder_id,
            )


def _system_prompt(sample_id: str) -> str:
    """Choose 80% canonical and 20% equivalent prompts deterministically."""
    bucket = hashlib.sha256(sample_id.encode("utf-8")).digest()[0] % 10
    if bucket < 8:
        return SYSTEM_PROMPT_VARIANTS[0]
    return SYSTEM_PROMPT_VARIANTS[1 + (bucket - 8) % 2]


def _normalise_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("dialogue_history must be a list")
    messages: list[dict[str, str]] = []
    role_map = {
        "user": "user",
        "human": "user",
        "system": "assistant",
        "assistant": "assistant",
    }
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"dialogue_history item {index} must be an object")
        raw_role = str(item.get("role", item.get("tag", ""))).lower()
        if raw_role not in role_map:
            raise ValueError(f"unsupported dialogue_history role: {raw_role!r}")
        content = str(item.get("text", item.get("content", ""))).strip()
        if not content:
            raise ValueError(f"dialogue_history item {index} has empty text")
        messages.append({"role": role_map[raw_role], "content": content})
    return messages


def _chat_token_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, generation: bool
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("LLM tokenizer must expose the official apply_chat_template")
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("apply_chat_template returned more than one sequence")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _drop_oldest_history_turn(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if not history:
        return []
    if (
        len(history) >= 2
        and history[0]["role"] == "user"
        and history[1]["role"] == "assistant"
    ):
        return history[2:]
    return history[1:]


def _build_supervised_sequence(
    row: dict[str, Any],
    tokenizer: Any,
    audio_placeholder_id: int,
    audio_token_count: int,
    max_length: int,
) -> dict[str, list[Any]]:
    missing = FORMAL_MANIFEST_FIELDS - row.keys()
    if missing:
        raise ValueError(
            f"manifest row lacks formal SpokenWOZ fields: {sorted(missing)}"
        )
    if audio_token_count <= 0:
        raise ValueError("audio token count must be positive")
    response = str(row["assistant_response"]).strip()
    if not response:
        raise ValueError("assistant_response cannot be empty")
    history = _normalise_history(row["dialogue_history"])
    audio_content = (
        AUDIO_START_TOKEN
        + AUDIO_PLACEHOLDER_TOKEN * audio_token_count
        + AUDIO_END_TOKEN
    )
    while True:
        messages = [
            {"role": "system", "content": _system_prompt(str(row["sample_id"]))},
            *history,
            {"role": "user", "content": audio_content},
        ]
        prompt_ids = _chat_token_ids(tokenizer, messages, generation=True)
        full_ids = _chat_token_ids(
            tokenizer,
            [*messages, {"role": "assistant", "content": response}],
            generation=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "official chat template does not preserve the assistant generation prefix"
            )
        if len(full_ids) <= max_length:
            break
        if not history:
            raise ValueError(
                f"sample {row['sample_id']} exceeds max_length without removable dialogue history"
            )
        history = _drop_oldest_history_turn(history)
    placeholder_mask = [token_id == audio_placeholder_id for token_id in full_ids]
    if sum(placeholder_mask) != audio_token_count:
        raise ValueError(
            "official chat template did not preserve the expected number of audio placeholders"
        )
    return {
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        "attention_mask": [1] * len(full_ids),
        "audio_placeholder_mask": placeholder_mask,
    }


def collate_cached_audio_conversations(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    feature_cache: Any,
    audio_placeholder_id: int,
    max_length: int = 4096,
) -> Stage2ConversationBatch:
    """Create text tensors plus padded cached ASR features and replacement masks."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not rows:
        raise ValueError("conversation batch cannot be empty")
    text_batches = []
    features = []
    for row in rows:
        cached = feature_cache.get(str(row["sample_id"]))
        feature = cached["features"]
        if feature.ndim != 2 or feature.shape[0] <= 0:
            raise ValueError("cached audio features must be rank-2 and non-empty")
        features.append(feature)
        text_batches.append(
            _build_supervised_sequence(
                row,
                tokenizer,
                audio_placeholder_id,
                int(feature.shape[0]),
                max_length,
            )
        )
    input_ids = pad_sequence(
        [torch.tensor(item["input_ids"], dtype=torch.long) for item in text_batches],
        batch_first=True,
        padding_value=getattr(tokenizer, "pad_token_id", 0) or 0,
    )
    labels = pad_sequence(
        [torch.tensor(item["labels"], dtype=torch.long) for item in text_batches],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = pad_sequence(
        [
            torch.tensor(item["attention_mask"], dtype=torch.bool)
            for item in text_batches
        ],
        batch_first=True,
        padding_value=False,
    )
    placeholder_mask = pad_sequence(
        [
            torch.tensor(item["audio_placeholder_mask"], dtype=torch.bool)
            for item in text_batches
        ],
        batch_first=True,
        padding_value=False,
    )
    audio_features = pad_sequence(features, batch_first=True)
    audio_attention_mask = torch.zeros(audio_features.shape[:2], dtype=torch.bool)
    for index, feature in enumerate(features):
        audio_attention_mask[index, : feature.shape[0]] = True
    return Stage2ConversationBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        sample_ids=[str(row["sample_id"]) for row in rows],
        dialogue_ids=[str(row["dialogue_id"]) for row in rows],
        audio_features=audio_features,
        audio_attention_mask=audio_attention_mask,
        audio_placeholder_mask=placeholder_mask,
    )


def collate_conversations(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 4096,
    audio_placeholder_id: int | None = None,
) -> Stage2ConversationBatch:
    """Build a one-placeholder CPU contract batch using the official chat template."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not rows:
        raise ValueError("conversation batch cannot be empty")
    if audio_placeholder_id is None:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            raise TypeError("LLM tokenizer must expose convert_tokens_to_ids")
        audio_placeholder_id = int(
            tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER_TOKEN)
        )
    sequences = [
        _build_supervised_sequence(row, tokenizer, audio_placeholder_id, 1, max_length)
        for row in rows
    ]
    input_ids = pad_sequence(
        [torch.tensor(item["input_ids"], dtype=torch.long) for item in sequences],
        batch_first=True,
        padding_value=getattr(tokenizer, "pad_token_id", 0) or 0,
    )
    labels = pad_sequence(
        [torch.tensor(item["labels"], dtype=torch.long) for item in sequences],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = pad_sequence(
        [torch.tensor(item["attention_mask"], dtype=torch.bool) for item in sequences],
        batch_first=True,
        padding_value=False,
    )
    placeholder_mask = pad_sequence(
        [
            torch.tensor(item["audio_placeholder_mask"], dtype=torch.bool)
            for item in sequences
        ],
        batch_first=True,
        padding_value=False,
    )
    return Stage2ConversationBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        sample_ids=[str(row["sample_id"]) for row in rows],
        dialogue_ids=[str(row["dialogue_id"]) for row in rows],
        audio_placeholder_mask=placeholder_mask,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1_projector_initialization(
    projector: Any,
    checkpoint: Path,
    expected_sha256: str = FROZEN_STAGE1_PROJECTOR_SHA256,
) -> dict[str, Any]:
    """Load the frozen stage-one projector and prove the exact checkpoint was used."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"stage-one projector checkpoint does not exist: {checkpoint}"
        )
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"stage-one projector SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    import torch

    from training.omni.projector_stage1.checkpoint_loading import (
        load_trusted_checkpoint,
    )

    state = load_trusted_checkpoint(checkpoint, torch)
    projector_config = state.get("config", {}).get("projector")
    if projector_config != projector.export_config():
        raise ValueError(
            "stage-one projector config does not match stage-two projector"
        )
    projector.load_state_dict(state["projector"], strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": actual_sha256,
        "format": state.get("format"),
        "config": projector_config,
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"manifest row {index} is not an object: {path}")
        missing = FORMAL_MANIFEST_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"manifest row {index} lacks formal SpokenWOZ fields {sorted(missing)}: {path}"
            )
        if not isinstance(row["dialogue_history"], list):
            raise TypeError(
                f"manifest row {index} dialogue_history is not a list: {path}"
            )
        if not str(row["assistant_response"]).strip():
            raise ValueError(
                f"manifest row {index} assistant_response is empty: {path}"
            )
    return rows


def _audit_formal_manifests(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    leaked = False
    for field in ("dialogue_id", "source_audio_sha256", "clip_sha256"):
        train_values = {row[field] for row in train_rows}
        dev_values = {row[field] for row in dev_rows}
        intersection = train_values & dev_values
        audit[field] = {
            "intersection_count": len(intersection),
            "examples": sorted(intersection, key=str)[:10],
        }
        leaked = leaked or bool(intersection)
    audit["leakage"] = leaked
    if leaked:
        fields = [
            field
            for field in ("dialogue_id", "source_audio_sha256", "clip_sha256")
            if audit[field]["intersection_count"]
        ]
        raise ValueError(f"train/dev leakage detected in: {', '.join(fields)}")
    return audit


def validate_manifests(train_manifest: Path, dev_manifest: Path) -> dict[str, Any]:
    train_rows = load_manifest(train_manifest)
    dev_rows = load_manifest(dev_manifest)
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("train manifest contains a non-train row")
    if any(row["split"] != "dev" for row in dev_rows):
        raise ValueError("dev manifest contains a non-dev row")
    audit = _audit_formal_manifests(train_rows, dev_rows)
    return {
        "train_samples": len(train_rows),
        "dev_samples": len(dev_rows),
        "train_manifest_sha256": _sha256(train_manifest),
        "dev_manifest_sha256": _sha256(dev_manifest),
        "leakage_audit": audit,
    }


def validate_config(config: Stage2Config) -> None:
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if config.learning_rate <= 0 or config.steps <= 0 or config.validation_every <= 0:
        raise ValueError(
            "learning rate, steps and validation interval must be positive"
        )
    if config.early_stopping_patience < 0:
        raise ValueError("early stopping patience cannot be negative")
    if config.max_validation_batches is not None and config.max_validation_batches <= 0:
        raise ValueError("max_validation_batches must be positive")
    if config.output_dir.exists() and config.resume_from is None:
        raise FileExistsError(
            f"refusing to reuse output directory: {config.output_dir}"
        )


def run_preflight(config: Stage2Config) -> dict[str, Any]:
    validate_config(config)
    manifest_info = validate_manifests(config.train_manifest, config.dev_manifest)
    config.output_dir.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "preflight-only",
        "asr_encoder": {"prefix": config.asr_encoder_prefix, "frozen": True},
        "llm_lora": {"rank": config.lora_rank, "alpha": config.lora_alpha},
        "prompt_contract": {
            "chat_template": "tokenizer.apply_chat_template",
            "canonical_system_prompt": CANONICAL_SYSTEM_PROMPT,
            "system_prompt_policy": "80% canonical, 20% deterministic equivalent variants",
            "current_user_text_input": False,
            "supervision": "assistant_response-only",
        },
        "config": {
            "train_manifest": str(config.train_manifest.resolve()),
            "dev_manifest": str(config.dev_manifest.resolve()),
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "validation_every": config.validation_every,
            "early_stopping_patience": config.early_stopping_patience,
        },
        "manifests": manifest_info,
    }
    (config.output_dir / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def freeze_asr_and_select_trainables(
    model: Any,
    asr_encoder_prefix: str = "audio_encoder",
    projector_prefix: str = "audio_projector",
) -> list[Any]:
    """Freeze the ASR encoder/base LLM and expose projector plus LoRA parameters only."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            name.startswith((projector_prefix + ".", "lora_"))
            or ".audio_projector." in name
            or ".lora_" in name
        )
        if name.startswith(asr_encoder_prefix + ".") and parameter.requires_grad:
            raise AssertionError(f"ASR encoder parameter became trainable: {name}")
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainables:
        raise ValueError("no projector or LoRA parameters are trainable")
    return trainables


def attach_lora(linear: Any, rank: int, alpha: float) -> Any:
    """Wrap a torch.nn.Linear without importing torch until this helper is used."""
    import torch
    from torch import nn

    if not isinstance(linear, nn.Linear):
        raise TypeError("attach_lora expects torch.nn.Linear")

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scale = alpha / rank

        def forward(self, inputs: Any) -> Any:
            update = (inputs @ self.lora_A.t()) @ self.lora_B.t()
            return self.base(inputs) + self.scale * update

    return LoRALinear(linear)


def inject_lora(
    model: Any, target_modules: Sequence[str], rank: int, alpha: float
) -> Any:
    """Replace named Linear leaves with LoRA wrappers and return the same model."""
    for module_name in target_modules:
        parent_name, _, leaf = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, leaf)
        setattr(parent, leaf, attach_lora(original, rank, alpha))
    return model


def build_stage2_model(
    asr_encoder: Any,
    llm: Any,
    projector_in_features: int,
    projector_out_features: int,
    lora_targets: Sequence[str],
    lora_rank: int,
    lora_alpha: float,
    stage1_projector_checkpoint: Path | None = None,
    stage1_projector_sha256: str = FROZEN_STAGE1_PROJECTOR_SHA256,
) -> Any:
    """Compose ASR encoder, projector and LLM, injecting LoRA into named LLM Linear leaves."""
    from torch import nn

    from training.omni.projector_stage1.model import CachedProjectorStage1Model
    from training.omni.projector_stage1.projector import (
        AudioProjector,
        AudioProjectorConfig,
    )

    if projector_in_features <= 0 or projector_out_features <= 0:
        raise ValueError("projector dimensions must be positive")
    inject_lora(llm, lora_targets, lora_rank, lora_alpha)
    projector = AudioProjector(
        AudioProjectorConfig(
            input_size=projector_in_features,
            output_size=projector_out_features,
            hidden_size=projector_out_features,
        )
    )
    projector_initialization = None
    if stage1_projector_checkpoint is not None:
        projector_initialization = load_stage1_projector_initialization(
            projector, stage1_projector_checkpoint, stage1_projector_sha256
        )
    core = CachedProjectorStage1Model(llm, projector)

    class Stage2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_encoder = asr_encoder
            self.core = core
            self.projector_initialization = projector_initialization

        def forward(self, batch: Any) -> Any:
            required = (
                "input_ids",
                "labels",
                "attention_mask",
                "audio_features",
                "audio_attention_mask",
                "audio_placeholder_mask",
            )
            if any(getattr(batch, name, None) is None for name in required):
                raise TypeError(
                    "Stage2 batch must include cached audio features and replacement masks"
                )
            input_ids = batch.input_ids.to(next(self.parameters()).device)
            labels = batch.labels.to(input_ids.device)
            embeddings = self.core.language_model.get_input_embeddings()(input_ids)
            attention_mask = batch.attention_mask.to(input_ids.device)
            projector_dtype = next(self.core.audio_projector.parameters()).dtype
            return self.core(
                audio_features=batch.audio_features.to(input_ids.device, dtype=projector_dtype),
                audio_attention_mask=batch.audio_attention_mask.to(input_ids.device),
                text_embeddings=embeddings,
                audio_placeholder_mask=batch.audio_placeholder_mask.to(
                    input_ids.device
                ),
                attention_mask=attention_mask,
                labels=labels,
            )

    model = Stage2Model()
    freeze_asr_and_select_trainables(model)
    return model


def build_model_from_pretrained(
    llm_model: str,
    projector_in_features: int,
    projector_out_features: int,
    lora_targets: Sequence[str],
    lora_rank: int,
    lora_alpha: float,
    stage1_projector_checkpoint: Path,
    stage1_projector_sha256: str,
    device: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = False,
) -> Any:
    """Load only the LLM: frozen ASR features are supplied by the validated cache."""
    import torch
    from torch import nn
    from transformers import AutoModelForImageTextToText

    torch_dtype = getattr(torch, dtype)
    llm = AutoModelForImageTextToText.from_pretrained(llm_model, dtype=torch_dtype)
    if gradient_checkpointing:
        if not hasattr(llm, "gradient_checkpointing_enable"):
            raise TypeError("LLM does not support gradient checkpointing")
        llm.gradient_checkpointing_enable()
        if hasattr(llm, "config"):
            llm.config.use_cache = False
    model = build_stage2_model(
        nn.Identity(),
        llm,
        projector_in_features,
        projector_out_features,
        lora_targets,
        lora_rank,
        lora_alpha,
        stage1_projector_checkpoint,
        stage1_projector_sha256,
    )
    return model.to(device=device, dtype=torch_dtype)


def _loss_value(output: Any) -> Any:
    return (
        output["loss"] if isinstance(output, dict) else getattr(output, "loss", output)
    )


def _write_loss_curve(records: list[dict[str, Any]], path: Path) -> None:
    width, height = 720, 360
    values = [
        value
        for row in records
        for value in (row.get("loss"), row.get("validation_loss"))
        if value is not None
    ]
    if not values:
        raise ValueError("cannot write loss curve without loss records")
    low, high = min(values), max(values)
    scale = max(high - low, 1e-12)

    def points(key: str) -> str:
        selected = [(row["step"], row[key]) for row in records if key in row]
        return " ".join(
            f"{40 + (step / max(1, records[-1]['step'])) * 640:.1f},"
            f"{320 - ((value - low) / scale) * 280:.1f}"
            for step, value in selected
        )

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<polyline fill="none" stroke="#2563eb" points="' + points("loss") + '"/>',
        '<polyline fill="none" stroke="#dc2626" points="'
        + points("validation_loss")
        + '"/>',
        "</svg>\n",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _save_checkpoint(
    path: Path, model: Any, optimizer: Any, scheduler: Any, step: int, best_loss: float
) -> None:
    import torch

    checkpoint_model = getattr(model, "module", model)
    trainable_names = {
        name
        for name, parameter in checkpoint_model.named_parameters()
        if parameter.requires_grad
    }
    state_dict = checkpoint_model.state_dict()
    trainable_state = {
        name: state_dict[name].detach().cpu() for name in trainable_names
    }
    if not trainable_state:
        raise RuntimeError("refusing to save a checkpoint without trainable weights")
    torch.save(
        {
            "format": "projector-stage2-v2",
            "step": step,
            "trainable_state": trainable_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_loss": best_loss,
            "projector_initialization": getattr(
                checkpoint_model, "projector_initialization", None
            ),
            "feature_cache": getattr(checkpoint_model, "feature_cache_metadata", None),
        },
        path,
    )


def train_model(
    model: Any,
    train_batches: Iterable[Any],
    dev_batches: Iterable[Any],
    config: Stage2Config,
) -> dict[str, Any]:
    """Train an injected model for CPU tests or a future authorized runtime."""
    import torch

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1

    if rank == 0:
        validate_config(config)
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        config.output_dir.mkdir(parents=True)
    if distributed:
        torch.distributed.barrier()
    base_model = getattr(model, "module", model)
    trainables = freeze_asr_and_select_trainables(
        base_model, config.asr_encoder_prefix, config.projector_prefix
    )
    optimizer = torch.optim.AdamW(trainables, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if not hasattr(train_batches, "__len__") or not hasattr(dev_batches, "__len__"):
        raise TypeError("train and dev batches must be re-iterable sized sources")
    if len(train_batches) == 0 or len(dev_batches) == 0:
        raise ValueError("train and dev batches must be non-empty")
    records: list[dict[str, Any]] = []
    best_validation = float("inf")
    bad_checks = 0
    start_step = 0
    if config.resume_from is not None:
        state = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        base_model.load_state_dict(state["trainable_state"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        best_validation = float(state.get("best_validation_loss", best_validation))
    train_iterator = iter(train_batches)
    progress = None
    if rank == 0 and not config.no_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=config.steps, desc="stage2 train", unit="step")
        except ImportError:
            pass
    for step in range(start_step + 1, config.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_batches)
            batch = next(train_iterator)
        model.train()
        optimizer.zero_grad()
        loss = _loss_value(model(batch))
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_value = loss.detach()
        if distributed:
            torch.distributed.all_reduce(loss_value)
            loss_value /= world_size
        record = {"step": step, "loss": float(loss_value)}
        if step % config.validation_every == 0 or step == config.steps:
            model.eval()
            with torch.no_grad():
                dev_losses = [
                    _loss_value(model(batch)).detach()
                    for batch in islice(iter(dev_batches), config.max_validation_batches)
                ]
            validation_total = torch.stack(dev_losses).sum()
            validation_count = torch.tensor(float(len(dev_losses)), device=validation_total.device)
            if distributed:
                torch.distributed.all_reduce(validation_total)
                torch.distributed.all_reduce(validation_count)
            validation_loss = float(validation_total / validation_count)
            record["validation_loss"] = validation_loss
            if validation_loss < best_validation:
                best_validation = validation_loss
                bad_checks = 0
                if rank == 0:
                    _save_checkpoint(
                        config.output_dir / "best.pt", model, optimizer, scheduler, step, best_validation
                    )
            else:
                bad_checks += 1
        if rank == 0:
            records.append(record)
        if progress is not None:
            progress.update(1)
            progress.set_postfix(loss=f"{record['loss']:.4f}")
        if (
            config.early_stopping_patience
            and bad_checks >= config.early_stopping_patience
        ):
            break
    if progress is not None:
        progress.close()
    if rank == 0:
        _save_checkpoint(
            config.output_dir / "last.pt", model, optimizer, scheduler, records[-1]["step"], best_validation
        )
        (config.output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
        )
        _write_loss_curve(records, config.output_dir / "loss_curve.svg")
    return {
        "steps": records[-1]["step"] if rank == 0 else config.steps,
        "best_validation_loss": best_validation,
        "records": records if rank == 0 else [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 projector plus LLM-LoRA CPU preflight."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument(
        "--feature-dir", type=Path, help="Frozen ASR feature cache used by --run"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-encoder-prefix", default="audio_encoder")
    parser.add_argument("--projector-prefix", default="audio_projector")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate only; real model loading is intentionally not performed",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Load models and train; requires explicit runtime authorization",
    )
    parser.add_argument("--asr-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--projector-in-features", type=int)
    parser.add_argument("--projector-out-features", type=int)
    parser.add_argument("--init-projector-checkpoint", type=Path)
    parser.add_argument(
        "--init-projector-sha256", default=FROZEN_STAGE1_PROJECTOR_SHA256
    )
    parser.add_argument("--lora-target", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-feature-shards", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distributed = args.distributed
    rank = 0
    world_size = 1
    if distributed:
        import torch

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = f"cuda:{local_rank}"
    config = Stage2Config(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        asr_encoder_prefix=args.asr_encoder_prefix,
        projector_prefix=args.projector_prefix,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        steps=args.steps,
        validation_every=args.validation_every,
        early_stopping_patience=args.early_stopping_patience,
        no_progress=args.no_progress,
        max_validation_batches=args.max_validation_batches,
    )
    if not args.run:
        result = run_preflight(config)
    else:
        if args.max_cached_feature_shards <= 0:
            raise ValueError("--max-cached-feature-shards must be positive")
        required = {
            "--llm-model": args.llm_model,
            "--projector-in-features": args.projector_in_features,
            "--projector-out-features": args.projector_out_features,
            "--feature-dir": args.feature_dir,
            "--init-projector-checkpoint": args.init_projector_checkpoint,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if not args.lora_target:
            missing.append("--lora-target")
        if missing:
            raise ValueError("--run requires " + ", ".join(missing))
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.llm_model, fix_mistral_regex=True
        )
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            raise ValueError("LLM tokenizer must expose convert_tokens_to_ids")
        audio_placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
        if audio_placeholder_id is None or audio_placeholder_id == getattr(
            tokenizer, "unk_token_id", None
        ):
            raise ValueError("LLM tokenizer has no <|vision_pad|> audio placeholder")
        train_rows = load_manifest(args.train_manifest)
        dev_rows = load_manifest(args.dev_manifest)
        cache_metadata = validate_feature_cache(
            args.feature_dir, [*train_rows, *dev_rows]
        )
        model = build_model_from_pretrained(
            args.llm_model,
            args.projector_in_features,
            args.projector_out_features,
            args.lora_target,
            args.lora_rank,
            args.lora_alpha,
            args.init_projector_checkpoint,
            args.init_projector_sha256,
            args.device,
            args.dtype,
            args.gradient_checkpointing,
        )
        model.feature_cache_metadata = cache_metadata
        freeze_asr_and_select_trainables(model, args.asr_encoder_prefix, args.projector_prefix)
        if distributed:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        train_rows = train_rows[rank::world_size]
        dev_rows = dev_rows[rank::world_size]
        train_batches = CachedConversationBatchSource(
            train_rows, tokenizer, args.feature_dir, int(audio_placeholder_id),
            args.batch_size, args.max_cached_feature_shards,
        )
        dev_batches = CachedConversationBatchSource(
            dev_rows, tokenizer, args.feature_dir, int(audio_placeholder_id),
            args.batch_size, args.max_cached_feature_shards,
        )
        result = train_model(model, train_batches, dev_batches, config)
    if rank == 0:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if distributed:
        import torch

        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
