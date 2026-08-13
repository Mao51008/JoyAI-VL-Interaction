import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:
    torch = None

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.train import (
    CachedConversationBatchSource,
    Stage2Config,
    _build_supervised_sequence,
    _optimizer_parameter_groups,
    build_parser,
    build_stage2_model,
    collate_vision_distillation,
    collate_cached_audio_conversations,
    collate_conversations,
    freeze_asr_and_select_trainables,
    load_vision_distillation_manifest,
    load_stage1_projector_initialization,
    MixedStage2BatchSource,
    run_preflight,
    WeightedAudioBatchSource,
    train_model,
)


def _manifest(path: Path, split: str, sample: str) -> None:
    row = {
        "sample_id": sample,
        "dialogue_id": sample,
        "turn_id": "0",
        "split": split,
        "audio_path": f"audio/{split}/{sample}.wav",
        "clip_duration_ms": 1000.0,
        "clip_sha256": f"clip-{sample}",
        "source_audio_sha256": f"source-{sample}",
        "user_text": "private current transcript",
        "assistant_response": "the supervised reply",
        "dialogue_history": [],
        "provenance": {
            "dataset": "SpokenWOZ",
            "version": "test",
            "split": split,
            "source_file": f"{split}.json",
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _config(tmp_path: Path) -> Stage2Config:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    _manifest(train, "train", "train-1")
    _manifest(dev, "dev", "dev-1")
    return Stage2Config(
        train, dev, tmp_path / "out", steps=4, validation_every=2, no_progress=True
    )


def _row(sample: str = "s1") -> dict:
    return {
        "sample_id": sample,
        "dialogue_id": "d1",
        "turn_id": "2",
        "split": "train",
        "audio_path": "audio/train/d1_turn0002.wav",
        "clip_duration_ms": 1000.0,
        "clip_sha256": "clip-s1",
        "source_audio_sha256": "source-d1",
        "user_text": "SECRET CURRENT TRANSCRIPT",
        "assistant_response": "target reply",
        "dialogue_history": [
            {"role": "user", "text": "previous request"},
            {"role": "system", "text": "previous response"},
        ],
        "provenance": {"dataset": "SpokenWOZ", "split": "train"},
    }


class ChatTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        assert token == "<|vision_pad|>"
        return 7

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        self.calls.append((messages, add_generation_prompt))
        role_ids = {"system": 2, "user": 3, "assistant": 4}
        ids = [1]
        for message in messages:
            ids.append(role_ids[message["role"]])
            content = message["content"]
            if "<|vision_pad|>" in content:
                assert content.startswith("<|vision_start|>")
                assert content.endswith("<|vision_end|>")
                ids.extend([6] + [7] * content.count("<|vision_pad|>") + [8])
            else:
                ids.append(10 if content == "target reply" else 9)
            ids.append(5)
        if add_generation_prompt:
            ids.append(4)
        return ids


class VisionProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict, return_tensors):
        assert tokenize and return_dict and return_tensors == "pt"
        self.calls.append(messages)
        for message in messages:
            if message["role"] == "assistant":
                assert isinstance(message["content"], list)
                assert len(message["content"]) == 1
                assert message["content"][0]["type"] == "text"
                assert isinstance(message["content"][0]["text"], str)
        has_response = any(message["role"] == "assistant" for message in messages)
        ids = [1, 2, 3, 4] if has_response else [1, 2, 3]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values": torch.ones((1, 3)),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


def test_cli_help_and_preflight_writes_contract(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--help"])
    assert error.value.code == 0
    result = run_preflight(_config(tmp_path))
    assert result["status"] == "preflight-only"
    assert result["prompt_contract"]["current_user_text_input"] is False
    assert (
        result["prompt_contract"]["supervision"]
        == "assistant_response-and-asr_transcription"
    )
    assert result["prompt_contract"]["asr_transcript_prompt_leakage"] is False
    assert result["config"]["asr_replay_ratio"] == pytest.approx(0.3)
    assert (
        json.loads((tmp_path / "out" / "preflight.json").read_text())["manifests"][
            "train_samples"
        ]
        == 1
    )


def test_sequence_uses_official_chat_template_without_current_transcript():
    tokenizer = ChatTokenizer()
    sequence = _build_supervised_sequence(_row(), tokenizer, 7, 1, 4096)
    assert sum(sequence["audio_placeholder_mask"]) == 1
    assert sequence["labels"][-2:] == [10, 5]
    assert sequence["labels"][:-2] == [-100] * (len(sequence["labels"]) - 2)
    prompt_messages = tokenizer.calls[0][0]
    assert [message["role"] for message in prompt_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "SECRET CURRENT TRANSCRIPT" not in json.dumps(tokenizer.calls)
    assert prompt_messages[-1]["content"].count("<|vision_pad|>") == 1


def test_asr_replay_supervises_transcript_without_leaking_it_into_prompt():
    tokenizer = ChatTokenizer()
    row = _row()
    row["training_task"] = "asr_transcription"
    sequence = _build_supervised_sequence(row, tokenizer, 7, 1, 4096)
    assert sequence["labels"][-2:] == [9, 5]
    prompt_messages = tokenizer.calls[0][0]
    assert [message["role"] for message in prompt_messages] == ["system", "user"]
    assert "SECRET CURRENT TRANSCRIPT" not in json.dumps(prompt_messages)
    assert "Transcribe" in prompt_messages[0]["content"]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_vision_distillation_uses_only_fixed_teacher_response(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    row = {
        "sample_id": "vision-1",
        "split": "train",
        "image_path": image.name,
        "prompt": "What is in the image?",
        "teacher_response": "A fixture image.",
        "provenance": {"teacher_model": "JoyAI-VL"},
    }
    manifest = tmp_path / "vision.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_vision_distillation_manifest(manifest)
    processor = VisionProcessor()
    batch = collate_vision_distillation(loaded[0], processor)
    assert batch.modality == "vision_distillation"
    assert batch.task_types == ["vision_distillation"]
    assert batch.labels.tolist() == [[-100, -100, -100, 4]]
    assert batch.vision_inputs["pixel_values"].shape == (1, 3)
    assert processor.calls[1][-1]["content"] == [{"type": "text", "text": "A fixture image."}]


def test_mixed_source_preserves_all_batches():
    source = MixedStage2BatchSource([["audio-1", "audio-2"], ["vision-1"]], shuffle=False)
    assert list(source) == ["audio-1", "audio-2", "vision-1"]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_collator_preserves_ids_masks_and_attention():
    batch = collate_conversations([_row()], ChatTokenizer())
    assert batch.sample_ids == ["s1"]
    assert batch.dialogue_ids == ["d1"]
    assert batch.audio_placeholder_mask.sum().item() == 1
    assert batch.labels[0, -2:].tolist() == [10, 5]
    assert batch.attention_mask.all()


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_cached_batch_source_defers_feature_reads_until_iteration():
    created = []

    class Cache:
        def get(self, sample_id):
            return {"features": torch.ones(2, 2)}

    def cache_factory(directory, max_loaded_shards):
        created.append((directory, max_loaded_shards))
        return Cache()

    source = CachedConversationBatchSource(
        [_row("s1"), _row("s2")],
        ChatTokenizer(),
        Path("cache"),
        7,
        batch_size=1,
        max_cached_shards=3,
        cache_factory=cache_factory,
    )
    assert len(source) == 2
    assert created == []
    first = next(iter(source))
    assert first.sample_ids == ["s1"]
    assert created == [(Path("cache"), 3)]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_cached_batch_source_shuffles_deterministically_by_epoch():
    class Cache:
        def get(self, sample_id):
            return {"features": torch.ones(2, 2)}

    rows = [_row(f"s{index}") for index in range(6)]

    def source():
        return CachedConversationBatchSource(
            rows,
            ChatTokenizer(),
            Path("cache"),
            7,
            batch_size=2,
            max_cached_shards=1,
            shuffle=True,
            seed=11,
            cache_factory=lambda *_: Cache(),
        )

    first_source = source()
    first_epoch = [sample for batch in first_source for sample in batch.sample_ids]
    second_epoch = [sample for batch in first_source for sample in batch.sample_ids]
    repeated_first_epoch = [sample for batch in source() for sample in batch.sample_ids]
    assert first_epoch == repeated_first_epoch
    assert first_epoch != second_epoch
    assert sorted(first_epoch) == sorted(row["sample_id"] for row in rows)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_cached_batch_source_applies_deterministic_asr_replay_ratio():
    class Cache:
        def get(self, sample_id):
            return {"features": torch.ones(2, 2)}

    source = CachedConversationBatchSource(
        [_row(f"s{index}") for index in range(10)],
        ChatTokenizer(),
        Path("cache"),
        7,
        batch_size=2,
        max_cached_shards=1,
        shuffle=True,
        seed=19,
        asr_replay_ratio=0.3,
        cache_factory=lambda *_: Cache(),
    )
    task_types = [task for batch in source for task in batch.task_types]
    assert task_types.count("asr_transcription") == 3
    assert task_types.count("dialogue_response") == 7


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_cached_batch_source_preserves_explicit_task_assignments():
    class Cache:
        def get(self, sample_id):
            return {"features": torch.ones(2, 2)}

    rows = [_row(f"s{index}") for index in range(10)]
    for index, row in enumerate(rows):
        row["training_task"] = (
            "asr_transcription" if index < 4 else "dialogue_response"
        )
    source = CachedConversationBatchSource(
        rows,
        ChatTokenizer(),
        Path("cache"),
        7,
        batch_size=2,
        max_cached_shards=1,
        shuffle=True,
        seed=19,
        asr_replay_ratio=0.9,
        cache_factory=lambda *_: Cache(),
    )
    task_types = [task for batch in source for task in batch.task_types]
    assert task_types.count("asr_transcription") == 4
    assert task_types.count("dialogue_response") == 6


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_weighted_audio_source_covers_voiceassistant_with_fixed_mix():
    class Cache:
        def get(self, sample_id):
            return {"features": torch.ones(1, 2)}

    def rows(dataset, count, task="dialogue_response"):
        result = []
        for index in range(count):
            row = _row(f"{dataset}-{index}")
            row["provenance"] = {"dataset": dataset}
            row["training_task"] = task
            result.append(row)
        return result

    source = WeightedAudioBatchSource(
        [
            *rows("shenyunhang/VoiceAssistant-400K", 10),
            *rows("Clotho-AQA", 3),
            *rows("LibriSpeech", 4, "asr_transcription"),
        ],
        ChatTokenizer(),
        Path("cache"),
        7,
        batch_size=1,
        max_cached_shards=1,
        seed=23,
        cache_factory=lambda *_: Cache(),
    )
    sample_ids = [batch.sample_ids[0] for batch in source]
    assert len(sample_ids) == 20
    assert sum(sample_id.startswith("shenyunhang/VoiceAssistant-400K") for sample_id in sample_ids) == 10
    assert sum(sample_id.startswith("Clotho-AQA") for sample_id in sample_ids) == 3
    assert sum(sample_id.startswith("LibriSpeech") for sample_id in sample_ids) == 7
    assert {
        sample_id for sample_id in sample_ids
        if sample_id.startswith("shenyunhang/VoiceAssistant-400K")
    } == {f"shenyunhang/VoiceAssistant-400K-{index}" for index in range(10)}


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_model_builder_injects_lora_and_freezes_asr():
    asr = torch.nn.Linear(2, 2)
    llm = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model = build_stage2_model(asr, llm, 2, 2, ["0"], 2, 4.0)
    names = dict(model.named_parameters())
    assert any("lora_A" in name for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in names.items()
        if name.startswith("audio_encoder.")
    )
    groups = _optimizer_parameter_groups(
        model, Stage2Config(Path("train"), Path("dev"), Path("out"))
    )
    assert [group["group_name"] for group in groups] == ["projector", "lora"]
    assert [group["lr"] for group in groups] == [3e-6, 1e-5]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_audio_features_change_inputs_and_loss():
    torch.manual_seed(0)

    class Cache:
        def __init__(self):
            self.features = torch.tensor([[1.0, 0.0]])

        def get(self, sample_id):
            return {"features": self.features}

    class TinyLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 2)
            self.proj = torch.nn.Linear(2, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, inputs_embeds, labels, **kwargs):
            return (
                self.proj(inputs_embeds).sum() - labels.float().mean() / 10
            ).square()

    model = build_stage2_model(torch.nn.Linear(2, 2), TinyLLM(), 2, 2, ["proj"], 1, 2.0)
    model.to(dtype=torch.bfloat16)
    tokenizer = ChatTokenizer()
    rows = [_row()]
    cache = Cache()
    first = collate_cached_audio_conversations(rows, tokenizer, cache, 7)
    loss_one = model(first)
    cache.features = torch.tensor([[1.0, 1.0]])
    second = collate_cached_audio_conversations(rows, tokenizer, cache, 7)
    loss_two = model(second)
    assert first.audio_placeholder_mask.sum().item() == 1
    assert first.labels[0, :-2].eq(-100).all()
    assert not torch.equal(first.audio_features, second.audio_features)
    assert not torch.equal(loss_one, loss_two)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_model_routes_vision_distillation_through_language_model(tmp_path):
    class TinyVisionLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 2)
            self.proj = torch.nn.Linear(2, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids=None, pixel_values=None, labels=None, **kwargs):
            assert input_ids is not None
            assert pixel_values is not None
            assert labels is not None
            return (self.proj(self.embedding(input_ids)).sum() + pixel_values.sum()).square()

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    batch = collate_vision_distillation(
        {
            "sample_id": "vision-1",
            "_image_path": str(image),
            "prompt": "Describe this.",
            "teacher_response": "A fixture.",
        },
        VisionProcessor(),
    )
    model = build_stage2_model(
        torch.nn.Linear(2, 2), TinyVisionLLM(), 2, 2, ["proj"], 1, 2.0
    )
    assert model(batch).isfinite()


def test_sequence_rejects_legacy_text_only_rows():
    with pytest.raises(ValueError, match="formal SpokenWOZ fields"):
        _build_supervised_sequence(
            {"sample_id": "s1", "dialogue_id": "d1", "text": "legacy target"},
            ChatTokenizer(),
            7,
            1,
            4096,
        )


def test_sequence_uses_explicit_system_prompt_override_only_when_requested():
    tokenizer = ChatTokenizer()
    row = _row()
    row["system_prompt_override"] = "Base prompt\nListen to the user's audio and answer the request directly."
    _build_supervised_sequence(row, tokenizer, 7, 1, 4096)
    assert tokenizer.calls[0][0][0]["content"] == row["system_prompt_override"]


def test_feature_cache_validation_rejects_manifest_fingerprint_mismatch(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    (cache / "index.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "s1",
                    "clip_sha256": "clip-s1",
                    "source_audio_sha256": "source-d1",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert validate_feature_cache(cache, [_row()])["schema_version"] == 1
    mismatched = _row()
    mismatched["clip_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatched"):
        validate_feature_cache(cache, [mismatched])


def test_preflight_rejects_leakage_and_invalid_parameters(tmp_path):
    config = _config(tmp_path)
    dev = config.dev_manifest
    row = json.loads(dev.read_text())
    row["dialogue_id"] = "train-1"
    dev.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        run_preflight(config)
    with pytest.raises(ValueError, match="positive"):
        run_preflight(
            Stage2Config(
                config.train_manifest, config.dev_manifest, tmp_path / "bad", steps=0
            )
        )


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_mock_training_freezes_asr_and_writes_metrics_checkpoints_curve(tmp_path):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder = torch.nn.Linear(1, 1)
            self.audio_projector = torch.nn.Linear(1, 1)
            self.lora_weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, batch):
            return (self.audio_projector(batch).sum() + self.lora_weight).square()

    model = MockModel()
    trainables = freeze_asr_and_select_trainables(model)
    assert model.audio_encoder.weight.requires_grad is False
    assert any(parameter is model.audio_projector.weight for parameter in trainables)
    result = train_model(
        model,
        [torch.ones(1, 1)],
        [torch.ones(1, 1)],
        _config(tmp_path),
    )
    assert result["steps"] == 4
    assert all(
        (tmp_path / "out" / name).is_file()
        for name in ("best.pt", "last.pt", "metrics.jsonl", "loss_curve.svg")
    )
    state = torch.load(
        tmp_path / "out" / "last.pt", map_location="cpu", weights_only=True
    )
    assert "model" not in state
    assert set(state["trainable_state"]) == {
        "audio_projector.weight",
        "audio_projector.bias",
        "lora_weight",
    }
    assert state["training_config"]["gradient_accumulation_steps"] == 8
    assert state["lora_targets"] == []
    assert [group["group_name"] for group in state["optimizer"]["param_groups"]] == [
        "projector",
        "lora",
    ]
    assert all(record["supervised_tokens"] == 8 for record in result["records"])
    assert all("gradient_norm" in record for record in result["records"])
    assert all(record["task_samples"] == {"unknown": 8} for record in result["records"])


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_training_and_validation_losses_are_weighted_by_supervised_tokens(tmp_path):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder = torch.nn.Linear(1, 1)
            self.audio_projector = torch.nn.Linear(1, 1)
            self.lora_weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, batch):
            return self.audio_projector.weight.sum() * 0 + batch.loss

    def batch(loss, supervised_tokens):
        labels = torch.full((1, supervised_tokens + 1), -100, dtype=torch.long)
        labels[:, 1:] = 1
        return SimpleNamespace(loss=torch.tensor(float(loss)), labels=labels)

    config = replace(
        _config(tmp_path),
        steps=1,
        validation_every=1,
        gradient_accumulation_steps=2,
        warmup_steps=0,
    )
    result = train_model(
        MockModel(),
        [batch(1, 1), batch(3, 3)],
        [batch(1, 1), batch(3, 3)],
        config,
    )
    assert result["records"][0]["loss"] == pytest.approx(2.5)
    assert result["records"][0]["supervised_tokens"] == 4
    assert result["records"][0]["validation_loss"] == pytest.approx(2.5)
    assert result["records"][0]["validation_supervised_tokens"] == 4


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_stage_one_projector_checkpoint_initializes_matching_projector(tmp_path):
    from training.omni.projector_stage1.projector import (
        AudioProjector,
        AudioProjectorConfig,
    )

    source = AudioProjector(AudioProjectorConfig(input_size=2, output_size=3))
    checkpoint = tmp_path / "stage1.pt"
    torch.save(
        {
            "format": "projector-stage1-v4",
            "config": {"projector": source.export_config()},
            "projector": source.state_dict(),
        },
        checkpoint,
    )
    target = AudioProjector(AudioProjectorConfig(input_size=2, output_size=3))
    import hashlib

    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance = load_stage1_projector_initialization(target, checkpoint, expected)
    assert provenance["sha256"] == expected
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.state_dict()
    )
