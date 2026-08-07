import json
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None

from training.omni.projector_stage2.cache_features import validate_feature_cache
from training.omni.projector_stage2.train import (
    Stage2Config,
    _build_supervised_sequence,
    build_parser,
    build_stage2_model,
    collate_cached_audio_conversations,
    collate_conversations,
    freeze_asr_and_select_trainables,
    load_stage1_projector_initialization,
    run_preflight,
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


def test_cli_help_and_preflight_writes_contract(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--help"])
    assert error.value.code == 0
    result = run_preflight(_config(tmp_path))
    assert result["status"] == "preflight-only"
    assert result["prompt_contract"]["current_user_text_input"] is False
    assert result["prompt_contract"]["supervision"] == "assistant_response-only"
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


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_collator_preserves_ids_masks_and_attention():
    batch = collate_conversations([_row()], ChatTokenizer())
    assert batch.sample_ids == ["s1"]
    assert batch.dialogue_ids == ["d1"]
    assert batch.audio_placeholder_mask.sum().item() == 1
    assert batch.labels[0, -2:].tolist() == [10, 5]
    assert batch.attention_mask.all()


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


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_audio_features_change_inputs_and_loss():
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
                self.proj(inputs_embeds).mean() - labels.float().mean() / 10
            ).square()

    model = build_stage2_model(torch.nn.Linear(2, 2), TinyLLM(), 2, 2, ["proj"], 1, 2.0)
    tokenizer = ChatTokenizer()
    rows = [_row()]
    cache = Cache()
    first = collate_cached_audio_conversations(rows, tokenizer, cache, 7)
    loss_one = model(first)
    cache.features = torch.tensor([[0.0, 1.0]])
    second = collate_cached_audio_conversations(rows, tokenizer, cache, 7)
    loss_two = model(second)
    assert first.audio_placeholder_mask.sum().item() == 1
    assert first.labels[0, :-2].eq(-100).all()
    assert not torch.equal(first.audio_features, second.audio_features)
    assert not torch.equal(loss_one, loss_two)


def test_sequence_rejects_legacy_text_only_rows():
    with pytest.raises(ValueError, match="formal SpokenWOZ fields"):
        _build_supervised_sequence(
            {"sample_id": "s1", "dialogue_id": "d1", "text": "legacy target"},
            ChatTokenizer(),
            7,
            1,
            4096,
        )


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
