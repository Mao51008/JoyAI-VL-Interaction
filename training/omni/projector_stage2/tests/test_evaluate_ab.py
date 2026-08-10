import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.omni.projector_stage2.evaluate_ab import (
    CheckpointParts,
    aggregate_records,
    apply_configuration,
    load_stage2_checkpoint_parts,
    paired_comparison,
    select_dialogue_balanced_rows,
)


def _parts() -> CheckpointParts:
    return CheckpointParts(
        projector={"core.audio_projector.weight": torch.tensor([[3.0]])},
        lora={
            "core.language_model.proj.lora_A": torch.tensor([[4.0]]),
            "core.language_model.proj.lora_B": torch.tensor([[5.0]]),
        },
        lora_targets=("proj",),
        lora_rank=1,
        metadata={},
    )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.audio_projector = torch.nn.Linear(1, 1, bias=False)
        self.core.language_model = torch.nn.Module()
        self.core.language_model.proj = torch.nn.Module()
        self.core.language_model.proj.lora_A = torch.nn.Parameter(torch.tensor([[1.0]]))
        self.core.language_model.proj.lora_B = torch.nn.Parameter(torch.tensor([[2.0]]))


def test_apply_configuration_switches_projector_and_lora_independently():
    model = TinyModel()
    stage1 = {"core.audio_projector.weight": torch.tensor([[1.0]])}
    apply_configuration(model, "stage2_projector_no_lora", stage1, _parts())
    assert model.core.audio_projector.weight.item() == 3.0
    assert model.core.language_model.proj.lora_A.item() == 0.0
    assert model.core.language_model.proj.lora_B.item() == 0.0
    apply_configuration(model, "stage1_projector_stage2_lora", stage1, _parts())
    assert model.core.audio_projector.weight.item() == 1.0
    assert model.core.language_model.proj.lora_A.item() == 4.0
    assert model.core.language_model.proj.lora_B.item() == 5.0


def test_checkpoint_loader_splits_and_validates_state(tmp_path: Path):
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "format": "projector-stage2-v2",
            "step": 7,
            "best_validation_loss": 1.25,
            "projector_initialization": {"sha256": "stage1"},
            "feature_cache": {"samples": 2},
            "trainable_state": {
                "core.audio_projector.weight": torch.ones(1, 1),
                "core.language_model.proj.lora_A": torch.ones(2, 3),
                "core.language_model.proj.lora_B": torch.ones(4, 2),
            },
        },
        checkpoint,
    )
    parts = load_stage2_checkpoint_parts(checkpoint, "stage1")
    assert parts.lora_targets == ("proj",)
    assert parts.lora_rank == 2
    assert parts.metadata["step"] == 7
    assert parts.metadata["feature_cache"] == {"samples": 2}


def test_dialogue_selection_is_deterministic_and_unique():
    rows = [
        {"dialogue_id": f"d{dialogue}", "sample_id": f"s{dialogue}-{turn}"}
        for dialogue in range(5)
        for turn in range(3)
    ]
    first = select_dialogue_balanced_rows(rows, 4, 9)
    second = select_dialogue_balanced_rows(rows, 4, 9)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len({row["dialogue_id"] for row in first}) == 4


def test_aggregate_and_paired_metrics_use_tokens_and_matching_ids():
    baseline = [
        {"sample_id": "a", "nll": 2.0, "supervised_tokens": 1},
        {"sample_id": "b", "nll": 1.0, "supervised_tokens": 3},
    ]
    candidate = [
        {"sample_id": "a", "nll": 1.0, "supervised_tokens": 1},
        {"sample_id": "b", "nll": 2.0, "supervised_tokens": 3},
    ]
    metrics = aggregate_records(baseline)
    assert metrics["token_weighted_nll"] == pytest.approx(1.25)
    assert metrics["sample_mean_nll"] == pytest.approx(1.5)
    paired = paired_comparison(baseline, candidate)
    assert paired == {
        "samples": 2,
        "mean_nll_delta": 0.0,
        "fraction_candidate_lower": 0.5,
    }
