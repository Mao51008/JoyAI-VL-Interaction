import pytest

torch = pytest.importorskip("torch")

from training.omni.midasheng.joint_train import (
    ENCODER_LR,
    LORA_LR,
    PROJECTOR_LR,
    _task_counts,
    configure_high_encoder_blocks,
    language_all_linear_targets,
)


def test_joint_sampling_is_exact_for_one_hundred_samples():
    assert _task_counts(100) == {
        "voiceassistant": 40,
        "clotho_aqa": 20,
        "librispeech": 40,
    }


def test_joint_encoder_unfreezes_final_eight_of_thirty_two_blocks_only():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.front_end = torch.nn.Linear(1, 1)
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(1, 1) for _ in range(32))

    encoder = Encoder()
    report = configure_high_encoder_blocks(encoder)
    assert report["total_blocks"] == 32
    assert report["unfrozen_block_indices"] == list(range(24, 32))
    assert not encoder.front_end.weight.requires_grad
    assert not encoder.blocks[23].weight.requires_grad
    assert encoder.blocks[24].weight.requires_grad


def test_all_linear_targets_are_scoped_to_language_model():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                setattr(self, name, torch.nn.Linear(2, 2))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module(); self.language_model.layer = Layer()
            self.visual = torch.nn.Module(); self.visual.q_proj = torch.nn.Linear(2, 2)

    targets = language_all_linear_targets(Model())
    assert len(targets) == 7
    assert all(target.startswith("language_model.") for target in targets)
    assert (ENCODER_LR, PROJECTOR_LR, LORA_LR) == (3e-7, 3e-6, 1e-5)
