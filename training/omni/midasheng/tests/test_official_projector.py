import pytest

torch = pytest.importorskip("torch")

from training.omni.midasheng.official_projector import build_frozen_projector_adapter, subsampled_token_count


def test_adapter_retains_complete_official_projector_and_freezes_it():
    source = torch.nn.Module()
    source.k = 5
    source.net = torch.nn.Sequential(torch.nn.Linear(6400, 3584), torch.nn.GELU(), torch.nn.Linear(3584, 3584))
    first_weight = source.net[0].weight.detach().clone()
    projector = build_frozen_projector_adapter(source)
    assert projector.official_projector.k == 5
    assert torch.equal(projector.official_projector.net[0].weight, first_weight)
    assert (projector.official_projector.net[2].in_features, projector.official_projector.net[2].out_features) == (3584, 3584)
    assert all(not parameter.requires_grad for parameter in projector.official_projector.parameters())
    assert all(parameter.requires_grad for parameter in projector.joyai_adapter.parameters())


def test_subsampled_token_count_matches_official_trailing_discard():
    assert subsampled_token_count(10) == 2
    assert subsampled_token_count(14) == 2
    with pytest.raises(ValueError):
        subsampled_token_count(4)
