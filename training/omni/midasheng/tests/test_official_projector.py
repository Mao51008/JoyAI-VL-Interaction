import pytest

torch = pytest.importorskip("torch")

from training.omni.midasheng.official_projector import adapt_official_projector, subsampled_token_count


def test_adapt_official_projector_retains_first_layer_and_replaces_only_head():
    source = torch.nn.Module()
    source.k = 5
    source.net = torch.nn.Sequential(torch.nn.Linear(6400, 3584), torch.nn.GELU(), torch.nn.Linear(3584, 3584))
    first_weight = source.net[0].weight.detach().clone()
    projector = adapt_official_projector(source)
    assert projector.k == 5
    assert torch.equal(projector.net[0].weight, first_weight)
    assert (projector.net[2].in_features, projector.net[2].out_features) == (3584, 4096)
    assert all(parameter.requires_grad for parameter in projector.parameters())


def test_subsampled_token_count_matches_official_trailing_discard():
    assert subsampled_token_count(10) == 2
    assert subsampled_token_count(14) == 2
    with pytest.raises(ValueError):
        subsampled_token_count(4)
