"""Unit tests for OMWU, ExtraGradient, MWU, and GDA 2D batched dynamics and numerical stability."""

import pytest
import torch


from src.dynamics.mwu import MultiplicativeWeightsUpdate
from src.dynamics.omwu import OptimisticMWU





@pytest.mark.parametrize(
    "dynamic_cls",
    [OptimisticMWU, MultiplicativeWeightsUpdate],
)
def test_heterogeneous_action_sizes_vectorized_step(dynamic_cls):
    """Test 2D vectorized dynamics with heterogeneous action sizes [3, 5, 2]."""
    action_sizes = [3, 5, 2]
    dyn = dynamic_cls(action_sizes=action_sizes, eta=0.05)

    u_vecs = [
        torch.tensor([1.0, -1.0, 0.5]),
        torch.tensor([0.5, 0.2, -0.7, 1.2, -0.3]),
        torch.tensor([-0.8, 0.8]),
    ]

    new_strats = dyn.step(u_vecs)

    assert len(new_strats) == 3
    for i, a_size in enumerate(action_sizes):
        assert new_strats[i].shape == (a_size,)
        assert torch.all(new_strats[i] >= 0.0)
        assert abs(new_strats[i].sum().item() - 1.0) < 1e-5


@pytest.mark.parametrize("dynamic_cls", [OptimisticMWU, MultiplicativeWeightsUpdate])
def test_numerical_stability_large_utilities(dynamic_cls):
    """Test Log-Sum-Exp stability with extremely large utility values (+5000.0)."""
    dyn = dynamic_cls(action_sizes=[2, 2], eta=1.0)
    # Huge utility difference that would overflow exp(5000) without log-sum-exp shift
    u_huge = [torch.tensor([5000.0, -5000.0]), torch.tensor([-5000.0, 5000.0])]

    new_strats = dyn.step(u_huge)

    for s in new_strats:
        assert not torch.isnan(s).any()
        assert not torch.isinf(s).any()
        assert abs(s.sum().item() - 1.0) < 1e-5


@pytest.mark.parametrize("dynamic_cls", [OptimisticMWU, MultiplicativeWeightsUpdate])
def test_numerical_stability_near_zero_probabilities(dynamic_cls):
    """Test log-clamping stability with near-zero strategy probabilities (1e-35)."""
    dyn = dynamic_cls(action_sizes=[2, 2], eta=0.01)
    near_zero_strats = [
        torch.tensor([1e-35, 1.0 - 1e-35]),
        torch.tensor([0.5, 0.5]),
    ]
    dyn.reset(initial_strategies=near_zero_strats)

    u_vecs = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    new_strats = dyn.step(u_vecs)

    for s in new_strats:
        assert not torch.isnan(s).any()
        assert not torch.isinf(s).any()
        assert abs(s.sum().item() - 1.0) < 1e-5
