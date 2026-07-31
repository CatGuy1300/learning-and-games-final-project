"""Unit tests for configuration schemas, utility bounds, and strategy validation."""

import pytest
import torch

from src.config.schemas import GameConfig
from src.config.validation import (
    clamp_utility_matrices,
    validate_game_config,
    validate_payoff_tensors,
    validate_strategies,
)


def test_validate_game_config_invalid():
    # Invalid num players
    with pytest.raises(ValueError):
        validate_game_config(GameConfig(num_players=1, action_sizes=[2]))

    # Mismatched action sizes length
    with pytest.raises(ValueError):
        validate_game_config(GameConfig(num_players=3, action_sizes=[2, 2]))

    # Invalid action count
    with pytest.raises(ValueError):
        validate_game_config(GameConfig(num_players=2, action_sizes=[1, 2]))


def test_validate_payoff_bounds_and_clamping():
    payoffs = [
        torch.tensor([[1.5, -0.5], [0.0, 0.5]]),  # 1.5 breaches (-1, 1) range
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    ]
    with pytest.raises(ValueError):
        validate_payoff_tensors(payoffs, 2, [2, 2], (-1.0, 1.0))

    clamped = clamp_utility_matrices(payoffs, (-1.0, 1.0))
    assert clamped[0].max().item() == 1.0


def test_validate_strategies():
    valid = [torch.tensor([0.4, 0.6]), torch.tensor([0.2, 0.8])]
    validate_strategies(valid, [2, 2])

    invalid_sum = [torch.tensor([0.5, 0.6]), torch.tensor([0.2, 0.8])]
    with pytest.raises(ValueError):
        validate_strategies(invalid_sum, [2, 2])

    invalid_neg = [torch.tensor([-0.1, 1.1]), torch.tensor([0.2, 0.8])]
    with pytest.raises(ValueError):
        validate_strategies(invalid_neg, [2, 2])
