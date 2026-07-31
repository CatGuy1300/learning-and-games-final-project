"""Unit tests for matrix and N-player game environments."""

import pytest
import torch

from src.games.generators import (
    create_matching_pennies,
    create_prisoners_dilemma,
    create_random_game,
)
from src.games.matrix_game import MatrixGame
from src.games.nplayer_game import NPlayerGame


def test_matching_pennies_payoffs():
    game = create_matching_pennies(utility_range=(-1.0, 1.0))
    strats = [
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.5, 0.5]),
    ]
    u_vecs = game.get_utility_vectors(strats)
    payoffs = game.get_expected_payoffs(strats)
    br = game.best_response_payoffs(strats)

    assert torch.allclose(u_vecs[0], torch.tensor([0.0, 0.0]))
    assert abs(payoffs[0]) < 1e-6
    assert abs(payoffs[1]) < 1e-6
    assert abs(br[0] - 0.0) < 1e-6


def test_nplayer_random_game_shape_and_bounds():
    game = create_random_game(
        num_players=3,
        action_sizes=[2, 3, 4],
        utility_range=(-2.0, 2.0),
        seed=42,
    )
    assert game.num_players == 3
    assert game.action_sizes == [2, 3, 4]

    strats = [
        torch.tensor([0.5, 0.5]),
        torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        torch.tensor([0.25, 0.25, 0.25, 0.25]),
    ]
    u_vecs = game.get_utility_vectors(strats)

    assert u_vecs[0].shape == (2,)
    assert u_vecs[1].shape == (3,)
    assert u_vecs[2].shape == (4,)
    assert all(u.min().item() >= -2.0 - 1e-5 for u in u_vecs)
    assert all(u.max().item() <= 2.0 + 1e-5 for u in u_vecs)
