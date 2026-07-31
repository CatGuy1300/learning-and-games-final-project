import pytest
import torch

from src.games.matrix_game import MatrixGame
from src.dynamics.omwu import OptimisticMWU
from src.dynamics.mirror_prox import MirrorProx

def test_matrix_game_batched():
    # B=50 games, 2 players, actions 3x3
    B = 50
    m, n = 3, 3
    payoff_a = torch.rand((B, m, n))
    payoff_b = torch.rand((B, m, n))

    game = MatrixGame(payoff_a=payoff_a, payoff_b=payoff_b, utility_range=(0.0, 1.0))
    assert game.batch_size == B
    
    # Init batched OMWU
    omwu = OptimisticMWU(action_sizes=[m, n], batch_size=B)
    
    # Test step_2d batched
    for _ in range(5):
        u_2d = game.get_stacked_utility_vectors(omwu.stacked_strategies)
        # u_2d should be shape (B, 2, 3)
        assert u_2d.shape == (B, 2, max(m, n))
        omwu.step_2d(u_2d)

    # Check that strategies are correctly bounded and sum to 1 along action dim
    assert omwu.stacked_strategies.shape == (B, 2, 3)
    sums = omwu.stacked_strategies.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))

def test_mirror_prox_batched():
    # B=50 games, 2 players, actions 3x3
    B = 50
    m, n = 3, 3
    payoff_a = torch.rand((B, m, n))
    payoff_b = torch.rand((B, m, n))

    game = MatrixGame(payoff_a=payoff_a, payoff_b=payoff_b, utility_range=(0.0, 1.0))
    mp = MirrorProx(action_sizes=[m, n], batch_size=B)
    
    for _ in range(5):
        # We need to manually simulate step_unrolled_block for mp
        stacked_u = game.get_stacked_utility_vectors(mp.stacked_strategies)
        mp.step_2d(stacked_u)
        
    sums = mp.stacked_strategies.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))

from src.games.nplayer_game import NPlayerGame

def test_nplayer_game_batched():
    # B=10 games, 3 players, actions 2x3x4
    B = 10
    action_sizes = [2, 3, 4]
    
    payoffs = [
        torch.rand((B, *action_sizes)),
        torch.rand((B, *action_sizes)),
        torch.rand((B, *action_sizes)),
    ]
    
    game = NPlayerGame(payoffs=payoffs, utility_range=(0.0, 1.0))
    omwu = OptimisticMWU(action_sizes=action_sizes, batch_size=B)
    
    for _ in range(3):
        u_2d = game.get_stacked_utility_vectors(omwu.stacked_strategies)
        assert u_2d.shape == (B, 3, 4)
        omwu.step_2d(u_2d)
        
    sums = omwu.stacked_strategies.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))

