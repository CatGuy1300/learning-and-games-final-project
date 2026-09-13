import torch
import pytest

from src.config.schemas import ExperimentConfig, GameConfig, DynamicConfig, ExecutionConfig, CheckpointConfig
from src.engine.runner import ExperimentRunner
from src.dynamics.mirror_prox import MirrorProx
from src.games.nplayer_game import NPlayerGame
from src.dynamics.continuous import ContinuousGameDynamics

def test_runner_checkpoint_k_steps(tmp_path):
    """Test that runner correctly clamps target_step using next_ckpt to avoid infinite loop."""
    config = ExperimentConfig(
        name="test_runner_k_steps",
        game=GameConfig(generator="random", num_actions=[2, 2]),
        dynamic=DynamicConfig(algorithm="omwu"),
        execution=ExecutionConfig(total_steps=1000, steps_per_call=1000, quiet=True),
        checkpoint=CheckpointConfig(enabled=True, save_interval=200, checkpoint_dir=str(tmp_path))
    )
    runner = ExperimentRunner(config)
    # The runner's _execute_loop will natively break up the 1000 steps_per_call into 200 step chunks
    # because of the `next_ckpt` deduplication logic. If the deduplication loop is broken, it will infinite loop or crash.
    summary = runner.run(target_steps=1000)
    assert isinstance(summary, dict)

def test_mirror_prox_unrolled_block():
    """Test that mirror prox double-evaluation executes safely in-place."""
    game = NPlayerGame(payoffs=[torch.rand(2, 2) * 2 - 1, torch.rand(2, 2) * 2 - 1])
    mp = MirrorProx(action_sizes=[2, 2], eta=0.1)
    
    # Initialize state
    mp.stacked_logits.zero_()
    mp.stacked_strategies.fill_(0.5)
    
    cum_u = torch.zeros(1, 2, 2)
    cum_p = torch.zeros(1, 2)
    
    # The **kwargs shouldn't crash it, and in-place .add_() should work
    mp.step_unrolled_block(game, cum_u, cum_p, k_steps=2, dummy_kwarg=True)
    
    # Check that strategies haven't exploded
    assert not torch.isnan(mp.stacked_strategies).any()
    assert torch.allclose(mp.stacked_strategies.sum(dim=-1), torch.ones(2))

def test_nplayer_batched_strings():
    """Test NPlayerGame batched utility vector math (using ... strings)."""
    # 2 players, 2x2 game (clamp between -1 and 1)
    payoffs = [torch.rand(2, 2) * 2 - 1, torch.rand(2, 2) * 2 - 1]
    game = NPlayerGame(payoffs=payoffs)
    
    # Create batched strategies (B=5, N=2, A=2)
    B = 5
    strategies = [
        torch.softmax(torch.randn(B, 2), dim=-1),
        torch.softmax(torch.randn(B, 2), dim=-1)
    ]
    
    # get_utility_vectors explicitly uses the `...` logic when strategies are 2D
    u_vecs = game.get_utility_vectors(strategies)
    
    assert len(u_vecs) == 2
    assert u_vecs[0].shape == (B, 2)
    assert u_vecs[1].shape == (B, 2)
    assert not torch.isnan(u_vecs[0]).any()

def test_continuous_nplayer_penalty():
    """Test ContinuousGameDynamics dynamic N-player penalty loop."""
    # We can now natively test a 3-player continuous dynamics simulation!
    # To pass validation, utility must be bounded
    payoffs = [torch.rand(2, 2, 2) * 2 - 1, torch.rand(2, 2, 2) * 2 - 1, torch.rand(2, 2, 2) * 2 - 1]
    
    dyn = ContinuousGameDynamics(payoffs, logit_penalty_threshold=10.0)
    
    # State needs to be big enough to unpack:
    # 3x w (2 elements), 3x Z (2 elements), 3x P (1 element), 1x Barrier
    # Total = (3*2) + (3*2) + (3*1) + 1 = 16
    
    # We mock the forward pass since compute_w_dot isn't implemented in the base class
    # but the penalty loop executes BEFORE w_dot is packed! Wait, w_dot is evaluated first.
    # So we'll just mock compute_w_dot to return zeros.
    def mock_w_dot(strategies):
        return torch.zeros(6)
    
    dyn.compute_w_dot = mock_w_dot
    
    state = torch.randn(16, dtype=torch.float32)
    state_dot = dyn.forward(t=0, state=state)
    
    assert state_dot.shape == (16,)
    assert not torch.isnan(state_dot).any()
