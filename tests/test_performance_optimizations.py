"""Unit tests for multi-step unrolling, GPU zero-sync pipeline, and batch game ensemble execution."""

import pytest
import torch

from src.config.schemas import ExperimentConfig, GameConfig, DynamicConfig, ExecutionConfig
from src.engine.runner import ExperimentRunner
from src.games.generators import create_matching_pennies, create_random_game
from src.games.nplayer_game import BatchNPlayerGame, NPlayerGame


def test_steps_per_call_math_equivalence():
    """Verify that steps_per_call unrolling produces exact identical strategy results as 1-step mode."""
    steps = 100
    # 1-step runner
    config1 = ExperimentConfig(
        name="test_1step",
        game=GameConfig(generator="matching_pennies"),
        dynamic=DynamicConfig(algorithm="omwu", eta=0.05, initial_strategy_type="custom", custom_initial_strategies=[[0.8, 0.2], [0.3, 0.7]]),
        execution=ExecutionConfig(total_steps=steps, seed=42, steps_per_call=1),
    )
    runner1 = ExperimentRunner(config=config1)
    runner1.run()
    strats1 = [s.clone() for s in runner1.dynamic.strategies]

    # 10-step unrolled runner
    config10 = ExperimentConfig(
        name="test_10step",
        game=GameConfig(generator="matching_pennies"),
        dynamic=DynamicConfig(algorithm="omwu", eta=0.05, initial_strategy_type="custom", custom_initial_strategies=[[0.8, 0.2], [0.3, 0.7]]),
        execution=ExecutionConfig(total_steps=steps, seed=42, steps_per_call=10),
    )
    runner10 = ExperimentRunner(config=config10)
    runner10.run()
    strats10 = [s.clone() for s in runner10.dynamic.strategies]

    for s1, s10 in zip(strats1, strats10):
        assert torch.allclose(s1, s10, atol=1e-5), f"Mismatch between 1-step and 10-step unrolling: {s1} vs {s10}"


def test_batch_nplayer_game_ensemble_math_equivalence():
    """Verify that BatchNPlayerGame evaluates parallel game instances identically to NPlayerGame."""
    B = 5
    game1 = create_random_game(num_players=3, action_sizes=[2, 3, 2], seed=42)
    payoffs = game1.get_payoff_tensors()

    # Stack payoffs into batch of size B
    batch_payoffs = [p.unsqueeze(0).repeat(B, *([1] * p.ndim)) for p in payoffs]
    batch_game = BatchNPlayerGame(batch_payoffs=batch_payoffs)

    single_strats = [
        torch.tensor([0.6, 0.4]),
        torch.tensor([0.2, 0.5, 0.3]),
        torch.tensor([0.7, 0.3]),
    ]
    batch_strats = [s.unsqueeze(0).repeat(B, 1) for s in single_strats]

    single_u = game1.get_utility_vectors(single_strats)
    batch_u = batch_game.get_utility_vectors(batch_strats)

    for i in range(len(payoffs)):
        for b in range(B):
            assert torch.allclose(single_u[i], batch_u[i][b], atol=1e-5)
