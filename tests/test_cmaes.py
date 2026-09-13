import pytest
import torch
import numpy as np

from src.config.schemas import ExperimentConfig, GameConfig, DynamicConfig, ExecutionConfig, CMAESConfig
from src.engine.optimizer import CMAESGameOptimizer

def test_cmaes_optimizer_smoke():
    # Setup a small configuration for 2x2 games
    B = 10
    config = ExperimentConfig(
        name="test_cmaes",
        game=GameConfig(
            generator="custom",
            utility_range=(-1.0, 1.0),
            payoffs=[
                [[0.0, 0.0], [0.0, 0.0]], # Player 1 base
                [[0.0, 0.0], [0.0, 0.0]], # Player 2 base
            ]
        ),
        dynamic=DynamicConfig(algorithm="omwu", eta=0.1),
        execution=ExecutionConfig(total_steps=10, batch_size=B),
        cmaes=CMAESConfig(sigma=0.5, seed=42, gamma_volatility=2.0)
    )

    optimizer = CMAESGameOptimizer(base_config=config)
    
    # Check dimensions
    assert optimizer.num_players == 2
    assert optimizer.dim == 8 # 2x2 = 4 per player
    
    # Check that batch size matches population size
    assert optimizer.base_config.execution.batch_size == optimizer.population_size
    
    best_payoffs, best_factors = optimizer.optimize(generations=2)
    
    assert len(best_payoffs) == 2
    assert best_payoffs[0].shape == (2, 2)
    assert best_payoffs[1].shape == (2, 2)
    assert isinstance(best_factors, dict)
    assert "regret" in best_factors

def test_cmaes_optimizer_delta_reg():
    B = 10
    config = ExperimentConfig(
        name="test_cmaes_delta",
        game=GameConfig(
            generator="custom",
            utility_range=(-1.0, 1.0),
            payoffs=[
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        dynamic=DynamicConfig(algorithm="omwu", eta=0.1),
        execution=ExecutionConfig(total_steps=10, batch_size=B),
        cmaes=CMAESConfig(
            sigma=0.5, 
            seed=42,
            objective_type="delta_reg",
            T1_ratio=0.5,
            lambda_reg=0.1,
            gamma_volatility=2.0
        )
    )

    optimizer = CMAESGameOptimizer(base_config=config)
    
    best_payoffs, best_factors = optimizer.optimize(generations=2)
    assert len(best_payoffs) == 2
    assert isinstance(best_factors, dict)
    assert "delta" in best_factors

def test_cmaes_optimizer_chunked_envelope_trend():
    config = ExperimentConfig(
        name="test_cmaes_chunked",
        game=GameConfig(
            generator="custom",
            utility_range=(-1.0, 1.0),
            payoffs=[
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        dynamic=DynamicConfig(algorithm="omwu", eta=0.1),
        execution=ExecutionConfig(total_steps=100, batch_size=4),
        cmaes=CMAESConfig(
            sigma=0.5, 
            seed=42,
            objective_type="chunked_envelope_trend",
            num_chunks=5,
            T1_ratio=0.5,
            lambda_reg=0.1,
            gamma_volatility=2.0
        )
    )

    optimizer = CMAESGameOptimizer(base_config=config)
    
    best_payoffs, best_factors = optimizer.optimize(generations=2)
    assert len(best_payoffs) == 2
    assert isinstance(best_factors, dict)
    assert "delta" in best_factors
    assert "peak_weight" in best_factors
