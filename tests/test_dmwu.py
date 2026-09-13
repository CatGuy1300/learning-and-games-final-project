"""Tests for Dissipative Multiplicative Weights Update (DMWU)."""

import torch
import pytest

from src.dynamics.dmwu import DMWU
from src.config.schemas import ExperimentConfig
from src.engine.runner import ExperimentRunner

def test_dmwu_initialization():
    dynamic = DMWU(action_sizes=[2, 2], dmwu_gamma=0.2)
    assert dynamic.gamma == 0.2
    assert dynamic.stacked_w_hat.shape == dynamic.stacked_logits.shape
    assert dynamic.diff.shape == dynamic.stacked_logits.shape

def test_dmwu_step_shapes():
    dynamic = DMWU(action_sizes=[3, 4], dmwu_gamma=0.1)
    
    u_p1 = torch.rand(3)
    u_p2 = torch.rand(4)
    
    new_strats = dynamic.step([u_p1, u_p2])
    
    assert len(new_strats) == 2
    assert new_strats[0].shape == (3,)
    assert new_strats[1].shape == (4,)
    
    # Probabilities should sum to 1
    assert torch.allclose(new_strats[0].sum(), torch.tensor(1.0))
    assert torch.allclose(new_strats[1].sum(), torch.tensor(1.0))

def test_dmwu_runner_integration():
    config = ExperimentConfig()
    config.dynamic.algorithm = "dmwu"
    config.dynamic.dmwu_gamma = 0.05
    config.execution.total_steps = 10
    config.execution.batch_size = 2
    config.execution.quiet = True
    
    runner = ExperimentRunner(config)
    summary = runner.run()
    
    # Check that runner successfully ran and returned a summary
    assert summary["total_steps"] == 10
    assert "final_avg_regrets" in summary
