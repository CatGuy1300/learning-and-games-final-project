"""Unit tests for regret calculation routines."""

import pytest
import torch

from src.metrics.regret import (
    compute_average_regret,
    compute_cumulative_regret,
    compute_step_metrics,
)


def test_regret_math():
    exp_payoffs = [0.2, -0.4]
    br_payoffs = [0.8, 0.0]

    instant_regrets, _ = compute_step_metrics(exp_payoffs, br_payoffs)
    assert abs(instant_regrets[0] - 0.6) < 1e-6
    assert abs(instant_regrets[1] - 0.4) < 1e-6

    cum_u = [torch.tensor([10.0, 5.0]), torch.tensor([2.0, 8.0])]
    cum_actual = [4.0, 3.0]

    cum_r = compute_cumulative_regret(cum_u, cum_actual)
    assert abs(cum_r[0] - (10.0 - 4.0)) < 1e-6  # 6.0
    assert abs(cum_r[1] - (8.0 - 3.0)) < 1e-6   # 5.0

    avg_r = compute_average_regret(cum_r, total_steps=10)
    assert abs(avg_r[0] - 0.6) < 1e-6
    assert abs(avg_r[1] - 0.5) < 1e-6
