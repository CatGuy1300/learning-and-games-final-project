"""Rigorous mathematical correctness and exactness test suite for learning dynamics algorithms.

Verifies:
1. Hand-calculated analytical step exactness for MWU, GDA, OMWU, and ExtraGradient.
2. Step 1 mathematical equivalences: OMWU(step 1) == MWU(step 1) and ExtraGradient(step 1) == GDA(step 1).
3. Zero learning rate invariance (eta = 0.0 => x_{t+1} == x_t).
4. Dual interface exactness: step() and step_2d() produce 100% identical outputs.
"""

import math
import pytest
import torch

from src.dynamics.omwu import OptimisticMWU
from src.dynamics.extra_gradient import ExtraGradient
from src.dynamics.mwu import MultiplicativeWeightsUpdate
from src.dynamics.gda import GradientDescentAscent
from src.games.matrix_game import MatrixGame


def test_hand_calculated_mwu_and_gda_exactness():
    """Verify MWU and GDA updates against hand-calculated analytical solutions."""
    # Matching Pennies game: A = [[1, -1], [-1, 1]]
    payoff_a = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    payoff_b = -payoff_a
    game = MatrixGame(payoff_a=payoff_a, payoff_b=payoff_b)

    # Initial strategies: x0 = [0.6, 0.4], y0 = [0.3, 0.7]
    x0 = torch.tensor([0.6, 0.4])
    y0 = torch.tensor([0.3, 0.7])
    eta = 0.1

    # Hand math for u1 = A y0 = [1*0.3 - 1*0.7, -1*0.3 + 1*0.7] = [-0.4, 0.4]
    # Hand math for u2 = B^T x0 = (-A^T) x0 = [-1*0.6 + 1*0.4, 1*0.6 - 1*0.4] = [-0.2, 0.2]
    #
    # MWU Step 1:
    # Player 0: x1_raw = [0.6 * exp(0.1 * -0.4), 0.4 * exp(0.1 * 0.4)]
    # Player 1: y1_raw = [0.3 * exp(0.1 * -0.2), 0.7 * exp(0.1 * 0.2)]
    expected_x1_unnorm = [0.6 * math.exp(-0.04), 0.4 * math.exp(0.04)]
    expected_x1 = torch.tensor([
        expected_x1_unnorm[0] / sum(expected_x1_unnorm),
        expected_x1_unnorm[1] / sum(expected_x1_unnorm),
    ])

    mwu = MultiplicativeWeightsUpdate(action_sizes=[2, 2], eta=eta)
    mwu.reset(initial_strategies=[x0, y0])
    u_vecs = game.get_utility_vectors(mwu.strategies)
    mwu_next = mwu.step(u_vecs)

    assert torch.allclose(mwu_next[0], expected_x1, atol=1e-6)

    # GDA Step 1:
    # Player 0 unconstrained: x0 + eta * u1 = [0.6 - 0.04, 0.4 + 0.04] = [0.56, 0.44] (already on simplex!)
    # Player 1 unconstrained: y0 + eta * u2 = [0.3 - 0.02, 0.7 + 0.02] = [0.28, 0.72] (already on simplex!)
    gda = GradientDescentAscent(action_sizes=[2, 2], eta=eta)
    gda.reset(initial_strategies=[x0, y0])
    gda_next = gda.step(u_vecs)

    assert torch.allclose(gda_next[0], torch.tensor([0.56, 0.44]), atol=1e-6)
    assert torch.allclose(gda_next[1], torch.tensor([0.28, 0.72]), atol=1e-6)


def test_step_1_equivalences():
    """Verify OMWU(step 1) == MWU(step 1) and ExtraGradient(step 1) == GDA(step 1)."""
    actions = [3, 4]
    eta = 0.05
    x0 = torch.tensor([0.2, 0.3, 0.5])
    y0 = torch.tensor([0.1, 0.4, 0.2, 0.3])
    u0_x = torch.tensor([1.2, -0.5, 0.3])
    u0_y = torch.tensor([-0.8, 0.1, 1.5, -0.2])

    # OMWU vs MWU step 1
    omwu = OptimisticMWU(action_sizes=actions, eta=eta)
    mwu = MultiplicativeWeightsUpdate(action_sizes=actions, eta=eta)
    omwu.reset(initial_strategies=[x0, y0])
    mwu.reset(initial_strategies=[x0, y0])

    omwu_step1 = omwu.step([u0_x, u0_y])
    mwu_step1 = mwu.step([u0_x, u0_y])

    assert torch.allclose(omwu_step1[0], mwu_step1[0], atol=1e-7)
    assert torch.allclose(omwu_step1[1], mwu_step1[1], atol=1e-7)

    # ExtraGradient vs GDA step 1
    eg = ExtraGradient(action_sizes=actions, eta=eta)
    gda = GradientDescentAscent(action_sizes=actions, eta=eta)
    eg.reset(initial_strategies=[x0, y0])
    gda.reset(initial_strategies=[x0, y0])

    eg_step1 = eg.step([u0_x, u0_y])
    gda_step1 = gda.step([u0_x, u0_y])

    assert torch.allclose(eg_step1[0], gda_step1[0], atol=1e-7)
    assert torch.allclose(eg_step1[1], gda_step1[1], atol=1e-7)


def test_zero_learning_rate_invariance():
    """Verify eta = 0.0 leaves strategies 100% unchanged across all algorithms."""
    actions = [3, 3]
    x0 = torch.tensor([0.1, 0.4, 0.5])
    y0 = torch.tensor([0.3, 0.3, 0.4])
    u = [torch.tensor([10.0, -5.0, 2.0]), torch.tensor([-3.0, 4.0, -1.0])]

    for cls in [OptimisticMWU, ExtraGradient, MultiplicativeWeightsUpdate, GradientDescentAscent]:
        dyn = cls(action_sizes=actions, eta=0.0)
        dyn.reset(initial_strategies=[x0, y0])
        next_strats = dyn.step(u)
        assert torch.allclose(next_strats[0], x0, atol=1e-7)
        assert torch.allclose(next_strats[1], y0, atol=1e-7)


def test_dual_interface_exactness():
    """Verify step() and step_2d() produce 100% identical outputs."""
    actions = [2, 3]
    x0 = torch.tensor([0.7, 0.3])
    y0 = torch.tensor([0.2, 0.5, 0.3])
    u_list = [torch.tensor([1.5, -0.5]), torch.tensor([0.1, -1.2, 0.8])]
    u_2d = torch.nn.utils.rnn.pad_sequence(u_list, batch_first=True, padding_value=0.0)

    for cls in [OptimisticMWU, ExtraGradient, MultiplicativeWeightsUpdate, GradientDescentAscent]:
        dyn1 = cls(action_sizes=actions, eta=0.02)
        dyn2 = cls(action_sizes=actions, eta=0.02)

        dyn1.reset(initial_strategies=[x0, y0])
        dyn2.reset(initial_strategies=[x0, y0])

        res1 = dyn1.step(u_list)
        res2_2d = dyn2.step_2d(u_2d)

        for i in range(len(actions)):
            assert torch.allclose(res1[i], res2_2d[i, : actions[i]], atol=1e-7)
