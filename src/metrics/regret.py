"""Vectorized regret and payoff performance metrics."""

from typing import Dict, List, Tuple
import torch


def compute_step_metrics(
    expected_payoffs: List[float],
    best_response_payoffs: List[float],
) -> Tuple[List[float], List[float]]:
    """Compute instantaneous regret and best-response gap for single step t.

    Instantaneous regret for player i:
    r_i(t) = max_{a_i} u_i(a_i, x^{-i}_t) - E_{a~x_t}[U^{(i)}(a)]

    Returns
    -------
    Tuple[List[float], List[float]]
        (instant_regrets, best_response_payoffs)
    """
    instant_regrets = [
        br - exp for br, exp in zip(best_response_payoffs, expected_payoffs)
    ]
    return instant_regrets, best_response_payoffs


def compute_cumulative_regret(
    cumulative_utility_vectors: List[torch.Tensor],
    cumulative_actual_payoffs: List[float],
) -> List[float]:
    r"""Compute cumulative regret R_i(T) for each player up to step T.

    R_i(T) = max_{a_i} \sum_{t=1}^T u_i(a_i, x^{-i}_t) - \sum_{t=1}^T u_i(x_t)

    Parameters
    ----------
    cumulative_utility_vectors : List[torch.Tensor]
        Sum of utility vectors \sum_t u_i(x_t^{-i}) of shape (A_i,).
    cumulative_actual_payoffs : List[float]
        Sum of actual expected payoffs \sum_t E_{x_t}[U^{(i)}].

    Returns
    -------
    List[float]
        Cumulative regret per player R_i(T).
    """
    num_players = len(cumulative_utility_vectors)
    cum_regrets = []
    for i in range(num_players):
        max_fixed_action_payoff = cumulative_utility_vectors[i].max().item()
        actual_payoff = cumulative_actual_payoffs[i]
        regret_i = max_fixed_action_payoff - actual_payoff
        cum_regrets.append(regret_i)
    return cum_regrets


def compute_average_regret(
    cumulative_regrets: List[float], total_steps: int
) -> List[float]:
    """Compute average regret R_i(T) / T for each player.

    Parameters
    ----------
    cumulative_regrets : List[float]
        Cumulative regrets R_i(T).
    total_steps : int
        Current step count T.

    Returns
    -------
    List[float]
        Average regrets per player.
    """
    if total_steps <= 0:
        return [0.0] * len(cumulative_regrets)
    return [r / float(total_steps) for r in cumulative_regrets]
