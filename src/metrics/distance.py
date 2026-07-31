"""Strategy trajectory distance and convergence metrics."""

import torch


def compute_strategy_movement(
    prev_strategies: list[torch.Tensor], curr_strategies: list[torch.Tensor]
) -> list[float]:
    """Compute Euclidean norm L2 distance ||x_t - x_{t-1}||_2 per player.

    Parameters
    ----------
    prev_strategies : List[torch.Tensor]
        Previous step strategies x_{t-1}.
    curr_strategies : List[torch.Tensor]
        Current step strategies x_t.

    Returns
    -------
    List[float]
        L2 strategy delta per player.
    """
    movements = []
    for prev, curr in zip(prev_strategies, curr_strategies):
        dist = torch.norm(curr - prev, p=2).item()
        movements.append(dist)
    return movements


def compute_strategy_entropy(strategies: list[torch.Tensor]) -> list[float]:
    r"""Compute Shannon entropy H(x_i) = - \sum_a x_{i,a} \log(x_{i,a}) per player.

    Parameters
    ----------
    strategies : List[torch.Tensor]
        Strategy distributions.

    Returns
    -------
    List[float]
        Entropy values per player.
    """
    entropies = []
    eps = 1e-30
    for strat in strategies:
        p = torch.clamp(strat, min=eps)
        ent = -torch.sum(p * torch.log(p)).item()
        entropies.append(ent)
    return entropies
