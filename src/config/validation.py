"""Validation utilities and rule checking for games, configurations, and strategies."""

import torch

from src.config.schemas import CMAESConfig, ExperimentConfig, GameConfig


def validate_game_config(config: GameConfig) -> None:
    """Validate game parameters, action sizes, and utility ranges.

    Parameters
    ----------
    config : GameConfig
        Game configuration object.

    Raises
    ------
    ValueError
        If parameters breach shape, dimension, or utility range constraints.
    """
    if config.num_players < 2:
        raise ValueError(f"Number of players must be >= 2, got {config.num_players}")

    if len(config.action_sizes) != config.num_players:
        raise ValueError(
            f"action_sizes length ({len(config.action_sizes)}) must equal num_players ({config.num_players})"
        )

    for i, a_size in enumerate(config.action_sizes):
        if a_size < 2:
            raise ValueError(f"Player {i} action size must be >= 2, got {a_size}")

    u_min, u_max = config.utility_range
    if u_min >= u_max:
        raise ValueError(
            f"utility_range u_min ({u_min}) must be strictly less than u_max ({u_max})"
        )


def validate_payoff_tensors(
    payoffs: list[torch.Tensor],
    num_players: int,
    action_sizes: list[int],
    utility_range: tuple[float, float],
) -> None:
    """Validate that payoff tensors conform to expected shapes and range bounds.

    Parameters
    ----------
    payoffs : List[torch.Tensor]
        List of payoff tensors per player.
    num_players : int
        Expected number of players N.
    action_sizes : List[int]
        Expected action tuple (A_1, ..., A_N).
    utility_range : Tuple[float, float]
        Allowed payoff range (u_min, u_max).
    """
    if len(payoffs) != num_players:
        raise ValueError(f"Expected {num_players} payoff tensors, got {len(payoffs)}")

    expected_shape = tuple(action_sizes)
    u_min, u_max = utility_range

    for i, p_tensor in enumerate(payoffs):
        if tuple(p_tensor.shape) != expected_shape and tuple(p_tensor.shape)[1:] != expected_shape:
            raise ValueError(
                f"Player {i} payoff tensor shape {tuple(p_tensor.shape)} does not match expected {expected_shape} or (B,) + {expected_shape}"
            )
        # Check numerical bounds (allowing small floating point epsilon)
        min_val = p_tensor.min().item()
        max_val = p_tensor.max().item()
        eps = 1e-5
        if min_val < u_min - eps or max_val > u_max + eps:
            raise ValueError(
                f"Player {i} payoff values [{min_val:.4f}, {max_val:.4f}] breach utility_range [{u_min}, {u_max}]"
            )


def validate_strategies(strategies: list[torch.Tensor], action_sizes: list[int]) -> None:
    """Validate that strategy vectors form valid probability distributions on the simplex.

    Parameters
    ----------
    strategies : List[torch.Tensor]
        List of strategy vectors per player.
    action_sizes : List[int]
        Expected action sizes per player.
    """
    for i, (strat, a_size) in enumerate(zip(strategies, action_sizes)):
        if strat.shape[-1] != a_size or (strat.dim() != 1 and strat.dim() != 2):
            raise ValueError(
                f"Player {i} strategy shape {tuple(strat.shape)} does not match action size {a_size} or (B, {a_size})"
            )
        if torch.any(strat < -1e-6):
            raise ValueError(f"Player {i} strategy has negative probabilities: {strat}")
        prob_sum = strat.sum(dim=-1)
        if not torch.all(torch.abs(prob_sum - 1.0) < 1e-4):
            raise ValueError(f"Player {i} strategy does not sum to 1.0 (got {prob_sum}): {strat}")


def clamp_utility_matrices(
    payoffs: list[torch.Tensor], utility_range: tuple[float, float]
) -> list[torch.Tensor]:
    """Project / clamp payoff tensors into the allowed utility range [u_min, u_max].

    Parameters
    ----------
    payoffs : List[torch.Tensor]
        List of payoff tensors.
    utility_range : Tuple[float, float]
        Bounds (u_min, u_max).

    Returns
    -------
    List[torch.Tensor]
        Clamped payoff tensors.
    """
    u_min, u_max = utility_range
    return [torch.clamp(p, min=u_min, max=u_max) for p in payoffs]


def validate_cmaes_config(config: CMAESConfig) -> None:
    if config.objective_type not in ["delta_reg", "raw"]:
        raise ValueError(f"objective_type must be 'delta_reg' or 'raw', got {config.objective_type}")
    if config.T1_ratio <= 0.0 or config.T1_ratio >= 1.0:
        raise ValueError(f"T1_ratio must be in (0, 1), got {config.T1_ratio}")
    if config.sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {config.sigma}")


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate master experiment configuration."""
    validate_game_config(config.game)
    if config.execution.total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {config.execution.total_steps}")
    if config.dynamic.eta is not None and config.dynamic.eta <= 0:
        raise ValueError(f"eta must be > 0, got {config.dynamic.eta}")
    validate_cmaes_config(config.cmaes)
