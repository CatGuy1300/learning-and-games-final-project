"""Preset game generators and random matrix/tensor game initializers."""

import torch

from src.games.matrix_game import MatrixGame
from src.games.nplayer_game import NPlayerGame


def scale_tensor_to_range(
    tensor: torch.Tensor, utility_range: tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """Linearly scale tensor values to lie inside [u_min, u_max].

    Parameters
    ----------
    tensor : torch.Tensor
        Input raw tensor.
    utility_range : Tuple[float, float]
        Target bounds (u_min, u_max).

    Returns
    -------
    torch.Tensor
        Scaled tensor.
    """
    u_min, u_max = utility_range
    t_min = tensor.min()
    t_max = tensor.max()

    if torch.abs(t_max - t_min) < 1e-8:
        return torch.full_like(tensor, (u_min + u_max) / 2.0)

    # Scale [t_min, t_max] -> [0, 1] -> [u_min, u_max]
    normed = (tensor - t_min) / (t_max - t_min)
    return u_min + normed * (u_max - u_min)


def create_matching_pennies(
    utility_range: tuple[float, float] = (-1.0, 1.0), device: torch.device = torch.device("cpu")
) -> MatrixGame:
    """Create zero-sum Matching Pennies matrix game.

    A = [[ 1, -1],
         [-1,  1]]
    B = -A
    """
    u_min, u_max = utility_range
    val = min(abs(u_min), abs(u_max))
    payoff_a = torch.tensor([[val, -val], [-val, val]], dtype=torch.get_default_dtype())
    payoff_b = -payoff_a
    return MatrixGame(payoff_a, payoff_b, utility_range=utility_range, device=device)


def create_prisoners_dilemma(
    utility_range: tuple[float, float] = (-1.0, 1.0), device: torch.device = torch.device("cpu")
) -> MatrixGame:
    """Create classic Prisoner's Dilemma matrix game.

    Cooperate=0, Defect=1
    P1 Payoff A: [[-1, -3], [ 0, -2]]
    P2 Payoff B: [[-1,  0], [-3, -2]]
    """
    raw_a = torch.tensor([[-1.0, -3.0], [0.0, -2.0]], dtype=torch.get_default_dtype())
    raw_b = torch.tensor([[-1.0, 0.0], [-3.0, -2.0]], dtype=torch.get_default_dtype())

    payoff_a = scale_tensor_to_range(raw_a, utility_range)
    payoff_b = scale_tensor_to_range(raw_b, utility_range)
    return MatrixGame(payoff_a, payoff_b, utility_range=utility_range, device=device)


def create_rock_paper_scissors(
    utility_range: tuple[float, float] = (-1.0, 1.0), device: torch.device = torch.device("cpu")
) -> MatrixGame:
    """Create zero-sum Rock-Paper-Scissors matrix game."""
    raw_a = torch.tensor(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=torch.get_default_dtype()
    )
    raw_b = -raw_a

    payoff_a = scale_tensor_to_range(raw_a, utility_range)
    payoff_b = scale_tensor_to_range(raw_b, utility_range)
    return MatrixGame(payoff_a, payoff_b, utility_range=utility_range, device=device)


def create_shapley_game(
    utility_range: tuple[float, float] = (-1.0, 1.0), device: torch.device = torch.device("cpu")
) -> MatrixGame:
    """Create Shapley non-zero sum game with cyclic best-response trajectory."""
    raw_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.get_default_dtype())
    raw_b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.get_default_dtype())

    payoff_a = scale_tensor_to_range(raw_a, utility_range)
    payoff_b = scale_tensor_to_range(raw_b, utility_range)
    return MatrixGame(payoff_a, payoff_b, utility_range=utility_range, device=device)


def create_random_game(
    num_players: int = 2,
    action_sizes: list[int] | None = None,
    utility_range: tuple[float, float] = (-1.0, 1.0),
    seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> NPlayerGame:
    """Generate random N-player general-sum game with payoffs scaled to utility_range.

    Parameters
    ----------
    num_players : int
        Number of players N.
    action_sizes : Optional[List[int]]
        Action sizes (A_1, ..., A_N). Defaults to [2] * num_players.
    utility_range : Tuple[float, float]
        Target utility bounds (u_min, u_max).
    seed : Optional[int]
        RNG seed.
    device : torch.device
        PyTorch device.

    Returns
    -------
    NPlayerGame
        Random NPlayerGame instance.
    """
    if action_sizes is None:
        action_sizes = [2] * num_players

    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
    else:
        gen = None

    shape = tuple(action_sizes)
    u_min, u_max = utility_range

    payoffs: list[torch.Tensor] = []
    for _ in range(num_players):
        raw = torch.rand(shape, generator=gen, dtype=torch.get_default_dtype())
        scaled = u_min + raw * (u_max - u_min)
        payoffs.append(scaled)

    return NPlayerGame(payoffs, utility_range=utility_range, device=device)
