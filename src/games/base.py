"""Abstract base class for finite general-sum games."""

from abc import ABC, abstractmethod

import torch


class BaseGame(ABC):
    """Abstract interface for N-player general-sum finite games."""

    def __init__(
        self,
        num_players: int,
        action_sizes: list[int],
        utility_range: tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize base game properties."""
        self.num_players = num_players
        self.action_sizes = action_sizes
        self.utility_range = utility_range
        self.device = device

    @abstractmethod
    def get_payoff_tensors(self) -> list[torch.Tensor]:
        """Return list of payoff tensors [U^(1), ..., U^(N)] per player."""

    @abstractmethod
    def get_utility_vectors(self, strategies: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute expected utility vector u_i(x^{-i}) for each player i given strategies x."""

    def get_stacked_utility_vectors(self, stacked_strategies: torch.Tensor) -> torch.Tensor:
        """Compute 2D utility tensor (N, max_action_size) directly from 2D strategy tensor (N, max_action_size).

        Parameters
        ----------
        stacked_strategies : torch.Tensor
            2D strategy tensor of shape (N, max_action_size).

        Returns
        -------
        torch.Tensor
            2D utility tensor of shape (N, max_action_size).
        """
        strats = [stacked_strategies[i, : self.action_sizes[i]] for i in range(self.num_players)]
        u_vecs = self.get_utility_vectors(strats)
        return torch.nn.utils.rnn.pad_sequence(u_vecs, batch_first=True, padding_value=0.0)

    @abstractmethod
    def get_expected_payoffs(self, strategies: list[torch.Tensor]) -> list[float]:
        """Compute scalar expected payoff E_{a ~ x}[U^{(i)}(a)] for each player i."""

    @abstractmethod
    def best_response_payoffs(self, strategies: list[torch.Tensor]) -> list[float]:
        """Compute max_a_i u_i(a_i, x^{-i}) best response payoff for each player i."""

    @abstractmethod
    def to(self, device: torch.device) -> "BaseGame":
        """Move game payoff tensors to target device."""
