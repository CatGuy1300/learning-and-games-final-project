"""N-player general-sum finite game environment with PyTorch tensor backends."""

from typing import List, Tuple
import torch

from src.config.validation import validate_payoff_tensors, validate_strategies
from src.games.base import BaseGame

# Subscript letters for einsum contracts (up to 26 action dimensions)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


class NPlayerGame(BaseGame):
    """General N-player finite game defined by payoff tensors U^(i) in R^(A_1 x ... x A_N)."""

    def __init__(
        self,
        payoffs: List[torch.Tensor],
        utility_range: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize NPlayerGame with payoff tensors.

        Parameters
        ----------
        payoffs : List[torch.Tensor]
            List of N payoff tensors, each of shape (A_1, ..., A_N).
        utility_range : Tuple[float, float]
            Expected range bounds (u_min, u_max).
        device : torch.device
            Target PyTorch device.
        """
        num_players = len(payoffs)
        if num_players < 2:
            raise ValueError(f"NPlayerGame requires >= 2 players, got {num_players}")
        if num_players > len(_LETTERS):
            raise ValueError(f"Max supported players is {len(_LETTERS)}, got {num_players}")

        action_sizes = list(payoffs[0].shape)
        validate_payoff_tensors(payoffs, num_players, action_sizes, utility_range)

        super().__init__(
            num_players=num_players,
            action_sizes=action_sizes,
            utility_range=utility_range,
            device=device,
        )

        self.payoffs = [p.to(device=device, dtype=torch.float32) for p in payoffs]

        # Pre-cache einsum string equations for fast 2D utility contractions
        indices = _LETTERS[: self.num_players]
        self._einsum_strs: List[str] = []
        for i in range(self.num_players):
            target_letter = indices[i]
            operand_indices = [
                f",{indices[j]}" for j in range(self.num_players) if j != i
            ]
            self._einsum_strs.append(f"{indices}{''.join(operand_indices)}->{target_letter}")

    def get_payoff_tensors(self) -> List[torch.Tensor]:
        """Return list of payoff tensors [U^(1), ..., U^(N)]."""
        return self.payoffs

    def get_utility_vectors(self, strategies: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute expected utility vector u_i(x^{-i}) for each player i.

        Parameters
        ----------
        strategies : List[torch.Tensor]
            Current strategy probability vectors x = (x^1, ..., x^N).

        Returns
        -------
        List[torch.Tensor]
            List of utility vectors u_i of shape (A_i,).
        """
        validate_strategies(strategies, self.action_sizes)

        utility_vectors: List[torch.Tensor] = []
        for i in range(self.num_players):
            other_strats = [
                strategies[j].to(device=self.device, dtype=torch.float32)
                for j in range(self.num_players)
                if j != i
            ]
            u_i = torch.einsum(self._einsum_strs[i], self.payoffs[i], *other_strats)
            utility_vectors.append(u_i)

        return utility_vectors

    def get_stacked_utility_vectors(self, stacked_strategies: torch.Tensor) -> torch.Tensor:
        """Compute 2D utility tensor (N, max_action_size) directly from 2D strategy tensor.

        Parameters
        ----------
        stacked_strategies : torch.Tensor
            2D strategy tensor of shape (N, max_action_size).

        Returns
        -------
        torch.Tensor
            2D utility tensor of shape (N, max_action_size).
        """
        u_vecs = []
        for i in range(self.num_players):
            other_strats = [
                stacked_strategies[j, : self.action_sizes[j]]
                for j in range(self.num_players)
                if j != i
            ]
            u_i = torch.einsum(self._einsum_strs[i], self.payoffs[i], *other_strats)
            u_vecs.append(u_i)

        return torch.nn.utils.rnn.pad_sequence(u_vecs, batch_first=True, padding_value=0.0)

    def get_expected_payoffs(self, strategies: List[torch.Tensor]) -> List[float]:
        """Compute scalar expected payoff per player.

        Parameters
        ----------
        strategies : List[torch.Tensor]
            Strategy distributions.

        Returns
        -------
        List[float]
            Expected payoff E_{a~x}[U^{(i)}(a)] for each player.
        """
        u_vecs = self.get_utility_vectors(strategies)
        return [
            torch.dot(u_vecs[i], strategies[i].to(device=self.device)).item()
            for i in range(self.num_players)
        ]

    def best_response_payoffs(self, strategies: List[torch.Tensor]) -> List[float]:
        """Compute max_a_i u_i(a_i, x^{-i}) best response payoff per player.

        Parameters
        ----------
        strategies : List[torch.Tensor]
            Strategy distributions.

        Returns
        -------
        List[float]
            Best response payoffs for each player.
        """
        u_vecs = self.get_utility_vectors(strategies)
        return [u_vecs[i].max().item() for i in range(self.num_players)]

    def to(self, device: torch.device) -> "NPlayerGame":
        """Move payoff tensors to target device.

        Parameters
        ----------
        device : torch.device
            Target PyTorch device.
        """
        self.device = device
        self.payoffs = [p.to(device=device) for p in self.payoffs]
        return self


class BatchNPlayerGame:
    """Ensemble game environment evaluating B parallel game instances simultaneously on GPU."""

    def __init__(
        self,
        batch_payoffs: List[torch.Tensor],
        utility_range: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize BatchNPlayerGame.

        Parameters
        ----------
        batch_payoffs : List[torch.Tensor]
            List of N batch payoff tensors, each of shape (batch_size, A_1, ..., A_N).
        utility_range : Tuple[float, float]
            Utility range bounds (u_min, u_max).
        device : torch.device
            Target PyTorch device.
        """
        self.num_players = len(batch_payoffs)
        self.batch_size = batch_payoffs[0].shape[0]
        self.action_sizes = list(batch_payoffs[0].shape[1:])
        self.utility_range = utility_range
        self.device = device
        self.payoffs = [p.to(device=device, dtype=torch.float32) for p in batch_payoffs]

    def get_utility_vectors(self, batch_strategies: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute expected utility vectors for B parallel game instances in one GPU einsum pass.

        Parameters
        ----------
        batch_strategies : List[torch.Tensor]
            List of N strategy tensors, each of shape (batch_size, A_i).

        Returns
        -------
        List[torch.Tensor]
            List of N batch utility tensors, each of shape (batch_size, A_i).
        """
        indices = _LETTERS[: self.num_players]
        utility_vectors: List[torch.Tensor] = []

        for i in range(self.num_players):
            target_letter = indices[i]
            # Construct batch einsum equation: e.g. for B=100, N=3, i=0: 'zabc,zb,zc->za'
            operand_indices = [
                f",z{indices[j]}" for j in range(self.num_players) if j != i
            ]
            einsum_str = f"z{indices}{''.join(operand_indices)}->z{target_letter}"

            other_strats = [
                batch_strategies[j].to(device=self.device, dtype=torch.float32)
                for j in range(self.num_players)
                if j != i
            ]

            u_i = torch.einsum(einsum_str, self.payoffs[i], *other_strats)
            utility_vectors.append(u_i)

        return utility_vectors
