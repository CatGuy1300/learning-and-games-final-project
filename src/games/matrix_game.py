"""Two-player general-sum matrix game environment."""

from typing import List, Tuple
import torch

from src.config.validation import validate_payoff_tensors, validate_strategies
from src.games.base import BaseGame


class MatrixGame(BaseGame):
    """Optimized 2-player matrix game with payoffs A, B in R^(m x n) for Player 1 and Player 2."""

    def __init__(
        self,
        payoff_a: torch.Tensor,
        payoff_b: torch.Tensor,
        utility_range: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize 2-player MatrixGame.

        Parameters
        ----------
        payoff_a : torch.Tensor
            Payoff matrix A of shape (m, n) for Player 1.
        payoff_b : torch.Tensor
            Payoff matrix B of shape (m, n) for Player 2.
        utility_range : Tuple[float, float]
            Expected payoff value range (u_min, u_max).
        device : torch.device
            Target PyTorch device.
        """
        payoffs = [payoff_a, payoff_b]
        if payoff_a.dim() != 2 or payoff_b.dim() != 2:
            raise ValueError("MatrixGame payoff matrices must be 2D tensors of shape (m, n)")
        if payoff_a.shape != payoff_b.shape:
            raise ValueError(
                f"Payoff A shape {payoff_a.shape} must equal payoff B shape {payoff_b.shape}"
            )

        m, n = payoff_a.shape
        validate_payoff_tensors(payoffs, 2, [m, n], utility_range)

        super().__init__(
            num_players=2,
            action_sizes=[m, n],
            utility_range=utility_range,
            device=device,
        )

        self.payoff_a = payoff_a.to(device=device, dtype=torch.float32)
        self.payoff_b = payoff_b.to(device=device, dtype=torch.float32)
        self.is_square = (m == n)
        if self.is_square:
            self.payoff_tensor_3d = torch.stack([self.payoff_a, self.payoff_b.T], dim=0)

    def get_payoff_tensors(self) -> List[torch.Tensor]:
        """Return list of payoff matrices [A, B] for Player 1 and Player 2."""
        return [self.payoff_a, self.payoff_b]

    def get_utility_vectors(self, strategies: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute u_1 = A * y and u_2 = B^T * x.

        Parameters
        ----------
        strategies : List[torch.Tensor]
            [x, y] where x has dim m and y has dim n.

        Returns
        -------
        List[torch.Tensor]
            [u_1, u_2]
        """
        validate_strategies(strategies, self.action_sizes)
        x = strategies[0].to(device=self.device, dtype=torch.float32)
        y = strategies[1].to(device=self.device, dtype=torch.float32)

        u1 = torch.matmul(self.payoff_a, y)  # shape (m,)
        u2 = torch.matmul(self.payoff_b.T, x)  # shape (n,)
        return [u1, u2]

    def get_stacked_utility_vectors(self, stacked_strategies: torch.Tensor) -> torch.Tensor:
        """Compute 2D utility tensor directly from 2D strategy tensor using single-kernel torch.bmm.

        Parameters
        ----------
        stacked_strategies : torch.Tensor
            2D strategy tensor of shape (2, max_action_size).

        Returns
        -------
        torch.Tensor
            2D utility tensor of shape (2, max_action_size).
        """
        if self.is_square:
            # Single-kernel batched matrix multiplication across all players on GPU
            strats_perm = stacked_strategies.flip(dims=[0]).unsqueeze(2)
            return torch.bmm(self.payoff_tensor_3d, strats_perm).squeeze(2)

        m, n = self.action_sizes
        x = stacked_strategies[0, :m]
        y = stacked_strategies[1, :n]

        u1 = torch.matmul(self.payoff_a, y)
        u2 = torch.matmul(self.payoff_b.T, x)
        return torch.nn.utils.rnn.pad_sequence([u1, u2], batch_first=True, padding_value=0.0)

    def get_expected_payoffs(self, strategies: List[torch.Tensor]) -> List[float]:
        """Compute x^T A y and x^T B y."""
        u_vecs = self.get_utility_vectors(strategies)
        x = strategies[0].to(device=self.device, dtype=torch.float32)
        y = strategies[1].to(device=self.device, dtype=torch.float32)

        payoff1 = torch.dot(x, u_vecs[0]).item()
        payoff2 = torch.dot(y, u_vecs[1]).item()
        return [payoff1, payoff2]

    def best_response_payoffs(self, strategies: List[torch.Tensor]) -> List[float]:
        """Compute max_i (A y)_i and max_j (B^T x)_j."""
        u_vecs = self.get_utility_vectors(strategies)
        return [u_vecs[0].max().item(), u_vecs[1].max().item()]

    def to(self, device: torch.device) -> "MatrixGame":
        """Move matrix tensors to target device."""
        self.device = device
        self.payoff_a = self.payoff_a.to(device=device)
        self.payoff_b = self.payoff_b.to(device=device)
        if self.is_square:
            self.payoff_tensor_3d = self.payoff_tensor_3d.to(device=device)
        return self
