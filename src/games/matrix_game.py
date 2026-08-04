"""Two-player general-sum matrix game environment."""

import torch

from src.config.validation import validate_payoff_tensors, validate_strategies
from src.games.base import BaseGame


class MatrixGame(BaseGame):
    """Optimized 2-player matrix game with payoffs A, B in R^(m x n) for Player 1 and Player 2."""

    def __init__(
        self,
        payoff_a: torch.Tensor,
        payoff_b: torch.Tensor,
        utility_range: tuple[float, float] = (-1.0, 1.0),
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
        if payoff_a.dim() not in [2, 3] or payoff_b.dim() not in [2, 3]:
            raise ValueError("MatrixGame payoff matrices must be 2D or 3D tensors")
        if payoff_a.shape != payoff_b.shape:
            raise ValueError(
                f"Payoff A shape {payoff_a.shape} must equal payoff B shape {payoff_b.shape}"
            )

        self.batch_size = payoff_a.shape[0] if payoff_a.dim() == 3 else 1
        m, n = payoff_a.shape[-2:]
        validate_payoff_tensors(payoffs, 2, [m, n], utility_range)

        super().__init__(
            num_players=2,
            action_sizes=[m, n],
            utility_range=utility_range,
            device=device,
        )

        self.payoff_a = payoff_a.to(device=device, dtype=torch.get_default_dtype())
        self.payoff_b = payoff_b.to(device=device, dtype=torch.get_default_dtype())
        self.is_square = m == n
        if self.is_square:
            # stack dimension depends on whether batched (B, 2, m, n) or (2, m, n)
            dim_to_stack = 1 if self.batch_size > 1 else 0
            self.payoff_tensor_3d = torch.stack(
                [self.payoff_a, self.payoff_b.transpose(-2, -1)], dim=dim_to_stack
            )

    def get_payoff_tensors(self) -> list[torch.Tensor]:
        """Return list of payoff matrices [A, B] for Player 1 and Player 2."""
        return [self.payoff_a, self.payoff_b]

    def get_utility_vectors(self, strategies: list[torch.Tensor]) -> list[torch.Tensor]:
        validate_strategies(strategies, self.action_sizes)
        x = strategies[0].to(device=self.device, dtype=torch.get_default_dtype())
        y = strategies[1].to(device=self.device, dtype=torch.get_default_dtype())

        y_ = y.unsqueeze(-1) if y.dim() == 2 else y
        x_ = x.unsqueeze(-1) if x.dim() == 2 else x

        u1 = (
            torch.matmul(self.payoff_a, y_).squeeze(-1)
            if y.dim() == 2
            else torch.matmul(self.payoff_a, y)
        )
        u2 = (
            torch.matmul(self.payoff_b.transpose(-2, -1), x_).squeeze(-1)
            if x.dim() == 2
            else torch.matmul(self.payoff_b.transpose(-2, -1), x)
        )
        return [u1, u2]

    def get_stacked_utility_vectors(self, stacked_strategies: torch.Tensor) -> torch.Tensor:
        is_batched = stacked_strategies.dim() == 3

        if self.is_square:
            if is_batched:
                strats_perm = stacked_strategies.flip(dims=[1]).unsqueeze(-1)
                if self.payoff_tensor_3d.dim() == 4:
                    return torch.matmul(self.payoff_tensor_3d, strats_perm).squeeze(-1)
                else:
                    return torch.matmul(self.payoff_tensor_3d.unsqueeze(0), strats_perm).squeeze(-1)
            else:
                strats_perm = stacked_strategies.flip(dims=[0]).unsqueeze(2)
                return torch.bmm(self.payoff_tensor_3d, strats_perm).squeeze(2)

        m, n = self.action_sizes
        if is_batched:
            x = stacked_strategies[:, 0, :m]
            y = stacked_strategies[:, 1, :n]
            p_a = (
                self.payoff_a.unsqueeze(0).expand(x.shape[0], -1, -1)
                if self.payoff_a.dim() == 2
                else self.payoff_a
            )
            p_b = (
                self.payoff_b.unsqueeze(0).expand(x.shape[0], -1, -1)
                if self.payoff_b.dim() == 2
                else self.payoff_b
            )
            u1 = torch.bmm(p_a, y.unsqueeze(-1)).squeeze(-1)
            u2 = torch.bmm(x.unsqueeze(1), p_b).squeeze(1)
            # pad
            if u1.shape[1] < self.max_action_size:
                u1 = torch.nn.functional.pad(u1, (0, self.max_action_size - u1.shape[1]))
            if u2.shape[1] < self.max_action_size:
                u2 = torch.nn.functional.pad(u2, (0, self.max_action_size - u2.shape[1]))
            return torch.stack([u1, u2], dim=1)
        else:
            x = stacked_strategies[0, :m]
            y = stacked_strategies[1, :n]
            u1 = torch.matmul(self.payoff_a, y)
            u2 = torch.matmul(self.payoff_b.transpose(-2, -1), x)
            return torch.nn.utils.rnn.pad_sequence([u1, u2], batch_first=True, padding_value=0.0)

    def get_expected_payoffs(self, strategies: list[torch.Tensor]) -> list[float]:
        """Compute x^T A y and x^T B y."""
        u_vecs = self.get_utility_vectors(strategies)
        x = strategies[0].to(device=self.device, dtype=torch.get_default_dtype())
        y = strategies[1].to(device=self.device, dtype=torch.get_default_dtype())

        payoff1 = (
            torch.sum(x * u_vecs[0], dim=-1).tolist()
            if x.dim() == 2
            else torch.dot(x, u_vecs[0]).item()
        )
        payoff2 = (
            torch.sum(y * u_vecs[1], dim=-1).tolist()
            if y.dim() == 2
            else torch.dot(y, u_vecs[1]).item()
        )
        return [payoff1, payoff2]

    def best_response_payoffs(self, strategies: list[torch.Tensor]) -> list[float]:
        """Compute max_i (A y)_i and max_j (B^T x)_j."""
        u_vecs = self.get_utility_vectors(strategies)
        if strategies[0].dim() == 2:
            return [u_vecs[0].max(dim=-1).values.tolist(), u_vecs[1].max(dim=-1).values.tolist()]
        return [u_vecs[0].max().item(), u_vecs[1].max().item()]

    def to(self, device: torch.device) -> "MatrixGame":
        """Move matrix tensors to target device."""
        self.device = device
        self.payoff_a = self.payoff_a.to(device=device)
        self.payoff_b = self.payoff_b.to(device=device)
        if self.is_square:
            self.payoff_tensor_3d = self.payoff_tensor_3d.to(device=device)
        return self
