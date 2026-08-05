"""Optimistic Multiplicative Weights Update (OMWU / Optimistic Hedge) learning dynamic."""

import math
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic
from src.utils.logging import setup_logger

logger = setup_logger("omwu")

class OptimisticMWU(BaseLearningDynamic):
    """Optimistic Multiplicative Weights Update (OMWU) for general N-player finite games.

    Update rule per player i and action a:
    log(x_{t+1, a}) = log(x_{t, a}) + 2 * eta * u_{t, a} - eta * u_{t-1, a}

    Vectorization & Numerical Stability Rationale:
    1. Zero-Loop PyTorch C++ Stacking: Utility vectors are batched into a 2D tensor via PyTorch C++ native
       `pad_sequence(utility_vectors, batch_first=True)`, completely avoiding Python for-loops.
    2. Fully Vectorized 2D Tensor Operations: Strategy updates for all N players are computed
       simultaneously in a single 2D PyTorch tensor operation of shape (num_players, max_action_size).
    3. Log-Domain Updates & Epsilon Clamping: Maintains strategies in log-space and clamps to min=1e-30
       before torch.log() to prevent log(0) = -inf crashes.
    4. Optimistic Extrapolation: Uses 2 * u_t - u_{t-1} prediction momentum.
    5. Log-Sum-Exp Normalization: Subtracts max(log_next) along dim=-1 to prevent float overflow (+inf/NaN).
    """

    def __init__(
        self,
        action_sizes: list[int],
        eta: float | None = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        T: int = 10000,
        strict_theory_eta: bool = False,
        logit_penalty_threshold: float | None = None,
        logit_penalty_norm: int = 2,
    ) -> None:
        """Initialize OMWU dynamic."""
        if eta is not None:
            inferred_eta = eta
        elif strict_theory_eta:
            inferred_eta = 1.0 / (16.0 * len(action_sizes) * (math.log(T) ** 4))
            logger.info(f"Inferred OMWU strict theory eta = {inferred_eta:.6f} (1 / (16 * N * log^4(T)))")
        else:
            inferred_eta = 1.0 / (8.0 * max(action_sizes))
            logger.info(f"Inferred OMWU empirical eta = {inferred_eta:.6f} (1 / (8 * max(A)))")
            
        super().__init__(action_sizes=action_sizes, eta=inferred_eta, device=device, batch_size=batch_size)
        self.stacked_prev_utilities = torch.zeros_like(self.stacked_strategies)
        self.has_prev = False
        self.log_strategies = torch.zeros_like(self.stacked_strategies)
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.reset()

    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset strategy distributions to uniform or custom, and clear utility history."""
        if initial_strategies is not None:
            self.strategies = [
                s.clone().to(device=self.device, dtype=torch.get_default_dtype()) for s in initial_strategies
            ]
        else:
            self.strategies = [
                torch.full((a,), 1.0 / a, device=self.device, dtype=torch.get_default_dtype())
                for a in self.action_sizes
            ]

        eps = 1e-30
        self.log_strategies.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))
        self.stacked_prev_utilities.zero_()
        self.has_prev = False
        self.cumulative_logit_penalty.zero_()

    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Update strategies using 2D vectorized OMWU step rule across all N players simultaneously."""
        # 1. Zero-loop C++ batched 2D tensor conversion (shape: num_players x max_action_size)
        u_tensors = [u.to(device=self.device, dtype=torch.get_default_dtype()) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        # Pad to max_action_size if required
        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        self.step_2d(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Perform 2D in-place vectorized OMWU step directly on persistent log-strategies."""
        if not self.has_prev:
            self.stacked_prev_utilities.copy_(stacked_u_curr)
            self.has_prev = True

        stacked_u_prev = self.stacked_prev_utilities

        # 1. Update log-domain strategies directly
        self.log_strategies.add_(stacked_u_curr, alpha=2.0 * self.eta)
        self.log_strategies.sub_(stacked_u_prev, alpha=self.eta)

        # 2. Max-Centering in-place
        max_vals = self.log_strategies.max(dim=-1, keepdim=True).values
        self.log_strategies.sub_(max_vals)
        
        # 3. Logit Penalty accumulation
        if self.logit_penalty_threshold is not None:
            excess = torch.nn.functional.relu(torch.abs(self.log_strategies) - self.logit_penalty_threshold)
            self.cumulative_logit_penalty += (excess ** self.logit_penalty_norm).sum(dim=(-1, -2))

        # Apply mask
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))

        # 4. Softmax exponentiation in-place
        torch.exp(self.log_strategies, out=self.stacked_strategies)
        self.stacked_strategies.masked_fill_(~self.mask, 0.0)

        # 5. Normalize in-place
        self.stacked_strategies.div_(self.stacked_strategies.sum(dim=-1, keepdim=True))

        self.stacked_prev_utilities.copy_(stacked_u_curr)
        return self.stacked_strategies

    def get_state(self) -> dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.cpu() for s in self.strategies],
            "logits": [
                (
                    self.log_strategies[0, i, : self.action_sizes[i]].cpu()
                    if self.batch_size == 1
                    else self.log_strategies[:, i, : self.action_sizes[i]].cpu()
                )
                for i in range(self.num_players)
            ],
            "prev_utilities": (
                [
                    (
                        self.stacked_prev_utilities[0, i, : self.action_sizes[i]].cpu()
                        if self.batch_size == 1
                        else self.stacked_prev_utilities[:, i, : self.action_sizes[i]].cpu()
                    )
                    for i in range(self.num_players)
                ]
                if self.stacked_prev_utilities is not None
                else None
            ),
            "eta": self.eta,
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary."""
        self.strategies = [s.to(device=self.device) for s in state_dict["strategies"]]
        eps = 1e-30
        self.log_strategies.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))
        if state_dict["prev_utilities"] is not None:
            self.has_prev = True
            self.stacked_prev_utilities = torch.zeros_like(self.stacked_strategies)
            for i, u in enumerate(state_dict["prev_utilities"]):
                self.stacked_prev_utilities[
                    0 if self.batch_size == 1 else slice(None), i, : self.action_sizes[i]
                ] = u.to(device=self.device)
        else:
            self.stacked_prev_utilities.zero_()
            self.has_prev = False
        self.eta = state_dict.get("eta", self.eta)
