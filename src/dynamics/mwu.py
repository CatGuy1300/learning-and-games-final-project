"""Multiplicative Weights Update (MWU / Hedge) learning dynamic baseline."""

import math
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic
from src.utils.logging import setup_logger

logger = setup_logger("mwu")

class MultiplicativeWeightsUpdate(BaseLearningDynamic):
    """Vanilla Multiplicative Weights Update (MWU / Hedge)."""

    def __init__(
        self,
        action_sizes: list[int],
        eta: float | None = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        T: int = 10000,
        logit_penalty_threshold: float | None = None,
        logit_penalty_norm: int = 2,
    ) -> None:
        """Initialize MWU dynamic."""
        if eta is not None:
            inferred_eta = eta
        else:
            inferred_eta = math.sqrt(math.log(max(action_sizes)) / T)
            logger.info(f"Inferred MWU empirical eta = {inferred_eta:.6f} (1 / (4 * max(A)))")
            
        super().__init__(action_sizes=action_sizes, eta=inferred_eta, device=device, batch_size=batch_size)
        self.log_strategies = torch.zeros_like(self.stacked_strategies)
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.reset()

    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset strategy distributions."""
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

    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Update strategies using 2D vectorized MWU step across all N players simultaneously."""
        u_tensors = [u.to(device=self.device, dtype=torch.get_default_dtype()) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        self.step_2d(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Perform 2D in-place vectorized MWU step directly on persistent log-strategies."""
        # 1. Update log-domain strategies directly
        self.log_strategies.add_(stacked_u_curr, alpha=self.eta)

        # 2. Max-Centering in-place
        max_vals = self.log_strategies.max(dim=-1, keepdim=True).values
        self.log_strategies.sub_(max_vals)
        
        # 3. Logit Penalty accumulation
        if self.logit_penalty_threshold is not None:
            excess = torch.nn.functional.relu(torch.abs(self.log_strategies) - self.logit_penalty_threshold)
            self.cumulative_logit_penalty += (excess ** self.logit_penalty_norm).sum(dim=(-1, -2))

        # Apply mask
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))

        # 4. Softmax probability evaluation (in-place)
        torch.exp(self.log_strategies, out=self.stacked_strategies)
        self.stacked_strategies.masked_fill_(~self.mask, 0.0)

        # 4. Normalize in-place
        self.stacked_strategies.div_(self.stacked_strategies.sum(dim=-1, keepdim=True))
        return self.stacked_strategies

    def get_state(self) -> dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.cpu() for s in self.strategies],
            "eta": self.eta,
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary."""
        self.strategies = [s.to(device=self.device) for s in state_dict["strategies"]]
        eps = 1e-30
        self.log_strategies.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))
        self.eta = state_dict.get("eta", self.eta)
