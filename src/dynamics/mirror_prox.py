"""MirrorProx (Entropy-Regularized ExtraGradient) learning dynamic."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic


class MirrorProx(BaseLearningDynamic):
    """MirrorProx (Extragradient adapted for probability simplex using Entropy regularization).

    Predictor: x_half = softmax(logits + eta * u(x))
    Corrector: x_next = softmax(logits + eta * u(x_half))
    """

    def __init__(
        self,
        action_sizes: list[int],
        eta: float = 0.01,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
    ) -> None:
        """Initialize MirrorProx dynamic."""
        super().__init__(action_sizes=action_sizes, eta=eta, device=device, batch_size=batch_size)
        self.stacked_logits = torch.zeros_like(self.stacked_strategies)
        self.logits_half = torch.zeros_like(self.stacked_logits)
        self.is_half_step = False
        self.reset()

    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset strategies and logits."""
        if initial_strategies is not None:
            self.strategies = [
                s.clone().to(device=self.device, dtype=torch.float32) for s in initial_strategies
            ]
        else:
            self.strategies = [
                torch.full((a,), 1.0 / a, device=self.device, dtype=torch.float32)
                for a in self.action_sizes
            ]

        eps = 1e-30
        self.stacked_logits.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.stacked_logits.masked_fill_(~self.mask, -float("inf"))
        self.is_half_step = False

    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Update strategies using 2D batched MirrorProx rule."""
        u_tensors = [u.to(device=self.device, dtype=torch.float32) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        self.step_2d(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Alternating Predictor-Corrector step logic.

        Phase 1: Predictor (is_half_step == False)
        Phase 2: Corrector (is_half_step == True)
        """
        if not self.is_half_step:
            # Predictor step
            self.logits_half = self.stacked_logits + self.eta * stacked_u_curr

            # Max-center to prevent overflow
            max_val = self.logits_half.max(dim=-1, keepdim=True).values
            self.logits_half = self.logits_half - max_val

            # Handle masked actions using torch.where as requested
            self.logits_half = torch.where(self.mask, self.logits_half, -float("inf"))

            # Calculate half-step probabilities
            x_half = torch.softmax(self.logits_half, dim=-1)

            self.is_half_step = True
            return x_half
        else:
            # Corrector step
            self.stacked_logits = self.stacked_logits + self.eta * stacked_u_curr

            # Max-center to prevent overflow
            max_val = self.stacked_logits.max(dim=-1, keepdim=True).values
            self.stacked_logits = self.stacked_logits - max_val

            # Handle masked actions using torch.where as requested
            self.stacked_logits = torch.where(self.mask, self.stacked_logits, -float("inf"))

            # Calculate full-step probabilities
            self.stacked_strategies = torch.softmax(self.stacked_logits, dim=-1)

            self.is_half_step = False
            return self.stacked_strategies

    def step_unrolled_block(
        self, game: Any, cum_u_2d: torch.Tensor, cum_p_1d: torch.Tensor, k_steps: int
    ) -> None:
        """Override to implement the Predictor-Corrector double-evaluation graph without branching."""
        for _ in range(k_steps):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # 1. Predictor (is_half_step == False)
            # Evaluate utility at current state x_t
            u_curr = game.get_stacked_utility_vectors(self.stacked_strategies)

            # Accumulate metrics based on current full state x_t
            cum_u_2d += u_curr
            cum_p_1d += (u_curr * self.stacked_strategies).sum(dim=-1)

            # Step to get half-step probabilities
            x_half = self.step_2d(u_curr)

            # 2. Corrector (is_half_step == True)
            # Evaluate utility at x_{t+1/2}
            u_half = game.get_stacked_utility_vectors(x_half)

            # Step to get full-step probabilities x_{t+1}
            self.step_2d(u_half)

    def get_state(self) -> dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.cpu() for s in self.strategies],
            "logits": self.stacked_logits.cpu(),
            "is_half_step": self.is_half_step,
            "eta": self.eta,
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary."""
        self.strategies = [s.to(device=self.device) for s in state_dict["strategies"]]
        if "logits" in state_dict:
            self.stacked_logits.copy_(state_dict["logits"].to(device=self.device))
        else:
            eps = 1e-30
            self.stacked_logits.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
            self.stacked_logits.masked_fill_(~self.mask, -float("inf"))
        self.is_half_step = state_dict.get("is_half_step", False)
        self.eta = state_dict.get("eta", self.eta)
