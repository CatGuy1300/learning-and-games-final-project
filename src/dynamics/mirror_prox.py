"""MirrorProx (Entropy-Regularized ExtraGradient) learning dynamic."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic
from src.utils.logging import setup_logger

logger = setup_logger("mirror_prox")


class MirrorProx(BaseLearningDynamic):
    """MirrorProx (Extragradient adapted for probability simplex using Entropy regularization).

    Predictor: x_half = softmax(logits + eta * u(x))
    Corrector: x_next = softmax(logits + eta * u(x_half))
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
        logit_penalty_mode: str = "absolute",
    ) -> None:
        """Initialize MirrorProx dynamic."""
        if eta is not None:
            inferred_eta = eta
        else:
            inferred_eta = 1.0 / (16.0 * len(action_sizes) * max(action_sizes))
            logger.info(f"Inferred MirrorProx empirical eta = {inferred_eta:.6f} (1 / (16 * N * max(A)))")
            
        super().__init__(action_sizes=action_sizes, eta=inferred_eta, device=device, batch_size=batch_size)
        self.stacked_logits = torch.zeros_like(self.stacked_strategies)
        self.logits_half = torch.zeros_like(self.stacked_logits)
        self.x_half = torch.zeros_like(self.stacked_strategies)
        self.is_half_step = False
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.logit_penalty_mode = logit_penalty_mode
        self.reset()

    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset strategies and logits."""
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
        self.stacked_logits.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.stacked_logits.masked_fill_(~self.mask, -float("inf"))
        self.is_half_step = False
        self.cumulative_logit_penalty.zero_()
        
        if self.logit_penalty_threshold is not None:
            self.target_logs = torch.zeros_like(self.stacked_logits)
            self.excess = torch.zeros_like(self.stacked_logits)

    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Update strategies using 2D batched MirrorProx rule."""
        u_tensors = [u.to(device=self.device, dtype=torch.get_default_dtype()) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        self.step_2d(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Alternating Predictor-Corrector step logic.

        MirrorProx evaluates the gradient twice per step. To map this into a standard
        for-loop cleanly, we track an `is_half_step` boolean state.

        Phase 1: Predictor (is_half_step == False)
            Computes a tentative step half-way:
            x_{t+1/2} = softmax(logits_t + eta * u(x_t))
            Saves this half-step state to evaluate u(x_{t+1/2}) next.

        Phase 2: Corrector (is_half_step == True)
            Computes the actual full step using the half-step gradient:
            x_{t+1} = softmax(logits_t + eta * u(x_{t+1/2}))
        """
        if not self.is_half_step:
            # Predictor step
            self.logits_half.copy_(self.stacked_logits).add_(stacked_u_curr, alpha=self.eta)

            # Max-Centering in-placevent overflow
            max_val = self.logits_half.max(dim=-1, keepdim=True).values
            self.logits_half.sub_(max_val)

            # Handle masked actions
            self.logits_half.masked_fill_(~self.mask, -float("inf"))

            # Calculate half-step probabilities (in-place)
            torch.exp(self.logits_half, out=self.x_half)
            self.x_half.div_(self.x_half.sum(dim=-1, keepdim=True))

            self.is_half_step = True
            return self.x_half
        else:
            # Corrector step
            self.stacked_logits.add_(stacked_u_curr, alpha=self.eta)

            # Max-center to prevent overflow
            max_val = self.stacked_logits.max(dim=-1, keepdim=True).values
            self.stacked_logits.sub_(max_val)

            # Handle masked actions
            self.stacked_logits.masked_fill_(~self.mask, -float("inf"))

            # Logit Penalty accumulation (only on the full step)
            if self.logit_penalty_threshold is not None:
                if self.logit_penalty_mode == "centered":
                    valid_counts = self.mask.sum(dim=-1, keepdim=True).to(dtype=self.stacked_logits.dtype)
                    safe_logs = torch.where(self.mask, self.stacked_logits, torch.zeros_like(self.stacked_logits))
                    means = safe_logs.sum(dim=-1, keepdim=True) / valid_counts
                    torch.sub(self.stacked_logits, means, out=self.target_logs)
                else:
                    self.target_logs.copy_(self.stacked_logits)
                    
                torch.abs(self.target_logs, out=self.excess)
                self.excess.sub_(self.logit_penalty_threshold)
                torch.nn.functional.relu(self.excess, inplace=True)
                self.excess.masked_fill_(~self.mask, 0.0)
                self.excess.pow_(self.logit_penalty_norm)
                self.cumulative_logit_penalty += self.excess.sum(dim=(-1, -2))

            # Calculate full-step probabilities (in-place)
            torch.exp(self.stacked_logits, out=self.stacked_strategies)
            self.stacked_strategies.div_(self.stacked_strategies.sum(dim=-1, keepdim=True))

            self.is_half_step = False
            return self.stacked_strategies

    def step_unrolled_block(
        self, game: Any, cum_u_2d: torch.Tensor, cum_p_1d: torch.Tensor, k_steps: int,
        hist_strats: torch.Tensor | None = None,
        hist_logits: torch.Tensor | None = None,
        hist_stacked_u: torch.Tensor | None = None,
        hist_cum_u: torch.Tensor | None = None,
        hist_cum_p: torch.Tensor | None = None,
        **kwargs: Any
    ) -> None:
        """Override to implement the Predictor-Corrector double-evaluation graph without branching."""
        for i in range(k_steps):
            # 1. Predictor (is_half_step == False)
            # Evaluate utility at current state x_t
            u_curr = game.get_stacked_utility_vectors(self.stacked_strategies)

            # Accumulate metrics based on current full state x_t
            cum_u_2d += u_curr
            cum_p_1d += (u_curr * self.stacked_strategies).sum(dim=-1)
            
            if hist_strats is not None:
                hist_strats[i] = self.stacked_strategies
                if hist_stacked_u is not None:
                    hist_stacked_u[i] = u_curr
                if hist_cum_u is not None:
                    hist_cum_u[i] = cum_u_2d
                if hist_cum_p is not None:
                    hist_cum_p[i] = cum_p_1d
                if hist_logits is not None:
                    hist_logits[i] = self.stacked_logits

            # Step to get half-step probabilities.
            # We must assign this explicitly or use it, but since x_half is computed 
            # and returned directly from step_2d, we store it for the next evaluation.
            x_half = self.step_2d(u_curr)

            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # 2. Corrector (is_half_step == True)
            # Evaluate utility at x_{t+1/2}
            u_half = game.get_stacked_utility_vectors(x_half)

            # Step to get full-step probabilities x_{t+1}.
            # self.stacked_strategies is updated strictly IN-PLACE inside step_2d.
            # We do NOT clone or re-assign self.stacked_strategies, which would orphan the PyTorch memory reference.
            self.step_2d(u_half)

    def get_state(self) -> dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.clone() for s in self.strategies],
            "logits": self.stacked_logits.clone(),
            "is_half_step": self.is_half_step,
            "eta": self.eta,
            "cumulative_logit_penalty": self.cumulative_logit_penalty.clone(),
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
