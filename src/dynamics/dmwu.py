"""Dissipative Multiplicative Weights Update (DMWU) dynamic."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic


class DMWU(BaseLearningDynamic):
    """Dissipative Multiplicative Weights Update (DMWU).
    
    Implements a discrete-time variation of DGDA with a momentum-like anchor.
    """

    def __init__(
        self,
        action_sizes: list[int],
        eta: float | None = None,
        device: torch.device | str = "cpu",
        batch_size: int = 1,
        T: int = 10000,
        dmwu_gamma: float = 0.1,
        logit_penalty_threshold: float | None = None,
        logit_penalty_norm: int = 2,
        logit_penalty_mode: str = "absolute",
    ) -> None:
        """Initialize DMWU dynamic."""
        self.gamma = dmwu_gamma
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.logit_penalty_mode = logit_penalty_mode

        super().__init__(action_sizes, eta=eta, device=device, batch_size=batch_size)

        if eta is not None:
            self.eta = eta
        else:
            self.eta = 1.0 / (4.0 * max(action_sizes))

        self.stacked_logits = torch.zeros_like(self.stacked_strategies)
        self.stacked_w_hat = torch.zeros_like(self.stacked_logits)
        self.diff = torch.zeros_like(self.stacked_logits)
        self.reset()

    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset strategies, logits, and auxiliary logits."""
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
        
        self.stacked_w_hat.copy_(self.stacked_logits)
        self.cumulative_logit_penalty.zero_()
        
        if self.logit_penalty_threshold is not None:
            self.target_logs = torch.zeros_like(self.stacked_logits)
            self.excess = torch.zeros_like(self.stacked_logits)

    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Update strategies using 2D vectorized DMWU step rule."""
        u_tensors = [u.to(device=self.device, dtype=torch.get_default_dtype()) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        self.step_2d(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Execute a single fully vectorized DMWU step."""
        
        # 1. diff = gamma * (w_k - \hat{w}_k)
        self.diff.copy_(self.stacked_logits).sub_(self.stacked_w_hat)
        self.diff.masked_fill_(~self.mask, 0.0)
        self.diff.mul_(self.gamma)
        
        # 2. w_{k+1} = w_k + eta * V_k - diff
        self.stacked_logits.add_(stacked_u_curr, alpha=self.eta).sub_(self.diff)
        
        # 3. \hat{w}_{k+1} = \hat{w}_k + diff
        self.stacked_w_hat.add_(self.diff)

        # 4. Max-center to prevent overflow (must center BOTH w and w_hat to preserve relative differences)
        max_val = self.stacked_logits.max(dim=-1, keepdim=True).values
        self.stacked_logits.sub_(max_val)
        self.stacked_w_hat.sub_(max_val)

        # Handle masked actions
        self.stacked_logits.masked_fill_(~self.mask, -float("inf"))
        self.stacked_w_hat.masked_fill_(~self.mask, -float("inf"))

        # Logit Penalty accumulation
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

        # Calculate probabilities (in-place)
        torch.exp(self.stacked_logits, out=self.stacked_strategies)
        self.stacked_strategies.div_(self.stacked_strategies.sum(dim=-1, keepdim=True))

        return self.stacked_strategies

    def get_state(self) -> dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.clone() for s in self.strategies],
            "logits": self.stacked_logits.clone(),
            "w_hat": self.stacked_w_hat.clone(),
            "eta": self.eta,
            "gamma": self.gamma,
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
            
        if "w_hat" in state_dict:
            self.stacked_w_hat.copy_(state_dict["w_hat"].to(device=self.device))
        else:
            self.stacked_w_hat.copy_(self.stacked_logits)
            
        self.eta = state_dict.get("eta", self.eta)
        self.gamma = state_dict.get("gamma", self.gamma)
