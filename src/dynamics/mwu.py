"""Multiplicative Weights Update (MWU / Hedge) learning dynamic baseline."""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic


class MultiplicativeWeightsUpdate(BaseLearningDynamic):
    """Vanilla Multiplicative Weights Update (MWU / Hedge)."""

    def __init__(
        self,
        action_sizes: List[int],
        eta: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize MWU dynamic."""
        super().__init__(action_sizes=action_sizes, eta=eta, device=device)
        self.log_strategies = torch.zeros(
            (self.num_players, self.max_action_size), device=self.device, dtype=torch.float32
        )
        self.reset()

    def reset(self, initial_strategies: Optional[List[torch.Tensor]] = None) -> None:
        """Reset strategy distributions."""
        self.stacked_strategies.zero_()
        for i, a_size in enumerate(self.action_sizes):
            if initial_strategies is not None:
                s = initial_strategies[i].clone().to(device=self.device, dtype=torch.float32)
            else:
                s = torch.full((a_size,), 1.0 / a_size, device=self.device, dtype=torch.float32)
            self.stacked_strategies[i, :a_size] = s

        eps = 1e-30
        self.log_strategies.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))

    def step(self, utility_vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Update strategies using 2D vectorized MWU step across all N players simultaneously."""
        u_tensors = [u.to(device=self.device, dtype=torch.float32) for u in utility_vectors]
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
        max_val = self.log_strategies.max(dim=1, keepdim=True).values
        self.log_strategies.sub_(max_val)
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))

        # 3. Softmax exponentiation in-place
        torch.exp(self.log_strategies, out=self.stacked_strategies)
        self.stacked_strategies.masked_fill_(~self.mask, 0.0)
        
        # 4. Normalize in-place
        self.stacked_strategies.div_(self.stacked_strategies.sum(dim=1, keepdim=True))
        return self.stacked_strategies

    def get_state(self) -> Dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.cpu() for s in self.strategies],
            "eta": self.eta,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.strategies = [s.to(device=self.device) for s in state_dict["strategies"]]
        eps = 1e-30
        self.log_strategies.copy_(torch.log(torch.clamp(self.stacked_strategies, min=eps)))
        self.log_strategies.masked_fill_(~self.mask, -float("inf"))
        self.eta = state_dict.get("eta", self.eta)
