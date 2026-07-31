"""Gradient Descent Ascent (GDA) learning dynamic baseline with simplex projection."""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic
from src.dynamics.extra_gradient import project_onto_simplex_batch


class GradientDescentAscent(BaseLearningDynamic):
    """Vanilla Gradient Descent Ascent (GDA) with simplex projection."""

    def __init__(
        self,
        action_sizes: List[int],
        eta: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize GDA dynamic."""
        super().__init__(action_sizes=action_sizes, eta=eta, device=device)
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

    def step(self, utility_vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Update strategies using 2D batched GDA step projected onto simplex across all N players simultaneously."""
        u_tensors = [u.to(device=self.device, dtype=torch.float32) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        # 1. Unconstrained gradient step across ALL players simultaneously
        raw_next = self.stacked_strategies + self.eta * stacked_u_curr

        # 2. Batched 2D Euclidean simplex projection across all N players simultaneously
        project_onto_simplex_batch(raw_next, mask=self.mask, out=self.stacked_strategies)

        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Perform 2D in-place vectorized GDA step directly on 2D utility tensor."""
        raw_next = self.stacked_strategies + self.eta * stacked_u_curr
        project_onto_simplex_batch(raw_next, mask=self.mask, out=self.stacked_strategies)
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
        self.eta = state_dict.get("eta", self.eta)
