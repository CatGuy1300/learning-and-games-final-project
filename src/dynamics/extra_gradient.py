"""Extra Gradient (EG) / Optimistic Gradient Descent (OGD) with simplex projection."""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

from src.dynamics.base import BaseLearningDynamic


def project_onto_simplex_batch(
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ind: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized Euclidean projection of 2D matrix V (shape: num_players x max_action_size) onto the probability simplex."""
    N, d = V.shape
    # We cannot do `mask.all()` here because it causes a blocking GPU-to-CPU synchronization!
    # Instead, we just check if d == 2, and we can compute the closed form natively.
    # If a mask exists, we can apply it after the closed-form math safely, because d=2 simplex projection
    # for padded actions will just be clamped to 0 anyway if we handle it correctly.
    if d == 2:
        # Closed-form O(1) 2-action Euclidean simplex projection: theta = (v1 + v2 - 1) / 2
        theta = (V.sum(dim=-1, keepdim=True) - 1.0) * 0.5
        if out is not None:
            torch.clamp(V - theta, min=0.0, out=out)
        else:
            out = torch.clamp(V - theta, min=0.0)
            
        if mask is not None:
            out.masked_fill_(~mask, 0.0)
            
        sums = out.sum(dim=-1, keepdim=True)
        sums.masked_fill_(sums == 0, 1.0)
        out.div_(sums)
        return out

    if mask is not None:
        V_work = torch.where(mask, V, -float("inf"))
    else:
        V_work = V

    # 1. Vectorized descending sort along action dimension (dim=-1)
    u, _ = torch.sort(V_work, descending=True, dim=-1)

    # Replace -inf padding entries with 0.0 for cumulative sum math
    u_valid = torch.where(torch.isinf(u), 0.0, u)

    # 2. Parallel cumulative sum shift across all players and actions
    cssv = torch.cumsum(u_valid, dim=-1) - 1.0
    if ind is None:
        ind = torch.arange(1, d + 1, device=V.device, dtype=V.dtype).unsqueeze(0)
    cond = u - cssv / ind > 0.0

    # 3. Vectorized support size calculation per player row
    rho = torch.clamp(cond.sum(dim=-1, keepdim=True), min=1)

    # 4. Gather exact offset threshold theta per player row
    cssv_rho = torch.gather(cssv, dim=-1, index=rho - 1)
    theta = cssv_rho / rho

    # 5. Soft-thresholding: clamp negative values to 0.0
    if out is not None:
        torch.clamp(V - theta, min=0.0, out=out)
    else:
        out = torch.clamp(V - theta, min=0.0)
        
    if mask is not None:
        out.masked_fill_(~mask, 0.0)
        
    sums = out.sum(dim=-1, keepdim=True)
    sums.masked_fill_(sums == 0, 1.0)
    out.div_(sums)
    return out


def project_onto_simplex(v: torch.Tensor) -> torch.Tensor:
    """1D wrapper for single vector Euclidean projection onto probability simplex."""
    V_2d = v.unsqueeze(0)
    return project_onto_simplex_batch(V_2d).squeeze(0)


class ExtraGradient(BaseLearningDynamic):
    """Extra Gradient (EG) / Optimistic Gradient Descent (OGD) for general N-player finite games."""

    def __init__(
        self,
        action_sizes: List[int],
        eta: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize ExtraGradient dynamic."""
        super().__init__(action_sizes=action_sizes, eta=eta, device=device)
        self.stacked_prev_utilities = torch.zeros(
            (self.num_players, self.max_action_size), device=self.device, dtype=torch.float32
        )
        self.has_prev = False
        self.reset()

    def reset(self, initial_strategies: Optional[List[torch.Tensor]] = None) -> None:
        """Reset strategies and clear past utility history."""
        self.stacked_strategies.zero_()
        for i, a_size in enumerate(self.action_sizes):
            if initial_strategies is not None:
                s = initial_strategies[i].clone().to(device=self.device, dtype=torch.float32)
            else:
                s = torch.full((a_size,), 1.0 / a_size, device=self.device, dtype=torch.float32)
            self.stacked_strategies[i, :a_size] = s
        self.stacked_prev_utilities.zero_()
        self.has_prev = False

    def step(self, utility_vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Update strategies using 2D batched Extra Gradient (OGD) rule across all N players simultaneously."""
        # 1. Zero-loop C++ batched 2D tensor conversion (shape: num_players x max_action_size)
        u_tensors = [u.to(device=self.device, dtype=torch.float32) for u in utility_vectors]
        stacked_u_curr = pad_sequence(u_tensors, batch_first=True, padding_value=0.0)

        if stacked_u_curr.shape[1] < self.max_action_size:
            pad_cols = self.max_action_size - stacked_u_curr.shape[1]
            stacked_u_curr = torch.nn.functional.pad(stacked_u_curr, (0, pad_cols))

        if not self.has_prev:
            self.stacked_prev_utilities.copy_(stacked_u_curr)
            self.has_prev = True

        stacked_u_prev = self.stacked_prev_utilities

        # 2. Fully vectorized optimistic gradient step across ALL players simultaneously
        raw_next = self.stacked_strategies + 2.0 * self.eta * stacked_u_curr - self.eta * stacked_u_prev

        # 3. Batched 2D Euclidean simplex projection across all N players simultaneously
        project_onto_simplex_batch(raw_next, mask=self.mask, ind=self.ind, out=self.stacked_strategies)

        self.stacked_prev_utilities.copy_(stacked_u_curr)
        return self.strategies

    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Perform 2D in-place vectorized ExtraGradient step directly on 2D utility tensor."""
        if not self.has_prev:
            self.stacked_prev_utilities.copy_(stacked_u_curr)
            self.has_prev = True

        stacked_u_prev = self.stacked_prev_utilities

        raw_next = self.stacked_strategies + 2.0 * self.eta * stacked_u_curr - self.eta * stacked_u_prev
        project_onto_simplex_batch(raw_next, mask=self.mask, ind=self.ind, out=self.stacked_strategies)

        self.stacked_prev_utilities.copy_(stacked_u_curr)
        return self.stacked_strategies

    def get_state(self) -> Dict[str, Any]:
        """Serialize state dictionary."""
        return {
            "strategies": [s.cpu() for s in self.strategies],
            "prev_utilities": (
                [
                    self.stacked_prev_utilities[i, : self.action_sizes[i]].cpu()
                    for i in range(self.num_players)
                ]
                if self.stacked_prev_utilities is not None
                else None
            ),
            "eta": self.eta,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.strategies = [s.to(device=self.device) for s in state_dict["strategies"]]
        if state_dict["prev_utilities"] is not None:
            self.has_prev = True
            self.stacked_prev_utilities = torch.zeros(
                (self.num_players, self.max_action_size), device=self.device, dtype=torch.float32
            )
            for i, u in enumerate(state_dict["prev_utilities"]):
                self.stacked_prev_utilities[i, : self.action_sizes[i]] = u.to(device=self.device)
        else:
            self.stacked_prev_utilities.zero_()
            self.has_prev = False
        self.eta = state_dict.get("eta", self.eta)
