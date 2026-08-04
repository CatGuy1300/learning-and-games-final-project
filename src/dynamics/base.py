"""Abstract base class for learning dynamics algorithms."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseLearningDynamic(ABC):
    """Abstract interface for multi-player learning dynamics with vectorized 2D strategy tensors."""

    def __init__(
        self,
        action_sizes: list[int],
        eta: float | None = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
    ) -> None:
        """Initialize learning dynamic.

        Parameters
        ----------
        action_sizes : List[int]
            Tuple of action counts (A_1, ..., A_N).
        eta : float | None
            Learning rate step size. If None, it is automatically inferred.
        device : torch.device
            PyTorch device allocation.
        batch_size : int
            Number of independent games to simulate simultaneously.
        """
        self.num_players = len(action_sizes)
        self.action_sizes = action_sizes
        self.max_action_size = max(action_sizes)
        self.eta = eta
        self.device = device
        self.batch_size = batch_size

        # Create boolean mask tensor of shape (1, num_players, max_action_size) for padded actions
        self.mask = torch.zeros(
            (1, self.num_players, self.max_action_size), device=self.device, dtype=torch.bool
        )
        for i, a_size in enumerate(action_sizes):
            self.mask[0, i, :a_size] = True

        # Batched 3D strategy tensor of shape (batch_size, num_players, max_action_size)
        self.stacked_strategies = torch.zeros(
            (self.batch_size, self.num_players, self.max_action_size),
            device=self.device,
            dtype=torch.float32,
        )

        # Pre-cached index tensor (1, 1, max_action_size) for zero-allocation simplex projections
        self.ind = torch.arange(
            1, self.max_action_size + 1, device=self.device, dtype=torch.float32
        ).view(1, 1, self.max_action_size)

    @property
    def strategies(self) -> list[torch.Tensor]:
        """Return strategies as a list of 1D/2D probability tensors [x^1, ..., x^N]."""
        if self.batch_size == 1:
            return [
                self.stacked_strategies[0, i, : self.action_sizes[i]]
                for i in range(self.num_players)
            ]
        return [
            self.stacked_strategies[:, i, : self.action_sizes[i]] for i in range(self.num_players)
        ]

    @strategies.setter
    def strategies(self, strats: list[torch.Tensor]) -> None:
        """Set strategies from a list of probability tensors."""
        self.stacked_strategies.zero_()
        for i, s in enumerate(strats):
            if self.batch_size == 1:
                self.stacked_strategies[0, i, : self.action_sizes[i]] = s.to(device=self.device)
            else:
                self.stacked_strategies[:, i, : self.action_sizes[i]] = s.to(device=self.device)

    @abstractmethod
    def reset(self, initial_strategies: list[torch.Tensor] | None = None) -> None:
        """Reset player strategy vectors and internal algorithm state."""

    @abstractmethod
    def step(self, utility_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform one step update of learning dynamics given current utility vectors u_i(x^{-i})."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Serialize internal state dictionary for checkpoint saving."""

    @abstractmethod
    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary from saved checkpoint."""

    @abstractmethod
    def step_2d(self, stacked_u_curr: torch.Tensor) -> torch.Tensor:
        """Perform 2D in-place vectorized step directly on 2D utility tensor."""

    def step_multi(self, game: Any, num_steps: int = 1) -> list[torch.Tensor]:
        """Perform num_steps unrolled updates in PyTorch GPU execution loop.

        Parameters
        ----------
        game : Any
            Game environment instance for computing utility vectors.
        num_steps : int
            Number of unrolled steps to execute.

        Returns
        -------
        List[torch.Tensor]
            Updated strategies after num_steps.
        """
        for _ in range(num_steps):
            u_vecs = game.get_utility_vectors(self.strategies)
            self.step(u_vecs)
        return self.strategies

    def step_unrolled_block(
        self, game: Any, cum_u_2d: torch.Tensor, cum_p_1d: torch.Tensor, k_steps: int,
        hist_strats: torch.Tensor | None = None,
        hist_logits: torch.Tensor | None = None,
        hist_stacked_u: torch.Tensor | None = None,
        hist_cum_u: torch.Tensor | None = None,
        hist_cum_p: torch.Tensor | None = None,
    ) -> None:
        """Execute k_steps dynamic updates natively on 2D GPU tensors.

        Parameters
        ----------
        game : Any
            Game instance with get_stacked_utility_vectors method.
        cum_u_2d : torch.Tensor
            2D cumulative utility tensor (N, max_A) to mutate in-place.
        cum_p_1d : torch.Tensor
            1D cumulative actual payoffs tensor (N,) to mutate in-place.
        k_steps : int
            Number of unrolled steps.
        """
        for i in range(k_steps):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            stacked_u = game.get_stacked_utility_vectors(self.stacked_strategies)
            cum_u_2d += stacked_u
            cum_p_1d += (stacked_u * self.stacked_strategies).sum(dim=-1)
            
            if hist_strats is not None:
                hist_strats[i] = self.stacked_strategies
                if hist_stacked_u is not None:
                    hist_stacked_u[i] = stacked_u
                if hist_cum_u is not None:
                    hist_cum_u[i] = cum_u_2d
                if hist_cum_p is not None:
                    hist_cum_p[i] = cum_p_1d
                if hist_logits is not None and hasattr(self, "log_strategies"):
                    hist_logits[i] = self.log_strategies
                    
            self.step_2d(stacked_u)
