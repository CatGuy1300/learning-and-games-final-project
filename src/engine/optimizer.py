import copy

import numpy as np
import torch
from cmaes import CMA

from src.config.schemas import ExperimentConfig
from src.engine.runner import ExperimentRunner


class CMAESGameOptimizer:
    """Finds the worst-case (highest regret) payoff matrices for a given learning dynamic using CMA-ES.

    This optimizer uses the lightweight `cmaes` package (from Optuna) and leverages the
    native batched GPU execution in `ExperimentRunner` to evaluate an entire generation
    of games simultaneously on the GPU.
    """

    def __init__(self, base_config: ExperimentConfig, sigma: float = 0.5, seed: int = 42):
        self.base_config = copy.deepcopy(base_config)
        self.sigma = sigma
        self.seed = seed
        np.random.seed(self.seed)

        # Determine shapes
        if self.base_config.game.generator != "custom":
            # For optimization, we require a custom base structure to mutate
            raise ValueError(
                "CMA-ES requires base_config.game.generator='custom' to define the base payoff shape."
            )

        # Get shape from the base custom payoffs
        self.payoff_shapes = [np.array(p).shape for p in self.base_config.game.payoffs]
        self.num_players = len(self.payoff_shapes)

        # Flattened dimension size
        self.dim = sum(np.prod(s) for s in self.payoff_shapes)

        # CMA-ES instance
        # We start centered at 0.0 with the specified sigma
        self.optimizer = CMA(mean=np.zeros(self.dim), sigma=self.sigma, seed=self.seed)

        # Override config batch_size to match CMA population size
        self.population_size = self.optimizer.population_size
        self.base_config.execution.batch_size = self.population_size

    def _unflatten_payoffs(self, flat_batch: np.ndarray) -> list[torch.Tensor]:
        """Convert a (population_size, dim) numpy array into a list of batched payoff tensors."""
        B = flat_batch.shape[0]
        payoffs = []
        offset = 0
        for shape in self.payoff_shapes:
            size = np.prod(shape)
            # Slice out this player's parameters and reshape to (B, *shape)
            p_flat = flat_batch[:, offset : offset + size]
            p_tensor = torch.tensor(p_flat, dtype=torch.float32).reshape(B, *shape)

            # Clip payoffs to utility_range if specified
            if self.base_config.game.utility_range is not None:
                u_min, u_max = self.base_config.game.utility_range
                p_tensor = torch.clamp(p_tensor, u_min, u_max)

            payoffs.append(p_tensor)
            offset += size

        return payoffs

    def step(self) -> tuple[np.ndarray, float]:
        """Execute one generation of CMA-ES.

        Returns
        -------
        Tuple[np.ndarray, float]
            The best solution found in this generation and its average regret.
        """
        # Ask for population_size solutions
        solutions_flat = np.array([self.optimizer.ask() for _ in range(self.population_size)])

        # Convert to batched payoff tensors
        batched_payoffs = self._unflatten_payoffs(solutions_flat)

        # Run batched simulation
        config = copy.deepcopy(self.base_config)
        # We pass the batched tensors directly to bypass JSON schema validation overhead
        config.game.payoffs = batched_payoffs

        runner = ExperimentRunner(config)
        metrics = runner.run()

        # We want to MAXIMIZE regret. CMA-ES MINIMIZES.
        # metrics["final_avg_regrets"] is (B, N). We take the sum across players.
        # So fitness = -sum(regrets)

        # Extract regret per batch element
        # final_avg_regrets shape: [N] if B=1, but here B=population_size so it's [B, N]?
        # Wait, compute_average_regret in runner.py:
        # if B=1: returns [float, float]
        # if B>1: returns [[float, float], [float, float], ...] where outer is players!
        # Ah, let's verify runner.py's compute_average_regret format.

        # Actually it's easier to just sum over players for each batch element.
        regrets_by_player = metrics["final_avg_regrets"]
        # regrets_by_player is List[List[float]] where len is N, and inner len is B.
        # Let's sum across players:
        B = self.population_size
        sum_regrets = np.zeros(B)
        for i in range(self.num_players):
            sum_regrets += np.array(regrets_by_player[i])

        fitnesses = -sum_regrets  # Negative because CMA-ES minimizes

        # Tell
        solutions_with_fitness = [(solutions_flat[b], fitnesses[b]) for b in range(B)]
        self.optimizer.tell(solutions_with_fitness)

        # Best in this generation
        best_idx = np.argmin(fitnesses)
        return solutions_flat[best_idx], sum_regrets[best_idx]

    def optimize(self, generations: int = 50) -> tuple[list[torch.Tensor], float]:
        """Run CMA-ES for the specified number of generations.

        Returns
        -------
        Tuple[List[torch.Tensor], float]
            The best payoff matrices found and their regret sum.
        """
        best_solution = None
        best_regret = -float("inf")

        for g in range(generations):
            sol, regret = self.step()
            if regret > best_regret:
                best_regret = regret
                best_solution = sol

        # Return unflattened batched tensor but index 0 to make it unbatched
        batched_best = self._unflatten_payoffs(np.array([best_solution]))
        unbatched_best = [p[0] for p in batched_best]
        return unbatched_best, best_regret
