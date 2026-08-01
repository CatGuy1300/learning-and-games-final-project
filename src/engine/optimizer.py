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

    def __init__(
        self,
        base_config: ExperimentConfig,
        sigma: float = 0.5,
        seed: int = 42,
        objective_type: str = "raw",
        T1_ratio: float = 0.8,
        lambda_reg: float = 0.1,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.sigma = sigma
        self.seed = seed
        self.objective_type = objective_type
        self.T1_ratio = T1_ratio
        self.lambda_reg = lambda_reg
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
        T = config.execution.total_steps
        B = self.population_size

        if self.objective_type == "delta_reg":
            T1 = int(self.T1_ratio * T)
            metrics_t1 = runner.run(target_steps=T1)
            metrics_t = runner.run(target_steps=T)

            regrets_t1_by_player = metrics_t1["final_cum_regrets"]
            regrets_t_by_player = metrics_t["final_cum_regrets"]

            sum_regrets_t1 = np.zeros(B)
            sum_regrets_t = np.zeros(B)
            avg_regrets_t = np.zeros(B)

            for i in range(self.num_players):
                sum_regrets_t1 += np.array(regrets_t1_by_player[i])
                sum_regrets_t += np.array(regrets_t_by_player[i])
                avg_regrets_t += np.array(metrics_t["final_avg_regrets"][i])

            delta = sum_regrets_t - sum_regrets_t1
            # Minimize negative fitness
            fitnesses = -(delta + self.lambda_reg * np.log(np.maximum(sum_regrets_t, 0.0) + 1e-8))
            report_regrets = avg_regrets_t

        else:
            metrics = runner.run(target_steps=T)
            regrets_by_player = metrics["final_cum_regrets"]

            sum_regrets = np.zeros(B)
            avg_regrets_t = np.zeros(B)
            for i in range(self.num_players):
                sum_regrets += np.array(regrets_by_player[i])
                avg_regrets_t += np.array(metrics["final_avg_regrets"][i])

            fitnesses = -sum_regrets  # Negative because CMA-ES minimizes
            report_regrets = avg_regrets_t

        # Tell
        solutions_with_fitness = [(solutions_flat[b], fitnesses[b]) for b in range(B)]
        self.optimizer.tell(solutions_with_fitness)

        # Best in this generation
        best_idx = np.argmin(fitnesses)
        return solutions_flat[best_idx], report_regrets[best_idx]

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
