import copy
import os
import uuid

import numpy as np
import torch
from cmaes import CMA

from src.config.schemas import ExperimentConfig
from src.engine.runner import ExperimentRunner
from src.utils.logging import setup_logger

logger = setup_logger("cmaes_optimizer")


class CMAESGameOptimizer:
    """Finds the worst-case (highest regret) payoff matrices for a given learning dynamic using CMA-ES.

    This optimizer uses the lightweight `cmaes` package (from Optuna) and leverages the
    native batched GPU execution in `ExperimentRunner` to evaluate an entire generation
    of games simultaneously on the GPU.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.sigma = self.base_config.cmaes.sigma
        self.seed = self.base_config.cmaes.seed
        self.objective_type = self.base_config.cmaes.objective_type
        self.T1_ratio = self.base_config.cmaes.T1_ratio
        self.lambda_reg = self.base_config.cmaes.lambda_reg
        self.session_id = self.base_config.session_id or uuid.uuid4().hex[:8]
        self.base_config.session_id = self.session_id
        np.random.seed(self.seed)

        # Determine shapes and initial mean
        if self.base_config.game.generator == "custom" and self.base_config.game.payoffs is not None:
            self.payoff_shapes = [np.array(p).shape for p in self.base_config.game.payoffs]
            self.num_players = len(self.payoff_shapes)
            self.dim = sum(np.prod(s) for s in self.payoff_shapes)
            # Center CMA-ES at the provided custom payoffs
            initial_mean = np.concatenate([np.array(p).flatten() for p in self.base_config.game.payoffs])
        else:
            self.num_players = self.base_config.game.num_players
            sizes = self.base_config.game.action_sizes
            # In an N-player game, every player's payoff tensor has shape (A_1, A_2, ..., A_N)
            self.payoff_shapes = [tuple(sizes) for _ in range(self.num_players)]
            self.dim = sum(np.prod(s) for s in self.payoff_shapes)
            # Center CMA-ES at 0 for random explorations
            initial_mean = np.zeros(self.dim)

        # CMA-ES instance
        # We start centered at initial_mean with the specified sigma
        self.initial_mean = initial_mean
        self._init_cma(self.base_config.cmaes.population_size)

    def _init_cma(self, pop_size: int | None = None):
        """Initializes or restarts the CMA optimizer with the given population size."""
        # Explicitly destroy the old runner and clear PyTorch's JIT compiler cache 
        # to prevent CUDAGraphs/Inductor memory leaks from the old batch size.
        if hasattr(self, 'runner'):
            del self.runner
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.compiler, 'reset'):
                torch.compiler.reset()

        self.optimizer = CMA(mean=self.initial_mean, sigma=self.sigma, seed=self.seed, population_size=pop_size)
        self.population_size = self.optimizer.population_size
        self.base_config.execution.batch_size = self.population_size
        self.base_config.execution.quiet = True
        
        # Instantiate the ExperimentRunner ONCE for this population size.
        # By re-using this runner in evaluate_population, PyTorch can cache and reuse the CUDAGraph,
        # completely bypassing the massive torch.compile warmup overhead in every generation.
        import copy
        import uuid
        config = copy.deepcopy(self.base_config)
        config.parent_session_id = self.base_config.session_id
        config.session_id = uuid.uuid4().hex[:8]
        config.game.generator = "custom"
        config.logging.save_stats_interval = 999999999
        config.checkpoint.enabled = False
        # Pre-fill with zeros of the correct batch size
        dummy_flat = np.zeros((self.population_size, self.dim))
        config.game.payoffs = self._unflatten_payoffs(dummy_flat)
        self.runner = ExperimentRunner(config)

    def _unflatten_payoffs(self, flat_batch: np.ndarray) -> list[torch.Tensor]:
        """Convert a (population_size, dim) numpy array into a list of batched payoff tensors."""
        B = flat_batch.shape[0]
        payoffs = []
        offset = 0
        for shape in self.payoff_shapes:
            size = np.prod(shape)
            # Slice out this player's parameters and reshape to (B, *shape)
            p_flat = flat_batch[:, offset : offset + size]
            p_tensor = torch.tensor(p_flat, dtype=torch.get_default_dtype()).reshape(B, *shape)

            # Clip payoffs to utility_range if specified
            if self.base_config.game.utility_range is not None:
                u_min, u_max = self.base_config.game.utility_range
                p_tensor = torch.clamp(p_tensor, u_min, u_max)

            payoffs.append(p_tensor)
            offset += size

        return payoffs

    def step(self, progress_context=None) -> tuple[np.ndarray, dict]:
        """Perform a single CMA-ES step (one generation).

        Returns
        -------
        Tuple[np.ndarray, dict]
            The best solution in this generation and its corresponding objective factors.
        """
        # Ask for population_size solutions
        solutions_flat = np.array([self.optimizer.ask() for _ in range(self.population_size)])

        # Convert to batched payoff tensors
        batched_payoffs = self._unflatten_payoffs(solutions_flat)

        # Instead of instantiating a new ExperimentRunner (which would trigger torch.compile),
        # we reset the cached runner and copy the new payoffs directly into the existing GPU tensors.
        self.runner.reset()
        self.runner.game.update_payoffs(batched_payoffs)

        T = self.base_config.execution.total_steps
        B = self.population_size

        if self.objective_type == "delta_reg":
            T1 = int(self.T1_ratio * T)
            metrics_t1 = self.runner.run(target_steps=T1, progress_context=progress_context)
            metrics_t = self.runner.run(target_steps=T, window_ranges=[(T1, T)], progress_context=progress_context)

            regrets_t1_by_player = metrics_t1["final_cum_regrets"]
            regrets_t_by_player = metrics_t["final_cum_regrets"]

            p1_regrets_t1 = np.array(regrets_t1_by_player[0])
            p1_regrets_t = np.array(regrets_t_by_player[0])

            delta = p1_regrets_t - p1_regrets_t1
            volatility = metrics_t["window_volatilities"][0]
            
            # Minimize negative fitness
            fitnesses = -(delta + self.lambda_reg * np.log(np.maximum(p1_regrets_t, 0.0) + 1e-8) + self.base_config.cmaes.gamma_volatility * volatility)
            report_regrets = p1_regrets_t
            
        elif self.objective_type in ["envelope_trend", "envelope_trend_log"]:
            w_past = (int((self.T1_ratio - 0.2) * T), int(self.T1_ratio * T))
            w_future = (int(self.T1_ratio * T), T)
            metrics_t = self.runner.run(target_steps=T, window_ranges=[w_past, w_future], progress_context=progress_context)
            peak_past = metrics_t["window_maxes"][0]
            peak_future = metrics_t["window_maxes"][1]
            
            if peak_past is None: peak_past = np.zeros(B)
            if peak_future is None: peak_future = np.zeros(B)
            
            delta_trend = peak_future - peak_past
            volatility = metrics_t["window_volatilities"][1]
            
            if self.objective_type == "envelope_trend_log":
                # softplus(x) = log(1 + exp(x))
                softplus_peak = np.logaddexp(0, peak_future)
                fitnesses = -(delta_trend + self.lambda_reg * np.log(softplus_peak + 1e-8) + self.base_config.cmaes.gamma_volatility * volatility)
            else:
                fitnesses = -(self.lambda_reg * peak_future + delta_trend + self.base_config.cmaes.gamma_volatility * volatility)
                
            report_regrets = peak_future
            delta = delta_trend

        elif self.objective_type in ["chunked_envelope_trend", "chunked_envelope_trend_log"]:
            num_chunks = self.base_config.cmaes.num_chunks
            eval_start = int(self.T1_ratio * T)
            eval_length = T - eval_start
            chunk_size = eval_length // num_chunks
            
            window_ranges = []
            for i in range(num_chunks):
                w_start = eval_start + i * chunk_size
                w_end = eval_start + (i + 1) * chunk_size
                if i == num_chunks - 1:
                    w_end = eval_start + num_chunks * chunk_size
                window_ranges.append((w_start, w_end))
                
            metrics_t = self.runner.run(target_steps=T, window_ranges=window_ranges, progress_context=progress_context)
            
            # Extract envelope peaks (num_chunks, B)
            envelope_peaks = np.array([metrics_t["window_maxes"][i] for i in range(num_chunks)])
            
            # Linear Regression
            x = np.linspace(0, 1, num=num_chunks)
            x_mean = np.mean(x)
            y_mean = np.mean(envelope_peaks, axis=0) # shape: (B,)
            
            x_diff = x - x_mean
            y_diff = envelope_peaks - y_mean
            
            numerator = np.sum(x_diff[:, None] * y_diff, axis=0)
            denominator = np.sum(x_diff**2)
            
            delta_trend = numerator / denominator
            base_regret = np.mean(envelope_peaks, axis=0)
            
            # Use mean volatility across all chunks
            volatility = np.mean([metrics_t["window_volatilities"][i] for i in range(num_chunks)], axis=0)
            
            if self.objective_type == "chunked_envelope_trend_log":
                softplus_base = np.logaddexp(0, base_regret)
                fitnesses = -(delta_trend + self.lambda_reg * np.log(softplus_base + 1e-8) + self.base_config.cmaes.gamma_volatility * volatility)
            else:
                fitnesses = -(self.lambda_reg * base_regret + delta_trend + self.base_config.cmaes.gamma_volatility * volatility)
                
            report_regrets = base_regret
            delta = delta_trend

        else:
            w_future = (int(self.T1_ratio * T), T)
            metrics_t = self.runner.run(target_steps=T, window_ranges=[w_future], progress_context=progress_context)
            regrets_by_player = metrics_t["final_cum_regrets"]
            volatility = metrics_t["window_volatilities"][0]

            p1_regrets = np.array(regrets_by_player[0])

            fitnesses = -(p1_regrets + self.base_config.cmaes.gamma_volatility * volatility)
            report_regrets = p1_regrets

        # Extract logit penalty if enabled
        penalty_array = np.zeros_like(fitnesses)
        if self.base_config.cmaes.logit_penalty_weight > 0.0 and "logit_penalty" in metrics_t:
            pen_tensor = metrics_t["logit_penalty"]
            if self.base_config.cmaes.logit_penalty_average:
                pen_tensor = pen_tensor / T
            penalty_array = pen_tensor.cpu().numpy() * self.base_config.cmaes.logit_penalty_weight
            # Penalty makes fitness higher, causing CMA-ES to avoid it
            fitnesses += penalty_array

        # Tell
        solutions_with_fitness = [(solutions_flat[b], fitnesses[b]) for b in range(B)]
        self.optimizer.tell(solutions_with_fitness)

        # Best in this generation
        best_idx = np.argmin(fitnesses)
        
        # Extract raw volatility
        raw_volatility = volatility[best_idx] if 'volatility' in locals() else 0.0
        
        factors = {
            "regret": report_regrets[best_idx],
            "penalty": penalty_array[best_idx],
            "volatility": raw_volatility,
            "volatility_penalty": self.base_config.cmaes.gamma_volatility * raw_volatility,
            "fitness": -fitnesses[best_idx]
        }
        if self.objective_type in ["delta_reg", "envelope_trend_log", "chunked_envelope_trend_log"]:
            factors["delta"] = delta[best_idx]
            factors["log_reg"] = self.lambda_reg * np.log(np.maximum(report_regrets[best_idx], 0.0) + 1e-8)
        elif self.objective_type in ["envelope_trend", "chunked_envelope_trend"]:
            factors["delta"] = delta[best_idx]
            factors["peak_weight"] = self.lambda_reg * report_regrets[best_idx]
            
        return solutions_flat[best_idx], factors

    def save_results(self, best_payoffs: list[torch.Tensor], best_regret: float, out_dir: str = "outputs") -> str:
        """Save the best payoff matrices and their regret to disk."""
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"cmaes_best_{self.session_id}.pt")
        torch.save({
            "payoffs": best_payoffs,
            "regret": best_regret,
            "config": self.base_config.model_dump()
        }, save_path)
        logger.info(f"Saved CMA-ES best results to {save_path}")
        return save_path

    @classmethod
    def load_results(cls, session_id: str, out_dir: str = "outputs") -> dict:
        """Load previously saved CMA-ES best payoff matrices."""
        save_path = os.path.join(out_dir, f"cmaes_best_{session_id}.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No CMA-ES results found at {save_path}")
        return torch.load(save_path, weights_only=False)

    def optimize(self, generations: int | None = None) -> tuple[list[torch.Tensor], dict]:
        """Run CMA-ES for the specified number of generations with IPOP restarts.

        Returns
        -------
        Tuple[List[torch.Tensor], dict]
            The best payoff matrices found and their corresponding objective factors.
        """
        from collections import deque
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
        from rich.console import Console
        
        console = Console()
        
        best_fitness = -float("inf")
        best_sol = None
        best_factors = None

        config_maxiter = generations if generations is not None else self.base_config.cmaes.maxiter
        maxfevals = self.base_config.cmaes.maxfevals
        tolfun = self.base_config.cmaes.tolfun
        max_restarts = self.base_config.cmaes.restarts
        tolfun_hist_len = self.base_config.cmaes.tolfun_hist

        evals = 0
        restarts = 0

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[status]}"),
            TextColumn("{task.fields[postfix]}"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
            disable=False
        ) as progress:
            task_restart = progress.add_task(f"[bold magenta]IPOP Restarts", total=max_restarts + 1, status="", postfix="")
            task_gen = progress.add_task(f"[bold blue]Generations", total=config_maxiter, status="Best Regret: N/A", postfix="")

            while restarts <= max_restarts and evals < maxfevals:
                fitness_history = deque(maxlen=tolfun_hist_len)
                
                # Reset generation bar
                progress.update(task_gen, completed=0, total=config_maxiter)
                if restarts > 0:
                    progress.console.print(f"[bold magenta]=== IPOP Restart {restarts}/{max_restarts} (Pop size: {self.population_size}) ===[/bold magenta]")

                for g in range(config_maxiter):
                    if evals >= maxfevals:
                        progress.console.print(f"[yellow]Budget of {maxfevals} evaluations exhausted.[/yellow]")
                        break
                        
                    sol, factors = self.step(progress_context=progress)
                    evals += self.population_size
                    
                    if factors["fitness"] > best_fitness:
                        best_fitness = factors["fitness"]
                        best_sol = sol
                        best_factors = factors

                    fitness_history.append(factors["fitness"])

                    # Format status text
                    status = f"Global Best: {best_factors['fitness']:.3f} | Local: {factors['fitness']:.3f}"
                    postfix = f"Local Reg: {factors['regret']:.3f}"
                    if "delta" in factors:
                        postfix += f" | d: {factors['delta']:.3f}"
                    if "log_reg" in factors:
                        postfix += f" | lreg: {factors['log_reg']:.3f}"
                    if factors["penalty"] > 0:
                        postfix += f" | Pen: {factors['penalty']:.3f}"
                        
                    progress.update(task_gen, advance=1, status=status, postfix=postfix)

                    # Native CMA-ES early stopping
                    if self.optimizer.should_stop():
                        progress.console.print(f"[yellow]Early stopping triggered by CMA-ES native criteria at generation {g+1}.[/yellow]")
                        progress.update(task_gen, completed=config_maxiter)
                        break
                        
                    # Tolfun early stopping
                    if len(fitness_history) == tolfun_hist_len:
                        if max(fitness_history) - min(fitness_history) < tolfun:
                            progress.console.print(f"[yellow]Flat fitness (tolfun={tolfun}) detected at generation {g+1}.[/yellow]")
                            progress.update(task_gen, completed=config_maxiter)
                            break

                if evals >= maxfevals:
                    break

                progress.update(task_restart, advance=1)
                restarts += 1
                if restarts <= max_restarts:
                    # Double population size for IPOP restart
                    new_pop = self.population_size * 2
                    self._init_cma(new_pop)
                    # Advance seed for restart to avoid identical trajectory
                    self.seed += 1

        # Save to disk at the end
        if best_sol is not None:
            batched_best = self._unflatten_payoffs(np.array([best_sol]))
            unbatched_best = [p[0] for p in batched_best]
            self.save_results(unbatched_best, best_factors["regret"])
            
            try:
                from src.utils.tracking import ExperimentTracker
                tracker = ExperimentTracker(out_dir=self.base_config.logging.output_dir)
                tracker.log_run(
                    config=self.base_config,
                    run_type="cmaes",
                    metrics=best_factors,
                    parent_session_id=None
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to log CMA-ES run to tracker: {e}")
        
        # Final cleanup to ensure the environment is pristine for any subsequent simulations
        if hasattr(self, 'runner'):
            del self.runner
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.compiler, 'reset'):
                torch.compiler.reset()
        
        return unbatched_best, best_factors
