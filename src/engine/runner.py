"""Master experiment runner for executing and checkpointing learning dynamics."""

import random
import uuid
from typing import Any

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config.schemas import ExperimentConfig
from src.config.validation import validate_experiment_config
from src.dynamics.base import BaseLearningDynamic
from src.dynamics.dmwu import DMWU
from src.dynamics.mirror_prox import MirrorProx
from src.dynamics.mwu import MultiplicativeWeightsUpdate
from src.dynamics.omwu import OptimisticMWU
from src.engine.checkpoint import CheckpointManager
from src.engine.statistics import StatsCollector
from src.games.base import BaseGame
from src.games.generators import (
    create_matching_pennies,
    create_prisoners_dilemma,
    create_random_game,
    create_rock_paper_scissors,
    create_shapley_game,
)
from src.games.matrix_game import MatrixGame
from src.games.nplayer_game import NPlayerGame
from src.metrics.regret import (
    compute_average_regret,
    compute_cumulative_regret,
    compute_step_metrics,
)
from src.utils.device import enable_gpu_optimizations, get_device, setup_dtype
from src.utils.logging import console, setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("runner")


def instantiate_game(config: ExperimentConfig, device: torch.device) -> BaseGame:
    """Instantiate game environment from config settings."""
    gen = config.game.generator.lower()
    u_range = config.game.utility_range

    if gen == "matching_pennies":
        return create_matching_pennies(utility_range=u_range, device=device)
    elif gen == "prisoners_dilemma":
        return create_prisoners_dilemma(utility_range=u_range, device=device)
    elif gen == "rock_paper_scissors":
        return create_rock_paper_scissors(utility_range=u_range, device=device)
    elif gen == "shapley":
        return create_shapley_game(utility_range=u_range, device=device)
    elif gen == "random":
        return create_random_game(
            num_players=config.game.num_players,
            action_sizes=config.game.action_sizes,
            utility_range=u_range,
            seed=config.game.seed,
            device=device,
        )
    elif config.game.payoffs is not None:
        payoffs = [
            p.clone().detach().to(dtype=torch.get_default_dtype())
            if isinstance(p, torch.Tensor)
            else torch.tensor(p, dtype=torch.get_default_dtype())
            for p in config.game.payoffs
        ]
        if config.game.num_players == 2 and len(payoffs) == 2 and payoffs[0].dim() in [2, 3]:
            return MatrixGame(payoffs[0], payoffs[1], utility_range=u_range, device=device)
        return NPlayerGame(payoffs, utility_range=u_range, device=device)
    else:
        raise ValueError(f"Unknown game generator or configuration: {config.game.generator}")


def instantiate_dynamic(
    config: ExperimentConfig, action_sizes: list[int], device: torch.device
) -> BaseLearningDynamic:
    """Instantiate learning dynamic algorithm from config."""
    algo = config.dynamic.algorithm.lower()
    eta = config.dynamic.eta
    batch_size = config.execution.batch_size

    if algo == "omwu":
        return OptimisticMWU(
            action_sizes=action_sizes,
            eta=eta,
            device=device,
            batch_size=batch_size,
            T=config.execution.total_steps,
            strict_theory_eta=config.dynamic.strict_theory_eta,
            logit_penalty_threshold=config.dynamic.logit_penalty_threshold,
            logit_penalty_norm=config.dynamic.logit_penalty_norm,
        )
    elif algo == "mwu":
        return MultiplicativeWeightsUpdate(
            action_sizes=action_sizes,
            eta=eta,
            device=device,
            batch_size=batch_size,
            T=config.execution.total_steps,
            logit_penalty_threshold=config.dynamic.logit_penalty_threshold,
            logit_penalty_norm=config.dynamic.logit_penalty_norm,
        )
    elif algo == "mirror_prox":
        return MirrorProx(
            action_sizes=action_sizes,
            eta=eta,
            device=device,
            batch_size=batch_size,
            T=config.execution.total_steps,
            strict_theory_eta=config.dynamic.strict_theory_eta,
        )
    elif algo == "dmwu":
        return DMWU(
            action_sizes=action_sizes,
            eta=eta,
            device=device,
            batch_size=batch_size,
            T=config.execution.total_steps,
            dmwu_gamma=config.dynamic.dmwu_gamma,
            logit_penalty_threshold=config.dynamic.logit_penalty_threshold,
            logit_penalty_norm=config.dynamic.logit_penalty_norm,
            logit_penalty_mode=config.dynamic.logit_penalty_mode,
        )
    else:
        raise ValueError(f"Unsupported learning dynamic algorithm: '{algo}'")


class ExperimentRunner:
    """Executes long-horizon learning dynamics simulation with statistics tracking and checkpointing."""

    def __init__(
        self,
        config: ExperimentConfig,
        resume_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize runner.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration container.
        resume_checkpoint_path : Optional[str]
            Path to .pt checkpoint file to resume from.
        """
        validate_experiment_config(config)
        self.config = config
        setup_dtype(config.execution.dtype)
        self.device = get_device(config.execution.device)
        enable_gpu_optimizations(self.device, fp32_precision=config.execution.fp32_precision)
        set_seed(config.execution.seed)

        self.session_id = config.session_id or str(uuid.uuid4())
        self.config.session_id = self.session_id

        self.game = instantiate_game(config, self.device)
        self.dynamic = instantiate_dynamic(config, self.game.action_sizes, self.device)

        if config.execution.compile:
            try:
                import sys

                if hasattr(torch, "_dynamo"):
                    torch._dynamo.config.suppress_errors = True
                    torch._dynamo.config.cache_size_limit = 64
                backend = (
                    "cudagraphs"
                    if (sys.platform == "win32" and self.device.type == "cuda")
                    else "inductor"
                )
                if not config.execution.quiet:
                    logger.info(
                        f"Enabling PyTorch JIT compilation via torch.compile(backend='{backend}')..."
                    )
                self.dynamic.step_unrolled_block = torch.compile(
                    self.dynamic.step_unrolled_block, backend=backend, dynamic=False
                )
            except Exception as e:
                logger.warning(
                    f"torch.compile failed to initialize (falling back to eager mode): {e}"
                )

        # Parse initial strategies based on config
        initial_strats = None
        strat_mode = config.dynamic.initial_strategy_type.lower()
        if strat_mode == "random":
            initial_strats = []
            generator = torch.Generator(device=self.device)
            if self.config.execution.seed is not None:
                generator.manual_seed(self.config.execution.seed)
            for a_size in self.game.action_sizes:
                r = torch.rand(a_size, generator=generator, device=self.device, dtype=torch.get_default_dtype())
                initial_strats.append(r / r.sum())
        elif strat_mode == "custom" and config.dynamic.custom_initial_strategies:
            initial_strats = [
                torch.tensor(s, device=self.device, dtype=torch.get_default_dtype())
                for s in config.dynamic.custom_initial_strategies
            ]

        self.dynamic.reset(initial_strategies=initial_strats)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint.checkpoint_dir,
            keep_top_k=config.checkpoint.keep_top_k,
            session_id=self.session_id,
        )

        self.stats_collector = StatsCollector(
            output_dir=config.logging.output_dir,
            session_id=self.session_id,
        )
        self.stats_collector.set_payoffs(self.game.get_payoff_tensors())

        self.start_step = 0
        self.max_action_size = max(self.game.action_sizes)
        self.stacked_cumulative_utility_vectors = torch.zeros(
            (self.config.execution.batch_size, self.game.num_players, self.max_action_size),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        self.cumulative_actual_payoffs = torch.zeros(
            (self.config.execution.batch_size, self.game.num_players),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )

        # Pre-allocate GPU history tensors for the unrolled block
        self.hist_strats = torch.zeros(
            (self.config.execution.steps_per_call, self.config.execution.batch_size, self.game.num_players, self.max_action_size),
            device=self.device, dtype=torch.get_default_dtype()
        )
        self.hist_logits = torch.zeros_like(self.hist_strats)
        self.hist_stacked_u = torch.zeros_like(self.hist_strats)
        self.hist_cum_u = torch.zeros_like(self.hist_strats)
        self.hist_cum_p = torch.zeros(
            (self.config.execution.steps_per_call, self.config.execution.batch_size, self.game.num_players),
            device=self.device, dtype=torch.get_default_dtype()
        )

        if resume_checkpoint_path:
            self._resume_from_checkpoint(resume_checkpoint_path)

    def reset(self) -> None:
        """Reset the runner and its simulation states for a new run in-place without triggering recompilation."""
        # Reset dynamic states
        initial_strats = None
        strat_mode = self.config.dynamic.initial_strategy_type.lower()
        if strat_mode == "random":
            initial_strats = []
            generator = torch.Generator(device=self.device)
            if self.config.execution.seed is not None:
                generator.manual_seed(self.config.execution.seed)
            for a_size in self.game.action_sizes:
                r = torch.rand(a_size, generator=generator, device=self.device, dtype=torch.get_default_dtype())
                initial_strats.append(r / r.sum())
        elif strat_mode == "custom" and self.config.dynamic.custom_initial_strategies:
            initial_strats = [
                torch.tensor(s, device=self.device, dtype=torch.get_default_dtype())
                for s in self.config.dynamic.custom_initial_strategies
            ]
        self.dynamic.reset(initial_strategies=initial_strats)
        
        # Zero out accumulated payoffs/utilities
        self.stacked_cumulative_utility_vectors.zero_()
        self.cumulative_actual_payoffs.zero_()
        
        # We don't necessarily need to zero hist arrays since they are overwritten,
        # but zeroing them avoids any potential leakage to stats collector if steps don't align.
        self.hist_strats.zero_()
        self.hist_logits.zero_()
        self.hist_stacked_u.zero_()
        self.hist_cum_u.zero_()
        self.hist_cum_p.zero_()
        
        # Reset step counters and stats collector
        self.start_step = 0
        
        # Re-initialize the stats collector cleanly
        self.stats_collector = StatsCollector(
            output_dir=self.config.logging.output_dir,
            session_id=self.session_id,
        )
        self.stats_collector.set_payoffs(self.game.get_payoff_tensors())

    @property
    def cumulative_utility_vectors(self) -> list[torch.Tensor]:
        """Return cumulative utility vectors as a list of 1D/2D tensors [U^1, ..., U^N]."""
        if self.config.execution.batch_size == 1:
            return [
                self.stacked_cumulative_utility_vectors[0, i, : self.game.action_sizes[i]]
                for i in range(self.game.num_players)
            ]
        return [
            self.stacked_cumulative_utility_vectors[:, i, : self.game.action_sizes[i]]
            for i in range(self.game.num_players)
        ]

    def _resume_from_checkpoint(self, filepath: str) -> None:
        """Resume experiment from checkpoint file."""
        data = self.checkpoint_manager.load(filepath)
        self.start_step = data["step"]
        self.session_id = data.get("session_id", self.session_id)
        self.dynamic.load_state(data["dynamic_state"])

        self.stacked_cumulative_utility_vectors.zero_()
        for i, u in enumerate(data["cumulative_utility_vectors"]):
            u_t = u.to(self.device)
            if u_t.dim() == 1:
                if self.config.execution.batch_size == 1:
                    self.stacked_cumulative_utility_vectors[0, i, : self.game.action_sizes[i]] = u_t
                else:
                    self.stacked_cumulative_utility_vectors[:, i, : self.game.action_sizes[i]] = (
                        u_t.unsqueeze(0).expand(self.config.execution.batch_size, -1)
                    )
            else:
                self.stacked_cumulative_utility_vectors[:, i, : self.game.action_sizes[i]] = u_t

        self.cumulative_actual_payoffs.zero_()
        if "cumulative_actual_payoffs" in data:
            if isinstance(data["cumulative_actual_payoffs"], list):
                loaded_tensor = torch.tensor(
                    data["cumulative_actual_payoffs"], device=self.device, dtype=torch.get_default_dtype()
                )
                if loaded_tensor.dim() == 1:
                    if self.config.execution.batch_size == 1:
                        self.cumulative_actual_payoffs[0] = loaded_tensor
                    else:
                        self.cumulative_actual_payoffs[:] = loaded_tensor.unsqueeze(0)
                else:
                    self.cumulative_actual_payoffs.copy_(loaded_tensor)
            else:
                self.cumulative_actual_payoffs.copy_(
                    data["cumulative_actual_payoffs"].to(self.device)
                )

        rng_state = data.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        self.last_ckpt = self.start_step
        logger.info(f"Resumed experiment '{self.config.name}' from step {self.start_step}")

    def run(self, target_steps: int | None = None, window_ranges: list[tuple[int, int]] | None = None, progress_context: Progress | None = None) -> dict[str, Any]:
        """Run simulation loop from current step to target_steps.

        Parameters
        ----------
        target_steps : int | None, optional
            Step to run up to. If None, uses config.execution.total_steps.
        window_ranges : list[tuple[int, int]] | None, optional
            List of (start_step, end_step) tuples. The runner will track and return the max cumulative regret inside each window.
        progress_context : Progress | None, optional
            Optional rich progress context for nested task rendering.

        Returns
        -------
        Dict[str, Any]
            Final metrics summary dictionary.
        """
        total_steps = target_steps if target_steps is not None else self.config.execution.total_steps
        if not self.config.execution.quiet:
            logger.info(
                f"Starting simulation '{self.config.name}' [Session: {self.session_id}] "
                f"on device '{self.device}' for T={total_steps} steps."
            )

        steps_per_call = max(1, self.config.execution.steps_per_call)
        window_maxes = {i: None for i in range(len(window_ranges))} if window_ranges else {}
        window_strat_maxes = {i: None for i in range(len(window_ranges))} if window_ranges else {}
        window_strat_mins = {i: None for i in range(len(window_ranges))} if window_ranges else {}
        record_history = not self.config.execution.quiet
        def _execute_loop(progress=None, task_id=None):
            step = self.start_step
            while step < total_steps:
                target_step = min(step + steps_per_call, total_steps)
                
                # Check if we need to checkpoint inside this unrolled block
                if self.config.checkpoint.enabled:
                    next_ckpt = (step // self.config.checkpoint.save_interval + 1) * self.config.checkpoint.save_interval
                    target_step = min(target_step, next_ckpt)
                
                k_steps = target_step - step
                
                if k_steps <= 0:
                    break
                
                with torch.no_grad():
                    if not record_history:
                        self.dynamic.step_unrolled_block(
                            game=self.game,
                            cum_u_2d=self.stacked_cumulative_utility_vectors,
                            cum_p_1d=self.cumulative_actual_payoffs,
                            k_steps=k_steps,
                        )
                    else:
                        self.dynamic.step_unrolled_block(
                            game=self.game,
                            cum_u_2d=self.stacked_cumulative_utility_vectors,
                            cum_p_1d=self.cumulative_actual_payoffs,
                            k_steps=k_steps,
                            hist_strats=self.hist_strats,
                            hist_logits=self.hist_logits,
                            hist_stacked_u=self.hist_stacked_u,
                            hist_cum_u=self.hist_cum_u,
                            hist_cum_p=self.hist_cum_p,
                        )
                
                step += k_steps
                
                if window_ranges is not None:
                    current_u = self.cumulative_utility_vectors
                    current_p = self.cumulative_actual_payoffs.T.tolist()
                    if self.config.execution.batch_size == 1:
                        current_p = [l[0] for l in current_p]
                    current_regret = compute_cumulative_regret(current_u, current_p)
                    p1_regret = current_regret[0]
                    if isinstance(p1_regret, list):
                        p1_regret = np.array(p1_regret)
                    else:
                        p1_regret = np.array([p1_regret])
                        
                    current_strat = self.dynamic.get_state()["strategies"]
                    # shape is usually list of (B, A)
                    # We can stack them to (B, num_players, max_A) if needed, but it's easier to just use a numpy array of whatever it is.
                    # Actually, self.dynamic.strategies might just be a list of tensors.
                    # Wait, let's just stack the strategies into a single numpy array if possible.
                    # Or we can just grab self.dynamic.strategies (it's a list) and concatenate them along the action dimension.
                    current_strat_np = np.concatenate([s.cpu().numpy() for s in current_strat], axis=-1)
                    
                    for w_idx, (w_start, w_end) in enumerate(window_ranges):
                        if w_start <= step <= w_end:
                            if window_maxes[w_idx] is None:
                                window_maxes[w_idx] = p1_regret
                                window_strat_maxes[w_idx] = current_strat_np
                                window_strat_mins[w_idx] = current_strat_np
                            else:
                                window_maxes[w_idx] = np.maximum(window_maxes[w_idx], p1_regret)
                                window_strat_maxes[w_idx] = np.maximum(window_strat_maxes[w_idx], current_strat_np)
                                window_strat_mins[w_idx] = np.minimum(window_strat_mins[w_idx], current_strat_np)
                
                if progress is not None:
                    progress.update(task_id, advance=k_steps)
                    
                is_log_step = step % self.config.logging.log_interval == 0
                is_flush_step = step % self.config.logging.save_stats_interval == 0
                
                if record_history:
                    s_int = self.config.logging.sample_interval
                    start_step = step - k_steps + 1
                    first_idx = (s_int - (start_step % s_int)) % s_int
                    indices = torch.arange(first_idx, k_steps, s_int)
                    
                    if len(indices) > 0:
                        cpu_strats = self.hist_strats[indices].cpu()
                        cpu_logits = self.hist_logits[indices].cpu() if hasattr(self.dynamic, "log_strategies") else None
                        cpu_u = self.hist_stacked_u[indices].cpu()
                        cpu_cum_u = self.hist_cum_u[indices].cpu()
                        cpu_cum_p = self.hist_cum_p[indices].cpu()
                        
                        actual_steps = [start_step + idx.item() for idx in indices]
                        self.stats_collector.record_batch(
                            steps=actual_steps,
                            strats=cpu_strats,
                            logits=cpu_logits,
                            u_vecs=cpu_u,
                            cum_u=cpu_cum_u,
                            cum_p=cpu_cum_p,
                            action_sizes=self.game.action_sizes
                        )

                    if is_log_step and progress is not None:
                        last_cum = self.stats_collector.history_cum_regrets[0][-1]
                        if isinstance(last_cum, (torch.Tensor, np.ndarray)):
                            flat_cum = last_cum.flatten().tolist()
                        elif isinstance(last_cum, list):
                            flat_cum = last_cum
                        else:
                            flat_cum = [float(last_cum)]
                            
                        flat_avg = [c / step for c in flat_cum]
                        postfix = f"max_cum_regret={max(flat_cum):.4f} max_avg_regret={max(flat_avg):.4f}"
                        if getattr(self, "last_ckpt", None) is not None:
                            postfix += f" | ckpt={self.last_ckpt}"
                        progress.update(task_id, postfix=postfix)
                else:
                    if is_log_step and progress is not None:
                        current_u = self.cumulative_utility_vectors
                        current_p = self.cumulative_actual_payoffs.T.tolist()
                        if self.config.execution.batch_size == 1:
                            current_p = [l[0] for l in current_p]
                        current_regret = compute_cumulative_regret(current_u, current_p)[0]
                        flat_cum = current_regret if isinstance(current_regret, list) else (current_regret.flatten().tolist() if isinstance(current_regret, (torch.Tensor, np.ndarray)) else [float(current_regret)])
                        flat_avg = [c / step for c in flat_cum]
                        postfix = f"max_cum_regret={max(flat_cum):.4f} max_avg_regret={max(flat_avg):.4f}"
                        if getattr(self, "last_ckpt", None) is not None:
                            postfix += f" | ckpt={self.last_ckpt}"
                        progress.update(task_id, postfix=postfix)

                    if is_flush_step or (step == total_steps):
                        self.stats_collector.flush_to_disk()

                is_ckpt_step = self.config.checkpoint.enabled and step % self.config.checkpoint.save_interval == 0
                if is_ckpt_step and step > getattr(self, "last_ckpt", 0):
                    rng_state = {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                    }
                    self.checkpoint_manager.save(
                        step=step,
                        config_dict=self.config.model_dump(),
                        dynamic_state=self.dynamic.get_state(),
                        rng_state=rng_state,
                        cumulative_utility_vectors=self.cumulative_utility_vectors,
                        cumulative_actual_payoffs=self.cumulative_actual_payoffs,
                    )
                    self.last_ckpt = step

        if progress_context is not None:
            task_id = progress_context.add_task(
                f"[yellow]Simulation ({self.config.name})",
                total=total_steps,
                completed=self.start_step,
                status="",
                postfix=""
            )
            _execute_loop(progress=progress_context, task_id=task_id)
            progress_context.remove_task(task_id)
        elif self.config.execution.quiet:
            _execute_loop()
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[postfix]}"),
                console=console,
                refresh_per_second=10,
            ) as progress:
                task_id = progress.add_task(
                    f"Running {self.config.dynamic.algorithm.upper()}",
                    total=total_steps,
                    completed=self.start_step,
                    postfix="",
                )
                _execute_loop(progress=progress, task_id=task_id)

        # Final flush & summary
        last_chunk_path = self.stats_collector.flush_to_disk()
        # Compute final end-of-run exact regrets
        final_cum_actual_payoffs_list = self.cumulative_actual_payoffs.T.tolist()
        if self.config.execution.batch_size == 1:
            final_cum_actual_payoffs_list = [l[0] for l in final_cum_actual_payoffs_list]
        final_cum_regrets = compute_cumulative_regret(
            self.cumulative_utility_vectors, final_cum_actual_payoffs_list
        )
        final_avg_regrets = compute_average_regret(final_cum_regrets, total_steps)

        # Update start_step so we can call run() multiple times sequentially
        self.start_step = total_steps

        summary = {
            "session_id": self.session_id,
            "device": str(self.device),
            "total_steps": total_steps,
            "final_cum_regrets": final_cum_regrets,
            "final_avg_regrets": final_avg_regrets,
            "output_dir": self.config.logging.output_dir,
            "last_chunk_file": last_chunk_path,
        }
        
        if window_ranges is not None:
            summary["window_maxes"] = window_maxes
            summary["window_volatilities"] = {}
            for w_idx in range(len(window_ranges)):
                if window_strat_maxes[w_idx] is not None:
                    amp = window_strat_maxes[w_idx] - window_strat_mins[w_idx]
                    # Cap the amplitude to prevent the optimizer from getting distracted
                    capped_amp = np.clip(amp, a_min=None, a_max=self.config.cmaes.volatility_cap)
                    # Sum capped amplitude across all actions for all players. Shape is (B, total_actions)
                    # Sum along axis -1 to get shape (B,)
                    summary["window_volatilities"][w_idx] = np.sum(capped_amp, axis=-1)
                else:
                    summary["window_volatilities"][w_idx] = np.zeros(self.config.execution.batch_size)
        
        if hasattr(self.dynamic, "cumulative_logit_penalty"):
            summary["logit_penalty"] = self.dynamic.cumulative_logit_penalty.clone()

        flat_final_avg = [
            item
            for sublist in final_avg_regrets
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        if not self.config.execution.quiet:
            try:
                final_val = float(torch.cat(flat_final_avg).max().item())
            except Exception:
                final_val = float(max([x.max().item() if hasattr(x, 'max') else x for x in flat_final_avg]))
            logger.info(
                f"Simulation completed cleanly! Final Max Avg Regret: {final_val:.6f}. "
                f"Stats session ID '{self.session_id}' in '{self.config.logging.output_dir}'"
            )
            
        try:
            from src.utils.tracking import ExperimentTracker
            tracker = ExperimentTracker(out_dir=self.config.logging.output_dir)
            
            try:
                final_val = float(torch.cat(flat_final_avg).max().item())
            except Exception:
                final_val = float(max([x.max().item() if hasattr(x, 'max') else x for x in flat_final_avg]))
                
            tracker.log_run(
                config=self.config,
                run_type="dynamic",
                metrics={"final_max_avg_regret": final_val},
                parent_session_id=self.config.parent_session_id
            )
        except Exception as e:
            logger.warning(f"Failed to log run to tracker: {e}")
            
        return summary
