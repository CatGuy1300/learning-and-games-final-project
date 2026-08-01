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
from src.utils.device import enable_gpu_optimizations, get_device
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
            p.clone().detach().to(dtype=torch.float32)
            if isinstance(p, torch.Tensor)
            else torch.tensor(p, dtype=torch.float32)
            for p in config.game.payoffs
        ]
        if config.game.num_players == 2 and len(payoffs) == 2 and payoffs[0].dim() == 2:
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
        )
    elif algo == "mwu":
        return MultiplicativeWeightsUpdate(
            action_sizes=action_sizes,
            eta=eta,
            device=device,
            batch_size=batch_size,
            T=config.execution.total_steps,
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
                backend = (
                    "cudagraphs"
                    if (sys.platform == "win32" and self.device.type == "cuda")
                    else "inductor"
                )
                logger.info(
                    f"Enabling PyTorch JIT compilation via torch.compile(backend='{backend}')..."
                )
                self.dynamic.step_2d = torch.compile(self.dynamic.step_2d, backend=backend)
                # Ensure torch.no_grad() is used or mark step begins to satisfy cudagraphs fast path
                self.game.get_stacked_utility_vectors = torch.compile(
                    self.game.get_stacked_utility_vectors, backend=backend
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
            for a_size in self.game.action_sizes:
                r = torch.rand(a_size, device=self.device, dtype=torch.float32)
                initial_strats.append(r / r.sum())
        elif strat_mode == "custom" and config.dynamic.custom_initial_strategies:
            initial_strats = [
                torch.tensor(s, device=self.device, dtype=torch.float32)
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
            dtype=torch.float32,
        )
        self.cumulative_actual_payoffs = torch.zeros(
            (self.config.execution.batch_size, self.game.num_players),
            device=self.device,
            dtype=torch.float32,
        )

        if resume_checkpoint_path:
            self._resume_from_checkpoint(resume_checkpoint_path)

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
                    data["cumulative_actual_payoffs"], device=self.device, dtype=torch.float32
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
        logger.info(f"Resumed experiment '{self.config.name}' from step {self.start_step}")

    def run(self, target_steps: int | None = None) -> dict[str, Any]:
        """Run simulation loop from current step to target_steps.

        Parameters
        ----------
        target_steps : int | None, optional
            Step to run up to. If None, uses config.execution.total_steps.

        Returns
        -------
        Dict[str, Any]
            Final metrics summary dictionary.
        """
        total_steps = target_steps if target_steps is not None else self.config.execution.total_steps
        logger.info(
            f"Starting simulation '{self.config.name}' [Session: {self.session_id}] "
            f"on device '{self.device}' for T={total_steps} steps."
        )

        steps_per_call = max(1, self.config.execution.steps_per_call)
        step = self.start_step

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
            while step < total_steps:
                target_step = min(step + steps_per_call, total_steps)
                k_steps = target_step - step

                with torch.no_grad():
                    self.dynamic.step_unrolled_block(
                        game=self.game,
                        cum_u_2d=self.stacked_cumulative_utility_vectors,
                        cum_p_1d=self.cumulative_actual_payoffs,
                        k_steps=k_steps,
                    )
                step += k_steps

                progress.update(task_id, advance=k_steps)

                # Heavy metrics, logging, stats buffering, and checkpointing at configured intervals
                is_sample_step = step % self.config.logging.sample_interval == 0
                is_log_step = step % self.config.logging.log_interval == 0
                is_flush_step = step % self.config.logging.save_stats_interval == 0
                is_ckpt_step = self.config.checkpoint.enabled and (
                    step % self.config.checkpoint.save_interval == 0
                )

                if is_sample_step or is_log_step or is_ckpt_step or (step == total_steps):
                    curr_strats = self.dynamic.strategies
                    u_vecs = self.game.get_utility_vectors(curr_strats)
                    cum_actual_payoffs_list = self.cumulative_actual_payoffs.T.tolist()
                    if self.config.execution.batch_size == 1:
                        cum_actual_payoffs_list = [l[0] for l in cum_actual_payoffs_list]
                    expected_payoffs = [
                        torch.sum(u_vecs[i] * curr_strats[i], dim=-1).tolist()
                        if self.config.execution.batch_size > 1
                        else torch.dot(u_vecs[i], curr_strats[i]).item()
                        for i in range(self.game.num_players)
                    ]
                    br_payoffs = self.game.best_response_payoffs(curr_strats)
                    instant_regrets, _ = compute_step_metrics(expected_payoffs, br_payoffs)
                    cum_regrets = compute_cumulative_regret(
                        self.cumulative_utility_vectors, cum_actual_payoffs_list
                    )
                    avg_regrets = compute_average_regret(cum_regrets, step)

                    if is_sample_step or (step == total_steps):
                        self.stats_collector.record_step(
                            step=step,
                            strategies=curr_strats,
                            expected_payoffs=expected_payoffs,
                            instant_regrets=instant_regrets,
                            cum_regrets=cum_regrets,
                        )

                    if is_flush_step or (step == total_steps):
                        self.stats_collector.flush_to_disk()

                    if is_log_step:
                        flat_cum = [
                            item
                            for sublist in cum_regrets
                            for item in (sublist if isinstance(sublist, list) else [sublist])
                        ]
                        flat_avg = [
                            item
                            for sublist in avg_regrets
                            for item in (sublist if isinstance(sublist, list) else [sublist])
                        ]
                        postfix = (
                            f"max_cum_regret={max(flat_cum):.4f} max_avg_regret={max(flat_avg):.4f}"
                        )
                        if getattr(self, "last_ckpt", None) is not None:
                            postfix += f" | ckpt={self.last_ckpt}"
                        progress.update(task_id, postfix=postfix)

                    if is_ckpt_step:
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

        flat_final_avg = [
            item
            for sublist in final_avg_regrets
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        logger.info(
            f"Simulation completed cleanly! Final Max Avg Regret: {max(flat_final_avg):.6f}. "
            f"Stats session ID '{self.session_id}' in '{self.config.logging.output_dir}'"
        )
        return summary
