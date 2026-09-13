"""Hyperparameter sweeper with multiprocessing and checkpointing."""

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import uuid
from copy import deepcopy
from typing import Any

import optuna
import yaml
from optuna.samplers import RandomSampler, TPESampler
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from src.config.schemas import ExperimentConfig
from src.engine.optimizer import CMAESGameOptimizer
from src.utils.tracking import ExperimentTracker

logger = logging.getLogger(__name__)
console = Console()


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation (e.g. 'cmaes.sigma')."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _hash_config(override_dict: dict[str, Any]) -> str:
    """Generate a stable MD5 hash for a dictionary of overrides."""
    # Sort keys to ensure stable hashing
    stable_str = json.dumps(override_dict, sort_keys=True)
    return hashlib.md5(stable_str.encode("utf-8")).hexdigest()


def generate_grid(base_config: ExperimentConfig, grid: dict[str, list[Any]]) -> list[tuple[str, ExperimentConfig, dict[str, Any]]]:
    """Generate a list of configs from a parameter grid.
    
    Returns:
        List of tuples: (config_hash, ExperimentConfig, override_dict)
    """
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    
    results = []
    for combo in combinations:
        override_dict = dict(zip(keys, combo))
        config_hash = _hash_config(override_dict)
        
        # Deepcopy base config and apply overrides
        new_config = deepcopy(base_config)
        for k, v in override_dict.items():
            _set_nested_attr(new_config, k, v)
            
        results.append((config_hash, new_config, override_dict))
        
    return results


def run_single_config(config_dump: dict, override_dict: dict, sweep_session_id: str) -> tuple[str, dict]:
    """Worker function to run a single CMA-ES optimization."""
    # Re-instantiate from dump to avoid pickling issues
    config = ExperimentConfig(**config_dump)
    
    # We assign a new session ID for this specific CMA-ES run
    config.session_id = str(uuid.uuid4())
    
    # Run optimization
    optimizer = CMAESGameOptimizer(config)
    _, best_factors = optimizer.optimize()
    
    # After the run finishes, we log the sweep result
    tracker = ExperimentTracker(out_dir=config.logging.output_dir)
    tracker.log_run(
        config=config,
        run_type="cmaes-tune",
        metrics={
            "best_fitness": best_factors["fitness"],
            "best_regret": best_factors["regret"],
            "best_volatility": best_factors.get("volatility"),
            "overrides": override_dict
        },
        parent_session_id=sweep_session_id
    )
    
    return _hash_config(override_dict), best_factors


def run_sweep(sweep_yaml_path: str, num_workers: int = 1, resume: bool = True) -> None:
    """Run a hyperparameter sweep using ProcessPoolExecutor."""
    with open(sweep_yaml_path, "r", encoding="utf-8") as f:
        sweep_data = yaml.safe_load(f)
        
    mode = sweep_data.get("mode", "grid")
    
    base_config_path = sweep_data["base_config"]
    grid = sweep_data.get("grid", {})
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}
        
    if mode == "optuna":
        sweep_data["base_config"] = base_raw
        return run_optuna_sweep(sweep_data, num_workers)
    base_config = ExperimentConfig(**base_raw)
    
    sweep_name = sweep_data.get("sweep_name", "cmaes_sweep")
    sweep_session_id = str(uuid.uuid4())
    
    configs_to_run = generate_grid(base_config, grid)
    
    # Load state
    state_file = os.path.join(base_config.logging.output_dir, f"{sweep_name}_state.json")
    state = {}
    if resume and os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            console.print(f"[bold green]Resuming from {state_file}[/bold green]")
        except Exception:
            pass
            
    # Filter pending
    pending_configs = []
    for c_hash, conf, overrides in configs_to_run:
        if state.get(c_hash, {}).get("status") != "DONE":
            pending_configs.append((c_hash, conf, overrides))
            state[c_hash] = {"status": "PENDING", "overrides": overrides}
            
    # Save initial state
    os.makedirs(base_config.logging.output_dir, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
        
    console.print(f"Starting sweep '{sweep_name}' with {len(pending_configs)} remaining configs on {num_workers} workers.")
    
    if len(pending_configs) == 0:
        console.print("[bold green]Sweep fully completed![/bold green]")
        return
        
    # We must use 'spawn' for CUDA multiprocessing
    ctx = mp.get_context("spawn")
    
    # Run parallel
    with ctx.Pool(processes=num_workers) as pool:
        results = []
        for c_hash, conf, overrides in pending_configs:
            # We must pass the dump, since pydantic models might be tricky to pickle depending on version
            conf.execution.quiet = True
            res = pool.apply_async(run_single_config, (conf.model_dump(), overrides, sweep_session_id))
            results.append((c_hash, res))
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Sweeping Hyperparameters[/bold cyan]"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "({task.completed}/{task.total})",
            "•",
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Running...", total=len(pending_configs))
            
            try:
                for c_hash, res in results:
                    try:
                        _, metrics = res.get()
                        state[c_hash]["status"] = "DONE"
                        state[c_hash]["metrics"] = metrics
                    except Exception as e:
                        console.print(f"[bold red]Config {c_hash} FAILED:[/bold red] {e}")
                        state[c_hash]["status"] = "FAILED"
                        state[c_hash]["error"] = str(e)
                        
                    # Update state file dynamically
                    with open(state_file, "w") as f:
                        json.dump(state, f, indent=2)
                    progress.advance(task)
            except KeyboardInterrupt:
                console.print("\n[bold red]Received Ctrl+C. Forcefully terminating pending workers...[/bold red]")
                pool.terminate()
                pool.join()
                console.print("[yellow]Sweep successfully paused. You can resume later without losing data.[/yellow]")
                import sys
                sys.exit(0)

    console.print("[bold green]Sweep finished![/bold green]")


def _run_trial_worker(db_url: str, sweep_name: str, base_config_dict: dict, search_space: dict, sweep_session_id: str) -> None:
    # This runs inside the spawned process

    import optuna

    from src.config.schemas import ExperimentConfig
    
    study = optuna.load_study(study_name=sweep_name, storage=db_url)
    
    # We ask optuna for 1 trial
    trial = study.ask()
    
    overrides = {}
    for key, space in search_space.items():
        stype = space.get("type", "float")
        if stype == "float":
            overrides[key] = trial.suggest_float(key, space["low"], space["high"])
        elif stype == "log_float":
            overrides[key] = trial.suggest_float(key, space["low"], space["high"], log=True)
        elif stype == "int":
            overrides[key] = trial.suggest_int(key, space["low"], space["high"])
        elif stype == "categorical":
            overrides[key] = trial.suggest_categorical(key, space["choices"])
            
    conf = ExperimentConfig(**base_config_dict)
    for k, v in overrides.items():
        _set_nested_attr(conf, k, v)
        
    conf.execution.quiet = True
    
    try:
        _, metrics = run_single_config(conf.model_dump(), overrides, sweep_session_id)
        study.tell(trial, metrics["regret"])
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)

def run_optuna_sweep(sweep_config: dict[str, Any], workers: int = 4) -> None:
    base_config_dict = sweep_config["base_config"]
    sweep_name = sweep_config.get("sweep_name", "optuna_sweep")
    search_space = sweep_config.get("search_space", {})
    num_trials = sweep_config.get("num_trials", 20)
    sampler_type = sweep_config.get("optuna_sampler", "tpe")
    
    sweep_session_id = uuid.uuid4().hex[:8]
    console.print(f"[bold green]Starting Optuna Sweep '{sweep_name}' ({sweep_session_id})[/bold green]")
    console.print(f"Workers: {workers} | Trials: {num_trials} | Sampler: {sampler_type}")
    
    sampler = TPESampler() if sampler_type == "tpe" else RandomSampler()
    
    os.makedirs("outputs", exist_ok=True)
    db_url = f"sqlite:///outputs/{sweep_name}.db"
    
    study = optuna.create_study(
        study_name=sweep_name,
        storage=db_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )
    
    completed = len(study.trials)
    remaining = max(0, num_trials - completed)
    if remaining == 0:
        console.print("[yellow]Sweep already completed based on database![/yellow]")
        return
        
    console.print(f"Resuming {remaining} trials...")
    
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Optuna Sweeping[/bold cyan]"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "({task.completed}/{task.total})",
            "•",
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Running...", total=remaining)
            
            results = []
            for _ in range(remaining):
                res = pool.apply_async(_run_trial_worker, (db_url, sweep_name, base_config_dict, search_space, sweep_session_id))
                results.append(res)
                
            try:
                for res in results:
                    try:
                        res.get()
                    except Exception as e:
                        console.print(f"[bold red]Worker FAILED:[/bold red] {e}")
                    progress.advance(task)
            except KeyboardInterrupt:
                console.print("\n[bold red]Received Ctrl+C. Forcefully terminating pending workers...[/bold red]")
                pool.terminate()
                pool.join()
                console.print("[yellow]Sweep successfully paused. You can resume later without losing data.[/yellow]")
                import sys
                sys.exit(0)
