import json

path = "src/engine/sweeper.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add optuna import
if "import optuna" not in content:
    content = content.replace("import multiprocessing as mp", "import multiprocessing as mp\nimport optuna\nfrom optuna.samplers import TPESampler, RandomSampler")

optuna_code = """

def _optuna_objective(trial: optuna.Trial, base_config: ExperimentConfig, search_space: dict[str, Any], sweep_session_id: str) -> float:
    # Build overrides
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
    
    # Create config
    conf = deepcopy(base_config)
    for k, v in overrides.items():
        _set_nested_attr(conf, k, v)
    
    conf.execution.quiet = True
    
    try:
        # Run optimizer
        _, metrics = run_single_config(conf.model_dump(), overrides, sweep_session_id)
        # We maximize regret
        return metrics["best_regret"]
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        raise optuna.TrialPruned()

def run_optuna_sweep(sweep_config: dict[str, Any], workers: int = 4) -> None:
    base_config = ExperimentConfig(**sweep_config["base_config"])
    sweep_name = sweep_config.get("sweep_name", "optuna_sweep")
    search_space = sweep_config.get("search_space", {})
    num_trials = sweep_config.get("num_trials", 20)
    sampler_type = sweep_config.get("optuna_sampler", "tpe")
    
    sweep_session_id = uuid.uuid4().hex[:8]
    console.print(f"[bold green]Starting Optuna Sweep '{sweep_name}' ({sweep_session_id})[/bold green]")
    console.print(f"Workers: {workers} | Trials: {num_trials} | Sampler: {sampler_type}")
    
    sampler = TPESampler() if sampler_type == "tpe" else RandomSampler()
    
    # Use SQLite for multiprocessing support and checkpointing
    os.makedirs("outputs", exist_ok=True)
    db_url = f"sqlite:///outputs/{sweep_name}.db"
    
    # Create or load study
    study = optuna.create_study(
        study_name=sweep_name,
        storage=db_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )
    
    # Run optimization (Optuna natively handles multiprocessing if we just run it in parallel, 
    # but since we are launching from CLI, the easiest way to multiprocess is optuna's n_jobs)
    # Wait, optuna's n_jobs uses threads or joblib. For heavy PyTorch, we need ProcessPoolExecutor.
    # Optuna's study.optimize can use n_jobs=workers but it might crash CUDA if it doesn't use spawn.
    # To be safe and reuse our robust process pool:
    
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        # We can dispatch trials to the pool manually, but Optuna needs to track them.
        # Actually, if we just use Optuna's n_jobs, it uses threading which is BAD for PyTorch CUDA.
        # The best robust way is to use joblib with 'loky' or our own pool.
        # But wait, optuna.optimize(n_jobs) natively uses threading which doesn't spawn new processes by default.
        pass
"""

# Let's write a smarter approach for optuna multiprocessing that respects mp.get_context("spawn")
optuna_code2 = """
def _run_trial_worker(db_url: str, sweep_name: str, base_config_dict: dict, search_space: dict, sweep_session_id: str) -> None:
    # This runs inside the spawned process
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from src.config.schemas import ExperimentConfig
    import copy
    
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
        study.tell(trial, metrics["best_regret"])
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
                
            for res in results:
                try:
                    res.get()
                except Exception as e:
                    console.print(f"[bold red]Worker FAILED:[/bold red] {e}")
                progress.advance(task)
"""

if "def run_optuna_sweep" not in content:
    content += "\n" + optuna_code2

# Now modify run_sweep to check mode
replace_target = """    base_config = ExperimentConfig(**base_config_data)
    sweep_name = sweep_config.get("sweep_name", "hyperparam_sweep")"""

replace_with = """    mode = sweep_config.get("mode", "grid")
    if mode == "optuna":
        # Pass base_config_data dict instead of string
        sweep_config["base_config"] = base_config_data
        return run_optuna_sweep(sweep_config, workers)
        
    base_config = ExperimentConfig(**base_config_data)
    sweep_name = sweep_config.get("sweep_name", "hyperparam_sweep")"""

content = content.replace(replace_target, replace_with)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
