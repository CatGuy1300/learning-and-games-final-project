"""Typer CLI interface for running and managing learning in games experiments."""

import os
from typing import Optional
import yaml
import typer
from rich.console import Console
from rich.table import Table

from src.config.schemas import ExperimentConfig
from src.config.validation import validate_experiment_config
from src.engine.runner import ExperimentRunner

app = typer.Typer(
    name="learning-games",
    help="Learning Dynamics in General-Sum Games: Simulation & Optimization CLI",
    add_completion=False,
)
console = Console()


def load_yaml_config(config_path: str) -> ExperimentConfig:
    """Load and parse YAML configuration file into ExperimentConfig."""
    if not os.path.exists(config_path):
        raise typer.BadParameter(f"Config file not found: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}
    config = ExperimentConfig(**raw_dict)
    return config


@app.command()
def run(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", "-r", help="Path to .pt checkpoint file to resume simulation from"
    ),
    steps: Optional[int] = typer.Option(
        None, "--steps", "-s", help="Override total simulation horizon steps T"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Override execution device ('cuda', 'cpu', 'auto')"
    ),
    compile: Optional[bool] = typer.Option(
        None, "--compile", help="Enable PyTorch JIT graph compilation via torch.compile()"
    ),
    sample_interval: Optional[int] = typer.Option(
        None, "--sample-interval", help="Sampling step interval for statistics recording"
    ),
    steps_per_call: Optional[int] = typer.Option(
        None, "--steps-per-call", help="Number of unrolled dynamic steps per loop invocation"
    ),
) -> None:
    """Run learning dynamics simulation experiment."""
    if config:
        exp_config = load_yaml_config(config)
    else:
        exp_config = ExperimentConfig()

    if steps is not None:
        exp_config.execution.total_steps = steps
    if device is not None:
        exp_config.execution.device = device
    if compile is not None:
        exp_config.execution.compile = compile
    if sample_interval is not None:
        exp_config.logging.sample_interval = sample_interval
    if steps_per_call is not None:
        exp_config.execution.steps_per_call = steps_per_call

    console.print(
        f"[bold green]Initializing Experiment:[/bold green] {exp_config.name} "
        f"([cyan]Algorithm:[/cyan] {exp_config.dynamic.algorithm.upper()}, "
        f"[cyan]Game Generator:[/cyan] {exp_config.game.generator})"
    )

    runner = ExperimentRunner(config=exp_config, resume_checkpoint_path=resume)
    summary = runner.run()

    table = Table(title="Simulation Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Session ID", summary["session_id"])
    table.add_row("Execution Device", summary.get("device", "cpu"))
    table.add_row("Total Steps T", str(summary["total_steps"]))
    for i, (cum_r, avg_r) in enumerate(
        zip(summary["final_cum_regrets"], summary["final_avg_regrets"])
    ):
        table.add_row(f"Player {i} Cum Regret", f"{cum_r:.6f}")
        table.add_row(f"Player {i} Avg Regret", f"{avg_r:.6f}")
    table.add_row("Stats Output Dir", summary["output_dir"])

    console.print(table)


@app.command()
def validate_config(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML configuration file")
) -> None:
    """Validate YAML configuration file structure, shapes, and utility bounds."""
    try:
        exp_config = load_yaml_config(config)
        validate_experiment_config(exp_config)
        console.print(f"[bold green]SUCCESS:[/bold green] Config file '{config}' is valid!")
    except Exception as e:
        console.print(f"[bold red]VALIDATION ERROR:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Display environment hardware and version details."""
    import torch

    console.print("[bold yellow]Learning in Games Framework Info[/bold yellow]")
    console.print(f"PyTorch Version: {torch.__version__}")
    console.print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")


@app.command()
def show_game(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML configuration file")
) -> None:
    """Display generated game payoff matrices/tensors."""
    from src.engine.runner import instantiate_game
    from src.utils.device import get_device

    exp_config = load_yaml_config(config)
    device = get_device(exp_config.execution.device)
    game = instantiate_game(exp_config, device)

    console.print(
        f"[bold green]Game Environment:[/bold green] {exp_config.game.generator} "
        f"([cyan]Players:[/cyan] {game.num_players}, [cyan]Actions:[/cyan] {game.action_sizes})"
    )

    payoffs = game.get_payoff_tensors()
    for i, p_tensor in enumerate(payoffs):
        console.print(f"\n[bold yellow]Player {i} Payoff Matrix/Tensor (shape {tuple(p_tensor.shape)}):[/bold yellow]")
        console.print(p_tensor.cpu().numpy())


if __name__ == "__main__":
    app()
