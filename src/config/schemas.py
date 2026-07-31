"""Pydantic configuration schemas for games, learning dynamics, and experiments."""

from typing import Any

from pydantic import BaseModel, Field


class GameConfig(BaseModel):
    """Configuration for N-player general-sum game environments."""

    num_players: int = Field(default=2, ge=2, description="Number of players N")
    action_sizes: list[int] = Field(
        default_factory=lambda: [2, 2],
        description="Action counts (A_1, ..., A_N) for each player",
    )
    utility_range: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Allowed range (u_min, u_max) for payoffs"
    )
    generator: str = Field(
        default="random",
        description="Preset game generator ('matching_pennies', 'prisoners_dilemma', 'shapley', 'random', 'custom')",
    )
    payoffs: list[Any] | None = Field(
        default=None, description="Explicit payoff matrices/tensors for all players"
    )
    seed: int | None = Field(default=42, description="Random seed for game generation")


class DynamicConfig(BaseModel):
    """Configuration for learning dynamics algorithms."""

    algorithm: str = Field(
        default="omwu",
        description="Learning dynamic type ('omwu', 'extra_gradient', 'mwu', 'gda', 'mirror_prox')",
    )
    eta: float = Field(default=0.01, gt=0.0, description="Learning rate step size eta")
    initial_strategy_type: str = Field(
        default="uniform",
        description="Initial strategy mode ('uniform', 'random', 'custom')",
    )
    custom_initial_strategies: list[list[float]] | None = Field(
        default=None, description="Custom initial probability distributions per player"
    )


class CheckpointConfig(BaseModel):
    """Configuration for experiment checkpointing and state persistence."""

    enabled: bool = Field(default=True, description="Enable periodic checkpoint saving")
    checkpoint_dir: str = Field(
        default="checkpoints", description="Directory path for saved checkpoints"
    )
    save_interval: int = Field(
        default=1000, ge=1, description="Step frequency K for saving checkpoints"
    )
    keep_top_k: int = Field(
        default=5, ge=1, description="Maximum number of recent checkpoints to retain"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and progress metrics."""

    log_level: str = Field(default="INFO", description="Logging verbosity level")
    log_interval: int = Field(default=100, ge=1, description="Step frequency for progress logging")
    save_stats_interval: int = Field(
        default=5000, ge=1, description="Step chunk size for incremental disk flushing"
    )
    sample_interval: int = Field(
        default=1, ge=1, description="Sampling step interval for statistics recording"
    )
    output_dir: str = Field(
        default="outputs", description="Directory path for statistics and metrics"
    )


class ExecutionConfig(BaseModel):
    """Execution runtime parameters."""

    total_steps: int = Field(default=10000, ge=1, description="Total horizon horizon steps T")
    device: str = Field(default="auto", description="Device preference ('auto', 'cuda', 'cpu')")
    seed: int = Field(default=42, description="Global random seed")
    compile: bool = Field(default=False, description="Enable torch.compile JIT graph fusion")
    fp32_precision: str = Field(
        default="high", description="TensorFloat-32 precision on GPU ('high', 'highest')"
    )
    steps_per_call: int = Field(
        default=1, ge=1, description="Number of unrolled dynamic steps per inner loop invocation"
    )
    batch_size: int = Field(
        default=1, ge=1, description="Number of independent games to simulate simultaneously"
    )


class ExperimentConfig(BaseModel):
    """Master experiment configuration container."""

    name: str = Field(default="game_experiment", description="Experiment identification name")
    session_id: str | None = Field(default=None, description="Session token / UUID run identifier")
    game: GameConfig = Field(default_factory=GameConfig)
    dynamic: DynamicConfig = Field(default_factory=DynamicConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
