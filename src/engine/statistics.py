"""Statistics buffer, incremental chunked disk flushing, and dataset loader for super long horizons."""

import glob
import os
from typing import Any, Dict, List, Optional
import torch

from src.utils.logging import setup_logger

logger = setup_logger("statistics")


class StatsCollector:
    """Buffers step metrics and flushes incremental fixed-size chunk files to disk, maintaining O(1) RAM."""

    def __init__(self, output_dir: str = "outputs", session_id: str = "default_session") -> None:
        """Initialize StatsCollector.

        Parameters
        ----------
        output_dir : str
            Output directory path.
        session_id : str
            Unique session token run ID.
        """
        self.output_dir = output_dir
        self.session_id = session_id
        os.makedirs(self.output_dir, exist_ok=True)

        self.chunk_index: int = 0
        self.history_steps: List[int] = []
        self.history_strategies: List[List[List[float]]] = []
        self.history_expected_payoffs: List[List[float]] = []
        self.history_instant_regrets: List[List[float]] = []
        self.history_cum_regrets: List[List[float]] = []
        self.payoffs: List[torch.Tensor] = []

    def set_payoffs(self, payoffs: List[torch.Tensor]) -> None:
        """Set game payoff tensors to be saved with statistics."""
        self.payoffs = [p.cpu().clone() for p in payoffs]

    def record_step(
        self,
        step: int,
        strategies: List[torch.Tensor],
        expected_payoffs: List[float],
        instant_regrets: List[float],
        cum_regrets: List[float],
    ) -> None:
        """Record single step metrics into active RAM chunk buffer."""
        self.history_steps.append(step)
        self.history_strategies.append(
            [s.detach().to(device="cpu", non_blocking=True).tolist() for s in strategies]
        )
        self.history_expected_payoffs.append(expected_payoffs)
        self.history_instant_regrets.append(instant_regrets)
        self.history_cum_regrets.append(cum_regrets)

    def flush_to_disk(self) -> Optional[str]:
        """Flush active chunk buffer to an incremental PyTorch .pt chunk file and clear RAM.

        Returns
        -------
        Optional[str]
            Saved chunk filepath, or None if buffer was empty.
        """
        if not self.history_steps:
            return None

        chunk_filename = f"stats_{self.session_id}_chunk_{self.chunk_index:04d}.pt"
        filepath = os.path.join(self.output_dir, chunk_filename)

        num_players = len(self.history_strategies[0]) if self.history_strategies else 0
        player_strategy_tensors = [
            torch.tensor(
                [step_strats[i] for step_strats in self.history_strategies], dtype=torch.float32
            )
            for i in range(num_players)
        ]

        chunk_data = {
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "payoffs": self.payoffs,
            "steps": torch.tensor(self.history_steps, dtype=torch.int64),
            "strategies": player_strategy_tensors,
            "expected_payoffs": torch.tensor(self.history_expected_payoffs, dtype=torch.float32),
            "instant_regrets": torch.tensor(self.history_instant_regrets, dtype=torch.float32),
            "cum_regrets": torch.tensor(self.history_cum_regrets, dtype=torch.float32),
        }

        torch.save(chunk_data, filepath)
        logger.debug(
            f"Flushed chunk {self.chunk_index} ({len(self.history_steps)} steps) to '{filepath}'"
        )

        # Clear RAM chunk buffer to guarantee O(1) constant memory footprint
        self.history_steps.clear()
        self.history_strategies.clear()
        self.history_expected_payoffs.clear()
        self.history_instant_regrets.clear()
        self.history_cum_regrets.clear()
        self.chunk_index += 1

        return filepath


def load_experiment_stats(output_dir: str = "outputs", session_id: str = "") -> Dict[str, Any]:
    """Load and concatenate all incremental stats chunk files for a given session.

    Parameters
    ----------
    output_dir : str
        Directory containing output stats files.
    session_id : str
        Session run identifier.

    Returns
    -------
    Dict[str, Any]
        Unified concatenated dataset containing full steps, strategies, and regrets.
    """
    chunk_pattern = os.path.join(output_dir, f"stats_{session_id}_chunk_*.pt")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        # Fallback to single file legacy format if present
        legacy_file = os.path.join(output_dir, f"stats_{session_id}.pt")
        if os.path.exists(legacy_file):
            return torch.load(legacy_file, map_location="cpu", weights_only=False)
        raise FileNotFoundError(f"No statistics files found for session_id '{session_id}' in '{output_dir}'")

    all_steps = []
    all_strategies = []
    all_expected_payoffs = []
    all_instant_regrets = []
    all_cum_regrets = []
    payoffs = None

    for f in chunk_files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        if payoffs is None and "payoffs" in data:
            payoffs = data["payoffs"]
        all_steps.append(data["steps"])
        all_strategies.append(data["strategies"])
        all_expected_payoffs.append(data["expected_payoffs"])
        all_instant_regrets.append(data["instant_regrets"])
        all_cum_regrets.append(data["cum_regrets"])

    combined_steps = torch.cat(all_steps, dim=0)
    combined_expected_payoffs = torch.cat(all_expected_payoffs, dim=0)
    combined_instant_regrets = torch.cat(all_instant_regrets, dim=0)
    combined_cum_regrets = torch.cat(all_cum_regrets, dim=0)

    # Combine player strategy trajectories per player
    num_players = len(all_strategies[0])
    combined_strategies = [
        torch.cat([chunk_strats[i] for chunk_strats in all_strategies], dim=0)
        for i in range(num_players)
    ]

    return {
        "session_id": session_id,
        "payoffs": payoffs,
        "steps": combined_steps,
        "strategies": combined_strategies,
        "expected_payoffs": combined_expected_payoffs,
        "instant_regrets": combined_instant_regrets,
        "cum_regrets": combined_cum_regrets,
    }
