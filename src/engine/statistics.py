"""Statistics buffer, incremental chunked disk flushing, and dataset loader for super long horizons."""

import glob
import os
from typing import Any

import torch

from src.utils.logging import setup_logger

logger = setup_logger("statistics")


class StatsCollector:
    """Buffers step metrics and flushes incremental fixed-size chunk files to disk, maintaining O(1) RAM."""

    def __init__(self, output_dir: str = "outputs", session_id: str = "default_session") -> None:
        """Initialize StatsCollector."""
        self.output_dir = output_dir
        self.session_id = session_id
        os.makedirs(self.output_dir, exist_ok=True)

        self.chunk_index: int = 0
        self.history_steps: list[int] = []
        
        # We initialize these lazily when we know the number of players
        self.num_players = 0
        self.history_strategies: list[list[torch.Tensor]] = []
        self.history_expected_payoffs: list[torch.Tensor] = []
        self.history_instant_regrets: list[torch.Tensor] = []
        self.history_cum_regrets: list[list[torch.Tensor]] = []
        self.history_logits: list[list[torch.Tensor]] = []
        self.history_instant_payoffs: list[list[torch.Tensor]] = []
        self.payoffs: list[torch.Tensor] = []

    def set_payoffs(self, payoffs: list[torch.Tensor]) -> None:
        """Set game payoff tensors to be saved with statistics."""
        self.payoffs = [p.cpu().clone() for p in payoffs]

    def _init_buffers(self, num_players: int):
        if self.num_players == 0:
            self.num_players = num_players
            for _ in range(num_players):
                self.history_strategies.append([])
                self.history_cum_regrets.append([])
                self.history_logits.append([])
                self.history_instant_payoffs.append([])

    def record_batch(
        self,
        steps: list[int],
        strats: torch.Tensor,
        logits: torch.Tensor | None,
        u_vecs: torch.Tensor,
        cum_u: torch.Tensor,
        cum_p: torch.Tensor,
        action_sizes: list[int]
    ) -> None:
        """Record an entire unrolled block of k_steps at once."""
        num_players = strats.shape[2]
        self._init_buffers(num_players)
        
        batch_size = strats.shape[1]
        self.history_steps.extend(steps)
        
        # expected payoffs: (K, B, P)
        expected_payoffs = (strats * u_vecs).sum(dim=-1) 
        
        br_payoffs = []
        for i in range(self.num_players):
            a = action_sizes[i]
            br_i = u_vecs[:, :, i, :a].max(dim=-1).values # (K, B)
            br_payoffs.append(br_i)
        
        br_payoffs_tensor = torch.stack(br_payoffs, dim=2) # (K, B, P)
        instant_regrets = br_payoffs_tensor - expected_payoffs # (K, B, P)
        
        if batch_size == 1:
            expected_payoffs = expected_payoffs.squeeze(1) # (K, P)
            instant_regrets = instant_regrets.squeeze(1) # (K, P)
            
        self.history_expected_payoffs.append(expected_payoffs)
        self.history_instant_regrets.append(instant_regrets)
        
        for i in range(self.num_players):
            a = action_sizes[i]
            
            s_i = strats[:, :, i, :a]
            u_i = u_vecs[:, :, i, :a]
            
            if batch_size == 1:
                s_i = s_i.squeeze(1)
                u_i = u_i.squeeze(1)
                
            self.history_strategies[i].append(s_i)
            self.history_instant_payoffs[i].append(u_i)
            
            if logits is not None:
                l_i = logits[:, :, i, :a]
                if batch_size == 1:
                    l_i = l_i.squeeze(1)
                self.history_logits[i].append(l_i)
                
            cum_u_i = cum_u[:, :, i, :a]
            cum_p_i = cum_p[:, :, i]
            regret_i = cum_u_i.max(dim=-1).values - cum_p_i
            if batch_size == 1:
                regret_i = regret_i.squeeze(1)
            self.history_cum_regrets[i].append(regret_i)

    def flush_to_disk(self) -> str | None:
        """Flush active chunk buffer to an incremental PyTorch .pt chunk file and clear RAM."""
        if not self.history_steps:
            return None

        chunk_filename = f"stats_{self.session_id}_chunk_{self.chunk_index:04d}.pt"
        filepath = os.path.join(self.output_dir, chunk_filename)

        chunk_data = {
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "payoffs": self.payoffs,
            "steps": torch.tensor(self.history_steps, dtype=torch.int64),
            "strategies": [torch.cat(hist, dim=0) for hist in self.history_strategies],
            "expected_payoffs": torch.cat(self.history_expected_payoffs, dim=0),
            "instant_regrets": torch.cat(self.history_instant_regrets, dim=0),
            "cum_regrets": [torch.cat(hist, dim=0) for hist in self.history_cum_regrets],
        }
        
        if any(self.history_logits):
            chunk_data["logits"] = [torch.cat(hist, dim=0) for hist in self.history_logits]
            
        if any(self.history_instant_payoffs):
            chunk_data["instant_payoffs"] = [torch.cat(hist, dim=0) for hist in self.history_instant_payoffs]

        torch.save(chunk_data, filepath)
        logger.debug(
            f"Flushed chunk {self.chunk_index} ({len(self.history_steps)} steps) to '{filepath}'"
        )

        self.history_steps.clear()
        self.history_expected_payoffs.clear()
        self.history_instant_regrets.clear()
        for i in range(self.num_players):
            self.history_strategies[i].clear()
            self.history_cum_regrets[i].clear()
            self.history_logits[i].clear()
            self.history_instant_payoffs[i].clear()
            
        self.chunk_index += 1
        return filepath


def load_experiment_stats(output_dir: str = "outputs", session_id: str = "") -> dict[str, Any]:
    """Load and concatenate all incremental stats chunk files for a given session."""
    chunk_pattern = os.path.join(output_dir, f"stats_{session_id}_chunk_*.pt")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        legacy_file = os.path.join(output_dir, f"stats_{session_id}.pt")
        if os.path.exists(legacy_file):
            return torch.load(legacy_file, map_location="cpu", weights_only=False)
        raise FileNotFoundError(f"No statistics files found")

    all_steps = []
    all_strategies = []
    all_expected_payoffs = []
    all_instant_regrets = []
    all_cum_regrets = []
    all_logits = []
    all_instant_payoffs = []
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
        if "logits" in data:
            all_logits.append(data["logits"])
        if "instant_payoffs" in data:
            all_instant_payoffs.append(data["instant_payoffs"])

    combined_steps = torch.cat(all_steps, dim=0)
    combined_expected_payoffs = torch.cat(all_expected_payoffs, dim=0)
    combined_instant_regrets = torch.cat(all_instant_regrets, dim=0)

    num_players = len(all_strategies[0])
    combined_strategies = [
        torch.cat([chunk_strats[i] for chunk_strats in all_strategies], dim=0)
        for i in range(num_players)
    ]
    
    if isinstance(all_cum_regrets[0], list):
        combined_cum_regrets = torch.stack([
            torch.cat([chunk_reg[i] for chunk_reg in all_cum_regrets], dim=0)
            for i in range(num_players)
        ], dim=-1)
    else:
        combined_cum_regrets = torch.cat(all_cum_regrets, dim=0)
    
    combined_logits = None
    if all_logits:
        combined_logits = [
            torch.cat([chunk_logits[i] for chunk_logits in all_logits], dim=0)
            for i in range(num_players)
        ]
        
    combined_instant_payoffs = None
    if all_instant_payoffs:
        combined_instant_payoffs = [
            torch.cat([chunk_payoffs[i] for chunk_payoffs in all_instant_payoffs], dim=0)
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
        "logits": combined_logits,
        "instant_payoffs": combined_instant_payoffs,
    }

def unpack_stats(stats_data: dict[str, Any]) -> tuple:
    """
    Unpacks stats dictionary into numpy arrays and tensors for visualization.
    Returns:
        tuple: (steps, cum_regrets, strats, logits, instant_payoffs)
    """
    steps = stats_data["steps"].numpy()
    cum_regrets = stats_data["cum_regrets"].numpy()
    strats = stats_data["strategies"]
    logits = stats_data.get("logits", None)
    instant_payoffs = stats_data.get("instant_payoffs", None)
    return steps, cum_regrets, strats, logits, instant_payoffs
