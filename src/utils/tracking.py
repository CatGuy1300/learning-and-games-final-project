import glob
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from filelock import FileLock

from src.config.schemas import ExperimentConfig


class ExperimentTracker:
    """Lightweight local tracker for experiment configurations and metrics."""

    def __init__(self, out_dir: str = "outputs"):
        self.out_dir = out_dir
        os.makedirs(os.path.abspath(self.out_dir), exist_ok=True)

    def _serialize_metrics(self, obj: Any) -> Any:
        # Check if it's a tensor-like object
        if hasattr(obj, 'numel') and hasattr(obj, 'tolist'):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
        elif hasattr(obj, 'item') and not hasattr(obj, 'numel'):
            # Fallback for numpy scalars or other scalar-like objects
            try:
                return obj.item()
            except ValueError:
                pass
        
        if isinstance(obj, dict):
            return {k: self._serialize_metrics(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_metrics(v) for v in obj]
        return obj

    def log_run(self, config: ExperimentConfig, run_type: str, metrics: dict[str, Any], parent_session_id: str | None = None) -> None:
        """Log a complete experiment run to the JSONL file.
        
        Parameters
        ----------
        config : ExperimentConfig
            The full configuration used for the run.
        run_type : str
            The type of run (e.g., 'dynamic', 'cmaes').
        metrics : dict
            A dictionary of metrics to log (e.g., regret, fitness).
        parent_session_id : str, optional
            The session ID of a parent run (e.g. if this is a validation run for a CMA-ES output).
        """
        # Convert metrics to native python types
        metrics = self._serialize_metrics(metrics)
        
        # Flatten out key parameters for easier pandas querying, but also store the full raw config dict
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": config.session_id,
            "parent_session_id": parent_session_id,
            "run_type": run_type,
            
            # Key searchable fields
            "algorithm": config.dynamic.algorithm,
            "eta": config.dynamic.eta,
            "total_steps": config.execution.total_steps,
            "game_generator": config.game.generator,
            
            # If CMA-ES config exists, we pull it up for easy querying
            "cmaes_objective": config.cmaes.objective_type if config.cmaes else None,
            "cmaes_sigma": config.cmaes.sigma if config.cmaes else None,
            
            # Full configuration and metrics
            "config": self._serialize_metrics(config.model_dump()),
            "metrics": metrics
        }
        log_file = os.path.join(self.out_dir, f"experiments_{run_type}_{config.dynamic.algorithm}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def list_available_tables(out_dir: str = "outputs", search_parent: bool = True) -> list[str]:
    """Returns a list of all available tracking tables in the specified output directory.
    
    If search_parent is True, also checks the parent directory's output folder (useful for notebooks).
    """
    patterns = [os.path.join(out_dir, "experiments_*.jsonl")]
    if search_parent:
        patterns.append(os.path.join("..", out_dir, "experiments_*.jsonl"))
        
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
        
    return sorted({os.path.basename(f) for f in files})

def load_jsonl_table(table_pattern: str, out_dir: str = "outputs", search_parent: bool = True) -> pd.DataFrame:
    """Loads and concatenates tables matching the pattern from the output directories.
    
    If search_parent is True, also searches the parent directory's output folder.
    """
    patterns = [os.path.join(out_dir, table_pattern)]
    if search_parent:
        patterns.append(os.path.join("..", out_dir, table_pattern))
        
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
        
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        records = []
        with open(f, "r", encoding="utf-8") as file:
            for line_idx, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping corrupted JSON line at {f}:{line_idx}")
                    
        if not records:
            continue
            
        df = pd.DataFrame(records)
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df.drop(columns=['metrics']), metrics_df], axis=1)
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)
