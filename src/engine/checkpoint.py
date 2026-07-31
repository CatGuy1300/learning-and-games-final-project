"""Checkpoint manager for saving and resuming long-horizon experiments."""

import glob
import os
import uuid
from typing import Any, Dict, List, Optional
import torch

from src.utils.logging import setup_logger

logger = setup_logger("checkpoint")


class CheckpointManager:
    """Manages experiment checkpoint serialization, loading, and directory cleanup."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize CheckpointManager.

        Parameters
        ----------
        checkpoint_dir : str
            Directory path for saving checkpoint files.
        keep_top_k : int
            Maximum number of recent checkpoint files to retain.
        session_id : Optional[str]
            Session token / run UUID identifier.
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self.session_id = session_id or str(uuid.uuid4())
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        step: int,
        config_dict: Dict[str, Any],
        dynamic_state: Dict[str, Any],
        rng_state: Dict[str, Any],
        cumulative_utility_vectors: List[torch.Tensor],
        cumulative_actual_payoffs: List[float],
    ) -> str:
        """Save experiment checkpoint dictionary atomically to disk.

        Returns
        -------
        str
            Saved checkpoint filepath.
        """
        checkpoint_filename = f"checkpoint_{self.session_id}_step_{step:08d}.pt"
        filepath = os.path.join(self.checkpoint_dir, checkpoint_filename)

        checkpoint_data = {
            "step": step,
            "session_id": self.session_id,
            "config": config_dict,
            "dynamic_state": dynamic_state,
            "rng_state": rng_state,
            "cumulative_utility_vectors": [u.cpu() for u in cumulative_utility_vectors],
            "cumulative_actual_payoffs": (
                cumulative_actual_payoffs.cpu().tolist()
                if isinstance(cumulative_actual_payoffs, torch.Tensor)
                else cumulative_actual_payoffs
            ),
        }

        # Atomic write to temporary file before rename
        tmp_path = filepath + ".tmp"
        torch.save(checkpoint_data, tmp_path)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_path, filepath)

        logger.debug(f"Saved checkpoint at step {step} to '{filepath}'")
        self._cleanup_old_checkpoints()
        return filepath

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint dictionary from file.

        Parameters
        ----------
        filepath : str
            Path to .pt checkpoint file.

        Returns
        -------
        Dict[str, Any]
            Loaded state dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        logger.info(f"Loading checkpoint from '{filepath}'")
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        return data

    def _cleanup_old_checkpoints(self) -> None:
        """Purge older checkpoints exceeding keep_top_k limit."""
        pattern = os.path.join(self.checkpoint_dir, f"checkpoint_{self.session_id}_step_*.pt")
        files = sorted(glob.glob(pattern))

        if len(files) > self.keep_top_k:
            to_delete = files[: -self.keep_top_k]
            for f in to_delete:
                try:
                    os.remove(f)
                    logger.debug(f"Removed old checkpoint '{f}'")
                except OSError as e:
                    logger.warning(f"Failed to remove checkpoint '{f}': {e}")
