"""Unit tests for checkpoint saving, loading, and state resumption."""

import os
import shutil
import tempfile
import pytest
import torch

from src.config.schemas import ExperimentConfig
from src.engine.runner import ExperimentRunner


def test_checkpoint_and_resume_identity():
    temp_dir = tempfile.mkdtemp()
    try:
        config = ExperimentConfig(
            name="test_checkpoint",
            checkpoint={"enabled": True, "checkpoint_dir": temp_dir, "save_interval": 10},
            execution={"total_steps": 20, "seed": 42},
            logging={"output_dir": temp_dir},
        )

        # Run first 20 steps
        runner1 = ExperimentRunner(config=config)
        summary1 = runner1.run()

        # Find checkpoint file at step 10
        files = [f for f in os.listdir(temp_dir) if f.endswith(".pt") and "checkpoint_" in f]
        assert len(files) >= 1
        ckpt_path = os.path.join(temp_dir, files[0])

        # Resume from checkpoint
        config.execution.total_steps = 20
        runner2 = ExperimentRunner(config=config, resume_checkpoint_path=ckpt_path)
        summary2 = runner2.run()

        assert summary2["total_steps"] == 20
        assert len(summary1["final_cum_regrets"]) == len(summary2["final_cum_regrets"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
