"""Unit tests for incremental chunked statistics flushing and dataset loading."""

import os
import shutil
import tempfile
import pytest
import torch

from src.engine.statistics import StatsCollector, load_experiment_stats


def test_incremental_chunk_flushing_and_ram_clearing():
    temp_dir = tempfile.mkdtemp()
    try:
        session_id = "test_chunking_session"
        collector = StatsCollector(output_dir=temp_dir, session_id=session_id)

        payoffs = [torch.tensor([[1.0, -1.0], [-1.0, 1.0]])]
        collector.set_payoffs(payoffs)

        # Record Chunk 0 (steps 1 to 5)
        for step in range(1, 6):
            strats = [torch.tensor([0.5, 0.5])]
            collector.record_step(
                step=step,
                strategies=strats,
                expected_payoffs=[0.0],
                instant_regrets=[0.0],
                cum_regrets=[0.0],
            )

        assert len(collector.history_steps) == 5

        # Flush Chunk 0
        chunk0_file = collector.flush_to_disk()
        assert chunk0_file is not None
        assert os.path.exists(chunk0_file)
        assert "chunk_0000" in chunk0_file

        # Verify RAM buffer was cleared to maintain O(1) memory
        assert len(collector.history_steps) == 0
        assert len(collector.history_strategies) == 0

        # Record Chunk 1 (steps 6 to 10)
        for step in range(6, 11):
            strats = [torch.tensor([0.6, 0.4])]
            collector.record_step(
                step=step,
                strategies=strats,
                expected_payoffs=[0.1],
                instant_regrets=[0.05],
                cum_regrets=[0.2],
            )

        chunk1_file = collector.flush_to_disk()
        assert chunk1_file is not None
        assert "chunk_0001" in chunk1_file

        # Test load_experiment_stats concatenates both chunks seamlessly
        dataset = load_experiment_stats(output_dir=temp_dir, session_id=session_id)

        assert dataset["session_id"] == session_id
        assert torch.all(dataset["steps"] == torch.arange(1, 11))
        assert len(dataset["strategies"]) == 1
        assert dataset["strategies"][0].shape == (10, 2)
        assert dataset["expected_payoffs"].shape == (10, 1)
        assert dataset["cum_regrets"].shape == (10, 1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
