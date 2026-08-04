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
        steps = list(range(1, 6))
        strats = torch.tensor([[[0.5, 0.5]]]).repeat(5, 1, 1, 1) # (5, 1, 1, 2)
        u_vecs = torch.zeros((5, 1, 1, 2))
        cum_u = torch.zeros((5, 1, 1, 2))
        cum_p = torch.zeros((5, 1, 1))
        
        collector.record_batch(
            steps=steps,
            strats=strats,
            logits=None,
            u_vecs=u_vecs,
            cum_u=cum_u,
            cum_p=cum_p,
            action_sizes=[2]
        )

        assert len(collector.history_steps) == 5

        # Flush Chunk 0
        chunk0_file = collector.flush_to_disk()
        assert chunk0_file is not None
        assert os.path.exists(chunk0_file)
        assert "chunk_0000" in chunk0_file

        # Verify RAM buffer was cleared to maintain O(1) memory
        assert len(collector.history_steps) == 0
        assert len(collector.history_strategies[0]) == 0

        # Record Chunk 1 (steps 6 to 10)
        steps = list(range(6, 11))
        strats = torch.tensor([[[0.6, 0.4]]]).repeat(5, 1, 1, 1)
        u_vecs = torch.tensor([[[0.1, 0.1]]]).repeat(5, 1, 1, 1) # E[u] = 0.1, BR = 0.1 -> regret = 0
        # for cum_regrets = 0.2, cum_u max - cum_p = 0.2
        cum_u = torch.tensor([[[0.2, 0.0]]]).repeat(5, 1, 1, 1) 
        cum_p = torch.zeros((5, 1, 1))

        collector.record_batch(
            steps=steps,
            strats=strats,
            logits=None,
            u_vecs=u_vecs,
            cum_u=cum_u,
            cum_p=cum_p,
            action_sizes=[2]
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
