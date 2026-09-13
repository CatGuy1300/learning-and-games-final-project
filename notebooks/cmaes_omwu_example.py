import sys
sys.path.append("..")

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config.schemas import ExperimentConfig, GameConfig, DynamicConfig, ExecutionConfig, CMAESConfig
from src.engine.optimizer import CMAESGameOptimizer
from src.engine.runner import ExperimentRunner
from src.engine.statistics import load_experiment_stats

# Set formatting for plots
plt.style.use('ggplot')




# Setup a small configuration for 2x2 games
# We use a batch size of 20 to run 20 games in parallel per generation

config = ExperimentConfig(
    name="cmaes_omwu_demo",
    game=GameConfig(
        generator="random",
        action_sizes = [2, 2],
        seed=42,
        utility_range=(-1.0, 1.0),
        # payoffs=[
        #     [[0.0, 0.0], [0.0, 0.0]], # Player 1 base
        #     [[0.0, 0.0], [0.0, 0.0]], # Player 2 base
        # ]
    ),
    dynamic=DynamicConfig(
        algorithm="omwu", 
        eta=0.05,
        logit_penalty_threshold=10,
        logit_penalty_norm=2,
        logit_penalty_mode="centered"

    ),
    execution=ExecutionConfig(
        total_steps=40000,
        steps_per_call=200,
        device="auto",
        dtype="float64", # Run CMA-ES in double precision to avoid numerical artifacts!
        compile=True
    ),
    cmaes=CMAESConfig(
        sigma=0.5,
        seed=42,
        objective_type="envelope_trend_log",
        # objective_type="raw",
        population_size=40,             # Number of games evaluated in parallel per generation
        maxiter=5,                      # Max generations per single IPOP restart
        maxfevals=50000,                # Total budget of function evaluations across all restarts
        tolfun=1e-4,                    # Stop generation early if flat-fitness tolerance is reached
        restarts=1,                     # Number of allowed IPOP (Increasing Population) restarts
        T1_ratio=0.8,
        lambda_reg=1,
        gamma_volatility=1.5,
        logit_penalty_weight=20,
        logit_penalty_average=True
    )
)

# Initialize CMA-ES Optimizer
# T1_ratio = 0.5 means we compute delta regret from step 500 to 1000
optimizer = CMAESGameOptimizer(
    base_config=config
)


# Run optimization
best_payoffs, best_factors = optimizer.optimize()

print("Optimization Complete!")
import json
print("\nCMA-ES Final Objective Breakdown:")
print(json.dumps({k: float(v) for k, v in best_factors.items()}, indent=2))
print("\nWorst-case Payoff Matrix Player 1:")
print(np.round(best_payoffs[0].numpy(), 4))
print("\nWorst-case Payoff Matrix Player 2:")
print(np.round(best_payoffs[1].numpy(), 4))
print(optimizer.session_id)


# The session_id was generated during initialization
session_id = optimizer.session_id

# We can load the exact dict saved to disk
saved_data = CMAESGameOptimizer.load_results(session_id)

# Verify it matches perfectly
assert torch.allclose(best_payoffs[0], saved_data["payoffs"][0])
print(f"Successfully reloaded results for session {session_id}!")

# To load from a completely past run, you would just type:
# past_data = CMAESGameOptimizer.load_results("1a2b3c4d")
# best_payoffs = past_data["payoffs"]

best_payoffs



import copy

# Create a validation config with the worst-case matrices
val_config = copy.deepcopy(config)
val_config.parent_session_id = session_id
val_config.dynamic.eta = 0.05
val_config.game.payoffs = [p.numpy().tolist() for p in best_payoffs]
val_config.execution.batch_size = 1
val_config.execution.total_steps = 400000
val_config.execution.steps_per_call = 200
val_config.game.generator = 'custom'
val_config.name = "worst_case_omwu"

# Run the single simulation
runner = ExperimentRunner(val_config)
summary = runner.run()

print(f"Validation Run Complete. Session ID: {summary['session_id']}")


from src.engine.statistics import unpack_stats
from src.utils.visualization import plot_static_trajectories

# Load and unpack the recorded statistics from disk
stats_data = load_experiment_stats(output_dir='outputs', session_id=summary['session_id'])
steps, cum_regrets, strats, logits, instant_payoffs = unpack_stats(stats_data)

# Plot the static N-player trajectories (Zoom in by setting start_step and end_step if desired!)
import torch
cum_action_payoffs = [torch.cumsum(p, dim=0) for p in instant_payoffs]
cum_expected_payoffs = [torch.cumsum(torch.sum(s * p, dim=-1), dim=0) for s, p in zip(strats, instant_payoffs)]
fig, axes = plot_static_trajectories(steps, cum_regrets, strats, title_prefix="OMWU", plot_all_actions=True, start_step=15000, end_step=None, cum_action_payoffs=cum_action_payoffs, cum_expected_payoffs=cum_expected_payoffs, logits=logits)
plt.show()



