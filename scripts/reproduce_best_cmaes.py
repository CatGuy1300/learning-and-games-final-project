"""
Reproduce and save the exact trajectory of a best CMA-ES solution.

Usage:
    uv run python scripts/reproduce_best_cmaes.py --session ec9bcf6d --steps-multiplier 2.0
"""

import argparse
import os
import uuid

from src.config.schemas import ExperimentConfig
from src.engine.optimizer import CMAESGameOptimizer
from src.engine.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Reproduce best CMA-ES run.")
    parser.add_argument("--session", type=str, required=True, help="Session ID of the CMA-ES run (e.g. ec9bcf6d)")
    parser.add_argument("--outputs", type=str, default="outputs", help="Directory where CMA-ES results are stored")
    parser.add_argument("--steps", type=int, default=None, help="Explicitly set the total number of steps to simulate")
    parser.add_argument("--steps-multiplier", type=float, default=None, help="Multiply the original optimization horizon by this factor")
    args = parser.parse_args()

    # Load the best results from disk
    print(f"[*] Loading best CMA-ES results for session: {args.session}...")
    try:
        results = CMAESGameOptimizer.load_results(args.session, out_dir=args.outputs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    best_payoffs = results["payoffs"]
    best_regret = results["regret"]
    raw_config = results["config"]

    print(f"[*] Best Regret from optimization: {best_regret:.4f}")

    # Explicitly force custom game generator and parse the optimized payoffs
    raw_config["game"]["generator"] = "custom"
    raw_config["game"]["payoffs"] = [p.numpy().tolist() for p in best_payoffs]

    # Reconstruct the configuration
    config = ExperimentConfig(**raw_config)
    
    # We want to run a single simulation visibly and force stats flushing
    config.execution.batch_size = 1
    config.execution.quiet = False
    config.logging.save_stats_interval = 5000  # Ensure we flush chunks
    
    # Apply step scaling logic
    if args.steps is not None:
        config.execution.total_steps = args.steps
        print(f"[*] Overriding horizon: explicitly set to {args.steps} steps.")
    elif args.steps_multiplier is not None:
        new_steps = int(config.execution.total_steps * args.steps_multiplier)
        config.execution.total_steps = new_steps
        print(f"[*] Scaling horizon: multiplied by {args.steps_multiplier} -> {new_steps} steps.")
    else:
        print(f"[*] Using original optimization horizon: {config.execution.total_steps} steps.")

    # Generate a new session ID for the reproduction run
    new_session = f"reprod_{args.session}_{uuid.uuid4().hex[:4]}"
    config.session_id = new_session
    config.parent_session_id = args.session
    print(f"[*] Starting reproduction run with new session ID: {new_session} (Parent: {args.session})")
    
    # Instantiate the runner with the loaded config
    runner = ExperimentRunner(config)
    
    # Inject the best payoff matrices found by CMA-ES
    runner.game.update_payoffs(best_payoffs)
    
    # Run the simulation!
    # Because we set quiet=False, the runner will pass the progress object
    # to the inner loop, which automatically triggers self.stats_collector.record_batch()
    # and flushes the full .pt trajectory to disk!
    summary = runner.run()
    
    print("\n" + "="*50)
    print("[SUCCESS] Reproduction complete!")
    print(f"Trajectory saved to: {summary['output_dir']}")
    print(f"You can now load and plot the stats in your notebook using:")
    print("-" * 50)
    print(f"stats_data = load_experiment_stats(output_dir='{summary['output_dir']}', session_id='{new_session}')")
    print(f"plot_dashboard(stats_data, title='Reproduction of {args.session}')")
    print("="*50)

if __name__ == "__main__":
    main()
