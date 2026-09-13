import argparse
import csv
import os
import yaml
from rich.console import Console

from src.config.schemas import ExperimentConfig
from src.engine.optimizer import CMAESGameOptimizer

console = Console()

def load_base_config() -> ExperimentConfig:
    with open("configs/default_omwu.yaml", "r") as f:
        return ExperimentConfig(**yaml.safe_load(f))

def main():
    parser = argparse.ArgumentParser(description="Deterministic Scaling Benchmark for Regret Dynamics")
    parser.add_argument("--min_actions", type=int, default=2, help="Minimum action size A (default: 2)")
    parser.add_argument("--max_actions", type=int, default=10, help="Maximum action size A (default: 10)")
    parser.add_argument("--algorithms", nargs="+", default=["omwu"], help="Learning algorithms (omwu, mwu, mirror_prox, dmwu)")
    parser.add_argument("--objectives", nargs="+", default=["delta_reg", "envelope_trend", "envelope_trend_log"], help="Objectives to run")
    parser.add_argument("--out", type=str, default="benchmark_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    csv_filename = args.out

    # Write CSV header if file doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "A", "Objective", "Peak_Regret", "Delta_Trend", "Best_Fitness", "Session_ID"])

    console.print(f"[bold cyan]Starting Benchmark: A={args.min_actions} to {args.max_actions} | Algos={args.algorithms}[/bold cyan]")

    for algo in args.algorithms:
        for A in range(args.min_actions, args.max_actions + 1):
            for obj_type in args.objectives:
                console.print(f"\n[bold green]Running Algo='{algo}' | A={A} | Objective='{obj_type}'[/bold green]")
                
                # 1. Load a fresh config to prevent mutation bleed
                conf = load_base_config()

                # 2. Apply Mathematical Rules of Thumb Overrides
                conf.game.action_sizes = [A, A]
                conf.cmaes.objective_type = obj_type
                conf.dynamic.algorithm = algo
                
                conf.execution.total_steps = int(50000 * A)
                conf.execution.quiet = False  # Show progress bars
                
                conf.cmaes.population_size = int(20 * A)
                conf.cmaes.maxfevals = int(2000 * A)  # Locks generations to exactly 100
                conf.cmaes.restarts = 1
                
                conf.cmaes.lambda_reg = 1.0 * A
                conf.cmaes.gamma_volatility = 1.0 * A
                conf.cmaes.logit_penalty_weight = 50.0 * A
                
                # 3. Run Optimization
                optimizer = CMAESGameOptimizer(conf)
                try:
                    best_game, best_factors = optimizer.optimize()
                    
                    # Extract True Regret Metrics
                    peak_regret = best_factors.get("regret", 0.0)
                    delta_trend = best_factors.get("delta_trend", best_factors.get("delta", 0.0))
                    best_fitness = best_factors.get("fitness", 0.0)

                    # 4. Append Results to CSV
                    with open(csv_filename, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([algo, A, obj_type, f"{peak_regret:.4f}", f"{delta_trend:.4f}", f"{best_fitness:.4f}", optimizer.session_id])
                    
                    console.print(f"[yellow]Result -> Peak Regret: {peak_regret:.4f} | Delta Trend: {delta_trend:.4f}[/yellow]")
                    
                except Exception as e:
                    console.print(f"[bold red]Failed on {algo}, A={A}, obj={obj_type}. Error: {e}[/bold red]")

    console.print(f"\n[bold cyan]Benchmark Complete! Results saved to {csv_filename}[/bold cyan]")

if __name__ == "__main__":
    main()
