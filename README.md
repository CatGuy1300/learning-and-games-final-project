# Learning in Games: Regret Dynamics & Worst-Game Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A PyTorch-accelerated framework for investigating learning dynamics in general-sum multiplayer games, analyzing regret behavior over large horizons $T$, and discovering worst-case game structures using continuous ODE and 0th-order optimization techniques.

---

## 📌 Project Overview

In finite general-sum multiplayer games, when all players employ no-regret learning algorithms like **Optimistic Multiplicative Weights Update (OMWU)** or **MirrorProx**, theoretical upper bounds guarantee low regret scaling such as $\mathcal{O}(\log^4 T)$ or $\mathcal{O}(\log T)$.

This project implements:
1. **$N$-Player General-Sum Game Framework**: PyTorch CUDA/CPU tensor backends for $N$-player finite games with arbitrary action dimensions $(A_1, \dots, A_N)$ and configurable utility ranges $[u_{\min}, u_{\max}]$.
2. **Generic Learning Dynamics**: Simplex-projected dynamics including OMWU (Optimistic Hedge), MirrorProx (Entropy-regularized Extragradient), and MWU.
3. **Long-Horizon Execution Engine**: Periodic checkpointing, session token tracking, streaming statistics serialization (`.pt`), and structured progress reporting (`rich`/`tqdm`).
4. **Reproducible Experiments**: CLI execution via Typer and YAML configuration files, alongside Jupyter Notebook integration for interactive analysis.

> [!TIP]
> **Adversarial Discovery:** For researchers looking to optimize game matrices to discover worst-case scenarios, please consult the [Optimization Guide](OPTIMIZATION_GUIDE.md) for a complete mathematical breakdown of the ODE and CMA-ES objective functions!

---

## 🚀 Quickstart

### Prerequisites
Install [uv](https://github.com/astral-sh/uv) (fast Python package installer & resolver).

### Installation
```bash
git clone https://github.com/CatGuy1300/learning-and-games-final-project.git
cd learning-and-games-final-project
uv sync
```

---

## 💻 Usage

### 1. Run Experiment via CLI
Run OMWU on a 2-player general-sum game using a configuration file:
```bash
uv run learning-games run --config configs/default_omwu.yaml
```

Run a 3-player random game:
```bash
uv run learning-games run --config configs/nplayer_random.yaml
```

### 2. Configure Theorem-Compliant Learning Rates
By default, the library infers a practical constant learning rate ($\eta$) suitable for discovering empirical chaos/cycles over long horizons. However, if you require rigorous mathematical bounds (e.g. $O(\text{polylog } T)$ regret in OMWU), you can activate theoretically-sound decay rates:
```yaml
# In your config.yaml
dynamic:
  algorithm: "omwu"
  strict_theory_eta: true
```

### 3. Resume Experiment from Checkpoint
```bash
uv run learning-games run --resume checkpoints/checkpoint_step_50000.pt
```

### 4. Validate Configuration File
```bash
uv run learning-games validate-config --config configs/default_omwu.yaml
```

### 5. Adversarial Hyperparameter Sweeps (Optuna)
To automatically discover hyperparameter configs that maximize adversarial regret, you can run Bayesian Optimization sweeps using the `sweep` command. We provide 3 distinct objective configurations in `configs/sweeps/`:

```bash
uv run learning-games sweep --config configs/sweeps/sweep_delta.yaml --workers 2
```
- `--workers`: Number of parallel CPU processes to spawn. *(Note: Do not exceed `--workers 2` on a single GPU setup, or the concurrent PyTorch CUDAGraph compilers will crash due to VRAM limits).*
- `--no-resume`: Add this flag to wipe the SQLite database and start the sweep completely from scratch.

*(For a full mathematical breakdown of these sweeps, see [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md))*

### 6. Deterministic Scaling Benchmarks (Paper Generation)
If Optuna sweeps are taking too long or falling victim to reward hacking, you can bypass them entirely using our deterministic Scaling Benchmark script. 

This script systematically iterates through game action sizes $A$, various learning algorithms, and objective functions, dynamically inferring mathematically safe hyperparameter budgets for each dimension. It logs the True Peak Regret directly to a CSV, making it perfect for generating rigorous research paper datasets.

**Run the Benchmark:**
```bash
uv run python scripts/benchmark_scaling.py --algorithms omwu mwu mirror_prox dmwu --out multi_algo_results.csv
```

**Configurable Parameters:**
- `--min_actions`: The starting size for the square symmetric game matrices (Default: 2).
- `--max_actions`: The maximum action size $A$ to scale up to (Default: 10).
- `--algorithms`: Space-separated list of learning algorithms to benchmark (`omwu`, `mwu`, `mirror_prox`, `dmwu`).
- `--objectives`: Space-separated list of objective functions to test.
- `--out`: Custom output CSV filename to prevent mixing column formats with older runs.

**Mathematical Scaling Decisions:**
The script dynamically overwrites `configs/default_omwu.yaml` based on the current action dimension $A$:
- `total_steps = 50,000 * A`: Because learning rate $\eta \propto 1/A$, the simulation requires more steps to reach the same continuous time horizon and test the logit penalty boundaries.
- `population_size = 20 * A`: High-dimensional spaces require a wider search radius.
- `maxfevals = 2,000 * A`: With the population scaling at $20A$, this locks the CMA-ES search budget to exactly **100 generations**, completely preventing exponential wall-clock time explosions.
- `lambda_reg` & `gamma_volatility = 1.0 * A`: Scales regularization penalties to keep up with raw Regret magnitude.
- `logit_penalty_weight = 50.0 * A`: Scales the "electric fence" boundary penalty to prevent float collapse.

---

## 📈 Experiment Tracking & Analysis

The framework features an automated, local JSONL tracking system (`outputs/experiments_log.jsonl`) that logs the full configuration, hyperparameters, and results of every run without requiring external servers like MLflow or Weights & Biases. 

You can explicitly link CMA-ES optimization runs directly to their downstream dynamic validation runs using the `parent_session_id` field in the configuration. This enables seamless end-to-end data analysis pipelines. 

An extensive guide on loading, sorting, and mathematically joining this data (`pd.merge`) using Pandas is provided in the new analysis notebook:
```bash
uv run jupyter notebook notebooks/experiment_analysis.ipynb
```

---

## 📓 Jupyter Notebooks

Launch Jupyter Notebook to explore learning dynamics interactively:
```bash
uv run jupyter notebook notebooks/demo_learning_dynamics.ipynb
```

---

## 🧪 Testing

Run pytest unit tests:
```bash
uv run pytest -v tests/
```

---

## 📂 Project Structure

```
├── configs/               # YAML experiment configurations
│   ├── default_omwu.yaml
│   └── nplayer_random.yaml
├── notebooks/             # Research & analysis notebooks
│   └── demo_learning_dynamics.ipynb
├── src/                   # Source package
│   ├── cli.py             # Typer CLI application
│   ├── config/            # Pydantic schemas & validation rules
│   ├── dynamics/          # OMWU, MirrorProx, MWU learning dynamics
│   ├── engine/            # Runner, Checkpointing, Statistics collector
│   ├── games/             # N-player & matrix game environments & generators
│   ├── metrics/           # Regret, best response, strategy distance calculations
│   └── utils/             # GPU device selection, seed reproducibility, rich logging
├── tests/                 # Automated pytest suite
├── AGENTS.md              # AI agent guidelines & coding standards
└── pyproject.toml         # UV package configuration
```
