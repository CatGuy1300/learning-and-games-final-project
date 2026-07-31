# Learning in Games: Regret Dynamics & Worst-Game Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A PyTorch-accelerated framework for investigating learning dynamics in general-sum multiplayer games, analyzing regret behavior over large horizons $T$, and discovering worst-case game structures using continuous ODE and 0th-order optimization techniques.

---

## 📌 Project Overview

In finite general-sum multiplayer games, when all players employ no-regret learning algorithms like **Optimistic Multiplicative Weights Update (OMWU)** or **Extra Gradient (EG)**, theoretical upper bounds guarantee low regret scaling such as $\mathcal{O}(\log^4 T)$ or $\mathcal{O}(\log T)$.

This project implements:
1. **$N$-Player General-Sum Game Framework**: PyTorch CUDA/CPU tensor backends for $N$-player finite games with arbitrary action dimensions $(A_1, \dots, A_N)$ and configurable utility ranges $[u_{\min}, u_{\max}]$.
2. **Generic Learning Dynamics**: Simplex-projected dynamics including OMWU (Optimistic Hedge), EG (Optimistic Gradient Descent), MWU, and GDA.
3. **Long-Horizon Execution Engine**: Periodic checkpointing, session token tracking, streaming statistics serialization (`.pt`), and structured progress reporting (`rich`/`tqdm`).
4. **Reproducible Experiments**: CLI execution via Typer and YAML configuration files, alongside Jupyter Notebook integration for interactive analysis.

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

### 2. Resume Experiment from Checkpoint
```bash
uv run learning-games run --resume checkpoints/checkpoint_step_50000.pt
```

### 3. Validate Configuration File
```bash
uv run learning-games validate-config --config configs/default_omwu.yaml
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
│   ├── dynamics/          # OMWU, EG, MWU, GDA learning dynamics
│   ├── engine/            # Runner, Checkpointing, Statistics collector
│   ├── games/             # N-player & matrix game environments & generators
│   ├── metrics/           # Regret, best response, strategy distance calculations
│   └── utils/             # GPU device selection, seed reproducibility, rich logging
├── tests/                 # Automated pytest suite
├── AGENTS.md              # AI agent guidelines & coding standards
└── pyproject.toml         # UV package configuration
```
