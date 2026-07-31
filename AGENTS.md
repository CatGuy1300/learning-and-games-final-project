# AGENTS.md - Developer & Agent Guidelines

This repository hosts a high-performance Python package for simulating learning dynamics in $N$-player general-sum games. Future AI subagents working on this codebase MUST follow the guidelines below.

---

## 🎯 Core Principles

1. **Token Frugality**:
   - Keep responses concise and focused.
   - Prefer precise target code edits (`replace_file_content` / `multi_replace_file_content`).
   - Do NOT output massive text dumps or redundant code snippets.

2. **Separation of Concerns & File Organization**:
   - Keep files small, clean, and single-purpose (< 300 lines per file).
   - Place schemas in `src/config/`, games in `src/games/`, dynamics in `src/dynamics/`, metrics in `src/metrics/`, engine logic in `src/engine/`.

3. **Strict Typing & Documentation**:
   - Use explicit Python type annotations (`torch.Tensor`, `Optional`, `Tuple`, `List`).
   - Add concise NumPy-style docstrings for all classes, methods, and functions.

4. **GPU Optimization & Vectorization**:
   - **CRITICAL**: You MUST read and follow `OPTIMIZATION_STANDARDS.md` before making any modifications to the mathematical engines, learning dynamics, or environment execution loops.
   - Perform tensor operations in PyTorch natively on GPU (`cuda` / `cpu`).
   - Use batched `einsum` or `matmul` for expected payoffs and best response vectors. Avoid Python `for` loops across action dimensions.

5. **Validation & Utility Bounds**:
   - Validate input configurations using `src/config/validation.py`.
   - Utility matrices MUST respect the configured range $[u_{\min}, u_{\max}]$ (e.g. $[-1.0, 1.0]$).

6. **Checkpointing & Reproducibility**:
   - Always log session run tokens (UUIDs) and step indices in state dictionaries.
   - Use `src/utils/reproducibility.py` to fix PyTorch/NumPy/Python random seeds.

---

## 🛠 Command Cheatsheet

- Run tests: `uv run pytest -v tests/`
- Run CLI: `uv run learning-games run --config configs/default_omwu.yaml`
- Format & Lint: `uv run ruff check src/`
