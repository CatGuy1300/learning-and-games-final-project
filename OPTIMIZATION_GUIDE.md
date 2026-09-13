# Optimization Guide: Discovering Worst-Case Games

This repository provides two advanced optimizers designed to automatically discover adversarial game matrices (payoffs) that force learning dynamics (like Optimistic Multiplicative Weights Update) to suffer maximum regret and prevent them from converging to Nash equilibria.

This guide details the mathematical objective functions, regularization parameters, and the differences between the two primary solvers.

---

## 1. Solvers: ODE vs CMA-ES

We provide two distinct optimization backends for searching the space of game matrices.

### `ODEAdjointOptimizer` (Gradient-Based)
Located in `src/engine/ode_optimizer.py`.
- **How it works**: Uses `torchdiffeq` to simulate the continuous-time ODE surrogate of the learning dynamics. It natively backpropagates through the ODE integration steps.
- **When to use**: Extremely fast and highly scalable for larger games (e.g., $10 \times 10$ matrices) where the continuous surrogate is a highly accurate reflection of the discrete dynamics.
- **Constraints**: Requires the objective and the dynamics to be perfectly differentiable.

### `CMAESGameOptimizer` (Derivative-Free)
Located in `src/engine/optimizer.py`.
- **How it works**: Uses the **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) 0th-order search algorithm. It runs the exact, discrete `Engine` simulation.
- **When to use**: Ideal for smaller games ($2 \times 2$, $3 \times 3$) when you want to optimize against the *exact* discrete update rule, which may contain non-differentiable elements, numerical noise, or discontinuous projections.

---

## 2. Objective Functions (`objective_type`)

The goal of the optimizer is to find a game matrix $U$ that maximizes some function of the regret $R(T)$. By default, we frame this as minimizing a loss $\mathcal{L}(U) = -\text{Objective}(U)$.

### `raw`
The simplest objective. It purely maximizes the final cumulative regret at step $T$.
$$ \mathcal{L}_{raw} = - R(T) $$
**Problem**: The optimizer often finds games that cause a massive initial spike in regret, but then the players converge and regret drops to zero. 

### `delta_reg`
Maximizes the *increase* in regret during the final portion of the simulation.
$$ \mathcal{L}_{delta} = - \left( R(T) - R(T_1) + \lambda \log(\max(0, R(T))) \right) $$
- **$T_1$**: The start of the evaluation window, defined by `T1_ratio` (e.g. $0.8 \times T$).
- **$\lambda$ (`lambda_reg`)**: A logarithmic barrier ensuring that the final regret remains strictly positive (preventing the optimizer from pushing both $R(T)$ and $R(T_1)$ deep into negative territory).

### `envelope_trend`
Because trajectories often oscillate wildly, measuring $R(T) - R(T_1)$ at exact timestamps is extremely vulnerable to **Phase-Hacking**. The optimizer will cheat by tweaking the oscillation frequency so that $T_1$ lands in a trough and $T$ lands on a peak, even if the absolute magnitude of the oscillations is shrinking.

The Envelope Trend objective measures the trend of the *peaks* (the envelope) of the regret curve:
$$ \text{Peak}_{past} = \max_{t \in [T_0, T_1]} R(t) $$
$$ \text{Peak}_{future} = \max_{t \in [T_1, T]} R(t) $$
$$ \mathcal{L}_{envelope} = - (\lambda \times \text{Peak}_{future} + (\text{Peak}_{future} - \text{Peak}_{past})) $$
*(Note: In the ODE solver, the `max` is implemented as a differentiable `logsumexp` smooth maximum).*

### `envelope_trend_log`
A numerically stable logarithmic barrier variant of `envelope_trend`.
$$ \mathcal{L}_{env\_log} = - \left( (\text{Peak}_{future} - \text{Peak}_{past}) + \lambda \log(\text{softplus}(\text{Peak}_{future})) \right) $$

### `chunked_envelope_trend`
To create a highly robust objective function that eliminates the "phase-hacking" loophole completely, we implement **Chunked Envelope Regression**. This approach extracts the signal envelope by slicing the evaluation window into many smaller chunks (e.g., 20), finding the peak regret in each chunk, and computing a Least Squares Linear Regression slope across these peaks. This extracts the true "envelope trend" regardless of wavelength.
$$ \text{Slope} = \frac{\sum (x - \bar{x})(\text{Peaks} - \bar{\text{Peaks}})}{\sum (x - \bar{x})^2} $$
$$ \mathcal{L}_{chunked} = - \left( \lambda \times \bar{\text{Peaks}} + \text{Slope} \right) $$

### `chunked_envelope_trend_log`
The logarithmic barrier variant of `chunked_envelope_trend`.
$$ \mathcal{L}_{chunked\_log} = - \left( \text{Slope} + \lambda \log(\text{softplus}(\bar{\text{Peaks}})) \right) $$

---

## 3. Regularization Parameters

### `gamma_volatility` & `volatility_cap` (The "Phase-Hacking" Solution)
Even with envelope tracking, optimizers may find microscopic limit cycles that appear to grow, but are functionally converged. 

**Volatility Score** measures the total amplitude (spread) of the probability strategies over the final evaluation window $[T_1, T]$. To prevent the optimizer from getting distracted by games that have massive volatility but very low regret (e.g., pure cyclic games), we apply a **Hard Cap** (`volatility_cap`, default `0.1`) to the maximum amplitude any single action can contribute:

$$ \text{Amp}_{i, a} = \max_{t \in [T_1, T]} p_{i, a}(t) - \min_{t \in [T_1, T]} p_{i, a}(t) $$
$$ \text{Volatility} = \sum_{i \in \text{Players}} \sum_{a \in A_i} \min(\text{Amp}_{i, a}, \text{volatility\_cap}) $$

By adding $\gamma \times \text{Volatility}$ to the objective, we force the optimizer to discover games where the probabilities swing wildly (up to the cap), completely preventing convergence to Nash equilibria while keeping the optimizer's primary focus on Regret.

### `logit_penalty_threshold`, `logit_penalty_weight`, & `logit_penalty_mode`
In continuous dynamics (like OMWU continuous), variables are often stored in logit space. If a strategy converges to a pure action, or if the logits mathematically drift together uniformly (which doesn't change probabilities but balloons the values), the logits approach $\pm\infty$. This causes severe numerical instability and NaN gradients in ODE solvers.

We apply a penalty to the ODE loss if the logits exceed a certain threshold. There are two modes controlled by `logit_penalty_mode`:

#### 1. `absolute` (Default)
Penalizes the absolute magnitude of the logits if they exceed the threshold $M$.
$$ \mathcal{L}_{penalty} = w_{penalty} \times \max(0, \| \text{logits} \| - M) $$

#### 2. `centered`
Softmax probabilities are invariant to uniform translation (i.e. $\text{softmax}(x) = \text{softmax}(x - c)$). Often, the ODE solver will allow all logits to drift to $+\infty$ uniformly. The `centered` mode subtracts the mean logit for each player before computing the penalty, ensuring we only penalize *actual* divergence toward pure strategies, not harmless uniform drift.
$$ \bar{x}_i = \frac{1}{A_i} \sum_{a=1}^{A_i} x_{i,a} $$
$$ \mathcal{L}_{penalty} = w_{penalty} \times \max(0, \| x_i - \bar{x}_i \| - M) $$

---

## 4. Advanced CMA-ES Control (IPOP & Early Stopping)

The `CMAESGameOptimizer` wrapper provides advanced control structures heavily inspired by the official `cma` package to help automate massive searches:

- `population_size`: Explicitly set the number of games evaluated in parallel per generation. If `None`, the Optuna `cmaes` backend infers it automatically based on dimension.
- `maxiter`: The maximum number of generations the optimizer will run *per single restart*. 
- `maxfevals`: The absolute maximum budget of function evaluations (e.g. population_size $\times$ generations) across *all* restarts.
- `tolfun`: Flat-fitness tolerance. If the difference between the maximum and minimum fitness over the last `tolfun_hist` generations falls below this threshold, the solver triggers an early stop (or an IPOP restart).
- `restarts`: Number of allowed **IPOP (Increasing Population) restarts**. When the solver stops early due to native `cmaes` convergence or custom `tolfun` flattening, it automatically re-initializes and *doubles* the `population_size` to escape local optima.

---

## 5. Example Configuration

```yaml
cmaes:
  sigma: 0.3
  seed: 42
  objective_type: "envelope_trend_log"
  population_size: 40
  maxiter: 500
  maxfevals: 50000
  tolfun: 1e-4
  restarts: 2
  T1_ratio: 0.8
  lambda_reg: 0.5
  gamma_volatility: 2.0
```
