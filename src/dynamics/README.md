# Learning Dynamics Library

This module (`src/dynamics/`) implements highly optimized, PyTorch-accelerated discrete-time learning dynamics for general-sum $N$-player games. 

These algorithms update mixed strategies on the simplex based on incoming utility vectors. All algorithms implemented here are subclasses of `BaseLearningDynamic` and support fully batched operations via CUDAGraphs.

---

## 1. Multiplicative Weights Update (MWU)
**File:** `mwu.py` | **Config Key:** `algorithm: "mwu"`

The standard exponential gradient descent algorithm on the simplex. MWU maps cumulative utility vectors to probabilities via the softmax function.

### Mathematical Formulation
$$ x_i^{(t+1)} = \frac{x_i^{(t)} \exp\left(\eta u_i^{(t)}\right)}{\sum_j x_j^{(t)} \exp\left(\eta u_j^{(t)}\right)} $$

### Parameters
- `eta` ($\eta$): The learning rate. If set to `null`, MWU strictly infers a horizon-dependent learning rate: $\eta = \sqrt{\frac{\log(\max A)}{T}}$.

---

## 2. Optimistic Multiplicative Weights Update (OMWU)
**File:** `omwu.py` | **Config Key:** `algorithm: "omwu"`

A predictive variant of MWU (also known as Optimistic Hedge) that achieves $\mathcal{O}(\text{polylog } T)$ regret in certain games by anticipating the opponent's next move. It uses the previous step's utility $u^{(t-1)}$ as an optimistic predictor of $u^{(t)}$.

### Mathematical Formulation
$$ x_i^{(t+1)} \propto x_i^{(t)} \exp\left( \eta \left[ 2 u_i^{(t)} - u_i^{(t-1)} \right] \right) $$

### Parameters
- `eta` ($\eta$): The learning rate. If set to `null`, OMWU infers a highly stable empirical rate $\eta = \frac{1}{8 \max A}$. If `strict_theory_eta: true` is set in the config, it uses the rigorous $\eta = \frac{1}{16 N \log^4(T)}$.

---

## 3. Dissipative Multiplicative Weights Update (DMWU)
**File:** `dmwu.py` | **Config Key:** `algorithm: "dmwu"`

An experimental variant that introduces a dissipation factor $\gamma$ to the logit updates. This effectively acts as $L_2$ regularization on the unnormalized log-probabilities, dragging the strategy towards the uniform distribution. It is highly effective at stabilizing explosive limit cycles.

### Mathematical Formulation
DMWU adapts the control-theoretic principles of **Dissipative Gradient Descent Ascent (DGDA)** (Zheng et al., 2024) to the probability simplex via Dual-Space Mirror Descent.

Instead of operating directly on probabilities $x$, DMWU operates on unconstrained logits $w$, utilizing an **augmented state space** ($\hat{w}$) to apply a high-pass friction filter:
$$ w_{k+1} = w_k + \eta u_k - \gamma(w_k - \hat{w}_k) $$
$$ \hat{w}_{k+1} = \hat{w}_k + \gamma(w_k - \hat{w}_k) $$
$$ x_{k+1} = \text{softmax}(w_{k+1}) $$

### Theoretical Properties & Connections

#### 1. Invariance of the Nash Equilibrium
A critical flaw of standard $L_2$ logit weight-decay (e.g., $w_{k+1} = (1-\gamma)w_k + \eta u_k$) is that it alters the fundamental game matrix, pulling the equilibrium point artificially toward the uniform distribution. 

DMWU completely avoids this via its augmented state $\hat{w}$. At convergence (steady state), the auxiliary update dictates:
$$ \hat{w}_{k+1} = \hat{w}_k \implies \gamma(w_k - \hat{w}_k) = 0 \implies w = \hat{w} $$
Substituting $w = \hat{w}$ into the main update rule reveals that the friction term explicitly vanishes to $0$. Thus, the fixed points of DMWU remain mathematically identical to the true Nash Equilibria of the underlying game!

#### 2. Shift-Invariant Numerical Safety
To prevent `float64` overflow during the Softmax exponentiation, DMWU dynamically subtracts the maximum logit value from the state. Because the high-pass filter relies entirely on the *difference* $(w - \hat{w})$, subtracting the exact same constant $c = \max(w)$ from *both* $w$ and $\hat{w}$ preserves the relative friction mathematically while rendering the algorithm immune to precision explosion.

### Parameters
- `dmwu_gamma` ($\gamma$): The dissipation rate (default: `0.1`). Higher values pull the strategy stronger toward uniformity. *Note: In benchmarking scripts, this should be inversely scaled by $A$ to prevent overpowered dissipation in high dimensions.*
- `eta` ($\eta$): If `null`, defaults to $\eta = \frac{1}{4 \max A}$.

---

## 4. Mirror Prox (Extragradient)
**File:** `mirror_prox.py` | **Config Key:** `algorithm: "mirror_prox"`

An implementation of Nemirovski's Mirror Prox algorithm (2004). Unlike the other algorithms, Mirror Prox requires a two-step "Predictor-Corrector" cycle per iteration. It takes a half-step to peek at the gradient, then applies that future gradient to the original position.

### Mathematical Formulation
1. **Predictor (Half-step):**
$$ x_{half}^{(t)} \propto x^{(t)} \exp\left( \eta u(x^{(t)}) \right) $$
2. **Corrector (Full-step):**
$$ x^{(t+1)} \propto x^{(t)} \exp\left( \eta u(x_{half}^{(t)}) \right) $$

### Parameters
- `eta` ($\eta$): Because Mirror Prox requires simulating intermediate gradients, it requires a much smaller step size to guarantee convergence bounds. If `null`, it infers $\eta = \frac{1}{16 N \max A}$.

---

## Logit Boundary Penalties

To prevent floating-point collapse (e.g. `NaN` generation from `torch.exp()` overflowing), all algorithms in this module support a universal **Logit Penalty** boundary framework.

If `logit_penalty_threshold` is configured (e.g., `12.0`), the algorithms will actively track if the underlying log-probabilities (logits) exceed the threshold. If they do, the framework computes a penalty scalar which is utilized by the optimization engine (e.g. `CMAESGameOptimizer`) to heavily penalize hyperparameter tuning trials that cause mathematical overflow.

- `logit_penalty_mode`: `"absolute"` (penalizes strict distance from zero) or `"centered"` (penalizes distance from the mean, accounting for softmax shift-invariance).
