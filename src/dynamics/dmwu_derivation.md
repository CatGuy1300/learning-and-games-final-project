

# Dissipative Multiplicative Weights Update (DMWU)

**File:** `dmwu.py` | **Config Key:** `algorithm: "dmwu"`

Dissipative Multiplicative Weights Update (DMWU) is a novel discrete-time learning dynamic designed to stabilize multi-agent learning in general-sum games. It adapts the control-theoretic principles of **Dissipative Gradient Descent Ascent (DGDA)** (Zheng et al., 2024) to the probability simplex via Dual-Space Mirror Descent.

Unlike standard $L_2$ regularization (which shifts the Nash Equilibrium towards a uniform distribution), DMWU utilizes an **augmented state space** to apply a high-pass friction filter. This successfully dissipates the "spurious energy" of limit cycles while keeping the true Nash Equilibrium perfectly invariant.

---

### 1. Theoretical Background: Euclidean DGDA
In continuous-time zero-sum and general-sum games, standard algorithms like Gradient Ascent and Multiplicative Weights Update (MWU) lack sufficient internal friction, causing trajectories to endlessly orbit unstable limit cycles (Poincaré recurrence). 

To force convergence, Zheng et al. (2024) proposed Dissipative GDA in Euclidean space. Instead of extrapolating gradients (like Optimistic algorithms), DGDA introduces an auxiliary tracking variable $\hat{x}$ that acts as a geometric moving average of the state. It then penalizes the difference between the current state and this moving average.

For utility maximization (where $V_k$ is the gradient of the expected payoff), the Euclidean DGDA updates are:
$$ x_{k+1} = x_k + \eta V_k - \gamma(x_k - \hat{x}_k) $$
$$ \hat{x}_{k+1} = \hat{x}_k + \gamma(x_k - \hat{x}_k) $$
Where $\eta > 0$ is the learning rate and $\gamma \in (0, 1)$ is the dissipation coefficient.

---

### 2. Dual-Space Derivation of DMWU
To project this dynamic onto the probability simplex $\Delta^A$, we utilize the Dual-Space formulation of Mirror Descent. 

Instead of operating directly on probabilities $x \in \Delta^A$, the algorithm operates on unconstrained logits $w \in \mathbb{R}^A$. The logits are mapped to the primal probability simplex via the Softmax function: 
$$ x^{(i)} = \frac{\exp(w^{(i)})}{\sum_j \exp(w^{(j)})} $$

Applying the DGDA high-pass filter directly in this unconstrained dual space yields the **DMWU** update rules:
$$ w_{k+1} = w_k + \eta V_k - \gamma(w_k - \hat{w}_k) $$
$$ \hat{w}_{k+1} = \hat{w}_k + \gamma(w_k - \hat{w}_k) $$
$$ x_{k+1} = \text{Softmax}(w_{k+1}) $$

*Note on Numerical Stability:* While this can be mathematically translated into a primal-space update involving geometric fractional powers of probabilities ($x_{k+1} \propto x_k^{1-\gamma} \hat{x}_k^\gamma e^{\eta V}$), doing so on digital hardware causes catastrophic floating-point underflow when probabilities approach zero. DMWU is strictly implemented in the dual logit space to guarantee absolute numerical stability.

---

### 3. Key Properties and Advantages

#### A. Invariance of the Nash Equilibrium
A critical flaw of standard logit weight-decay (e.g., $w_{k+1} = (1-\gamma)w_k + \eta V_k$) is that it alters the fundamental game matrix, pulling the equilibrium point artificially toward the uniform distribution. 

DMWU avoids this via its augmented state $\hat{w}$. At convergence (steady state), the auxiliary update dictates:
$$ \hat{w}_{k+1} = \hat{w}_k \implies \gamma(w_k - \hat{w}_k) = 0 \implies w = \hat{w} $$
Substituting $w = \hat{w}$ into the main update rule:
$$ w_{k+1} = w_k + \eta V_k - \gamma(0) \implies \eta V_k = 0 $$
Because the friction term explicitly vanishes to exactly $0$ at steady state, the fixed points of DMWU are mathematically identical to the true Nash Equilibria of the underlying game.

#### B. Shift-Invariant Numerical Safety
The Softmax function is shift-invariant: $\text{Softmax}(w) = \text{Softmax}(w - c)$. To prevent `float64` overflow during exponentiation, our implementation dynamically subtracts the maximum logit value from the state. Because DMWU relies on the difference $(w - \hat{w})$, subtracting the exact same constant $c = \max(w)$ from *both* $w$ and $\hat{w}$ preserves the relative friction mathematically while rendering the algorithm immune to precision explosion.

#### C. Computational Efficiency ($\mathcal{O}(A)$ vs $\mathcal{O}(A^2)$)
Unlike Optimistic MWU (OMWU) or Mirrored Extra-Gradient—which implicitly rely on Jacobian-vector products (extrapolating the gradient)—DMWU stabilizes the system using only historical state vectors. This removes the need for cross-derivative computations, making DMWU highly scalable for games with massive action spaces $A \gg 1$.

---

### 4. Parameters
*   `eta` ($\eta$): The learning rate. Defaults to the theoretically safe worst-case bound: $\frac{1}{4 \cdot \max(A)}$.
*   `dmwu_gamma` ($\gamma$): The dissipation rate (default: `0.1`). Controls the strength of the low-pass geometric filter. Higher values increase friction and dampen limit cycles more aggressively.