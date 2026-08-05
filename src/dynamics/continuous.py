import torch
from torch import nn


class ContinuousGameDynamics(nn.Module):
    """Generic ODE Surrogate Base Class for learning dynamics."""
    
    def __init__(self, U1: torch.Tensor, U2: torch.Tensor, logit_penalty_threshold: float | None = None, logit_penalty_norm: int = 2):
        super().__init__()
        # We assume U1 and U2 are parameters or tensors that require gradients
        self.U1 = U1
        self.U2 = U2
        self.A1, self.A2 = U1.shape
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm

    def compute_w_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the continuous gradient w_dot for the unconstrained logits.
        Subclasses must implement this based on their specific discrete algorithms.
        """
        raise NotImplementedError("Subclasses must implement w_dot logic.")

    def forward(self, t, state):
        """
        Forward pass for torchdiffeq.odeint.
        State packing: [w1, w2, Z1, P1, Z2, P2]
        """
        A1, A2 = self.A1, self.A2
        
        # Unpack state
        idx = 0
        w1 = state[idx : idx+A1]; idx += A1
        w2 = state[idx : idx+A2]; idx += A2
        Z1 = state[idx : idx+A1]; idx += A1
        P1 = state[idx : idx+1]; idx += 1
        Z2 = state[idx : idx+A2]; idx += A2
        P2 = state[idx : idx+1]; idx += 1
        
        has_barrier = len(state) > idx
        if has_barrier:
            B = state[idx : idx+1]; idx += 1
        
        # Compute probabilities
        x = torch.softmax(w1, dim=-1)
        y = torch.softmax(w2, dim=-1)
        
        # Compute specific dynamics for w_dot
        w_dot = self.compute_w_dot(x, y)
        
        # Expected Value Vectors (V)
        V1 = self.U1 @ y
        V2 = self.U2.T @ x
        
        # Continuous Regret integrands
        # Player 1
        Z1_dot = V1
        P1_dot = x @ V1
        
        # Player 2
        Z2_dot = V2
        P2_dot = y @ V2 # Equivalent to x @ self.U2 @ y
        
        # Penalty Integration
        if has_barrier and self.logit_penalty_threshold is not None:
            w_all = state[0 : A1 + A2]
            excess = torch.nn.functional.relu(torch.abs(w_all) - self.logit_penalty_threshold)
            B_dot = torch.sum(excess ** self.logit_penalty_norm).unsqueeze(0)
        elif has_barrier:
            B_dot = torch.zeros(1, device=state.device, dtype=state.dtype)

        # Pack state derivative
        state_components = [
            w_dot,
            Z1_dot,
            P1_dot.unsqueeze(0),
            Z2_dot,
            P2_dot.unsqueeze(0)
        ]
        if has_barrier:
            state_components.append(B_dot)
            
        state_dot = torch.cat(state_components)
        
        return state_dot


class OMWUContinuous(ContinuousGameDynamics):
    """
    Specific OMWU ODE Surrogate using High-Resolution M * w_dot = V math.
    """
    def __init__(self, U1: torch.Tensor, U2: torch.Tensor, eta: float, logit_penalty_threshold: float | None = None, logit_penalty_norm: int = 2):
        super().__init__(U1, U2, logit_penalty_threshold=logit_penalty_threshold, logit_penalty_norm=logit_penalty_norm)
        self.eta = eta

    def compute_w_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A1, A2 = self.A1, self.A2
        
        # Expected Value Vectors
        V1 = self.U1 @ y
        V2 = self.U2.T @ x
        V = torch.cat([V1, V2])
        
        # Softmax Jacobians
        Sigma_x = torch.diag(x) - torch.outer(x, x)
        Sigma_y = torch.diag(y) - torch.outer(y, y)
        
        # Cross-Derivative Blocks of Vector Field
        J12 = self.U1 @ Sigma_y
        J21 = self.U2.T @ Sigma_x
        
        # Mass Matrix M = I - (eta / 2) J
        # Solve M * w_dot = V via Schur Complement on blocks
        coef = self.eta / 2
        S = torch.eye(A1, device=x.device, dtype=x.dtype) - (coef**2) * (J12 @ J21)
        rhs = V1 + coef * (J12 @ V2)
        
        w1_dot = torch.linalg.solve(S, rhs)
        w2_dot = V2 + coef * (J21 @ w1_dot)
        
        w_dot = torch.cat([w1_dot, w2_dot])
        return w_dot
