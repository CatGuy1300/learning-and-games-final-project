import torch
from torch import nn
from src.games.nplayer_game import NPlayerGame


class ContinuousGameDynamics(nn.Module):
    """Generic ODE Surrogate Base Class for learning dynamics. Supports N players."""
    
    def __init__(self, payoffs: list[torch.Tensor], logit_penalty_threshold: float | None = None, logit_penalty_norm: int = 2, logit_penalty_mode: str = "absolute"):
        super().__init__()
        self.payoffs = [U for U in payoffs]
        self.num_players = len(self.payoffs)
        self.action_sizes = list(self.payoffs[0].shape[-self.num_players:])
        
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.logit_penalty_mode = logit_penalty_mode
        
        device = payoffs[0].device
        dtype = payoffs[0].dtype
        self.game = NPlayerGame(self.payoffs, dtype=dtype, device=device)

    def compute_w_dot(self, strategies: list[torch.Tensor]) -> torch.Tensor:
        """
        Computes the continuous gradient w_dot for the unconstrained logits.
        Subclasses must implement this based on their specific discrete algorithms.
        """
        raise NotImplementedError("Subclasses must implement w_dot logic.")

    def forward(self, t, state):
        """
        Forward pass for torchdiffeq.odeint.
        State packing: [w_1, ..., w_N, Z_1, P_1, ..., Z_N, P_N, B (optional)]
        """
        idx = 0
        
        # 1. Unpack logits
        logits = []
        for A in self.action_sizes:
            logits.append(state[idx : idx + A])
            idx += A
            
        # 2. Unpack continuous regret Z and continuous average probability P
        Z_list = []
        P_list = []
        for A in self.action_sizes:
            Z_list.append(state[idx : idx + A])
            idx += A
            P_list.append(state[idx : idx + 1])
            idx += 1
            
        has_barrier = len(state) > idx
        if has_barrier:
            B = state[idx : idx + 1]
            idx += 1
            
        # 3. Compute probabilities
        strategies = [torch.softmax(w, dim=-1) for w in logits]
        
        # 4. Compute algorithm specific logits gradient
        w_dot = self.compute_w_dot(strategies)
        
        # 5. Expected Value Vectors (V)
        V_list = self.game.get_utility_vectors(strategies)
        
        # 6. Continuous Regret integrands
        Z_dot_list = []
        P_dot_list = []
        for i in range(self.num_players):
            x = strategies[i]
            V = V_list[i]
            Z_dot_list.append(V)
            P_dot_list.append((x @ V).unsqueeze(0))
            
        # 7. Penalty Integration
        if has_barrier and self.logit_penalty_threshold is not None:
            penalty_sum = 0.0
            for i, w_i in enumerate(logits):
                if self.logit_penalty_mode == "centered":
                    w_target = w_i - torch.mean(w_i)
                else:
                    w_target = w_i
                
                excess = torch.nn.functional.relu(torch.abs(w_target) - self.logit_penalty_threshold)
                penalty_sum = penalty_sum + torch.sum(excess ** self.logit_penalty_norm)
            
            B_dot = penalty_sum.unsqueeze(0)
        elif has_barrier:
            B_dot = torch.zeros(1, device=state.device, dtype=state.dtype)

        # 8. Pack state derivative
        state_components = [w_dot]
        for i in range(self.num_players):
            state_components.append(Z_dot_list[i])
            state_components.append(P_dot_list[i])
            
        if has_barrier:
            state_components.append(B_dot)
            
        state_dot = torch.cat(state_components)
        return state_dot


class OMWUContinuous(ContinuousGameDynamics):
    """
    Specific OMWU ODE Surrogate using High-Resolution M * w_dot = V math.
    Generalized to N-players via full block-Jacobian mass matrix inversion.
    """
    def __init__(self, payoffs: list[torch.Tensor], eta: float, logit_penalty_threshold: float | None = None, logit_penalty_norm: int = 2, logit_penalty_mode: str = "absolute"):
        super().__init__(payoffs, logit_penalty_threshold=logit_penalty_threshold, logit_penalty_norm=logit_penalty_norm, logit_penalty_mode=logit_penalty_mode)
        self.eta = eta
        
        self.J_Vx_static = None
        if self.num_players == 2:
            # For 2-player multilinear games, the cross-derivative Jacobian is perfectly static!
            U1 = self.payoffs[0]
            U2 = self.payoffs[1]
            Z11 = torch.zeros((U1.shape[0], U1.shape[0]), device=U1.device, dtype=U1.dtype)
            Z22 = torch.zeros((U2.shape[1], U2.shape[1]), device=U1.device, dtype=U1.dtype)
            
            row1 = torch.cat([Z11, U1], dim=1)
            row2 = torch.cat([U2.T, Z22], dim=1)
            self.J_Vx_static = torch.cat([row1, row2], dim=0)

    def compute_w_dot(self, strategies: list[torch.Tensor]) -> torch.Tensor:
        V_list = self.game.get_utility_vectors(strategies)
        V = torch.cat(V_list)
        
        # Softmax Jacobians: Sigma_i = diag(x_i) - x_i x_i^T
        Sigmas = []
        for x in strategies:
            Sigmas.append(torch.diag(x) - torch.outer(x, x))
            
        # Build block diagonal Sigma matrix
        Sigma_block = torch.block_diag(*Sigmas)
        
        # Exact cross-derivative Jacobian J_Vx = dV/dx
        if self.J_Vx_static is not None:
            J_Vx = self.J_Vx_static
        else:
            # Fallback for N > 2 using generic autograd
            def compute_V(x_flat):
                strats = []
                offset = 0
                for A in self.action_sizes:
                    strats.append(x_flat[offset : offset+A])
                    offset += A
                return torch.cat(self.game.get_utility_vectors(strats))
                
            x_flat = torch.cat(strategies)
            from torch.func import jacrev
            J_Vx = jacrev(compute_V)(x_flat)
        
        # J_w = J_Vx @ Sigma
        J_w = J_Vx @ Sigma_block
        
        # Mass Matrix M = I - (eta / 2) J_w
        I = torch.eye(J_w.shape[0], device=strategies[0].device, dtype=strategies[0].dtype)
        M = I - (self.eta / 2.0) * J_w
        
        # Solve M * w_dot = V
        w_dot = torch.linalg.solve(M, V)
        return w_dot
