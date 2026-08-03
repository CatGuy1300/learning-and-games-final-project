
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from src.dynamics.continuous import OMWUContinuous


class ODEAdjointOptimizer:
    """Optimizes game matrices to maximize Continuous Regret using the ODE Adjoint method."""
    
    def __init__(
        self, 
        A1: int, 
        A2: int, 
        eta: float, 
        N_steps: int, 
        projection_mode: str = "tanh", 
        lr: float = 0.1, 
        device: torch.device = None
    ):
        self.A1 = A1
        self.A2 = A2
        self.eta = eta
        self.N_steps = N_steps
        self.T = N_steps * eta
        self.projection_mode = projection_mode
        self.lr = lr
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def optimize(self, epochs: int = 50, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs Adam optimization loop using backpropagation through the ODE solver.
        
        Returns:
            Tuple containing:
            - final_U1 (optimized Player 1 payoffs)
            - final_U2 (optimized Player 2 payoffs)
            - states (the final ODE trajectory tensor)
            - t (the time tensor used for integration)
        """
        torch.manual_seed(seed)
        A1, A2 = self.A1, self.A2
        
        if self.projection_mode == "tanh":
            # W represents unconstrained weights
            W1 = nn.Parameter(torch.randn(A1, A2, device=self.device) * 0.1)
            W2 = nn.Parameter(torch.randn(A2, A1, device=self.device) * 0.1)
            params = [W1, W2]
        else:
            # U represents exact payoff matrices
            U1 = nn.Parameter(torch.randn(A1, A2, device=self.device) * 0.1)
            U2 = nn.Parameter(torch.randn(A2, A1, device=self.device) * 0.1)
            params = [U1, U2]
            
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
        
        final_states = None
        final_t = torch.linspace(0, self.T, self.N_steps, device=self.device)
        
        with Progress(
            TextColumn(f"[bold blue]ODE Optimization ({self.projection_mode})"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.description}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Initializing...", total=epochs)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if self.projection_mode == "tanh":
                    curr_U1 = torch.tanh(W1)
                    curr_U2 = torch.tanh(W2)
                else:
                    curr_U1 = U1
                    curr_U2 = U2
                    
                dyn = OMWUContinuous(curr_U1, curr_U2, self.eta)
                state0 = torch.zeros(2*A1 + 2*A2 + 2, device=self.device)
                
                states = odeint(dyn, state0, final_t, method='dopri5')
                
                final_state = states[-1]
                idx = A1 + A2
                Z1_T = final_state[idx : idx+A1]; idx += A1
                P1_T = final_state[idx : idx+1]; idx += 1
                Z2_T = final_state[idx : idx+A2]; idx += A2
                P2_T = final_state[idx : idx+1]
                
                regret_1 = (Z1_T.max() - P1_T) / self.eta
                regret_2 = (Z2_T.max() - P2_T) / self.eta
                total_regret = regret_1 + regret_2
                
                loss = -total_regret
                loss.backward()
                
                optimizer.step()
                
                if self.projection_mode == "clamp":
                    with torch.no_grad():
                        U1.clamp_(-1.0, 1.0)
                        U2.clamp_(-1.0, 1.0)
                        
                progress.update(task, advance=1, description=f"Total Regret: {-loss.item():.4f}")
                final_states = states.detach()
                
        if self.projection_mode == "tanh":
            final_U1 = torch.tanh(W1).detach()
            final_U2 = torch.tanh(W2).detach()
        else:
            final_U1 = U1.detach()
            final_U2 = U2.detach()
            
        return final_U1, final_U2, final_states, final_t
