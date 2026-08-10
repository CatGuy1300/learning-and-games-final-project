import torch
from torch import nn
from torchdiffeq import odeint

from src.dynamics.continuous import OMWUContinuous


class ODEAdjointOptimizer:
    """Optimizes game matrices to maximize Continuous Regret using the ODE Adjoint method."""
    
    def __init__(
        self, 
        action_sizes: list[int], 
        eta: float, 
        N_steps: int, 
        projection_mode: str = "tanh", 
        lr: float = 0.1, 
        device: torch.device = None,
        logit_penalty_threshold: float | None = None,
        logit_penalty_norm: int = 2,
        logit_penalty_weight: float = 0.0,
        logit_penalty_average: bool = True,
        logit_penalty_mode: str = "absolute",
        objective_type: str = "final_regret",
        T1_ratio: float = 0.8,
        dtype: torch.dtype = torch.float64
    ):
        self.action_sizes = action_sizes
        self.num_players = len(action_sizes)
        self.eta = eta
        self.N_steps = N_steps
        self.T = N_steps * eta
        self.projection_mode = projection_mode
        self.lr = lr
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logit_penalty_threshold = logit_penalty_threshold
        self.logit_penalty_norm = logit_penalty_norm
        self.logit_penalty_weight = logit_penalty_weight
        self.logit_penalty_average = logit_penalty_average
        self.logit_penalty_mode = logit_penalty_mode
        self.objective_type = objective_type
        self.T1_ratio = T1_ratio
        self.dtype = dtype
        
    def optimize(self, epochs: int = 50, seed: int = 42) -> tuple[list[torch.Tensor], list[float], list[float], torch.Tensor, torch.Tensor]:
        """
        Runs Adam optimization loop using backpropagation through the ODE solver.
        """
        torch.manual_seed(seed)
        if self.projection_mode == "tanh":
            # W represents unconstrained weights
            params = nn.ParameterList([nn.Parameter(torch.randn(*self.action_sizes, device=self.device, dtype=self.dtype) * 0.1) for _ in range(self.num_players)])
        else:
            # U represents exact payoff matrices
            params = nn.ParameterList([nn.Parameter(torch.randn(*self.action_sizes, device=self.device, dtype=self.dtype) * 0.1) for _ in range(self.num_players)])
            
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
        
        final_states = None
        final_t = torch.linspace(0, self.T, self.N_steps, device=self.device, dtype=self.dtype)
        
        loss_history = []
        penalty_history = []
        
        with Progress(
            TextColumn(f"[bold blue]ODE Optimization ({self.projection_mode})"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("Regret: {task.fields[regret]:.4f}"),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Initializing...", total=epochs, regret=0.0)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if self.projection_mode == "tanh":
                    curr_payoffs = [torch.tanh(p) for p in params]
                else:
                    curr_payoffs = [p for p in params]
                
                # Instantiate dynamics exactly once per epoch with the projected payoffs.
                # This allows OMWUContinuous to build and cache the exact static Jacobian!
                dyn = OMWUContinuous(curr_payoffs, self.eta, logit_penalty_threshold=self.logit_penalty_threshold, logit_penalty_norm=self.logit_penalty_norm, logit_penalty_mode=self.logit_penalty_mode)
                
                state_size = 2 * sum(self.action_sizes) + self.num_players
                if self.logit_penalty_threshold is not None:
                    state_size += 1
                state0 = torch.zeros(state_size, device=self.device, dtype=self.dtype)
                
                states = odeint(dyn, state0, final_t, method='dopri5')
                
                if self.objective_type in ["envelope_trend", "envelope_trend_log"]:
                    idx_06 = int((self.T1_ratio - 0.2) * self.N_steps)
                    idx_08 = int(self.T1_ratio * self.N_steps)
                    
                    A = self.action_sizes[0]
                    idx = sum(self.action_sizes)
                    Z_array = states[:, idx : idx+A]
                    P_array = states[:, idx+A].unsqueeze(1)
                    
                    # Smooth max via logsumexp to provide dense gradients across the envelope
                    beta = 0.05
                    Regret_array = (beta * torch.logsumexp(Z_array / beta, dim=-1) - P_array.squeeze(-1)) / self.eta
                    
                    peak_past = beta * torch.logsumexp(Regret_array[idx_06 : idx_08] / beta, dim=0)
                    peak_future = beta * torch.logsumexp(Regret_array[idx_08 : ] / beta, dim=0)
                    
                    delta_trend = peak_future - peak_past
                    
                    if self.objective_type == "envelope_trend_log":
                        # lambda_reg is not configurable here yet, so we use a standard 0.1 scalar, similar to CMAES default
                        loss = -(delta_trend + 0.1 * torch.log(torch.nn.functional.softplus(peak_future) + 1e-8))
                    else:
                        loss = -(peak_future + delta_trend)
                    
                    idx_final = sum(self.action_sizes) + sum(self.action_sizes) + self.num_players
                else:
                    final_state = states[-1]
                    idx = sum(self.action_sizes)
                    
                    total_regret = 0.0
                    for A in self.action_sizes:
                        Z_T = final_state[idx : idx+A]; idx += A
                        P_T = final_state[idx]; idx += 1
                        total_regret = total_regret + (Z_T.max() - P_T) / self.eta
                    
                    loss = -total_regret
                    idx_final = idx
                
                if self.logit_penalty_threshold is not None and self.logit_penalty_weight > 0.0:
                    B_T = states[-1, idx_final : idx_final+1]
                    penalty = B_T[0]
                    if self.logit_penalty_average:
                        penalty = penalty / self.T
                    loss += self.logit_penalty_weight * penalty

                loss.backward()
                
                optimizer.step()
                
                if self.projection_mode == "clamp":
                    with torch.no_grad():
                        for p in params:
                            p.clamp_(-1.0, 1.0)
                        
                loss_history.append(-loss.item())
                if self.logit_penalty_threshold is not None and self.logit_penalty_weight > 0.0:
                    penalty_history.append(penalty.item())
                else:
                    penalty_history.append(0.0)
                        
                progress.update(task, advance=1, description=f"Total Regret: {-loss.item():.4f}")
                final_states = states.detach()
                
        if self.projection_mode == "tanh":
            final_payoffs = [torch.tanh(p).detach() for p in params]
        else:
            final_payoffs = [p.detach() for p in params]
            
        return final_payoffs, loss_history, penalty_history, final_states, final_t
