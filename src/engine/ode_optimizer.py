import os
import uuid
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from src.dynamics.continuous import OMWUContinuous


class ODEAdjointOptimizer:
    """Optimizes game matrices to maximize Continuous Regret using the ODE Adjoint method."""
    
    def __init__(
        self, 
        action_sizes: list[int], 
        eta: float | None = None, 
        N_steps: int = 2000, 
        projection_mode: str = "tanh", 
        lr: float = 0.1, 
        device: torch.device | None = None,
        logit_penalty_threshold: float | None = None,
        logit_penalty_norm: int = 2,
        logit_penalty_weight: float = 0.0,
        logit_penalty_average: bool = True,
        logit_penalty_mode: str = "absolute",
        objective_type: str = "final_regret",
        T1_ratio: float = 0.8,
        lambda_reg: float = 1.0,
        gamma_volatility: float = 2.0,
        ode_method: str = "rk4",
        dtype: torch.dtype = torch.float64,
        session_id: str | None = None
    ):
        self.action_sizes = action_sizes
        self.num_players = len(action_sizes)
        
        if eta is None:
            self.eta = 1.0 / (8.0 * max(action_sizes))
        else:
            self.eta = eta
            
        self.N_steps = N_steps
        self.T = self.N_steps * self.eta
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
        self.lambda_reg = lambda_reg
        self.gamma_volatility = gamma_volatility
        self.ode_method = ode_method
        self.dtype = dtype
        self.session_id = session_id or str(uuid.uuid4())
        
    def optimize(self, epochs: int = 50, seed: int = 42, num_restarts: int = 5) -> tuple[list[torch.Tensor], list[float], list[float], torch.Tensor, torch.Tensor]:
        """
        Runs Adam optimization loop using backpropagation through the ODE solver.
        """
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
        
        class ODEProgressWrapper(nn.Module):
            def __init__(self, dyn, progress, task_id, total_t, update_freq):
                super().__init__()
                self.dyn = dyn
                self.progress = progress
                self.task_id = task_id
                self.total_t = total_t
                self.update_freq = update_freq
                self.is_backward = False
                self.calls = 0

            def forward(self, t, y):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                    
                self.calls += 1
                if self.calls % self.update_freq == 0:
                    if self.is_backward:
                        self.progress.update(self.task_id, completed=self.total_t - t.item(), description='[magenta]ODE Backward')
                    else:
                        self.progress.update(self.task_id, completed=t.item(), description='[cyan]ODE Forward')
                
                # Clone the output to prevent CUDAGraphs from statically overwriting it during rk4 micro-steps!
                return self.dyn(t, y).clone()

        best_global_loss = float('inf')
        best_global_payoffs = None
        best_global_loss_history = []
        best_global_penalty_history = []
        best_global_final_states = None
        best_global_final_t = None
        
        # Calculate dynamic throttle to strictly enforce ~200 updates max per pass!
        update_freq = max(1, (self.N_steps * 4) // 200)
        
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[status]}"),
            TimeRemainingColumn()
        ) as progress:
            
            task_restart = progress.add_task(f"[bold blue]Restarts ({self.projection_mode})", total=num_restarts, status="")
            task_epoch = progress.add_task("[bold yellow]Epochs", total=epochs, status="Loss: N/A")
            task_ode = progress.add_task("[cyan]ODE Forward", total=self.T, status="")
            
            for restart in range(num_restarts):
                torch.manual_seed(seed + restart)
                progress.update(task_epoch, completed=0, status="Loss: N/A")
                
                if self.projection_mode == "tanh":
                    params = nn.ParameterList([nn.Parameter(torch.randn(*self.action_sizes, device=self.device, dtype=self.dtype) * 0.1) for _ in range(self.num_players)])
                else:
                    params = nn.ParameterList([nn.Parameter(torch.randn(*self.action_sizes, device=self.device, dtype=self.dtype) * 0.1) for _ in range(self.num_players)])
                    
                optimizer = torch.optim.Adam(params, lr=self.lr)
                
                final_states = None
                final_t = torch.linspace(0, self.T, self.N_steps, device=self.device, dtype=self.dtype)
                
                loss_history = []
                penalty_history = []
                
                # Instantiate dynamics EXACTLY ONCE per restart to preserve the JIT compilation cache!
                dummy_payoffs = [torch.zeros_like(p) for p in params]
                dyn = OMWUContinuous(dummy_payoffs, self.eta, logit_penalty_threshold=self.logit_penalty_threshold, logit_penalty_norm=self.logit_penalty_norm, logit_penalty_mode=self.logit_penalty_mode)
                
                # JIT Compile the ODE velocity function!
                import sys
                backend = "cudagraphs" if (sys.platform == "win32" and self.device.type == "cuda") else "inductor"
                dyn_compiled = torch.compile(dyn, backend=backend)
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    
                    if self.projection_mode == "tanh":
                        curr_payoffs = [torch.tanh(p) for p in params]
                    else:
                        curr_payoffs = [p for p in params]
                        
                    # Dynamically patch the compiled game references to the new differentiable payoffs
                    dyn.payoffs = curr_payoffs
                    dyn.game.payoffs = curr_payoffs
                    if dyn.num_players == 2:
                        U1, U2 = curr_payoffs[0], curr_payoffs[1]
                        row1 = torch.cat([torch.zeros((U1.shape[0], U1.shape[0]), device=U1.device, dtype=U1.dtype), U1], dim=1)
                        row2 = torch.cat([U2.T, torch.zeros((U2.shape[1], U2.shape[1]), device=U1.device, dtype=U1.dtype)], dim=1)
                        dyn.J_Vx_static = torch.cat([row1, row2], dim=0)
                    
                    state0 = dyn.get_initial_state(device=self.device, dtype=self.dtype)
                    
                    progress.update(task_ode, completed=0, description='[cyan]ODE Forward')
                    wrapper = ODEProgressWrapper(dyn_compiled, progress, task_ode, self.T, update_freq)
                    
                    states = odeint(wrapper, state0, final_t, method=self.ode_method, options={'step_size': self.eta}, adjoint_params=tuple(params))
                    
                    unpacked = dyn.unpack_states(states, eta=self.eta)
                    
                    if self.objective_type in ["envelope_trend", "envelope_trend_log"]:
                        idx_06 = int((self.T1_ratio - 0.2) * self.N_steps)
                        idx_08 = int(self.T1_ratio * self.N_steps)
                        
                        # Player 1's Regret trajectory
                        Z_array = unpacked["Z"][0]
                        P_array = unpacked["P"][0]
                        
                        beta = 0.05
                        Regret_array = (beta * torch.logsumexp(Z_array / beta, dim=-1) - P_array.squeeze(-1)) / self.eta
                        peak_past = beta * torch.logsumexp(Regret_array[idx_06 : idx_08] / beta, dim=0)
                        peak_future = beta * torch.logsumexp(Regret_array[idx_08 : ] / beta, dim=0)
                        
                        delta_trend = peak_future - peak_past
                        
                        x_late = unpacked["strategies"][0][idx_08:]
                        y_late = unpacked["strategies"][1][idx_08:]
                        
                        amp_x = torch.max(x_late, dim=0).values - torch.min(x_late, dim=0).values
                        amp_y = torch.max(y_late, dim=0).values - torch.min(y_late, dim=0).values
                        volatility_score = torch.sum(amp_x) + torch.sum(amp_y)
                        
                        if self.objective_type == "envelope_trend_log":
                            objective = delta_trend + self.lambda_reg * torch.log(torch.nn.functional.softplus(peak_future) + 1e-8) + (self.gamma_volatility * volatility_score)
                            loss = -objective
                        else:
                            objective = self.lambda_reg * peak_future + delta_trend + (self.gamma_volatility * volatility_score)
                            loss = -objective
                    else:
                        total_regret = sum([r[-1].max() for r in unpacked["regrets"]])
                        loss = -total_regret
                    
                    if self.logit_penalty_threshold is not None and self.logit_penalty_weight > 0.0:
                        penalty = unpacked["penalty"][-1].squeeze()
                        if self.logit_penalty_average:
                            penalty = penalty / self.T
                        loss += self.logit_penalty_weight * penalty
    
                    wrapper.is_backward = True
                    progress.update(task_ode, completed=0, description='[magenta]ODE Backward')
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
                            
                    progress.update(task_epoch, advance=1, status=f"Loss: {loss.item():.4f}")
                    final_states = states.detach()
                    
                final_loss = loss.item()
                
                if final_loss < best_global_loss:
                    best_global_loss = final_loss
                    if self.projection_mode == "tanh":
                        best_global_payoffs = [torch.tanh(p).detach() for p in params]
                    else:
                        best_global_payoffs = [p.detach() for p in params]
                    best_global_loss_history = loss_history
                    best_global_penalty_history = penalty_history
                    best_global_final_states = final_states
                    best_global_final_t = final_t
                    
                progress.update(task_restart, advance=1, status=f"Best Global Loss: {best_global_loss:.4f}")
                
        self.save_results(best_global_payoffs, best_global_loss_history, best_global_penalty_history)
            
        return best_global_payoffs, best_global_loss_history, best_global_penalty_history, best_global_final_states, best_global_final_t
        
    def save_results(self, best_payoffs: list[torch.Tensor], loss_history: list[float], penalty_history: list[float], out_dir: str = "outputs") -> str:
        """Save the best payoff matrices and histories to disk."""
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"ode_opt_best_{self.session_id}.pt")
        
        # Detach everything and convert to CPU for saving
        detached_payoffs = [p.detach().cpu() for p in best_payoffs]
        torch.save({
            "payoffs": detached_payoffs,
            "loss_history": loss_history,
            "penalty_history": penalty_history,
            "eta": self.eta,
            "N_steps": self.N_steps,
            "objective_type": self.objective_type
        }, save_path)
        return save_path

    @classmethod
    def load_results(cls, session_id: str, out_dir: str = "outputs") -> dict:
        """Load previously saved ODE Optimizer best payoff matrices."""
        save_path = os.path.join(out_dir, f"ode_opt_best_{session_id}.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No ODE optimization results found at {save_path}")
        return torch.load(save_path, map_location="cpu", weights_only=False)
