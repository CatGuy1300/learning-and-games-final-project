import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch
from mpl_toolkits.mplot3d import Axes3D

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# AGGRESSIVE PATH SIMPLIFICATION FOR O(N) LINE RENDERING
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0

def get_polygon_vertices(num_actions: int) -> np.ndarray:
    if num_actions == 2:
        return np.array([[0, 1], [1, 0]]) 
    elif num_actions == 3:
        return np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    else:
        angles = np.linspace(0, 2 * np.pi, num_actions, endpoint=False)
        angles = np.pi/2 - angles
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def project_simplex(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_actions = p.shape[1]
    if num_actions == 2:
        return p[:, 0], p[:, 1]
    elif num_actions == 3:
        x = p[:, 1] + 0.5 * p[:, 2]
        y = (np.sqrt(3) / 2) * p[:, 2]
        return x, y
    else:
        vertices = get_polygon_vertices(num_actions)
        projected = p @ vertices
        return projected[:, 0], projected[:, 1]

def project_vector(v: np.ndarray, num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = get_polygon_vertices(num_actions)
    projected = v @ vertices
    return projected[:, 0], projected[:, 1]

def draw_simplex_boundary(ax, num_actions):
    vertices = get_polygon_vertices(num_actions)
    if num_actions == 2:
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', alpha=0.3)
        ax.text(vertices[0, 0], vertices[0, 1] - 0.1, "Act 0", ha='center', va='center')
        ax.text(vertices[1, 0] + 0.1, vertices[1, 1], "Act 1", ha='center', va='center')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
    elif num_actions == 3:
        poly = np.vstack((vertices, vertices[0]))
        ax.plot(poly[:, 0], poly[:, 1], 'k-', alpha=0.3)
        ax.text(vertices[0, 0], vertices[0, 1] - 0.05, "Act 0", ha='center', va='top')
        ax.text(vertices[1, 0], vertices[1, 1] - 0.05, "Act 1", ha='center', va='top')
        ax.text(vertices[2, 0], vertices[2, 1] + 0.05, "Act 2", ha='center', va='bottom')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, np.sqrt(3)/2 + 0.2)
    else:
        poly = np.vstack((vertices, vertices[0]))
        ax.plot(poly[:, 0], poly[:, 1], 'k-', alpha=0.3)
        for i in range(num_actions):
            ax.text(vertices[i, 0]*1.15, vertices[i, 1]*1.15, f"Act {i}", ha='center', va='center')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
    ax.axis('off')

def format_probs(probs: np.ndarray):
    return "[" + ", ".join([f"{p:.3f}" for p in probs]) + "]"

def plot_phase_portrait_setup(fig, position, title, num_actions, data, xlabel, ylabel, zlabel):
    if num_actions <= 2:
        ax = fig.add_subplot(*position)
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if data is not None and len(data) > 0:
            ax.set_xlim(data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
            ax.set_ylim(data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
        line, = ax.plot([], [], alpha=0.6, linewidth=1.5)
        dot, = ax.plot([], [], marker='o', markersize=8)
        return ax, line, dot
    else:
        ax = fig.add_subplot(*position, projection='3d')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if data is not None and len(data) > 0:
            ax.set_xlim(data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
            ax.set_ylim(data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
            ax.set_zlim(data[:, 2].min() - 0.1, data[:, 2].max() + 0.1)
        line, = ax.plot([], [], [], alpha=0.6, linewidth=1.5)
        dot, = ax.plot([], [], [], marker='o', markersize=8)
        return ax, line, dot

def update_phase_line(line, dot, data_chunk, num_actions):
    if num_actions <= 2:
        line.set_data(data_chunk[:, 0], data_chunk[:, 1])
        dot.set_data([data_chunk[-1, 0]], [data_chunk[-1, 1]])
    else:
        line.set_data(data_chunk[:, 0], data_chunk[:, 1])
        line.set_3d_properties(data_chunk[:, 2])
        dot.set_data([data_chunk[-1, 0]], [data_chunk[-1, 1]])
        dot.set_3d_properties([data_chunk[-1, 2]])

def create_learning_dynamics_animation(
    steps: np.ndarray,
    cum_regrets: np.ndarray,
    strats: list[np.ndarray],
    logits: list[np.ndarray] = None,
    instant_payoffs: list[np.ndarray] = None,
    cum_action_payoffs: list[np.ndarray] = None,
    cum_expected_payoffs: list[np.ndarray] = None,
    start_step: int = None,
    end_step: int = None,
    separate_regret_plots: bool = False,
    max_frames: int = 300,
    fps: int = 30,
    tail_length: int = 50,
    save_path: str = None
) -> animation.FuncAnimation:
    
    # Filter by start_step and end_step
    mask = np.ones(len(steps), dtype=bool)
    if start_step is not None:
        mask &= (steps >= start_step)
    if end_step is not None:
        mask &= (steps <= end_step)
        
    f_steps = steps[mask]
    f_regrets = cum_regrets[mask]
    
    if len(f_steps) == 0:
        raise ValueError("No data points remain after applying start_step and end_step filters.")

    total_steps = len(f_steps)
    if total_steps > max_frames:
        indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_steps)
        
    num_players = len(strats)
    has_payoffs = instant_payoffs is not None
    has_cum_payoffs = cum_action_payoffs is not None
    
    rows = 3
    if has_cum_payoffs: rows += 1
    if has_payoffs: rows += 1
    
    row_regret = 0
    row_cumpay = 1 if has_cum_payoffs else -1
    row_strat = 2 if has_cum_payoffs else 1
    row_logit = 3 if has_cum_payoffs else 2
    row_pay = (4 if has_cum_payoffs else 3) if has_payoffs else -1
    
    fig = plt.figure(figsize=(4 * num_players + 2, rows * 4.5))
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    
    # Row 1: Regret curve
    line_regs = []
    reg_axes = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_players)))
    
    if separate_regret_plots:
        for i in range(num_players):
            ax = plt.subplot2grid((rows, num_players), (row_regret, i), fig=fig)
            reg_axes.append(ax)
            l, = ax.plot([], [], label=f"Player {i+1}", color=colors[i], linewidth=2)
            line_regs.append(l)
            ax.legend()
    else:
        reg_ax = plt.subplot2grid((rows, num_players), (row_regret, 0), colspan=num_players, fig=fig)
        reg_axes = [reg_ax] * num_players
        for i in range(num_players):
            l, = reg_ax.plot([], [], label=f"Player {i+1}", color=colors[i], linewidth=2)
            line_regs.append(l)
        reg_ax.legend()
    
    # Dynamic Phase Portraits Setup
    num_actions_list = [strats[i].shape[1] for i in range(num_players)]
    
    f_strats_list = []
    f_logits_list = []
    f_payoffs_list = []
    f_cum_payoffs_list = []
    f_cum_expected_list = []
    xy_full_list = []
    
    for i in range(num_players):
        if hasattr(strats[i], 'numpy'):
            f_strats = strats[i].numpy()[mask]
        else:
            f_strats = np.array(strats[i])[mask]
            
        if logits is not None:
            f_logs = logits[i].numpy()[mask] if hasattr(logits[i], 'numpy') else np.array(logits[i])[mask]
        else:
            f_logs = np.log(f_strats + 1e-12)
        f_logs = f_logs - np.mean(f_logs, axis=1, keepdims=True)
        
        if has_payoffs:
            f_pays = instant_payoffs[i].numpy()[mask] if hasattr(instant_payoffs[i], 'numpy') else np.array(instant_payoffs[i])[mask]
        else:
            f_pays = None
            
        f_strats_list.append(f_strats)
        f_logits_list.append(f_logs)
        f_payoffs_list.append(f_pays)
        xy_full_list.append(project_simplex(f_strats))
        
        if has_cum_payoffs:
            if hasattr(cum_action_payoffs[i], 'numpy'):
                f_cp = cum_action_payoffs[i].numpy()[mask]
            else:
                f_cp = np.array(cum_action_payoffs[i])[mask]
            f_cum_payoffs_list.append(f_cp)
            
            if cum_expected_payoffs is not None:
                if hasattr(cum_expected_payoffs[i], 'numpy'):
                    f_cep = cum_expected_payoffs[i].numpy()[mask]
                else:
                    f_cep = np.array(cum_expected_payoffs[i])[mask]
                f_cum_expected_list.append(f_cep)
    
    sub_steps = f_steps[indices]
    
    # Pre-format HUD Text
    hud_texts = []
    for i in range(len(indices)):
        cur_idx = indices[i]
        cur_t = f_steps[cur_idx]
        reg_str = " | ".join([f"P{j+1} Reg: {f_regrets[cur_idx, j]:.4f}" for j in range(num_players)])
        prob_str = " | ".join([f"P{j+1}: {format_probs(f_strats_list[j][cur_idx])}" for j in range(num_players)])
        info_str = f"Step T: {cur_t}  |  {reg_str}\n{prob_str}"
        hud_texts.append(info_str)

    # Cumulative Payoffs Row
    cum_pay_axes, line_cum_pays, dot_cum_pays = [], [], []
    line_cum_exp, dot_cum_exp = [], []
    has_cum_expected = cum_expected_payoffs is not None
    
    if has_cum_payoffs:
        for i in range(num_players):
            ax = plt.subplot2grid((rows, num_players), (row_cumpay, i), fig=fig)
            ax.set_title(f"Player {i+1} Cum. Payoffs")
            
            num_actions = f_cum_payoffs_list[i].shape[1]
            lines_i, dots_i = [], []
            for a in range(num_actions):
                l, = ax.plot([], [], label=f"Act {a}", linewidth=1.5)
                d, = ax.plot([], [], marker='o', markersize=5, color=l.get_color())
                lines_i.append(l)
                dots_i.append(d)
                
            if has_cum_expected:
                l_exp, = ax.plot([], [], label="Actual Exp", color='black', linestyle='--', linewidth=2.5, zorder=5)
                d_exp, = ax.plot([], [], marker='o', markersize=6, color='black', zorder=6)
                line_cum_exp.append(l_exp)
                dot_cum_exp.append(d_exp)
                
            ax.set_xlim(f_steps[0], f_steps[-1])
            local_min = f_cum_payoffs_list[i].min()
            local_max = f_cum_payoffs_list[i].max()
            if has_cum_expected:
                local_min = min(local_min, f_cum_expected_list[i].min())
                local_max = max(local_max, f_cum_expected_list[i].max())
            margin = (local_max - local_min) * 0.05 if local_max != local_min else 0.1
            ax.set_ylim(local_min - margin, local_max + margin)
            ax.set_xlabel("Step T")
            ax.set_ylabel("Cum. Payoffs")
            ax.grid(True)
            ax.legend(fontsize=8)
            
            cum_pay_axes.append(ax)
            line_cum_pays.append(lines_i)
            dot_cum_pays.append(dots_i)

    # Strategy Simplex
    simp_axes, line_strats, dot_strats = [], [], []
    vec_logits, vec_payoffs, track_texts = [], [], []
    
    for i in range(num_players):
        ax = fig.add_subplot(rows, num_players, row_strat * num_players + i + 1)
        ax.set_title(f"Player {i+1} Strategy Simplex")
        draw_simplex_boundary(ax, num_actions_list[i])
        l, = ax.plot([], [], color=colors[i], alpha=0.6, linewidth=1.5)
        d, = ax.plot([], [], marker='o', color=colors[i], markersize=8)
        
        v_log, = ax.plot([], [], color='red', alpha=0.8, linewidth=2, label="Logits Vector")
        v_pay, = ax.plot([], [], color='green', alpha=0.8, linewidth=2, label="Payoff Vector")
        t_track = ax.text(0, 0, "", fontsize=8, alpha=0.8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        simp_axes.append(ax)
        line_strats.append(l)
        dot_strats.append(d)
        vec_logits.append(v_log)
        vec_payoffs.append(v_pay)
        track_texts.append(t_track)
        
        if num_actions_list[i] > 3:
            ax.legend(loc='lower left', fontsize=8)
            
    # Logit Phase Portraits
    log_axes, line_logs, dot_logs = [], [], []
    for i in range(num_players):
        if num_actions_list[i] <= 3:
            ax, l, d = plot_phase_portrait_setup(
                fig, (rows, num_players, row_logit * num_players + i + 1), f"Player {i+1} Logits", num_actions_list[i], 
                f_logits_list[i][indices], "Logit A0", "Logit A1", "Logit A2"
            )
            l.set_color(colors[i])
            d.set_color(colors[i])
            log_axes.append(ax)
            line_logs.append(l)
            dot_logs.append(d)
        else:
            ax = fig.add_subplot(rows, num_players, row_logit * num_players + i + 1)
            ax.axis('off')
            ax.text(0.5, 0.5, "Logits (4D+)\nOmitted", ha='center', va='center')
            log_axes.append(ax)
            line_logs.append(None)
            dot_logs.append(None)
            
    # Payoff Phase Portraits
    pay_axes, line_pays, dot_pays = [], [], []
    if has_payoffs:
        for i in range(num_players):
            if num_actions_list[i] <= 3:
                ax, l, d = plot_phase_portrait_setup(
                    fig, (rows, num_players, row_pay * num_players + i + 1), f"Player {i+1} Payoffs", num_actions_list[i], 
                    f_payoffs_list[i][indices], "Payoff A0", "Payoff A1", "Payoff A2"
                )
                l.set_color(colors[i])
                d.set_color(colors[i])
                pay_axes.append(ax)
                line_pays.append(l)
                dot_pays.append(d)
            else:
                ax = fig.add_subplot(rows, num_players, row_pay * num_players + i + 1)
                ax.axis('off')
                ax.text(0.5, 0.5, "Payoffs (4D+)\nOmitted", ha='center', va='center')
                pay_axes.append(ax)
                line_pays.append(None)
                dot_pays.append(None)
                
    artists = line_regs + line_strats + dot_strats + vec_logits + vec_payoffs + track_texts
    artists.extend([l for l in line_logs if l is not None])
    artists.extend([d for d in dot_logs if d is not None])
    if has_payoffs:
        artists.extend([l for l in line_pays if l is not None])
        artists.extend([d for d in dot_pays if d is not None])
    if has_cum_payoffs:
        for i in range(num_players):
            artists.extend(line_cum_pays[i])
            artists.extend(dot_cum_pays[i])
        if has_cum_expected:
            artists.extend(line_cum_exp)
            artists.extend(dot_cum_exp)
        
    # Setup Regret Axis Properties
    min_reg_global = min(0, f_regrets.min())
    max_reg_global = max(1.0, f_regrets.max() * 1.05)
    
    for i in range(num_players):
        if i > 0 and not separate_regret_plots:
            continue
            
        ax = reg_axes[i]
        ax.set_xlim(f_steps[0], f_steps[-1])
        ax.set_xlabel("Step T")
        ax.set_ylabel("Cumulative Regret")
        ax.grid(True)
        
        if separate_regret_plots:
            local_min = min(0, f_regrets[:, i].min())
            local_max = max(1.0, f_regrets[:, i].max() * 1.05)
            ax.set_ylim(local_min, local_max)
            ax.set_title(f"Player {i+1} Regret Trajectory")
        else:
            ax.set_ylim(min_reg_global, max_reg_global)
            ax.set_title("Cumulative Regret Trajectory")
            
    # Text Annotation HUD
    hud_ax = reg_axes[num_players // 2] if separate_regret_plots else reg_axes[0]
    text_info = hud_ax.text(0.5, 1.05, "", transform=hud_ax.transAxes, 
                            ha='center', va='bottom', fontsize=11, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    def init():
        for a in artists:
            if hasattr(a, 'set_data_3d'):
                a.set_data_3d([], [], [])
            elif hasattr(a, 'set_data'):
                a.set_data([], [])
            if hasattr(a, 'set_3d_properties'):
                a.set_3d_properties([])
            if hasattr(a, 'set_text'):
                a.set_text("")
        text_info.set_text("")
        return artists + [text_info]
        
    def update(frame):
        cur_idx = indices[frame]
        
        # Update Regret Lines with FULL resolution data up to cur_idx
        for i in range(num_players):
            line_regs[i].set_data(f_steps[:cur_idx+1], f_regrets[:cur_idx+1, i])
        
        tail_start_frame = max(0, frame - tail_length)
        start_idx = indices[tail_start_frame]
        
        # Strategies, Logits, Payoffs with FULL resolution within the window
        if has_cum_payoffs:
            for i in range(num_players):
                f_cp = f_cum_payoffs_list[i][:cur_idx+1]
                num_actions = f_cp.shape[1]
                for a in range(num_actions):
                    line_cum_pays[i][a].set_data(f_steps[:cur_idx+1], f_cp[:, a])
                    dot_cum_pays[i][a].set_data([f_steps[cur_idx]], [f_cp[-1, a]])
                if has_cum_expected:
                    f_cep = f_cum_expected_list[i][:cur_idx+1]
                    line_cum_exp[i].set_data(f_steps[:cur_idx+1], f_cep)
                    dot_cum_exp[i].set_data([f_steps[cur_idx]], [f_cep[-1]])
                    
        for i in range(num_players):
            x, y = xy_full_list[i][0][:cur_idx+1], xy_full_list[i][1][:cur_idx+1]
            line_strats[i].set_data(x, y)
            
            px, py = x[-1], y[-1]
            dot_strats[i].set_data([px], [py])
            
            num_a = num_actions_list[i]
            if num_a > 3:
                # Add quiver vectors (normalize them visually)
                log_x, log_y = project_vector(f_logits_list[i][cur_idx:cur_idx+1], num_a)
                norm_log = np.hypot(log_x[0], log_y[0]) + 1e-8
                vec_logits[i].set_data([px, px + (log_x[0]/norm_log) * 0.2], [py, py + (log_y[0]/norm_log) * 0.2])
                
                if has_payoffs:
                    pay_x, pay_y = project_vector(f_payoffs_list[i][cur_idx:cur_idx+1], num_a)
                    norm_pay = np.hypot(pay_x[0], pay_y[0]) + 1e-8
                    vec_payoffs[i].set_data([px, px + (pay_x[0]/norm_pay) * 0.2], [py, py + (pay_y[0]/norm_pay) * 0.2])
                    
                # Add tracking text
                t_str = f"P: {format_probs(f_strats_list[i][cur_idx])}\nL: {format_probs(f_logits_list[i][cur_idx])}"
                if has_payoffs:
                    t_str += f"\nU: {format_probs(f_payoffs_list[i][cur_idx])}"
                track_texts[i].set_text(t_str)
                track_texts[i].set_position((px + 0.05, py + 0.05))
            else:
                vec_logits[i].set_data([], [])
                vec_payoffs[i].set_data([], [])
                track_texts[i].set_text("")
            
            if num_a <= 3:
                update_phase_line(line_logs[i], dot_logs[i], f_logits_list[i][start_idx:cur_idx+1], num_a)
                if has_payoffs:
                    update_phase_line(line_pays[i], dot_pays[i], f_payoffs_list[i][start_idx:cur_idx+1], num_a)
            
        # Update Text Info
        text_info.set_text(hud_texts[frame])
        
        return artists + [text_info]

    ani = animation.FuncAnimation(fig, update, frames=len(indices), init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        if tqdm is not None:
            pbar = tqdm(total=len(indices), desc="Rendering Animation")
            def progress_callback(current_frame, total_frames):
                pbar.update(1)
                if current_frame == total_frames - 1:
                    pbar.close()
        else:
            progress_callback = None
            
        ani.save(save_path, writer=writer, progress_callback=progress_callback)
        print(f"Animation successfully saved to {os.path.abspath(save_path)}")
        
    plt.close(fig)
    return ani

def plot_static_trajectories(steps: np.ndarray, 
                             cum_regrets: np.ndarray, 
                             strats: list[torch.Tensor], 
                             title_prefix: str = "Algorithm",
                             plot_all_actions: bool = False,
                             start_step: int = 0,
                             end_step: int = None,
                             separate_regret_plots: bool = False,
                             cum_action_payoffs: list[torch.Tensor] = None,
                             cum_expected_payoffs: list[torch.Tensor] = None):
    """Plots the static regret and strategy trajectories for an N-player, A-action game.
    
    Note: To plot cumulative action payoffs, pass `cum_action_payoffs`.
    If you only have instantaneous payoffs, you can compute this via `torch.cumsum(instant_payoffs, dim=0)`.
    """
    
    # Slice the temporal arrays for zooming
    start_idx = np.searchsorted(steps, start_step) if start_step is not None else 0
    end_idx = np.searchsorted(steps, end_step, side='right') if end_step is not None else len(steps)
    
    f_steps = steps[start_idx:end_idx]
    f_cum_regrets = cum_regrets[start_idx:end_idx]
    f_strats = [s[start_idx:end_idx] for s in strats]
    f_cum_payoffs = [cp[start_idx:end_idx] for cp in cum_action_payoffs] if cum_action_payoffs is not None else None
    f_cum_expected = [cep[start_idx:end_idx] for cep in cum_expected_payoffs] if cum_expected_payoffs is not None else None
    
    num_players = f_cum_regrets.shape[1]
    has_payoffs = f_cum_payoffs is not None
    rows = 2 if has_payoffs else 1
    
    if separate_regret_plots:
        cols = num_players + 1
        fig = plt.figure(figsize=(7 * cols, 5 * rows))
        reg_axes = [fig.add_subplot(rows, cols, i + 1) for i in range(num_players)]
        strat_ax = fig.add_subplot(rows, cols, cols)
        if has_payoffs:
            pay_axes = [fig.add_subplot(rows, cols, cols + i + 1) for i in range(num_players)]
    else:
        fig = plt.figure(figsize=(14, 5 * rows))
        reg_ax = fig.add_subplot(rows, 2, 1)
        reg_axes = [reg_ax] * num_players
        strat_ax = fig.add_subplot(rows, 2, 2)
        if has_payoffs:
            pay_axes = [fig.add_subplot(rows, num_players, num_players + i + 1) for i in range(num_players)]
        
    colors = plt.cm.tab10(np.linspace(0, 1, num_players))
    
    # 1. Regret Curve
    for i in range(num_players):
        reg_axes[i].plot(f_steps, f_cum_regrets[:, i], label=f"Player {i+1}", color=colors[i])
        reg_axes[i].set_xlabel("Step T")
        reg_axes[i].set_ylabel("Cumulative Regret")
        if separate_regret_plots:
            reg_axes[i].set_title(f"{title_prefix} Player {i+1} Regret")
            reg_axes[i].grid(True)
            reg_axes[i].legend()
            
    if not separate_regret_plots:
        reg_axes[0].set_title(f"{title_prefix} Cumulative Regret")
        reg_axes[0].grid(True)
        reg_axes[0].legend()
    
    # 2. Strategy Trajectory
    linestyles = ['-', '--', ':', '-.']
    for i in range(num_players):
        num_actions = f_strats[i].shape[1]
        actions_to_plot = range(num_actions) if plot_all_actions else [0]
        for a in actions_to_plot:
            ls = linestyles[a % len(linestyles)]
            label = f"P{i+1} Act {a}" if plot_all_actions or num_actions > 2 else f"Player {i+1} Pr(Act {a})"
            strat_ax.plot(f_steps, f_strats[i][:, a].numpy(), color=colors[i], linestyle=ls, label=label)
            
    strat_ax.set_xlabel("Step T")
    strat_ax.set_ylabel("Probability")
    strat_ax.set_title(f"{title_prefix} Strategy Evolution")
    strat_ax.grid(True)
    
    # Move legend outside if too many actions exist
    total_plotted_lines = num_players * (f_strats[0].shape[1] if plot_all_actions else 1)
    if total_plotted_lines > 6:
        strat_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        strat_ax.legend()
        
    # 3. Cumulative Action Payoffs
    if has_payoffs:
        for i in range(num_players):
            num_actions = f_cum_payoffs[i].shape[1]
            for a in range(num_actions):
                ls = linestyles[a % len(linestyles)]
                arr = f_cum_payoffs[i][:, a].numpy() if hasattr(f_cum_payoffs[i][:, a], 'numpy') else np.array(f_cum_payoffs[i][:, a])
                pay_axes[i].plot(f_steps, arr, color=colors[i], linestyle=ls, label=f"Act {a}")
            
            if f_cum_expected is not None:
                arr_exp = f_cum_expected[i].numpy() if hasattr(f_cum_expected[i], 'numpy') else np.array(f_cum_expected[i])
                pay_axes[i].plot(f_steps, arr_exp, color='black', linestyle='--', linewidth=2.5, label="Actual Exp")
                
            pay_axes[i].set_xlabel("Step T")
            pay_axes[i].set_ylabel("Cumulative Payoff")
            pay_axes[i].set_title(f"{title_prefix} Player {i+1} Cum. Payoffs")
            pay_axes[i].grid(True)
            pay_axes[i].legend()
            
    plt.tight_layout()
    return fig, None
