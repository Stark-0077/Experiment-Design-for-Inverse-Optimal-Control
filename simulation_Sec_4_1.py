#!/usr/bin/env python3
# ==============================================================
# IOC-LQR Full Pipeline (3 Trajectories + Fixed R=2 + IOC Learning)
# With time-series (x,u) plots & phase-plane plot saved as PDFs
# ==============================================================

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,
})

# =========================================================
# Create save folder
# =========================================================
save_dir = "figures_sec_4_1"
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# Utility
# =========================================================
def save_and_show(fig, filename, dpi=600, show=True):
    if not filename.lower().endswith(".pdf"):
        filename = filename.rsplit(".", 1)[0] + ".pdf"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"✅ Figure saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

def block_diag_repeat(M, T):
    return linalg.block_diag(*[M for _ in range(T)])

# =========================================================
# Core mapping: θ → Q (3-parameter PSD Q)
# =========================================================
def map_theta3_Q(theta3, eps_Q=1e-6):
    t1, t2, t3 = theta3
    Q = np.array([[t1**2 + eps_Q, t2 * t3],
                  [t2 * t3,       t3**2 + eps_Q]])
    return Q

# =========================================================
# LQR solver and simulation
# =========================================================
def lqr_finite_horizon_gains(A, B, Q, R, Qf, T):
    n, m = A.shape[0], B.shape[1]
    P = [None] * (T + 1)
    K = [None] * T
    P[T] = Qf.copy()
    for t in range(T - 1, -1, -1):
        Pn = P[t + 1]
        G = R + B.T @ Pn @ B
        Kt = np.linalg.solve(G, B.T @ Pn @ A)
        K[t] = Kt
        Pt = Q + A.T @ (Pn @ (A - B @ Kt))
        P[t] = 0.5 * (Pt + Pt.T)  # symmetrize
    return K, P

def simulate_closed_loop_finite(A, B, K_list, x0):
    T = len(K_list)
    n, m = A.shape[0], B.shape[1]
    x = np.zeros((T + 1, n))
    u = np.zeros((T, m))
    x[0] = x0
    for t in range(T):
        u[t] = -K_list[t] @ x[t]
        x[t + 1] = (A - B @ K_list[t]) @ x[t]
    return x, u

# =========================================================
# Build W(θ)
# =========================================================
def build_lifted_AB(A, B, T):
    n, m = A.shape[0], B.shape[1]
    A_powers = [np.eye(n)]
    for _ in range(1, T):
        A_powers.append(A_powers[-1] @ A)
    Gbar = np.vstack(A_powers)              # (T*n, n)
    Fbar = np.zeros((T * n, T * m))
    for t in range(T):
        for s in range(t):
            Fbar[t * n:(t + 1) * n, s * m:(s + 1) * m] = A_powers[t - 1 - s] @ B
    return Fbar, Gbar

def make_W(A, B, n, m, T, R_fixed, sigma_w=0.1):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    def W_func(theta3):
        Q = map_theta3_Q(theta3)
        Qbar, Rbar = block_diag_repeat(Q, T), block_diag_repeat(R_fixed, T)
        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S)
        W = (1/sigma_w) * (S_inv @ FtQG).T @ (S_inv @ FtQG)
        return 0.5 * (W + W.T)  # symmetrize
    return W_func

# =========================================================
# α★ = r * v_max(W)
# =========================================================
def find_alpha_star(theta3, W_func, r_ball):
    W = W_func(theta3)
    eigvals, eigvecs = np.linalg.eigh(W)
    v_max, v_min = eigvecs[:, -1], eigvecs[:, 0]
    alpha_max = r_ball * v_max / np.linalg.norm(v_max)
    alpha_min = r_ball * v_min / np.linalg.norm(v_min)
    alpha_mid = r_ball * (v_max + v_min) / np.linalg.norm(v_max + v_min)
    return alpha_max, alpha_min, alpha_mid, eigvals[-1], eigvals[0], W

# =========================================================
# IOC Loss and Gradient (finite-diff)
# =========================================================
def rollout_from_theta(A, B, Q, R, alpha, T):
    Qf = np.zeros_like(Q)
    K_list, _ = lqr_finite_horizon_gains(A, B, Q, R, Qf, T)
    return simulate_closed_loop_finite(A, B, K_list, alpha)

def compute_loss_fixedR(theta3, x_target, u_target, A, B, alpha, T, R_fixed):
    Q = map_theta3_Q(theta3)
    x_pred, u_pred = rollout_from_theta(A, B, Q, R_fixed, alpha, T)
    return np.mean((x_pred - x_target)**2) + np.mean((u_pred - u_target)**2)

def compute_grad_fixedR(theta3, x_target, u_target, A, B, alpha, T, R_fixed, eps=1e-6):
    grad = np.zeros_like(theta3)
    base = compute_loss_fixedR(theta3, x_target, u_target, A, B, alpha, T, R_fixed)
    for i in range(len(theta3)):
        t2 = theta3.copy()
        t2[i] += eps
        grad[i] = (compute_loss_fixedR(t2, x_target, u_target, A, B, alpha, T, R_fixed) - base) / eps
    return grad

def theta_rmse(theta_hat, theta_true):
    return np.sqrt(np.mean((theta_hat - theta_true)**2))

# =========================================================
# Plot helpers: (x,u) time series and phase-plane
# =========================================================
def plot_time_series_and_save(A, B, K_list, alpha_list, labels, colors, T):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    markers = ['o', '^', 'x']  # α1, α2, α3

    for i, (alpha, lbl, c) in enumerate(zip(alpha_list, labels, colors)):
        marker = markers[i % len(markers)]
        x_hist, u_hist = simulate_closed_loop_finite(A, B, K_list, alpha)

        ax[0].plot(np.arange(T), x_hist[:-1, 0],
                   '-', lw=2.0, color=c, marker=marker,
                   markersize=4, markevery=3, label=f"{lbl} $x_1$")
        ax[0].plot(np.arange(T), x_hist[:-1, 1],
                   '--', lw=1.8, color=c, marker=marker,
                   markersize=4, markevery=3,
                   markerfacecolor='none', label=f"{lbl} $x_2$")
        ax[1].plot(np.arange(T), u_hist[:, 0],
                   '-', lw=2.0, color=c, marker=marker,
                   markersize=4, markevery=3, label=lbl)

    ax[0].set_xlabel("Time step"); ax[0].set_ylabel("$x$")
    ax[1].set_xlabel("Time step"); ax[1].set_ylabel("$u$")
    ax[0].legend(ncol=2, frameon=True, handlelength=3, borderaxespad=0.6)
    ax[1].legend(frameon=True, handlelength=3, borderaxespad=0.6)
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout()


    # Save split PDFs copied from combined axes
    for i, name in enumerate(["state_trajectory", "control_sequence"]):
        single_fig, single_ax = plt.subplots(figsize=(5, 4))
        for line in ax[i].get_lines():
            single_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),                
                markersize=line.get_markersize(),       
                markerfacecolor=line.get_markerfacecolor(),
                markeredgecolor=line.get_markeredgecolor(),
                markevery=line.get_markevery()          
            )

        single_ax.set_xlabel(ax[i].get_xlabel())
        single_ax.set_ylabel(ax[i].get_ylabel())
        single_ax.grid(alpha=0.3)

        single_ax.legend(
            ncol=2 if i == 0 else 1,
            frameon=True,
            handlelength=3,
            borderaxespad=0.6
        )

        single_fig.tight_layout()
        save_and_show(single_fig, f"{name}.pdf", show=False)

    save_and_show(fig, "state-control-traj.pdf", show=False)


def plot_phase_plane_and_save(A, B, K_list, alpha_list, labels, colors, r_ball):
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))

    # α₁, α₂, α₃ marker 
    markers = ['o', '^', 'x']

    for i, (alpha, lbl, c) in enumerate(zip(alpha_list, labels, colors)):
        marker = markers[i % len(markers)]  

        x_hist, _ = simulate_closed_loop_finite(A, B, K_list, alpha)

        ax2.plot(
            x_hist[:, 0], x_hist[:, 1], '-', lw=1.8,
            marker=marker, markersize=3.8, color=c, label=lbl
        )

        ax2.plot(
            x_hist[0, 0], x_hist[0, 1],
            marker, color=c, markersize=7,
            markerfacecolor='none', markeredgewidth=1.4
        )

        ax2.plot(
            x_hist[-1, 0], x_hist[-1, 1],
            'x', color=c, markersize=7, mew=1.4
        )

    circle = plt.Circle((0, 0), r_ball, color='gray',
                        fill=False, linestyle='--', alpha=0.5)
    ax2.add_patch(circle)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal', 'box')
    ax2.legend(frameon=True, ncol=1)
    plt.tight_layout()
    save_and_show(fig2, "phase-plane.pdf", show=False)


# =========================================================
# Main Pipeline
# =========================================================
def main():
    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(1234)

    # Experiment settings
    r_ball = 3.0
    T = 50
    lr = 0.1
    num_iters = 6000
    R_fixed = np.array([[5]])  # Fixed R = 2 (matches script title/prints)

    # System A (double integrator)
    dt = 0.1
    A = np.array([[1, dt],
                  [0, 1]])
    B = np.array([[0.5 * dt * dt],
                  [dt]])

    n, m = A.shape

    print("\n=== Step 1. True parameters (R fixed = 5.0) ===")
    theta_true = np.array([5.0, 5.0, 5.0])
    # theta_true = np.array([3.0, 3.0, 3.0])
    print("θ_true =", theta_true)

    # noise level
    # sigma_w = 0.01  # standard deviation of measurement noise
    sigma_w = 0.001

    # Info geometry
    W_func = make_W(A, B, n, m, T, R_fixed, sigma_w=sigma_w)
    alpha_max, alpha_min, alpha_mid, λ_max, λ_min, W = find_alpha_star(theta_true, W_func, r_ball)
    print(f"λ_max={λ_max:.4f}, λ_min={λ_min:.4f}, ratio={λ_max/λ_min:.2f}, Tr(W)={np.trace(W):.4f}")

    info_vals = []
    for i, alpha in enumerate([alpha_max, alpha_min, alpha_mid], 1):
        val = float(alpha.T @ W @ alpha)
        info_vals.append(val)
        print(f"α{i}: info = {val:.6f}")
    print(f"Info ratio (max/min) = {max(info_vals)/min(info_vals):.3f}")

    # LQR gains under true Q
    Q_true = map_theta3_Q(theta_true)
    Qf = np.zeros_like(Q_true)
    K_list, _ = lqr_finite_horizon_gains(A, B, Q_true, R_fixed, Qf, T)

    # Three initial states (α’s)
    alpha_list = [alpha_max, alpha_min, alpha_mid]
    # labels = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$"]
    labels = [r"$\alpha_\max$", r"$\alpha_\min$", r"$\alpha_{\text{med}}$"]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # -----------------------------------------------------
    # NEW: Plot (x,u) time series & phase-plane (PDFs)
    # -----------------------------------------------------
    plot_time_series_and_save(A, B, K_list, alpha_list, labels, colors, T)
    plot_phase_plane_and_save(A, B, K_list, alpha_list, labels, colors, r_ball)

    # =====================================================
    # Step 2. IOC Learning (R fixed)
    # =====================================================
    print("\n=== Step 2. IOC Learning (R=5 fixed) ===")
    # theta_init = theta_true + 0.8 * rng.normal(size=theta_true.shape)
    

    sigma = 3
    np.random.seed(666) #333
    theta_init = theta_true + sigma * np.random.random(len(theta_true)) - sigma / 2

    # theta_init = np.array([4.5,5.5,4.5])/2
    theta_init = np.array([6.0, 6.0, 6.0])

    print("θ_init_guess (randomized) =", theta_init)

    results = {}
    for lbl, alpha_used, c in zip(labels, alpha_list, colors):
        x_true, u_true = rollout_from_theta(A, B, Q_true, R_fixed, alpha_used, T)
        wx = sigma_w * np.random.randn(*x_true.shape)  # same shape as x_true
        wu = sigma_w * np.random.randn(*u_true.shape)

        x_true = x_true + wx
        u_ture = u_true + wu

        theta_hat = theta_init.copy()
        losses, rmses = [], []
        for it in range(num_iters):
            loss = compute_loss_fixedR(theta_hat, x_true, u_true, A, B, alpha_used, T, R_fixed)
            grad = compute_grad_fixedR(theta_hat, x_true, u_true, A, B, alpha_used, T, R_fixed)
            theta_hat -= lr * grad
            losses.append(loss)
            rmses.append(theta_rmse(theta_hat, theta_true))
        results[lbl] = {"losses": np.array(losses), "rmses": np.array(rmses)}
        print(f"{lbl}: final RMSE={rmses[-1]:.4e}")

    # =====================================================
    # Plot: 4 error metrics, each saved separately + overview
    # =====================================================
    metric_names = ["absolute_loss", "relative_loss", "rmse", "relative_rmse"]
    ylabels = ["Loss of Trajecotry", "Relative Loss", "RMSE", "Relative RMSE"]

    fig3, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for lbl, c in zip(labels, colors):
        losses = results[lbl]["losses"]; rmses = results[lbl]["rmses"]
        axes[0].semilogy(losses, lw=2, color=c, label=lbl)
        axes[1].semilogy(losses / losses[0], lw=2, color=c, label=lbl)
        axes[2].plot(rmses, lw=2, color=c, label=lbl)
        axes[3].plot(rmses / rmses[0], lw=2, color=c, label=lbl)

    for i in range(4):
        fig_single, ax_single = plt.subplots(figsize=(5, 4))
        for lbl, c in zip(labels, colors):
            losses = results[lbl]["losses"]; rmses = results[lbl]["rmses"]
            if i == 0:
                y = losses; ax_single.semilogy(y, lw=2, color=c, label=lbl)
            elif i == 1:
                y = losses / losses[0]; ax_single.semilogy(y, lw=2, color=c, label=lbl)
            elif i == 2:
                y = rmses; ax_single.plot(y, lw=2, color=c, label=lbl)
            else:
                y = rmses / rmses[0]; ax_single.plot(y, lw=2, color=c, label=lbl)
        ax_single.set_xlabel("Iteration")
        ax_single.set_ylabel(ylabels[i])
        ax_single.grid(alpha=0.3)
        ax_single.legend()
        plt.tight_layout()
        save_and_show(fig_single, f"{metric_names[i]}.pdf", show=False)

    for i, a in enumerate(axes):
        a.set_xlabel("Iteration"); a.set_ylabel(ylabels[i])
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig3, "ioc_loss_rmse_overview.pdf", show=True)

    print("\n✅ IOC pipeline finished successfully.")

# =========================================================
if __name__ == "__main__":
    main()
