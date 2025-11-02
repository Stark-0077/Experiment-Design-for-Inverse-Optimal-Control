#!/usr/bin/env python3
# ==============================================================
# Motivation examples — 3 Alphas, Full Plots
# Outputs:
#   - figures_sec_4_1/state_trajectory.pdf
#   - figures_sec_4_1/control_sequence.pdf
#   - figures_sec_4_1/state-control-traj.pdf
#   - figures_sec_4_1/phase-plane.pdf
#   - figures_sec_4_1/ioc_loss.pdf
#   - figures_sec_4_1/ioc_rmse.pdf
# ==============================================================

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,
})

# =========================================================
# IO / Utils
# =========================================================
save_dir = "figures_sec_4_1"
os.makedirs(save_dir, exist_ok=True)

def save_and_show(fig, filename, dpi=600, show=False):
    if not filename.lower().endswith(".pdf"):
        filename = filename.rsplit(".", 1)[0] + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"✅ Figure saved to: {path}")
    if show:
        plt.show()
    plt.close(fig)

def block_diag_repeat(M, T):
    return linalg.block_diag(*[M for _ in range(T)])

# =========================================================
# Lifted dynamics:  X̄ = F̄ Ū + Ḡ α
#   X̄ = [x0; ...; x_{T-1}] ∈ R^{Tn}, Ū = [u0; ...; u_{T-1}] ∈ R^{Tm}
# =========================================================
def build_lifted_AB(A, B, T):
    n, m = A.shape[0], B.shape[1]
    A_powers = [np.eye(n)]
    for _ in range(1, T):
        A_powers.append(A_powers[-1] @ A)
    Gbar = np.vstack(A_powers)
    Fbar = np.zeros((T*n, T*m))
    for t in range(1, T):
        for s in range(t):
            Fbar[t*n:(t+1)*n, s*m:(s+1)*m] = A_powers[t-1-s] @ B
    return Fbar, Gbar

# =========================================================
# θ → Q (PSD with 3 params) and its first derivatives
# =========================================================
def map_theta_1_2_3_Q(theta3, eps_Q=1e-6):
    t1, t2, t3 = theta3
    Q = np.array([[t1**2 + eps_Q, t2 * t3],
                  [t2 * t3,       t3**2 + eps_Q]], dtype=float)
    return Q

def dQ_list(theta3):
    t1, t2, t3 = theta3
    dQ1 = np.array([[2*t1, 0.0],
                    [0.0,  0.0]], dtype=float)
    dQ2 = np.array([[0.0,  t3],
                    [t3,   0.0]], dtype=float)
    dQ3 = np.array([[0.0,  t2],
                    [t2, 2*t3]], dtype=float)
    return [dQ1, dQ2, dQ3]

# =========================================================
# Direct rollout (no Riccati): return x_{0:T-1}, u_{0:T-1}
# =========================================================
def rollout_from_theta(A, B, Q, R, alpha, T, ridge=1e-8, Qf=None):
    n, m = A.shape[0], B.shape[1]
    Fbar, Gbar = build_lifted_AB(A, B, T)
    Qbar = block_diag_repeat(Q, T)
    Rbar = block_diag_repeat(R, T)

    FtQF = Fbar.T @ Qbar @ Fbar
    FtQG = Fbar.T @ Qbar @ Gbar
    H = FtQF + Rbar
    rhs = FtQG @ alpha

    if Qf is not None:
        Phi_T = np.linalg.matrix_power(A, T)
        Gamma_T = np.hstack([np.linalg.matrix_power(A, T-1-k) @ B for k in range(T)])
        H += Gamma_T.T @ Qf @ Gamma_T
        rhs += Gamma_T.T @ Qf @ Phi_T @ alpha

    U_star = -np.linalg.solve(H + ridge*np.eye(H.shape[0]), rhs)
    X_star = Fbar @ U_star + Gbar @ alpha

    U_star = U_star.reshape(T, m)
    X_star = X_star.reshape(T, n)

    x_pred = np.zeros((T, n))
    x_pred[0, :] = alpha
    if T > 1:
        x_pred[1:, :] = X_star[:-1, :]
    u_pred = U_star
    return x_pred, u_pred

# =========================================================
# Trajectory + Jacobian wrt θ (for IOC)
# =========================================================
def traj_and_jacobian(A, B, theta3, R_fixed, alpha, T, ridge=1e-8, Qf=None):
    n, m = A.shape[0], B.shape[1]
    d = len(theta3)
    Fbar, Gbar = build_lifted_AB(A, B, T)
    Q = map_theta_1_2_3_Q(theta3)
    Qbar = block_diag_repeat(Q, T)
    Rbar = block_diag_repeat(R_fixed, T)

    FtQF = Fbar.T @ Qbar @ Fbar
    S = FtQF + Rbar
    FtQG = Fbar.T @ Qbar @ Gbar

    if Qf is not None:
        Phi_T = np.linalg.matrix_power(A, T)
        Gamma_T = np.hstack([np.linalg.matrix_power(A, T-1-k) @ B for k in range(T)])
        S += Gamma_T.T @ Qf @ Gamma_T
        FtQG = FtQG + Gamma_T.T @ Qf @ Phi_T

    Sinv = np.linalg.pinv(S + ridge*np.eye(S.shape[0]))
    M = Sinv @ FtQG

    dQs = dQ_list(theta3)
    dQbars = [block_diag_repeat(dQi, T) for dQi in dQs]

    M_list = []
    for i in range(d):
        A_i = Fbar.T @ dQbars[i] @ Fbar
        C_i = Fbar.T @ dQbars[i] @ Gbar
        M_i = Sinv @ (C_i - A_i @ M)
        M_list.append(M_i)

    alpha = alpha.reshape(-1, 1)
    U_star = -M @ alpha
    X_star = Fbar @ U_star + Gbar @ alpha

    U_star = U_star.reshape(T, m)
    X_star = X_star.reshape(T, n)
    x_pred = np.zeros((T, n))
    x_pred[0, :] = alpha.ravel()
    if T > 1:
        x_pred[1:, :] = X_star[:-1, :]
    u_pred = U_star

    du_dtheta = np.zeros((T, m, d))
    dx_dtheta = np.zeros((T, n, d))
    for i in range(d):
        dU_i = -(M_list[i] @ alpha).reshape(T, m)
        dX_i = (Fbar @ dU_i.reshape(-1,1)).reshape(T, n)
        du_dtheta[:, :, i] = dU_i
        if T > 1:
            dx_dtheta[1:, :, i] = dX_i[:-1, :]
    return x_pred, u_pred, dx_dtheta, du_dtheta

# =========================================================
# IOC: loss + analytic grad wrt θ
# =========================================================
def compute_loss_and_grad_fixedR(theta3, x_true, u_true, A, B, alpha, T, R_fixed,
                                 w_x=1.0, w_u=1.0, ridge=1e-8, Qf=None):
    x_pred, u_pred, dx_dtheta, du_dtheta = traj_and_jacobian(
        A, B, theta3, R_fixed, alpha, T, ridge=ridge, Qf=Qf
    )
    Tlen = T
    rx = x_pred - x_true
    ru = u_pred - u_true
    loss = (w_x / Tlen) * np.sum(rx**2) + (w_u / Tlen) * np.sum(ru**2)

    d = len(theta3)
    grad = np.zeros(d)
    for i in range(d):
        term_x = np.sum(rx * dx_dtheta[:, :, i])
        term_u = np.sum(ru * du_dtheta[:, :, i])
        grad[i] = (2.0 / Tlen) * (w_x * term_x + w_u * term_u)
    return float(loss), grad

def theta_rmse(theta_hat, theta_true):
    return float(np.sqrt(np.mean((theta_hat - theta_true)**2)))

# =========================================================
# Information metric W for alpha* selection
# =========================================================
def make_W(A, B, n, m, T, R_fixed, sigma_w=0.1):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    def W_func(theta3):
        Q = map_theta_1_2_3_Q(theta3)
        Qbar, Rbar = block_diag_repeat(Q, T), block_diag_repeat(R_fixed, T)
        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S)
        W = (1/sigma_w) * (S_inv @ FtQG).T @ (S_inv @ FtQG)
        return 0.5 * (W + W.T)
    return W_func

def find_alpha_star(theta3, W_func, r_ball):
    W = W_func(theta3)
    eigvals, eigvecs = np.linalg.eigh(W)
    v_max, v_min = eigvecs[:, -1], eigvecs[:, 0]
    alpha_max = r_ball * v_max / np.linalg.norm(v_max)
    alpha_min = r_ball * v_min / np.linalg.norm(v_min)
    alpha_mid = r_ball * (v_max + v_min) / np.linalg.norm(v_max + v_min)
    return alpha_max, alpha_min, alpha_mid, eigvals[-1], eigvals[0], W

# =========================================================
# Plot helpers
# =========================================================
def plot_time_series_and_save(A, B, Q_true, R_fixed, alpha_list, labels, colors, T):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    markers = ['o', '^', 'x']

    for i, (alpha, lbl, c) in enumerate(zip(alpha_list, labels, colors)):
        marker = markers[i % len(markers)]
        x_hist, u_hist = rollout_from_theta(A, B, Q_true, R_fixed, alpha, T)
        ax[0].plot(np.arange(T), x_hist[:, 0], '-', lw=2.0, color=c,
                   marker=marker, markersize=4, markevery=3, label=f"{lbl} $x_1$")
        ax[0].plot(np.arange(T), x_hist[:, 1], '--', lw=1.8, color=c,
                   marker=marker, markersize=4, markevery=3, markerfacecolor='none',
                   label=f"{lbl} $x_2$")
        ax[1].plot(np.arange(T), u_hist[:, 0], '-', lw=2.0, color=c,
                   marker=marker, markersize=4, markevery=3, label=lbl)

    ax[0].set_xlabel("Time step"); ax[0].set_ylabel("$x$")
    ax[1].set_xlabel("Time step"); ax[1].set_ylabel("$u$")
    ax[0].legend(ncol=2, frameon=True, handlelength=3, borderaxespad=0.6)
    ax[1].legend(frameon=True, handlelength=3, borderaxespad=0.6)
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout()

    for i, name in enumerate(["state_trajectory", "control_sequence"]):
        single_fig, single_ax = plt.subplots(figsize=(5, 4))
        for line in ax[i].get_lines():
            single_ax.plot(line.get_xdata(), line.get_ydata(),
                           label=line.get_label(),
                           color=line.get_color(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth(),
                           marker=line.get_marker(),
                           markersize=line.get_markersize(),
                           markerfacecolor=line.get_markerfacecolor(),
                           markeredgecolor=line.get_markeredgecolor(),
                           markevery=line.get_markevery())
        single_ax.set_xlabel(ax[i].get_xlabel())
        single_ax.set_ylabel(ax[i].get_ylabel())
        single_ax.grid(alpha=0.3)
        single_ax.legend(ncol=2 if i == 0 else 1, frameon=True, handlelength=3, borderaxespad=0.6)
        single_fig.tight_layout()
        save_and_show(single_fig, f"{name}.pdf", show=False)

    save_and_show(fig, "state-control-traj.pdf", show=False)

def plot_phase_plane_and_save(A, B, Q_true, R_fixed, alpha_list, labels, colors, r_ball, T):
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    markers = ['o', '^', 'x']

    for i, (alpha, lbl, c) in enumerate(zip(alpha_list, labels, colors)):
        marker = markers[i % len(markers)]
        x_hist, _ = rollout_from_theta(A, B, Q_true, R_fixed, alpha, T)
        ax2.plot(x_hist[:, 0], x_hist[:, 1], '-', lw=1.8,
                 marker=marker, markersize=3.8, color=c, label=lbl)
        ax2.plot(x_hist[0, 0], x_hist[0, 1], marker, color=c, markersize=7,
                 markerfacecolor='none', markeredgewidth=1.4)
        ax2.plot(x_hist[-1, 0], x_hist[-1, 1], 'x', color=c, markersize=7, mew=1.4)

    circle = plt.Circle((0, 0), r_ball, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax2.add_patch(circle)
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal', 'box')
    ax2.legend(frameon=True, ncol=1)
    plt.tight_layout()
    save_and_show(fig2, "phase-plane.pdf", show=False)

def plot_overview_loss_rmse(results, labels, colors):
    fig1, axL = plt.subplots(figsize=(5, 4.5))
    for lbl, c in zip(labels, colors):
        axL.semilogy(results[lbl]["losses"], lw=2, color=c, label=lbl)
    axL.set_xlabel("Iteration"); axL.set_ylabel("IOC Loss")
    axL.grid(alpha=0.3); axL.legend(frameon=True)
    fig1.tight_layout()
    save_and_show(fig1, "ioc_loss.pdf", show=False)

    fig2, axR = plt.subplots(figsize=(5, 4.5))
    for lbl, c in zip(labels, colors):
        axR.plot(results[lbl]["rmses"], lw=2, color=c, label=lbl)
    axR.set_xlabel("Iteration"); axR.set_ylabel("RMSE")
    axR.grid(alpha=0.3); axR.legend(frameon=True)
    fig2.tight_layout()
    save_and_show(fig2, "ioc_rmse.pdf", show=False)

    print("✅ Figures saved to:")
    print("   - ioc_loss.pdf")
    print("   - ioc_rmse.pdf")

# =========================================================
# Main
# =========================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(2025)

    T = 50
    r_ball = 3.0
    R_fixed = np.array([[5.0]])
    dt = 0.1
    noise = 1e-3

    A = np.array([[1, dt],
                  [0, 1]])
    B = np.array([[0.5*dt**2],
                  [dt]])
    n, m = A.shape

    theta_true = np.array([5.0, 5.0, 5.0], dtype=float)
    Q_true = map_theta_1_2_3_Q(theta_true)
    print("\n=== Alpha* selection via information metric W(θ_true) ===")
    W_func = make_W(A, B, n, m, T, R_fixed, sigma_w=0.001)
    alpha_max, alpha_min, alpha_med, lam_max, lam_min, W = find_alpha_star(theta_true, W_func, r_ball)
    print(f"λ_max={lam_max:.6f}, λ_min={lam_min:.6f}, ratio={lam_max/lam_min:.3f}, Tr(W)={np.trace(W):.6f}")

    alpha_list = [alpha_max, alpha_min, alpha_med]
    labels = [r"$\alpha_\max$", r"$\alpha_\min$", r"$\alpha_{\text{med}}$"]
    color_hex = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    plot_time_series_and_save(A, B, Q_true, R_fixed, alpha_list, labels, color_hex, T)
    plot_phase_plane_and_save(A, B, Q_true, R_fixed, alpha_list, labels, color_hex, r_ball, T)

    print("\n=== IOC Learning (R fixed = 5.0) ===")
    lr = 0.1
    num_iters = 10000

    traj_data = {}
    for lbl_tex, alpha in zip(labels, alpha_list):
        x_true, u_true = rollout_from_theta(A, B, Q_true, R_fixed, alpha, T)
        x_true = x_true + noise * rng.standard_normal(x_true.shape)
        u_true = u_true + noise * rng.standard_normal(u_true.shape)
        traj_data[lbl_tex] = (alpha, x_true, u_true)

    results = {}
    for lbl_tex, c in zip(labels, color_hex):
        alpha_used, x_true, u_true = traj_data[lbl_tex]
        theta_hat = np.array([6.0, 6.0, 6.0], dtype=float)
        losses, rmses = [], []

        for it in range(num_iters):
            loss, grad = compute_loss_and_grad_fixedR(
                theta_hat, x_true, u_true, A, B, alpha_used, T, R_fixed,
                w_x=1.0, w_u=1.0, ridge=1e-6, Qf=None
            )
            theta_hat -= lr * grad
            losses.append(loss)
            rmses.append(theta_rmse(theta_hat, theta_true))
            if (it+1) % 500 == 0:
                print(f"{lbl_tex}: iter {it+1:4d} | loss={loss:.6e} | θ̂={theta_hat}")

        results[lbl_tex] = {
            "theta_hat": theta_hat,
            "losses": np.array(losses),
            "rmses": np.array(rmses),
            "final_loss": float(losses[-1]),
            "final_rmse": float(rmses[-1]),
        }

    print("\n=== IOC Summary (per alpha) ===")
    for lbl_tex in labels:
        print(f"{lbl_tex:12s} -> final loss: {results[lbl_tex]['final_loss']:.6e} | "
              f"RMSE: {results[lbl_tex]['final_rmse']:.6e} | θ̂ = {results[lbl_tex]['theta_hat']}")

    plot_overview_loss_rmse(results, labels, color_hex)
    print("\n✅ All done. Figures saved under:", os.path.abspath(save_dir))

# =========================================================
if __name__ == "__main__":
    main()
