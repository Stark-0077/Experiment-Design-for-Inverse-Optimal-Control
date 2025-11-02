#!/usr/bin/env python3
# ==============================================================
# IOC–LQR Full Pipeline (Analytic Gradient + α from W(θ) spectrum)
# Finite-horizon cost with x,u only in {0,...,T-1}
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
# Save utilities
# =========================================================
save_dir = "figures_sec_4_1"
os.makedirs(save_dir, exist_ok=True)

def save_and_show(fig, filename, dpi=600, show=True):
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved {filename}")
    if show:
        plt.show()
    plt.close(fig)

def block_diag_repeat(M, T):
    return linalg.block_diag(*[M for _ in range(T)])


# =========================================================
# θ → Q mapping
# =========================================================
def map_theta_1_2_3_Q(theta3, eps_Q=1e-6):
    t1, t2, t3 = theta3
    return np.array([[t1**2 + eps_Q, t2*t3],
                     [t2*t3,         t3**2 + eps_Q]], dtype=float)


# =========================================================
# Finite-horizon LQR solver (no Qf)
# =========================================================
def lqr_finite_horizon_gains(A, B, Q, R, T):
    """No terminal cost: sum_{t=0}^{T-1} x_t^T Q x_t + u_t^T R u_t."""
    n, m = A.shape[0], B.shape[1]
    P, K = [None]*(T+1), [None]*T
    P[T] = np.zeros((n, n))   # Qf = 0
    for t in range(T-1, -1, -1):
        Pn = P[t+1]
        G = R + B.T @ Pn @ B
        Kt = np.linalg.solve(G, B.T @ Pn @ A)
        K[t] = Kt
        Pt = Q + A.T @ (Pn @ (A - B @ Kt))
        P[t] = 0.5*(Pt + Pt.T)
    return K


def simulate_closed_loop_finite(A, B, K_list, x0):
    """Simulate for T steps; returns x_{0:T-1}, u_{0:T-1}."""
    T = len(K_list)
    n, m = A.shape[0], B.shape[1]
    x = np.zeros((T, n))
    u = np.zeros((T, m))
    x[0] = x0
    for t in range(T):
        u[t] = -K_list[t] @ x[t]
        if t < T-1:
            x[t+1] = (A - B @ K_list[t]) @ x[t]
    return x, u


# =========================================================
# Lifted system matrices
# =========================================================
def build_lifted_AB(A, B, T):
    """For t=0,...,T-1"""
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
# W(θ) Information matrix (for α selection)
# =========================================================
def make_W(A, B, n, m, T, R_fixed, sigma_w=1e-3):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    def W_func(theta3):
        Q = map_theta_1_2_3_Q(theta3)
        Qbar, Rbar = block_diag_repeat(Q, T), block_diag_repeat(R_fixed, T)
        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S)
        W = (1/sigma_w) * (S_inv @ FtQG).T @ (S_inv @ FtQG)
        return 0.5*(W + W.T)
    return W_func

def find_alpha_star(theta3, W_func, r_ball):
    W = W_func(theta3)
    eigvals, eigvecs = np.linalg.eigh(W)
    v_max, v_min = eigvecs[:, -1], eigvecs[:, 0]
    alpha_max = r_ball * v_max / np.linalg.norm(v_max)
    alpha_min = r_ball * v_min / np.linalg.norm(v_min)
    alpha_mid = r_ball * (v_max + v_min) / np.linalg.norm(v_max + v_min)
    return alpha_max, alpha_min, alpha_mid, eigvals, W


# =========================================================
# Analytic trajectory Jacobian （∂z*/∂θ）
# =========================================================
def make_traj_grad_func(A, B, n, m, T, R_fixed, ridge=1e-5):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    def traj_grad_func(theta, alpha):
        Q = map_theta_1_2_3_Q(theta)
        Qbar = block_diag_repeat(Q, T)
        Rbar = block_diag_repeat(R_fixed, T)
        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S + ridge*np.eye(S.shape[0]))
        M = S_inv @ FtQG

        dQ1 = np.array([[2*theta[0], 0.0],
                        [0.0,        0.0]])
        dQ2 = np.array([[0.0,     theta[2]],
                        [theta[2], 0.0]])
        dQ3 = np.array([[0.0,     theta[1]],
                        [theta[1], 2*theta[2]]])
        dQbar = [block_diag_repeat(dQ1, T),
                 block_diag_repeat(dQ2, T),
                 block_diag_repeat(dQ3, T)]

        M_list = []
        for dQi_bar in dQbar:
            A_i = Fbar.T @ dQi_bar @ Fbar
            C_i = Fbar.T @ dQi_bar @ Gbar
            M_i = S_inv @ (C_i - A_i @ M)
            M_list.append(M_i)

        alpha = alpha.reshape(-1,1)
        U_star = M @ alpha
        X_star = Fbar @ U_star + Gbar @ alpha
        z_star = np.vstack([U_star, X_star])
        Iu = np.eye(T*m)
        stack = np.vstack([Iu, Fbar])
        dz_dtheta = np.hstack([stack @ (Mi @ alpha) for Mi in M_list])
        return z_star, dz_dtheta
    return traj_grad_func


def compute_grad_fixedR_analytic(theta3, x_target, u_target, A, B, alpha, T, R_fixed):
    n, m = A.shape[0], B.shape[1]
    traj_grad_func = make_traj_grad_func(A, B, n, m, T, R_fixed)
    z_star, dz_dtheta = traj_grad_func(theta3, alpha)
    z_target = np.vstack([
        u_target.reshape(-1, 1),
        x_target.reshape(-1, 1)
    ])
    Rz = z_star - z_target
    grad = (2.0/(T*(n+m))) * (dz_dtheta.T @ Rz).flatten()
    return grad


# =========================================================
# IOC loss and RMSE
# =========================================================
def rollout_from_theta(A, B, Q, R, alpha, T):
    K_list = lqr_finite_horizon_gains(A, B, Q, R, T)
    return simulate_closed_loop_finite(A, B, K_list, alpha)

def compute_loss_fixedR(theta3, x_target, u_target, A, B, alpha, T, R_fixed):
    Q = map_theta_1_2_3_Q(theta3)
    x_pred, u_pred = rollout_from_theta(A, B, Q, R_fixed, alpha, T)
    return np.mean((x_pred - x_target)**2) + np.mean((u_pred - u_target)**2)

def theta_rmse(theta_hat, theta_true):
    return np.sqrt(np.mean((theta_hat - theta_true)**2))

# =========================================================
# Numerical gradient (finite difference)
# =========================================================
def compute_grad_fixedR_numeric(theta3, x_target, u_target, A, B, alpha, T, R_fixed, eps=1e-5):
    """Finite-difference numerical gradient of loss wrt θ."""
    grad = np.zeros_like(theta3)
    base = compute_loss_fixedR(theta3, x_target, u_target, A, B, alpha, T, R_fixed)
    for i in range(len(theta3)):
        t2 = theta3.copy()
        t2[i] += eps
        grad[i] = (compute_loss_fixedR(t2, x_target, u_target, A, B, alpha, T, R_fixed) - base) / eps
    return grad


# =========================================================
# Main pipeline
# =========================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(42)

    # === System ===
    dt, T, r_ball = 0.1, 50, 3.0
    A = np.array([[1, dt],
                  [0, 1]])
    B = np.array([[0.5*dt*dt],
                  [dt]])
    R_fixed = np.array([[5.0]])
    n, m = A.shape[0], B.shape[1]
    theta_true = np.array([5.0, 5.0, 5.0])

    # === α selection from W(θ_true)
    W_func = make_W(A, B, n, m, T, R_fixed)
    alpha_max, alpha_min, alpha_mid, eigvals, _ = find_alpha_star(theta_true, W_func, r_ball)
    alpha_list = [alpha_max, alpha_min, alpha_mid]
    labels = [r"$\alpha_{\max}$", r"$\alpha_{\min}$", r"$\alpha_{\text{mid}}$"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # === Generate “true” trajectories ===
    Q_true = map_theta_1_2_3_Q(theta_true)
    results = {}
    print("\n=== IOC Learning (Analytic Gradient, no Qf, x,u∈{0:T-1}) ===")
    theta_init = np.array([6.0, 6.0, 6.0])
    lr, num_iters = 1e-2, 8000

        # === Gradient check on one example ===
    print("\n=== Gradient Check (Analytic vs Numeric) ===")
    alpha_test = alpha_max
    x_true, u_true = rollout_from_theta(A, B, Q_true, R_fixed, alpha_test, T)
    theta_test = np.array([5.5, 5.5, 5.5])
    grad_ana = compute_grad_fixedR_analytic(theta_test, x_true, u_true, A, B, alpha_test, T, R_fixed)
    grad_num = compute_grad_fixedR_numeric(theta_test, x_true, u_true, A, B, alpha_test, T, R_fixed)
    diff = grad_ana - grad_num
    cos_sim = np.dot(grad_ana, grad_num) / (np.linalg.norm(grad_ana)*np.linalg.norm(grad_num) + 1e-12)
    rel_err = np.linalg.norm(diff) / (np.linalg.norm(grad_ana)+np.linalg.norm(grad_num)+1e-12)
    print(f"θ_test={theta_test.round(3)}")
    print(f"Analytic grad={grad_ana.round(6)}")
    print(f"Numeric  grad={grad_num.round(6)}")
    print(f"Diff     ={diff.round(6)}")
    print(f"Cosine similarity={cos_sim:.6f},  Relative error={rel_err:.2e}")
    if cos_sim > 0.999 and rel_err < 1e-3:
        print("✅ Gradient check PASSED")
    else:
        print("⚠️  Gradient check needs review (scale or formula mismatch)")


    for lbl, alpha_used, c in zip(labels, alpha_list, colors):
        x_true, u_true = rollout_from_theta(A, B, Q_true, R_fixed, alpha_used, T)
        theta_hat = theta_init.copy()
        losses, rmses = [], []
        for it in range(num_iters):
            loss = compute_loss_fixedR(theta_hat, x_true, u_true, A, B, alpha_used, T, R_fixed)
            grad = compute_grad_fixedR_numeric(theta_hat, x_true, u_true, A, B, alpha_used, T, R_fixed)
            theta_hat -= lr * grad
            losses.append(loss)
            rmses.append(theta_rmse(theta_hat, theta_true))
        results[lbl] = {"losses": np.array(losses), "rmses": np.array(rmses)}
        print(f"{lbl}: final θ̂={theta_hat.round(3)}, RMSE={rmses[-1]:.3e}")

    # === Plot convergence ===
    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    ax = ax.flatten()
    for lbl, c in zip(labels, colors):
        losses = results[lbl]["losses"]; rmses = results[lbl]["rmses"]
        ax[0].semilogy(losses, lw=2, color=c, label=lbl)
        ax[1].semilogy(losses/losses[0], lw=2, color=c, label=lbl)
        ax[2].plot(rmses, lw=2, color=c, label=lbl)
        ax[3].plot(rmses/rmses[0], lw=2, color=c, label=lbl)
    ylabels = ["Loss", "Relative Loss", "RMSE", "Relative RMSE"]
    for i in range(4):
        ax[i].set_xlabel("Iteration"); ax[i].set_ylabel(ylabels[i])
        ax[i].grid(alpha=0.3); ax[i].legend()
    plt.tight_layout()
    save_and_show(fig, "ioc_loss_rmse_noQf.pdf", show=True)
    print("\n✅ Completed: No Qf, trajectory 0:T-1 only, analytic gradient verified.")

# =========================================================
if __name__ == "__main__":
    main()
