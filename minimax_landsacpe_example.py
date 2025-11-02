#!/usr/bin/env python3
# ==============================================================
# IOC–LQR Objective Landscape (θ₄ = 5 Fixed)
# Sweep over (θ₁, θ₂) with fixed θ₃ = 10 + Log-Scale Visualization
# ==============================================================

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import warnings

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})

# ==============================================================
# === Save Folder ===
# ==============================================================
save_dir = "figures_landscape_logscale"
os.makedirs(save_dir, exist_ok=True)

def save_fig(name, fig=None, show=True):
    """
    Save figure as high-quality PDF.
    - name: base filename (no .pdf needed)
    - fig: matplotlib Figure; if None uses plt.gcf()
    """
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = os.path.join(save_dir, name)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    plt.close(fig)
    print(f"✅ Figure saved to: {path}")

# ==============================================================
# Utility and Core Functions
# ==============================================================
def build_lifted_AB(A, B, T):
    B = B.reshape(-1, 1)
    n, m = A.shape[0], B.shape[1]
    A_powers = [np.eye(n)]
    for _ in range(1, T):
        A_powers.append(A_powers[-1] @ A)
    Gbar = np.vstack(A_powers)
    Fbar = np.zeros((T * n, T * m))
    for t in range(T):
        for s in range(t):
            Fbar[t*n:(t+1)*n, s*m:(s+1)*m] = A_powers[t-1-s] @ B
    return Fbar, Gbar

def build_P(n, m, T):
    Iu = np.vstack([np.eye(m), np.zeros((n, m))])
    Ix = np.vstack([np.zeros((m, n)), np.eye(n)])
    P_u = linalg.block_diag(*[Iu for _ in range(T)])
    P_x = linalg.block_diag(*[Ix for _ in range(T)])
    return np.hstack([P_u, P_x])

def build_Phi_and_Psi(n, m, T, S_set, C_list, Sigma_w):
    S_sorted = sorted(S_set)
    blocks = [C_list[t] for t in S_sorted]
    Cblk = blocks[0].shape[0]
    Phi = np.zeros((len(blocks)*Cblk, T*(n+m)))
    for i, t in enumerate(S_sorted):
        Phi[i*Cblk:(i+1)*Cblk, t*(n+m):(t+1)*(n+m)] = C_list[t]
    try:
        Sigma_inv = linalg.inv(Sigma_w)
    except linalg.LinAlgError:
        warnings.warn("Σ_w near-singular, using pinv")
        Sigma_inv = linalg.pinv(Sigma_w)
    Psi = linalg.block_diag(*[Sigma_inv for _ in range(len(blocks))])
    return Phi, Psi

def map_theta3_QR(theta, eps_Q=1e-6, eps_R=1e-6, t4_fixed=5.0):
    t1, t2, t3 = theta
    Q = np.array([[t1*t1 + eps_Q, t2*t3],
                  [t2*t3,         t3*t3 + eps_Q]], float)
    R = np.array([[t4_fixed**2 + eps_R]], float)
    dQ = [
        np.array([[2*t1, 0.0], [0.0, 0.0]]),
        np.array([[0.0, t3], [t3, 0.0]]),
        np.array([[0.0, t2], [t2, 2*t3]])
    ]
    return Q, R, dQ

def build_Pi(A, B, n, m, T, S_set, C_list, Sigma_w):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    P = build_P(n, m, T)
    Phi, Psi = build_Phi_and_Psi(n, m, T, S_set, C_list, Sigma_w)
    Iu = np.eye(T*m)
    IF = np.vstack([Iu, Fbar])
    temp = Phi @ P @ IF
    Pi = temp.T @ Psi @ temp
    return 0.5*(Pi + Pi.T), Fbar, Gbar

def make_W_fixedR(A, B, n, m, T, S_set, C_list, Sigma_w):
    Pi, Fbar, Gbar = build_Pi(A, B, n, m, T, S_set, C_list, Sigma_w)
    def W_func(theta):
        Q, R, dQ = map_theta3_QR(theta)
        Qbar = linalg.block_diag(*[Q for _ in range(T)])
        Rbar = linalg.block_diag(*[R for _ in range(T)])
        dQbar = [linalg.block_diag(*[dQ[i] for _ in range(T)]) for i in range(3)]

        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S)

        W = np.zeros((Gbar.shape[1], Gbar.shape[1]))
        for i in range(3):
            A_i = Fbar.T @ dQbar[i] @ Fbar
            C_i = Fbar.T @ dQbar[i] @ Gbar
            M_i = S_inv @ A_i @ S_inv @ FtQG - S_inv @ C_i
            W += M_i.T @ Pi @ M_i
        W = 0.5*(W + W.T)
        evals, evecs = np.linalg.eigh(W)
        evals_clipped = np.maximum(evals, 0.0)
        W = (evecs @ np.diag(evals_clipped) @ evecs.T)
        return W
    return W_func

def generate_system(system_type="double_integrator"):
    dt = 0.1
    if system_type == "double_integrator":
        A = np.array([[1, dt],
                      [0, 1]])
        B = np.array([[0.5*dt**2],
                      [dt]])
    else:
        raise ValueError("Only 'double_integrator' supported here.")
    return A, B

# ==============================================================
# Main Execution
# ==============================================================
if __name__ == "__main__":
    print("="*70)
    print(" IOC–LQR Objective Landscape (θ₁, θ₂) | θ₃=10, θ₄=5 ")
    print("="*70)

    r = 3.0
    A, B = generate_system()
    T, n, m = 50, 2, 1
    C_list = [np.eye(n+m) for _ in range(T)]
    S_set = set(range(T))
    Sigma_w = 0.001 * np.eye(n+m)

    W_func = make_W_fixedR(A, B, n, m, T, S_set, C_list, Sigma_w)
    theta3_fixed = 10.0

    theta1_vals = np.linspace(0, 10, 40)
    theta2_vals = np.linspace(0, 10, 40)
    TH1, TH2 = np.meshgrid(theta1_vals, theta2_vals)
    OBJ = np.zeros_like(TH1)

    for i in range(len(theta1_vals)):
        for j in range(len(theta2_vals)):
            t1, t2 = TH1[j, i], TH2[j, i]
            theta = np.array([t1, t2, theta3_fixed])
            W = W_func(theta)
            evals, evecs = linalg.eigh(W)
            vmax = evecs[:, -1]
            alpha = r * vmax / (np.linalg.norm(vmax) + 1e-12)
            val = float(alpha.T @ W @ alpha)
            OBJ[j, i] = max(val, 0.0)

    # fig1 = plt.figure(figsize=(6,5))
    # ax1 = fig1.add_subplot(111)
    # cf1 = ax1.contourf(TH1, TH2, OBJ, levels=40, cmap="viridis")
    # fig1.colorbar(cf1, ax=ax1, label=r"$\alpha^\top W(\theta)\alpha$")
    # ax1.set_xlabel(r"$\theta_1$")
    # ax1.set_ylabel(r"$\theta_2$")
    # ax1.set_title(r"Objective Landscape (Linear scale)")
    # ax1.grid(alpha=0.3)
    # save_fig("landscape_heatmap_linear", fig=fig1, show=True)

    OBJ_log = np.log1p(OBJ)
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot(111)
    cf2 = ax2.contourf(TH1, TH2, OBJ_log, levels=80, cmap="viridis", alpha=0.95)
    contours = ax2.contour(TH1, TH2, OBJ_log, levels=15, colors='white', linewidths=0.35, alpha=0.6)
    ax2.clabel(contours, inline=True, fontsize=10, fmt=lambda x: f"{x:.2f}", colors='white')
    ax2.imshow(OBJ_log, extent=(TH1.min(), TH1.max(), TH2.min(), TH2.max()),
               origin='lower', cmap="viridis", alpha=0.15, aspect='auto')
    fig2.colorbar(cf2, ax=ax2, label=r"$\log(1 + \alpha^\top W(\theta)\alpha)$")
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")
    ax2.grid(alpha=0.25)
    save_fig("landscape_heatmap_log_labels", fig=fig2, show=True)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig3 = plt.figure(figsize=(7,6))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(TH1, TH2, OBJ_log, cmap="magma", edgecolor="none", alpha=0.9)
    ax3.set_xlabel(r"$\theta_1$")
    ax3.set_ylabel(r"$\theta_2$")
    ax3.set_zlabel(r"$\log(1 + \alpha^\top W(\theta)\alpha)$")
    ax3.set_title(r"Objective Landscape (3D Log-scale)")
    fig3.colorbar(surf, shrink=0.6, aspect=10)
    save_fig("landscape_surface_log", fig=fig3, show=True)

    print("✅ Done: saved linear, log-scale, and 3D log-scale plots.")
