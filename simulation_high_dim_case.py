#!/usr/bin/env python3
# =============================================================================
# IOC–LQR Optimization (4D system, Q = U^T U with 10 params, scalar R fixed)
# Nested Alternating Optimization for (θ in R^10, α) with analytic dQ/dθ
# Figures saved to ./figures_4d
# System A has two complex-conjugate pole pairs (discrete-time), B is single input
# Also plots eigenvalue (poles) diagram in the complex plane
# =============================================================================

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# =============================================================================
# === Unified Save Folder ===
# =============================================================================
save_dir = "figures_4d"
os.makedirs(save_dir, exist_ok=True)

def save_fig(name, show=True):
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = os.path.join(save_dir, name)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    plt.close()
    print(f"✅ Figure saved to: {path}")

# =============================================================================
# Utility Functions
# =============================================================================

def build_lifted_AB(A, B, T):
    """Build lifted (Fbar, Gbar) such that x = Fbar u + Gbar α.
    Shapes: A∈R^{n×n}, B∈R^{n×m}; Fbar∈R^{Tn×Tm}, Gbar∈R^{Tn×n}
    """
    n, m = A.shape[0], B.shape[1]
    A_powers = [np.eye(n)]
    for _ in range(1, T):
        A_powers.append(A_powers[-1] @ A)
    Gbar = np.vstack(A_powers)                      # (T n) x n
    Fbar = np.zeros((T*n, T*m))                     # (T n) x (T m)
    for t in range(T):
        for s in range(t):
            Fbar[t*n:(t+1)*n, s*m:(s+1)*m] = A_powers[t-1-s] @ B
    return Fbar, Gbar

# =============================================================================
# Q parameterization: Q = U^T U with 10 parameters (upper-triangular U)
# R is scalar fixed (constant, not optimized)
# =============================================================================

def map_theta10_Q_and_grads(theta, eps_sym=0.0):
    """
    theta: length-10 vector for upper-triangular U.
    Returns:
      Q (4x4), dQ list of length-10 (each 4x4)
    """
    t = np.asarray(theta, dtype=float).ravel()
    assert t.size == 10, "theta must be length-10 for upper-triangular U."
    # Build U
    U = np.array([
        [t[0], t[1], t[2],  t[3]],
        [0.0,  t[4], t[5],  t[6]],
        [0.0,  0.0,  t[7],  t[8]],
        [0.0,  0.0,  0.0,   t[9]]
    ], dtype=float)
    Q = U.T @ U
    if eps_sym > 0:
        Q += eps_sym * np.eye(4)

    # Basis matrices E_k for dU/dθ_k
    E = []
    E.append(np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], float))  # θ1
    E.append(np.array([[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], float))  # θ2
    E.append(np.array([[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], float))  # θ3
    E.append(np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]], float))  # θ4
    E.append(np.array([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], float))  # θ5
    E.append(np.array([[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]], float))  # θ6
    E.append(np.array([[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], float))  # θ7
    E.append(np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]], float))  # θ8
    E.append(np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]], float))  # θ9
    E.append(np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], float))  # θ10

    dQ = []
    for Ek in E:
        dQk = Ek.T @ U + U.T @ Ek  # d(U^T U) = (dU)^T U + U^T (dU)
        dQ.append(dQk)
    return Q, dQ

# =============================================================================
# W(θ) construction and gradients (R fixed)
# =============================================================================

def make_W_and_grads_choleskyQ(A, B, n, m, T, R_fixed=25.0):
    """
    Fisher-like information W(θ) and gradients wrt θ (10 params in U).
    Uses Pi = I as in previous 4D scripts.
    """
    Pi = np.eye(T*m)
    Fbar, Gbar = build_lifted_AB(A, B, T)
    R = np.array([[R_fixed]], float)

    def W_and_grads(theta):
        Q, dQ = map_theta10_Q_and_grads(theta)

        # Block-diagonalize over horizon
        Qbar = linalg.block_diag(*[Q for _ in range(T)])          # (T n) x (T n)
        Rbar = linalg.block_diag(*[R for _ in range(T)])          # (T m) x (T m)
        dQbar = [linalg.block_diag(*[dQ[i] for _ in range(T)]) for i in range(10)]

        FtQF = Fbar.T @ Qbar @ Fbar              # (T m)×(T m)
        FtQG = Fbar.T @ Qbar @ Gbar              # (T m)×n
        S = FtQF + Rbar                          # (T m)×(T m)
        S_inv = np.linalg.pinv(S)

        W = np.zeros((Gbar.shape[1], Gbar.shape[1]))  # n×n
        W_grads = [np.zeros_like(W) for _ in range(10)]

        # Base M_i terms for each dQ_i (i=0..9)
        A_i_list, C_i_list, M_i_list = [], [], []
        for i in range(10):
            A_i = Fbar.T @ dQbar[i] @ Fbar
            C_i = Fbar.T @ dQbar[i] @ Gbar
            M_i = S_inv @ A_i @ S_inv @ FtQG - S_inv @ C_i
            A_i_list.append(A_i); C_i_list.append(C_i); M_i_list.append(M_i)
            W += M_i.T @ Pi @ M_i

        # Gradients wrt θ_j
        for j in range(10):
            dS_mat = (Fbar.T @ dQbar[j] @ Fbar)  # Rbar is fixed → dR = 0
            dS_eff = S_inv @ dS_mat @ S_inv
            dFtQG = Fbar.T @ dQbar[j] @ Gbar
            for i in range(10):
                A_i = A_i_list[i]; C_i = C_i_list[i]; M_i = M_i_list[i]
                dM_i = (-dS_eff @ A_i @ S_inv @ FtQG
                        - S_inv @ A_i @ dS_eff @ FtQG
                        + S_inv @ A_i @ S_inv @ dFtQG
                        + dS_eff @ C_i)
                W_grads[j] += dM_i.T @ Pi @ M_i + M_i.T @ Pi @ dM_i

        # Symmetrize
        W = 0.5*(W + W.T)
        W_grads = [0.5*(g + g.T) for g in W_grads]
        return W, W_grads

    return W_and_grads

# =============================================================================
# Optimization
# =============================================================================

def nested_theta_alpha_optimization(theta0, eta, j_max, k_max, r,
                                    W_and_grads, projector,
                                    tol=1e-6, verbose=True):
    theta = theta0.copy()
    hist_theta, hist_alpha, hist_obj, hist_W = [], [], [], []

    # Initialize α along the top eigenvector of W(θ0)
    W, _ = W_and_grads(theta)
    evals, evecs = linalg.eigh(W)
    vmax = evecs[:, -1]
    alpha = r * vmax / np.linalg.norm(vmax)
    hist_theta.append(theta.copy())
    hist_alpha.append(alpha.copy())
    hist_W.append(W.copy())
    hist_obj.append(float(alpha.T @ W @ alpha))

    for j in range(j_max):
        # Update α (best direction for current θ)
        W, _ = W_and_grads(theta)
        evals, evecs = linalg.eigh(W)
        vmax = evecs[:, -1]
        alpha = r * vmax / np.linalg.norm(vmax)

        # Gradient steps for θ holding α fixed
        for k in range(k_max):
            W, W_grads = W_and_grads(theta)
            grad = np.array([float(alpha.T @ g @ alpha) for g in W_grads])
            theta = projector(theta - eta * grad)

            hist_theta.append(theta.copy())
            hist_alpha.append(alpha.copy())
            hist_W.append(W.copy())
            hist_obj.append(float(alpha.T @ W @ alpha))
            if np.linalg.norm(grad) < tol:
                break

        if verbose and ((j+1) % max(1, j_max//10) == 0):
            print(f"[Outer] {j+1}/{j_max}: θ={np.round(theta,3)}  Obj={hist_obj[-1]:.6f}")

    return theta, alpha, np.array(hist_theta), np.array(hist_alpha), hist_W, hist_obj

# =============================================================================
# Visualization
# =============================================================================

def plot_alpha_components(hist_alpha):
    alphas = np.array(hist_alpha)
    steps = np.arange(len(alphas))
    plt.figure(figsize=(7, 4.2))
    for i in range(alphas.shape[1]):
        plt.plot(steps, alphas[:, i], lw=2, label=fr'$\alpha_{i+1}$')
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$ components")
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    save_fig("alpha_evolution_4d")

def plot_theta_evolution(hist_theta):
    steps = np.arange(len(hist_theta))
    plt.figure(figsize=(7.4,4.6))
    for i in range(hist_theta.shape[1]):
        plt.plot(steps, hist_theta[:,i], lw=1.6, label=fr'$\theta_{i+1}$')
    plt.xlabel("Iteration")
    plt.ylabel("θ Value")
    plt.legend(ncol=3)
    plt.grid(alpha=0.3)
    save_fig("theta_evolution_4d")

def plot_objective(hist_obj):
    plt.figure(figsize=(6.6,4.2))
    plt.plot(hist_obj, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha^\top W(\theta)\alpha$")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    save_fig("objective_4d")

def plot_condition_number(hist_W):
    conds = []
    for W in hist_W:
        s = np.linalg.svd(W, compute_uv=False)
        conds.append(s[0] / (s[-1] + 1e-12))
    plt.figure(figsize=(6.6,4.2))
    plt.plot(conds, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel(r"cond$(\mathcal{I})$")
    plt.title("Fisher Information Condition Number (4D)")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    save_fig("condition_number_4d")

def plot_poles(A):
    """Complex-plane pole plot with unit circle."""
    evals = np.linalg.eigvals(A)
    plt.figure(figsize=(5.2,5.2))
    ax = plt.gca()
    # unit circle
    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linestyle='--', alpha=0.5)
    # axes
    ax.axhline(0, color='0.5', lw=1)
    ax.axvline(0, color='0.5', lw=1)
    # eigenvalues
    ax.scatter(np.real(evals), np.imag(evals), s=60, marker='x', label='poles (eigvals)')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Discrete-time Poles (A)')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("poles_complex_plane")

def plot_objective_with_zoom(hist_obj, zoom_xlim=(0, 60), zoom_ylim=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(hist_obj, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\alpha^\top W(\theta)\alpha$")
    ax.grid(alpha=0.3)
    axins = inset_axes(ax, width="40%", height="40%", loc="center",
                       bbox_to_anchor=(0.15, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)
    axins.plot(hist_obj, lw=2)
    axins.set_xlim(*zoom_xlim)
    if zoom_ylim is not None:
        axins.set_ylim(*zoom_ylim)
    axins.grid(alpha=0.3)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", linestyle="--", ec="0.5", lw=1.2)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(save_dir, "objective_4d_zoom.pdf"), dpi=600, bbox_inches="tight")
    plt.show()

# =============================================================================
# 4D System with two complex pole pairs (discrete-time), single input (m=1)
# =============================================================================

def generate_system_4d_complex_poles(r1=0.95, w1=0.20, r2=0.90, w2=0.35):
    """
    Build A with eigenvalues {r1 e^{±j w1}, r2 e^{±j w2}} using real 2×2 blocks:
        R(r, w) = r * [[cos w, -sin w],[sin w, cos w]]
    Then A = block_diag(R(r1,w1), R(r2,w2)) ∈ R^{4×4}.
    Choose B ∈ R^{4×1} to ensure controllability for each block.
    """
    c1, s1 = np.cos(w1), np.sin(w1)
    c2, s2 = np.cos(w2), np.sin(w2)
    R1 = r1 * np.array([[c1, -s1],[s1, c1]])
    R2 = r2 * np.array([[c2, -s2],[s2, c2]])
    A = linalg.block_diag(R1, R2)  # 4×4

    # Single input affecting both 2D subsystems
    B = np.array([[1.0], [0.0], [1.0], [0.0]])  # 4×1

    # Controllability check
    Ctrb = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank = np.linalg.matrix_rank(Ctrb)
    if rank < 4:
        warnings.warn(f"(A,B) rank(Ctrb)={rank} < 4; consider changing B or pole params.")
    return A, B

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*76)
    print(" IOC–LQR Analytic Gradient Pipeline (4D, complex poles; Q=U^T U; R fixed) ")
    print("="*76)

    # --- System ---
    A, B = generate_system_4d_complex_poles(r1=0.95, w1=0.20, r2=0.90, w2=0.35)
    T, n, m = 50, 4, 1

    # --- θ projector (range clamp to keep numerics sane; Q PSD via U anyway) ---
    l, u = 0*np.ones(10), 10*np.ones(10)
    projector = lambda th: np.clip(th, l, u)

    # --- Init θ (upper-tri U entries): give non-degenerate diagonals ---
    theta0 = np.array([5.0,  5.0, 5.0,  5.0,
                       5.0,  5.0,  5.0,
                       5.0,  5.0,
                       5.0])

    # --- Build W and grads (R fixed = 25 = 5^2) ---
    W_and_grads = make_W_and_grads_choleskyQ(A, B, n, m, T, R_fixed=25.0)

    # --- Optimization ---
    theta_star, alpha_star, hist_theta, hist_alpha, hist_W, hist_obj = nested_theta_alpha_optimization(
        theta0, eta=1, j_max=5000, k_max=50, r=3.0,
        W_and_grads=W_and_grads, projector=projector,
        tol=1e-6, verbose=True)

    print("\nFinal θ* =", np.round(theta_star,6), "  (Q = U^T U from 10 params, R fixed)")
    print("Final α* =", np.round(alpha_star,6))
    print(f"‖α*‖={np.linalg.norm(alpha_star):.6f} (target r=3.0)")

    # --- Plots ---
    plot_alpha_components(hist_alpha)
    plot_theta_evolution(hist_theta)
    plot_objective(hist_obj)
    plot_condition_number(hist_W)
    plot_poles(A)
    # Optional zoom (tune as needed)
    # plot_objective_with_zoom(hist_obj, zoom_xlim=(0, 80), zoom_ylim=None)

    print("theta_list (first 5 shown):", np.round(hist_theta[:5],3), " ...")
    print(f"\n✅ All figures saved in: {os.path.abspath(save_dir)}")
