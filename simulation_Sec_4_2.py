#!/usr/bin/env python3
# ==============================================================
# IOC–LQR Optimization (θ₄ = 5 Fixed)
# Nested Alternating Optimization for (θ₁, θ₂, θ₃, α)
# Two initializations → six figures (no layout warnings)
# ==============================================================

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
    "legend.fontsize": 14,
})

# ==============================================================
# Save folder
# ==============================================================
save_dir = "figures_sec_4_2"
os.makedirs(save_dir, exist_ok=True)

def save_fig(name):
    """Save figure safely without triggering inset layout warnings"""
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = os.path.join(save_dir, name)
    plt.savefig(path, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"✅ Figure saved to: {path}")

# ==============================================================
# Core functions
# ==============================================================
def block_diag_repeat(M, T): 
    return linalg.block_diag(*[M for _ in range(T)])

def build_lifted_AB(A, B, T):
    B = B.reshape(-1,1)
    n, m = A.shape[0], B.shape[1]
    A_powers = [np.eye(n)]
    for _ in range(1, T):
        A_powers.append(A_powers[-1] @ A)
    Gbar = np.vstack(A_powers)
    Fbar = np.zeros((T*n, T*m))
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

def map_theta3_QR_and_grads(theta, eps_Q=1e-6, eps_R=1e-6):
    t1, t2, t3 = theta
    t4_fixed = 5
    Q = np.array([[t1*t1 + eps_Q, t2*t3],
                  [t2*t3,         t3*t3 + eps_Q]], float)
    R = np.array([[t4_fixed**2 + eps_R]], float)

    dQ = [
        np.array([[2*t1, 0.0], [0.0, 0.0]]),
        np.array([[0.0, t3], [t3, 0.0]]),
        np.array([[0.0, t2], [t2, 2*t3]])
    ]
    ddQ = [[np.zeros((2,2)) for _ in range(3)] for __ in range(3)]
    ddQ[0][0] = np.array([[2,0],[0,0]])
    ddQ[2][2] = np.array([[0,0],[0,2]])
    cross = np.array([[0,1],[1,0]])
    ddQ[1][2] = ddQ[2][1] = cross
    return Q, R, dQ, ddQ

def build_Pi(A, B, n, m, T, S_set, C_list, Sigma_w):
    Fbar, Gbar = build_lifted_AB(A, B, T)
    P = build_P(n, m, T)
    Phi, Psi = build_Phi_and_Psi(n, m, T, S_set, C_list, Sigma_w)
    Iu = np.eye(T*m)
    IF = np.vstack([Iu, Fbar])
    temp = Phi @ P @ IF
    Pi = temp.T @ Psi @ temp
    return 0.5*(Pi + Pi.T), Fbar, Gbar

def make_W_and_grads_fixedR(A, B, n, m, T, S_set, C_list, Sigma_w):
    Pi, Fbar, Gbar = build_Pi(A, B, n, m, T, S_set, C_list, Sigma_w)
    def W_and_grads(theta):
        Q, R, dQ, ddQ = map_theta3_QR_and_grads(theta)
        Qbar = linalg.block_diag(*[Q for _ in range(T)])
        Rbar = linalg.block_diag(*[R for _ in range(T)])
        dQbar = [linalg.block_diag(*[dQ[i] for _ in range(T)]) for i in range(3)]
        ddQbar = [[linalg.block_diag(*[ddQ[i][j] for _ in range(T)]) for j in range(3)] for i in range(3)]

        FtQF = Fbar.T @ Qbar @ Fbar
        FtQG = Fbar.T @ Qbar @ Gbar
        S = FtQF + Rbar
        S_inv = np.linalg.pinv(S)

        W = np.zeros((Gbar.shape[1], Gbar.shape[1]))
        W_grads = [np.zeros_like(W) for _ in range(3)]
        for i in range(3):
            A_i = Fbar.T @ dQbar[i] @ Fbar
            C_i = Fbar.T @ dQbar[i] @ Gbar
            M_i = S_inv @ A_i @ S_inv @ FtQG - S_inv @ C_i
            W += M_i.T @ Pi @ M_i
            for j in range(3):
                A_j = Fbar.T @ dQbar[j] @ Fbar
                dS = S_inv @ A_j @ S_inv
                dA_i = Fbar.T @ ddQbar[i][j] @ Fbar
                dB = Fbar.T @ dQbar[j] @ Gbar
                dC_i = Fbar.T @ ddQbar[i][j] @ Gbar
                dM_i = (-dS @ A_i @ S_inv @ FtQG
                        + S_inv @ dA_i @ S_inv @ FtQG
                        - S_inv @ A_i @ dS @ FtQG
                        + S_inv @ A_i @ S_inv @ dB
                        + dS @ C_i - S_inv @ dC_i)
                W_grads[j] += dM_i.T @ Pi @ M_i + M_i.T @ Pi @ dM_i
        W = 0.5*(W + W.T)
        W_grads = [0.5*(g + g.T) for g in W_grads]
        return W, W_grads
    return W_and_grads

def nested_theta_alpha_optimization(theta0, eta, j_max, k_max, r, W_and_grads, projector):
    theta = theta0.copy()
    hist_theta, hist_alpha, hist_obj, hist_W = [], [], [], []

    W, _ = W_and_grads(theta)
    _, evecs = linalg.eigh(W)
    vmax = evecs[:, -1]
    alpha = r * vmax / np.linalg.norm(vmax)

    hist_theta.append(theta.copy())
    hist_alpha.append(alpha.copy())
    hist_W.append(W.copy())
    hist_obj.append(float(alpha.T @ W @ alpha))

    for _ in range(j_max):
        W, _ = W_and_grads(theta)
        _, evecs = linalg.eigh(W)
        vmax = evecs[:, -1]
        alpha = r * vmax / np.linalg.norm(vmax)

        for _ in range(k_max):
            W, W_grads = W_and_grads(theta)
            grad = np.array([float(alpha.T @ g @ alpha) for g in W_grads])
            theta = projector(theta - eta * grad)

            hist_theta.append(theta.copy())
            hist_alpha.append(alpha.copy())
            hist_W.append(W.copy())
            hist_obj.append(float(alpha.T @ W @ alpha))

    return theta, alpha, np.array(hist_theta), np.array(hist_alpha), hist_W, hist_obj

# ==============================================================
# Visualization
# ==============================================================
def plot_alpha_with_two_zooms(hist_alpha, filename, zoom1_xlim, zoom1_ylim, zoom2_xlim, zoom2_ylim):
    alphas = np.array(hist_alpha); steps = np.arange(len(alphas))
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for i in range(alphas.shape[1]):
        ax.plot(steps, alphas[:, i], lw=2, label=fr'$\alpha_{i+1}$')
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\alpha$ components")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    # zoom 1
    axins1 = inset_axes(ax, width="20%", height="20%", loc="center",
                        bbox_to_anchor=(-0.08, 0.20, 1, 1), bbox_transform=ax.transAxes)
    for i in range(alphas.shape[1]):
        axins1.plot(steps, alphas[:, i], lw=2)
    axins1.set_xlim(*zoom1_xlim); axins1.set_ylim(*zoom1_ylim); axins1.grid(alpha=0.3)
    mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="0.5", lw=1.2)
    # zoom 2
    axins2 = inset_axes(ax, width="20%", height="20%", loc="center",
                        bbox_to_anchor=(-0.08, -0.20, 1, 1), bbox_transform=ax.transAxes)
    for i in range(alphas.shape[1]):
        axins2.plot(steps, alphas[:, i], lw=2)
    axins2.set_xlim(*zoom2_xlim); axins2.set_ylim(*zoom2_ylim); axins2.grid(alpha=0.3)
    mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
    save_fig(filename)

def plot_theta(hist_theta, filename):
    steps = np.arange(len(hist_theta))
    plt.figure(figsize=(6, 4), layout="constrained")
    for i in range(hist_theta.shape[1]):
        plt.plot(steps, hist_theta[:, i], lw=1.8, label=fr'$\theta_{i+1}$')
    plt.xlabel("Iteration"); plt.ylabel("θ Value"); plt.legend(); plt.grid(alpha=0.3)
    save_fig(filename)

def plot_objective_zoom(hist_obj, filename, zoom_xlim, zoom_ylim):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(hist_obj, lw=2)
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\alpha^\top W(\theta)\alpha$")
    ax.grid(alpha=0.3)
    axins = inset_axes(ax, width="40%", height="40%", loc="center",
                       bbox_to_anchor=(0.15, 0.15, 1, 1), bbox_transform=ax.transAxes)
    axins.plot(hist_obj, lw=2)
    axins.set_xlim(*zoom_xlim); axins.set_ylim(*zoom_ylim); axins.grid(alpha=0.3)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
    save_fig(filename)

# ==============================================================
# Main
# ==============================================================
if __name__=="__main__":
    print("="*70)
    print(" IOC–LQR Analytic Gradient (Fixed θ₄=5) — Two Cases ")
    print("="*70)

    r = 3.0
    A = np.array([[1, 0.1],[0, 1]])
    B = np.array([[0.005],[0.1]])
    T, n, m = 50, 2, 1
    C_list = [np.eye(n+m) for _ in range(T)]
    S_set = set(range(T))
    Sigma_w = 0.001*np.eye(n+m)
    l, u = np.zeros(3), 10*np.ones(3)
    projector = lambda th: np.clip(th, l, u)
    Wfun = make_W_and_grads_fixedR(A, B, n, m, T, S_set, C_list, Sigma_w)

    # ===== Case 1: θ₀ = [3,6,8]  → j_max=10, k_max=20 =====
    theta0 = np.array([3, 6, 8])
    print(f"\n=== Case 1: θ₀={theta0} ===")
    theta_star, alpha_star, hist_theta, hist_alpha, hist_W, hist_obj = \
        nested_theta_alpha_optimization(theta0, eta=0.002, j_max=10, k_max=20,
                                        r=r, W_and_grads=Wfun, projector=projector)
    plot_alpha_with_two_zooms(hist_alpha, "alpha_evolution_two_zooms.pdf",
        zoom1_xlim=(50, 100), zoom1_ylim=(2.976, 2.979),
        zoom2_xlim=(50, 100), zoom2_ylim=(0.360, 0.370))
    plot_theta(hist_theta, "theta_evolution_fixedR_1.pdf")
    plot_objective_zoom(hist_obj, "objective_fixedR_1_zoom.pdf",
                        zoom_xlim=(50, 100), zoom_ylim=(758, 764))

    # ===== Case 2: θ₀ = [5,8,5]  → j_max=70, k_max=20 =====
    theta0 = np.array([5, 8, 5])
    print(f"\n=== Case 2: θ₀={theta0} ===")
    theta_star, alpha_star, hist_theta, hist_alpha, hist_W, hist_obj = \
        nested_theta_alpha_optimization(theta0, eta=0.002, j_max=70, k_max=20,
                                        r=r, W_and_grads=Wfun, projector=projector)
    plot_alpha_with_two_zooms(hist_alpha, "alpha_evolution_two_zooms_another.pdf",
        zoom1_xlim=(800, 900), zoom1_ylim=(-1.15563991, -1.15561994),
        zoom2_xlim=(800, 900), zoom2_ylim=(-2.768490, -2.768485))
    plot_theta(hist_theta, "theta_evolution_fixedR.pdf")
    plot_objective_zoom(hist_obj, "objective_fixedR_zoom.pdf",
                        zoom_xlim=(800, 900), zoom_ylim=(1322.8, 1323.0))

    print(f"\n✅ All figures saved in: {os.path.abspath(save_dir)}")
