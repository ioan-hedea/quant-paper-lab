"""Module 2: MPC and horizon effect (Python version).

Python translation of mpc/Module_two.m.
Requires: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    from scipy.linalg import solve_discrete_are
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for Module_two.py. Install with: pip install scipy"
    ) from exc


# -------------------- System matrices (2D state) --------------------
A = np.array([[1.0, 0.1],
              [0.0, 1.0]])
B = np.array([[0.0],
              [1.0]])

nx = A.shape[0]
nu = B.shape[1]

# -------------------- Simulation settings ---------------------------
Nsim = 40
x0 = np.array([-3.0, 2.0])
Np = 3  # try 3, 10, 30

# -------------------- Cost matrices (students tune) -----------------
Q = np.diag([5.0, 1.0])
R = np.array([[0.1]])

# -------------------- Constraints ----------------------------------
umin = -0.8
umax = 0.8
xmin = np.array([-5.0, -5.0])
xmax = np.array([5.0, 5.0])


def build_prediction_matrices(a: np.ndarray, b: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    phi = np.zeros((nx * horizon, nx))
    gamma = np.zeros((nx * horizon, nu * horizon))

    for i in range(1, horizon + 1):
        phi[(i - 1) * nx:i * nx, :] = np.linalg.matrix_power(a, i)
        for j in range(1, i + 1):
            gamma[(i - 1) * nx:i * nx, (j - 1) * nu:j * nu] = np.linalg.matrix_power(a, i - j) @ b

    return phi, gamma


def solve_box_qp(h: np.ndarray, f: np.ndarray, lower: float, upper: float) -> np.ndarray | None:
    n = h.shape[0]

    def obj(u_vec: np.ndarray) -> float:
        return float(0.5 * u_vec @ h @ u_vec + f @ u_vec)

    # Gradient helps SLSQP converge faster and more reliably
    def jac(u_vec: np.ndarray) -> np.ndarray:
        return h @ u_vec + f

    u0 = np.zeros(n)
    bounds = [(lower, upper)] * n

    res = minimize(obj, u0, jac=jac, method="SLSQP", bounds=bounds, options={"disp": False, "maxiter": 200})
    if not res.success:
        return None
    return res.x


phi, gamma = build_prediction_matrices(A, B, Np)
qbar = np.kron(np.eye(Np), Q)
rbar = np.kron(np.eye(Np), R)


def run_mpc(use_terminal_cost: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qbar_local = qbar.copy()
    if use_terminal_cost:
        try:
            p_term = solve_discrete_are(A, B, Q, R)
        except Exception:
            p_term = Q.copy()
        qbar_local[-nx:, -nx:] = p_term

    x = np.zeros((nx, Nsim + 1))
    u_applied = np.zeros(Nsim)
    input_saturated = np.zeros(Nsim, dtype=bool)
    x[:, 0] = x0

    h = gamma.T @ qbar_local @ gamma + rbar
    h = 0.5 * (h + h.T) + 1e-8 * np.eye(h.shape[0])

    for k in range(Nsim):
        xk = x[:, k]
        f = gamma.T @ qbar_local @ phi @ xk

        u_opt = solve_box_qp(h, f, umin, umax)

        if u_opt is None:
            u = 0.0
            input_saturated[k] = False
            mode = "with terminal cost" if use_terminal_cost else "no terminal cost"
            print(f"[warning] QP infeasible at step {k + 1} ({mode}). Applying u = 0.")
        else:
            u = float(u_opt[0])
            input_saturated[k] = np.isclose(u, umax, atol=1e-6) or np.isclose(u, umin, atol=1e-6)

        u_applied[k] = u
        x[:, k + 1] = A @ x[:, k] + B[:, 0] * u

        # Optional clipping for readability (as in provided MATLAB script)
        x[:, k + 1] = np.minimum(np.maximum(x[:, k + 1], xmin), xmax)

    return x, u_applied, input_saturated


x, u_applied, input_saturated = run_mpc(use_terminal_cost=False)

k_state = np.arange(Nsim + 1)
k_input = np.arange(Nsim)

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)

axes[0].plot(k_state, x[0, :], linewidth=1.8, label="x1")
axes[0].plot(k_state, x[1, :], linewidth=1.8, label="x2")
axes[0].grid(True)
axes[0].set_ylabel("x1(k), x2(k)")
axes[0].legend(loc="best")
axes[0].set_title(f"Module 2: MPC closed-loop (Np={Np}, Q=diag[{Q[0,0]:g},{Q[1,1]:g}], R={R[0,0]:.3g})")

axes[1].step(k_input, u_applied, where="post", linewidth=2)
axes[1].axhline(umax, color="r", linestyle="--", label="u_max")
axes[1].axhline(umin, color="r", linestyle="--", label="u_min")
axes[1].grid(True)
axes[1].set_ylabel("u(k)")
axes[1].set_title("Applied control input")

axes[2].stem(k_input, input_saturated.astype(int), linefmt="C0-", markerfmt="C0o", basefmt="k-")
axes[2].grid(True)
axes[2].set_ylim(-0.1, 1.1)
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["inactive", "active"])
axes[2].set_xlabel("k")
axes[2].set_ylabel("sat?")
axes[2].set_title("Input-constraint activation")

fig.tight_layout()

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "module2_closed_loop.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure: {fig_path}")

# Set this True to reproduce Module 2 Question 5 (terminal-cost comparison)
run_terminal_cost_comparison = False

if run_terminal_cost_comparison:
    x_p, u_p, sat_p = run_mpc(use_terminal_cost=True)

    fig2, axes2 = plt.subplots(3, 1, figsize=(9, 9), sharex=False)

    axes2[0].plot(k_state, x[0, :], "b-", linewidth=1.6, label="x1 (no P)")
    axes2[0].plot(k_state, x[1, :], "r-", linewidth=1.6, label="x2 (no P)")
    axes2[0].plot(k_state, x_p[0, :], "b--", linewidth=1.6, label="x1 (with P)")
    axes2[0].plot(k_state, x_p[1, :], "r--", linewidth=1.6, label="x2 (with P)")
    axes2[0].grid(True)
    axes2[0].set_ylabel("x1(k), x2(k)")
    axes2[0].legend(loc="best")
    axes2[0].set_title(
        f"Comparison: Np={Np}, Q=diag[{Q[0,0]:g},{Q[1,1]:g}], R={R[0,0]:.3g} "
        "(solid=no P, dashed=with P)"
    )

    axes2[1].step(k_input, u_applied, where="post", color="k", linewidth=2, label="no P")
    axes2[1].step(k_input, u_p, where="post", color="m", linestyle="--", linewidth=2, label="with P")
    axes2[1].axhline(umax, color="r", linestyle="--", label="u_max")
    axes2[1].axhline(umin, color="r", linestyle="--", label="u_min")
    axes2[1].grid(True)
    axes2[1].set_ylabel("u(k)")
    axes2[1].legend(loc="best")
    axes2[1].set_title("Applied control input (solid=no terminal cost, dashed=with terminal cost)")

    axes2[2].stem(k_input, input_saturated.astype(int), linefmt="k-", markerfmt="ko", basefmt="k-", label="no P")
    axes2[2].stem(k_input, sat_p.astype(int), linefmt="m-", markerfmt="mo", basefmt="k-", label="with P")
    axes2[2].grid(True)
    axes2[2].set_ylim(-0.1, 1.1)
    axes2[2].set_yticks([0, 1])
    axes2[2].set_yticklabels(["inactive", "active"])
    axes2[2].set_xlabel("k")
    axes2[2].set_ylabel("sat?")
    axes2[2].legend(loc="best")
    axes2[2].set_title("Input-constraint activation (no P vs with P)")

    fig2.tight_layout()
    fig_path2 = fig_dir / "module2_terminal_cost_comparison.png"
    fig2.savefig(fig_path2, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path2}")

plt.show()
