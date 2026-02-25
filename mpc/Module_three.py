"""Module 3: Tube MPC intuition (Python version).

Python translation aligned with mpc/Module_three.m and assignment text:
- unconstrained nominal finite-horizon MPC,
- ancillary feedback u = u_nom - K e,
- bounded disturbances,
- tube visualization around nominal trajectory.

Requires: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.linalg import solve_discrete_are
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for Module_three.py. Install with: pip install scipy"
    ) from exc


# -------------------- System matrices (2D state vector) ----------
A = np.array([[1.0, 0.1],
              [0.0, 1.0]])
B = np.array([[0.0],
              [1.0]])

nx = A.shape[0]
nu = B.shape[1]

# -------------------- Tube MPC parameters -------------------------
Np = 10
Q = np.diag([5.0, 1.0])
R = np.array([[0.1]])

Qerror = np.diag([10.0, 1.0])
Rerror = np.array([[0.9]])

# Ancillary LQR for error dynamics
Perr = solve_discrete_are(A, B, Qerror, Rerror)
K = np.linalg.solve(Rerror + B.T @ Perr @ B, B.T @ Perr @ A)
Acl = A - B @ K
eig_cl = np.linalg.eigvals(Acl)
print("eig(A-BK) =", eig_cl)

disturbance_bound = 0.2  # assignment base-run default
Nsim = 40

x0 = np.array([-3.0, 2.0])
x_nom = np.zeros((nx, Nsim + 1))
x_act = np.zeros((nx, Nsim + 1))
u_nom_hist = np.zeros(Nsim)
u_act_hist = np.zeros(Nsim)

x_nom[:, 0] = x0
x_act[:, 0] = x0


def build_prediction_matrices(a: np.ndarray, b: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    phi = np.zeros((nx * horizon, nx))
    gamma = np.zeros((nx * horizon, nu * horizon))

    for i in range(1, horizon + 1):
        phi[(i - 1) * nx:i * nx, :] = np.linalg.matrix_power(a, i)
        for j in range(1, i + 1):
            gamma[(i - 1) * nx:i * nx, (j - 1) * nu:j * nu] = np.linalg.matrix_power(a, i - j) @ b

    return phi, gamma


phi, gamma = build_prediction_matrices(A, B, Np)
qbar = np.kron(np.eye(Np), Q)
rbar = np.kron(np.eye(Np), R)

h = gamma.T @ qbar @ gamma + rbar
h = 0.5 * (h + h.T) + 1e-9 * np.eye(h.shape[0])

rng = np.random.default_rng(seed=1)

for k in range(Nsim):
    xk_nom = x_nom[:, k]

    # Unconstrained nominal finite-horizon QP:
    # minimize 0.5*V'HV + f'V  => V* = -H^{-1} f
    f = gamma.T @ qbar @ phi @ xk_nom
    v_opt = -np.linalg.solve(h, f)
    u_nom = float(v_opt[0])

    # Tube policy
    e = x_act[:, k] - x_nom[:, k]
    u_act = u_nom - float(K @ e)

    # Disturbance
    w = disturbance_bound * (2.0 * rng.random(nx) - 1.0)

    # Updates
    x_nom[:, k + 1] = A @ x_nom[:, k] + B[:, 0] * u_nom
    x_act[:, k + 1] = A @ x_act[:, k] + B[:, 0] * u_act + w

    u_nom_hist[k] = u_nom
    u_act_hist[k] = u_act


# -------------------- Plot ----------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

ax = axes[0]
ax.plot(x_nom[0, :], x_nom[1, :], "b-", linewidth=2, label="Nominal trajectory")
ax.plot(x_act[0, :], x_act[1, :], "r-", linewidth=2, label="Actual trajectory")

# Assignment-style tube visualization: radius ~ disturbance_bound * ||K||
radius = float(disturbance_bound * np.linalg.norm(K))
for kk in range(0, Nsim + 1, 5):
    center = (x_nom[0, kk], x_nom[1, kk])
    circ = plt.Circle(center, radius, fill=False, color=(0.0, 0.6, 0.0), linewidth=0.7)
    ax.add_patch(circ)

ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Module 3: Tube MPC visualization")
ax.legend(loc="best")

ax = axes[1]
k = np.arange(Nsim)
ax.step(k, u_nom_hist, where="post", linewidth=1.8, label="u_nom")
ax.step(k, u_act_hist, where="post", linewidth=1.8, linestyle="--", label="u_act")
ax.grid(True)
ax.set_xlabel("k")
ax.set_ylabel("u")
ax.set_title("Nominal vs actual input")
ax.legend(loc="best")

fig.tight_layout()

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "module3_tube_mpc.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure: {fig_path}")

plt.show()
