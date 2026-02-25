"""Module 3: Tube MPC (Python version).

Python translation of mpc/Module_three.m.
Requires: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.linalg import solve_discrete_are
    from scipy.optimize import minimize
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

# -------------------- MPC weights and horizon ---------------------
Np = 10
Q = np.diag([5.0, 1.0])
R = np.array([[0.1]])
Nsim = 40

# Optional terminal weight
try:
    P = solve_discrete_are(A, B, Q, R)
except Exception:
    P = Q.copy()

# -------------------- Constraints ---------------------------------
xmin = np.array([-5.0, -5.0])
xmax = np.array([5.0, 5.0])
umin = -0.8
umax = 0.8

# -------------------- Ancillary error feedback K ------------------
Qerror = np.diag([10.0, 1.0])
Rerror = np.array([[0.9]])

Perr = solve_discrete_are(A, B, Qerror, Rerror)
K = np.linalg.solve(Rerror + B.T @ Perr @ B, B.T @ Perr @ A)
Acl = A - B @ K
eig_cl = np.linalg.eigvals(Acl)
print("eig(A-BK) =", eig_cl)

# -------------------- Disturbance and tube tightening -------------
disturbance_bound = 0.2  # assignment base run default
w_inf = disturbance_bound * np.ones(nx)

r_inf = np.zeros(nx)
for _ in range(500):
    r_next = np.abs(Acl) @ r_inf + w_inf
    if np.linalg.norm(r_next - r_inf, ord=np.inf) < 1e-10:
        r_inf = r_next
        break
    r_inf = r_next

kappa = np.abs(K) @ r_inf
kappa = float(kappa.item())

xmin_tight = xmin + r_inf
xmax_tight = xmax - r_inf
umin_tight = umin + kappa
umax_tight = umax - kappa

if np.any(xmin_tight >= xmax_tight) or (umin_tight >= umax_tight):
    raise RuntimeError(
        "Tube tightening infeasible. Reduce disturbance_bound or adjust constraints."
    )


def build_prediction_matrices(a: np.ndarray, b: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    phi = np.zeros((nx * horizon, nx))
    gamma = np.zeros((nx * horizon, nu * horizon))

    for i in range(1, horizon + 1):
        phi[(i - 1) * nx:i * nx, :] = np.linalg.matrix_power(a, i)
        for j in range(1, i + 1):
            gamma[(i - 1) * nx:i * nx, (j - 1) * nu:j * nu] = np.linalg.matrix_power(a, i - j) @ b

    return phi, gamma


def solve_qp_slsqp(
    h: np.ndarray,
    f: np.ndarray,
    a_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: float,
    ub: float,
) -> np.ndarray | None:
    n = h.shape[0]

    def obj(u_vec: np.ndarray) -> float:
        return float(0.5 * u_vec @ h @ u_vec + f @ u_vec)

    def jac(u_vec: np.ndarray) -> np.ndarray:
        return h @ u_vec + f

    # SLSQP expects inequality constraints c(u) >= 0
    cons = [{"type": "ineq", "fun": lambda u_vec, a=a_ineq, b=b_ineq: b - a @ u_vec}]
    bounds = [(lb, ub)] * n
    u0 = np.zeros(n)

    res = minimize(obj, u0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons, options={"disp": False, "maxiter": 400})
    if not res.success:
        return None
    return res.x


phi, gamma = build_prediction_matrices(A, B, Np)
qbar = np.kron(np.eye(Np), Q)
qbar[-nx:, -nx:] = P
rbar = np.kron(np.eye(Np), R)

h = gamma.T @ qbar @ gamma + rbar
h = 0.5 * (h + h.T) + 1e-9 * np.eye(h.shape[0])

x0 = np.array([-3.0, 2.0])
x_nom = np.zeros((nx, Nsim + 1))
x_act = np.zeros((nx, Nsim + 1))
u_nom_hist = np.zeros(Nsim)
u_act_hist = np.zeros(Nsim)
qp_ok = np.zeros(Nsim, dtype=bool)
e_norm_inf = np.zeros(Nsim + 1)

x_nom[:, 0] = x0
x_act[:, 0] = x0
e_norm_inf[0] = np.linalg.norm(x_act[:, 0] - x_nom[:, 0], ord=np.inf)

rng = np.random.default_rng(seed=1)

for k in range(Nsim):
    xk_nom = x_nom[:, k]

    # State constraints over prediction horizon
    a_ineq_x = np.vstack((gamma, -gamma))
    b_ineq_x = np.concatenate((
        np.kron(np.ones(Np), xmax_tight) - phi @ xk_nom,
        -np.kron(np.ones(Np), xmin_tight) + phi @ xk_nom,
    ))

    f = gamma.T @ qbar @ phi @ xk_nom

    v_opt = solve_qp_slsqp(h, f, a_ineq_x, b_ineq_x, umin_tight, umax_tight)

    if v_opt is None:
        v0 = 0.0
        print(f"[warning] Nominal QP infeasible at step {k + 1}. Applying fallback v=0.")
    else:
        qp_ok[k] = True
        v0 = float(v_opt[0])

    u_nom = v0
    x_nom[:, k + 1] = A @ x_nom[:, k] + B[:, 0] * u_nom

    e = x_act[:, k] - x_nom[:, k]
    u_act = u_nom - float(K @ e)
    u_act = float(np.clip(u_act, umin, umax))

    w = disturbance_bound * (2.0 * rng.random(nx) - 1.0)
    x_act[:, k + 1] = A @ x_act[:, k] + B[:, 0] * u_act + w

    u_nom_hist[k] = u_nom
    u_act_hist[k] = u_act
    e_norm_inf[k + 1] = np.linalg.norm(x_act[:, k + 1] - x_nom[:, k + 1], ord=np.inf)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(x_nom[0, :], x_nom[1, :], "b-", linewidth=2, label="Nominal trajectory")
ax.plot(x_act[0, :], x_act[1, :], "r-", linewidth=2, label="Actual trajectory")
for kk in range(0, Nsim + 1, 5):
    xk = x_nom[:, kk]
    rect = plt.Rectangle((xk[0] - r_inf[0], xk[1] - r_inf[1]), 2 * r_inf[0], 2 * r_inf[1],
                         fill=False, edgecolor=(0.0, 0.6, 0.0), linewidth=0.4)
    ax.add_patch(rect)
ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Module 3: Tube MPC (state space)")
ax.legend(loc="best")

ax = axes[0, 1]
k = np.arange(Nsim)
ax.step(k, u_nom_hist, where="post", linewidth=1.8, label="u_nom")
ax.step(k, u_act_hist, where="post", linewidth=1.8, linestyle="--", label="u_act")
ax.axhline(umax, color="k", linestyle=":", label="u_max")
ax.axhline(umin, color="k", linestyle=":", label="u_min")
ax.grid(True)
ax.set_xlabel("k")
ax.set_ylabel("u")
ax.set_title("Inputs")
ax.legend(loc="best")

ax = axes[1, 0]
k2 = np.arange(Nsim + 1)
ax.plot(k2, e_norm_inf, "m-", linewidth=1.8)
ax.axhline(np.max(r_inf), color="k", linestyle="--", label="max tube radius")
ax.grid(True)
ax.set_xlabel("k")
ax.set_ylabel("||e_k||_inf")
ax.set_title("Error vs. tube size")
ax.legend(loc="best")

ax = axes[1, 1]
ax.stem(np.arange(Nsim), qp_ok.astype(int), basefmt="k-")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(["infeasible", "feasible"])
ax.grid(True)
ax.set_xlabel("k")
ax.set_ylabel("QP status")
ax.set_title("Nominal QP feasibility")

fig.tight_layout()

print(f"Tightened state bounds: x1 in [{xmin_tight[0]:.3f}, {xmax_tight[0]:.3f}], x2 in [{xmin_tight[1]:.3f}, {xmax_tight[1]:.3f}]")
print(f"Tightened input bounds: u in [{umin_tight:.3f}, {umax_tight:.3f}]")
print(f"QP feasible at {int(np.sum(qp_ok))}/{Nsim} steps")

plt.show()
