"""Module 1: Prediction and system behavior (Python version).

Python translation of mpc/Module_one.m, including the optional
closed-loop preview requested in Question 5.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.linalg import solve_discrete_are
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for the optional closed-loop preview. "
        "Install with: pip install scipy"
    ) from exc


def format_vec(vec: np.ndarray) -> str:
    return "_".join(f"{float(v):g}".replace("-", "m").replace(".", "p") for v in vec)


# System matrices (2D state vector)
A = np.array([[1.0, 0.1],
              [0.0, 1.0]])
B = np.array([[0.0],
              [1.0]])

# Simulation settings
Nsim = 40
x0 = np.array([-2.0, 3.0])
u = 0.1

# Disturbance setup (assignment base run uses d = [0, 0])
disturbance_step = 15
d = np.array([0.0, 0.0])

# Set to True for Question 5
run_closed_loop_preview = True

# Simulate open-loop trajectory
x = np.zeros((2, Nsim + 1))
x[:, 0] = x0

for k in range(Nsim):
    if (k + 1) == disturbance_step:  # match MATLAB indexing semantics
        x[:, k] = x[:, k] + d
    x[:, k + 1] = A @ x[:, k] + (B[:, 0] * u)


# Plot open-loop trajectory
k = np.arange(Nsim + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(k, x[0, :], linewidth=2, label="x1")
ax.plot(k, x[1, :], linewidth=2, label="x2")
ax.grid(True)
ax.set_xlabel("k")
ax.set_ylabel("State components")
ax.set_title("Module 1: Open-loop state prediction")
ax.legend()
fig.tight_layout()

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
base_name = (
    f"module1_open_loop_x0_{format_vec(x0)}"
    f"_u_{u:g}_dstep_{disturbance_step}_d_{format_vec(d)}.png"
)
fig_path = fig_dir / base_name
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure: {fig_path}")
if run_closed_loop_preview:
    q_lqr = np.diag([1.0, 0.2])
    r_lqr = np.array([[0.5]])
    p_lqr = solve_discrete_are(A, B, q_lqr, r_lqr)
    k_fb = np.linalg.solve(r_lqr + B.T @ p_lqr @ B, B.T @ p_lqr @ A)

    x_ref = np.array([0.0, 0.0])
    x_cl = np.zeros((2, Nsim + 1))
    x_cl[:, 0] = x0

    for k_idx in range(Nsim):
        if (k_idx + 1) == disturbance_step:
            x_cl[:, k_idx] = x_cl[:, k_idx] + d

        u_fb = -k_fb @ (x_cl[:, k_idx] - x_ref)
        x_cl[:, k_idx + 1] = A @ x_cl[:, k_idx] + B[:, 0] * float(u_fb)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5))
    ax2.plot(k, x[0, :], "b-", linewidth=2, label="x1 (open-loop)")
    ax2.plot(k, x[1, :], "r-", linewidth=2, label="x2 (open-loop)")
    ax2.plot(k, x_cl[0, :], "b--", linewidth=2, label="x1 (closed-loop)")
    ax2.plot(k, x_cl[1, :], "r--", linewidth=2, label="x2 (closed-loop)")
    ax2.axvline(disturbance_step, color="k", linestyle=":", linewidth=1.5, label="disturbance time")
    ax2.grid(True)
    ax2.set_xlabel("k")
    ax2.set_ylabel("State components")
    ax2.set_title("Module 1: Open-loop vs closed-loop with disturbance")
    ax2.legend(loc="best")
    fig2.tight_layout()

    fig_path2 = fig_dir / (
        f"module1_open_vs_closed_x0_{format_vec(x0)}"
        f"_u_{u:g}_dstep_{disturbance_step}_d_{format_vec(d)}.png"
    )
    fig2.savefig(fig_path2, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path2}")


plt.show()
