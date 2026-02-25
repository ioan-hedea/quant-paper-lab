"""Module 1: Prediction and system behavior (Python version).

Python translation of mpc/Module_one.m.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

# Simulate open-loop trajectory
x = np.zeros((2, Nsim + 1))
x[:, 0] = x0

for k in range(Nsim):
    if (k + 1) == disturbance_step:  # match MATLAB indexing semantics
        x[:, k] = x[:, k] + d
    x[:, k + 1] = A @ x[:, k] + (B[:, 0] * u)

# Plot
k = np.arange(Nsim + 1)
plt.figure(figsize=(8, 4.5))
plt.plot(k, x[0, :], linewidth=2, label="x1")
plt.plot(k, x[1, :], linewidth=2, label="x2")
plt.grid(True)
plt.xlabel("k")
plt.ylabel("State components")
plt.title("Module 1: Open-loop state prediction")
plt.legend()
plt.tight_layout()

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "module1_open_loop.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure: {fig_path}")

plt.show()
