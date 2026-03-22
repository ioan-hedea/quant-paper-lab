"""
Section 1: MDP Dynamic Programming - Value Iteration & Policy Iteration
========================================================================
Visualizes the core DP algorithms from the SDM notes:
- Value Iteration with convergence animation
- Policy Iteration with policy evolution
- Bellman operator contraction visualization
- Comparison of VI vs PI convergence

Uses a configurable gridworld MDP.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

# ============================================================
# Gridworld MDP
# ============================================================

class GridWorldMDP:
    """
    NxN gridworld with stochastic transitions.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    def __init__(self, n=5, gamma=0.9, slip_prob=0.1,
                 goal_states=None, obstacle_states=None,
                 goal_reward=10.0, step_cost=-0.1, obstacle_penalty=-5.0):
        self.n = n
        self.gamma = gamma
        self.slip_prob = slip_prob
        self.n_states = n * n
        self.n_actions = 4

        self.goal_states = goal_states or [(n-1, n-1)]
        self.obstacle_states = obstacle_states or [(1, 1), (2, 3)]
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.obstacle_penalty = obstacle_penalty

        # Build transition and reward matrices
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        self._build_model()

    def _state_to_idx(self, r, c):
        return r * self.n + c

    def _idx_to_state(self, idx):
        return idx // self.n, idx % self.n

    def _build_model(self):
        for r in range(self.n):
            for c in range(self.n):
                s = self._state_to_idx(r, c)
                if (r, c) in self.goal_states:
                    # Terminal: all actions lead to self with 0 reward
                    for a in range(self.n_actions):
                        self.T[s, a, s] = 1.0
                        self.R[s, a] = 0.0
                    continue

                for a in range(self.n_actions):
                    self.R[s, a] = self.step_cost
                    # Intended direction
                    for a_actual in range(self.n_actions):
                        if a_actual == a:
                            prob = 1.0 - self.slip_prob
                        else:
                            prob = self.slip_prob / 3.0

                        dr, dc = self.ACTIONS[a_actual]
                        nr, nc = r + dr, c + dc
                        # Wall bounce
                        if nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
                            nr, nc = r, c
                        ns = self._state_to_idx(nr, nc)
                        self.T[s, a, ns] += prob

                    # Reward adjustments for reaching special states
                    for ns in range(self.n_states):
                        nr, nc = self._idx_to_state(ns)
                        if (nr, nc) in self.goal_states:
                            self.R[s, a] += self.T[s, a, ns] * self.goal_reward
                        elif (nr, nc) in self.obstacle_states:
                            self.R[s, a] += self.T[s, a, ns] * self.obstacle_penalty


# ============================================================
# Value Iteration
# ============================================================

def value_iteration(mdp, epsilon=1e-6, max_iters=500):
    """Returns Q-values, V-values history, and policy at each iteration."""
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    V_history = []
    policy_history = []
    delta_history = []

    for it in range(max_iters):
        Q_new = np.zeros_like(Q)
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q_new[s, a] = mdp.R[s, a] + mdp.gamma * np.sum(
                    mdp.T[s, a, :] * np.max(Q, axis=1)
                )

        delta = np.max(np.abs(Q_new - Q))
        Q = Q_new.copy()
        V = np.max(Q, axis=1)
        pi = np.argmax(Q, axis=1)

        V_history.append(V.copy())
        policy_history.append(pi.copy())
        delta_history.append(delta)

        if delta < epsilon:
            break

    return Q, V_history, policy_history, delta_history


# ============================================================
# Policy Iteration
# ============================================================

def policy_evaluation(mdp, pi, epsilon=1e-8, max_iters=1000):
    """Evaluate a deterministic policy by solving the linear system."""
    # Build P^pi and r^pi
    P_pi = np.zeros((mdp.n_states, mdp.n_states))
    r_pi = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        a = pi[s]
        P_pi[s, :] = mdp.T[s, a, :]
        r_pi[s] = mdp.R[s, a]

    # Solve (I - gamma * P^pi) V = r^pi
    A = np.eye(mdp.n_states) - mdp.gamma * P_pi
    V = np.linalg.solve(A, r_pi)
    return V


def policy_iteration(mdp, max_iters=100):
    """Returns optimal policy and history of policies and values."""
    pi = np.zeros(mdp.n_states, dtype=int)  # Start with action 0
    V_history = []
    policy_history = []

    for it in range(max_iters):
        # Policy Evaluation
        V = policy_evaluation(mdp, pi)
        V_history.append(V.copy())
        policy_history.append(pi.copy())

        # Policy Improvement
        Q = np.zeros((mdp.n_states, mdp.n_actions))
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q[s, a] = mdp.R[s, a] + mdp.gamma * np.sum(mdp.T[s, a, :] * V)

        pi_new = np.argmax(Q, axis=1)
        if np.all(pi_new == pi):
            break
        pi = pi_new

    return pi, V_history, policy_history


# ============================================================
# Visualization Functions
# ============================================================

def plot_value_grid(ax, V, mdp, title="", show_policy=True, policy=None):
    """Plot value function as a heatmap with optional policy arrows."""
    n = mdp.n
    V_grid = V.reshape(n, n)
    im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    ax.set_title(title, fontsize=11)

    # Annotate values
    for r in range(n):
        for c in range(n):
            color = 'white' if abs(V_grid[r, c]) > 0.5 * np.max(np.abs(V_grid)) else 'black'
            ax.text(c, r, f'{V_grid[r, c]:.1f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    # Goal and obstacle markers
    for gr, gc in mdp.goal_states:
        ax.add_patch(plt.Rectangle((gc-0.5, gr-0.5), 1, 1,
                                    fill=False, edgecolor='gold', linewidth=3))
    for or_, oc in mdp.obstacle_states:
        ax.add_patch(plt.Rectangle((oc-0.5, or_-0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=3))

    # Policy arrows
    if show_policy and policy is not None:
        arrow_map = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
        for s in range(mdp.n_states):
            r, c = mdp._idx_to_state(s)
            if (r, c) in mdp.goal_states:
                continue
            dx, dy = arrow_map[policy[s]]
            ax.annotate('', xy=(c + dx, r + dy), xytext=(c, r),
                       arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, alpha=0.3)
    return im


def demo_value_iteration():
    """Demonstrate Value Iteration with convergence visualization."""
    mdp = GridWorldMDP(n=5, gamma=0.9, slip_prob=0.1)
    Q, V_hist, pi_hist, deltas = value_iteration(mdp)

    # Plot 1: Value function evolution at selected iterations
    iters_to_show = [0, 1, 2, 5, min(10, len(V_hist)-1), len(V_hist)-1]
    iters_to_show = sorted(set(iters_to_show))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Value Iteration: Value Function Evolution', fontsize=14, fontweight='bold')

    for idx, (ax, it) in enumerate(zip(axes.flat, iters_to_show)):
        plot_value_grid(ax, V_hist[it], mdp,
                       title=f'Iteration {it+1}',
                       show_policy=True, policy=pi_hist[it])
    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mdp_dp/vi_evolution.png', dpi=150)
    plt.show()

    # Plot 2: Convergence (Bellman residual)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(range(1, len(deltas)+1), deltas, 'b-o', markersize=3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Bellman Residual (log scale)')
    ax1.set_title('Value Iteration Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-6, color='r', linestyle='--', label='ε = 1e-6')
    ax1.legend()

    # Track V(s0) over iterations
    s0_values = [V[0] for V in V_hist]
    ax2.plot(range(1, len(s0_values)+1), s0_values, 'g-o', markersize=3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('V(s₀)')
    ax2.set_title('Value of Start State Over Iterations')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mdp_dp/vi_convergence.png', dpi=150)
    plt.show()

    print(f"Value Iteration converged in {len(deltas)} iterations")
    print(f"Final V(start): {V_hist[-1][0]:.4f}")
    print(f"Optimal policy (actions): {[mdp.ACTION_NAMES[a] for a in pi_hist[-1]]}")


def demo_policy_iteration():
    """Demonstrate Policy Iteration with policy evolution."""
    mdp = GridWorldMDP(n=5, gamma=0.9, slip_prob=0.1)
    pi_opt, V_hist, pi_hist = policy_iteration(mdp)

    n_iters = len(V_hist)
    cols = min(n_iters, 4)
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
    if cols == 1:
        axes = [axes]
    fig.suptitle('Policy Iteration: Policy & Value Evolution', fontsize=14, fontweight='bold')

    for idx in range(cols):
        it = idx if idx < n_iters - 1 else n_iters - 1
        plot_value_grid(axes[idx], V_hist[it], mdp,
                       title=f'Iteration {it+1}',
                       show_policy=True, policy=pi_hist[it])

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mdp_dp/pi_evolution.png', dpi=150)
    plt.show()

    print(f"Policy Iteration converged in {n_iters} iterations")
    print(f"Final V(start): {V_hist[-1][0]:.4f}")


def demo_bellman_contraction():
    """Visualize the Bellman operator as a contraction mapping."""
    mdp = GridWorldMDP(n=4, gamma=0.9, slip_prob=0.0)

    # Start from two different Q initializations
    np.random.seed(42)
    Q1 = np.random.randn(mdp.n_states, mdp.n_actions) * 5
    Q2 = np.random.randn(mdp.n_states, mdp.n_actions) * 5

    diffs = [np.max(np.abs(Q1 - Q2))]
    theoretical = [diffs[0]]

    for _ in range(30):
        Q1_new = np.zeros_like(Q1)
        Q2_new = np.zeros_like(Q2)
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q1_new[s, a] = mdp.R[s, a] + mdp.gamma * np.sum(
                    mdp.T[s, a, :] * np.max(Q1, axis=1))
                Q2_new[s, a] = mdp.R[s, a] + mdp.gamma * np.sum(
                    mdp.T[s, a, :] * np.max(Q2, axis=1))
        Q1, Q2 = Q1_new, Q2_new
        diffs.append(np.max(np.abs(Q1 - Q2)))
        theoretical.append(theoretical[-1] * mdp.gamma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(diffs, 'bo-', markersize=4, label='||BⁿQ₁ - BⁿQ₂||∞ (actual)')
    ax.semilogy(theoretical, 'r--', label=f'γⁿ · ||Q₁ - Q₂||∞ (theoretical bound, γ={mdp.gamma})')
    ax.set_xlabel('Iteration n')
    ax.set_ylabel('L∞ distance (log scale)')
    ax.set_title('Bellman Operator Contraction: Two Q-functions converge to same Q*')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mdp_dp/bellman_contraction.png', dpi=150)
    plt.show()


def demo_vi_vs_pi():
    """Compare convergence speed of VI vs PI."""
    mdp = GridWorldMDP(n=6, gamma=0.95, slip_prob=0.1)

    # VI
    _, V_hist_vi, _, deltas_vi = value_iteration(mdp)
    # PI
    _, V_hist_pi, _ = policy_iteration(mdp)

    # Compute distance to optimal V
    V_star = V_hist_vi[-1]
    vi_errors = [np.max(np.abs(V - V_star)) for V in V_hist_vi]
    pi_errors = [np.max(np.abs(V - V_star)) for V in V_hist_pi]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(vi_errors)+1), vi_errors, 'b-o', markersize=3,
                label=f'Value Iteration ({len(vi_errors)} iters)')
    ax.semilogy(range(1, len(pi_errors)+1), pi_errors, 'r-s', markersize=5,
                label=f'Policy Iteration ({len(pi_errors)} iters)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||V - V*||∞ (log scale)')
    ax.set_title('Convergence Comparison: Value Iteration vs Policy Iteration (6×6 grid, γ=0.95)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mdp_dp/vi_vs_pi.png', dpi=150)
    plt.show()

    print(f"VI: {len(vi_errors)} iterations, PI: {len(pi_errors)} iterations")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Value Iteration")
    print("=" * 60)
    demo_value_iteration()

    print("\n" + "=" * 60)
    print("Demo 2: Policy Iteration")
    print("=" * 60)
    demo_policy_iteration()

    print("\n" + "=" * 60)
    print("Demo 3: Bellman Operator Contraction")
    print("=" * 60)
    demo_bellman_contraction()

    print("\n" + "=" * 60)
    print("Demo 4: VI vs PI Convergence Comparison")
    print("=" * 60)
    demo_vi_vs_pi()
