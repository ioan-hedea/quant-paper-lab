"""
Section 6: Safe and Robust Reinforcement Learning
===================================================
Visualizes core safe RL concepts from the SDM notes:
- CVaR (Conditional Value at Risk) optimization
- Robust MDPs with uncertainty sets
- Constrained MDPs with Lagrangian relaxation
- SPIBB: Safe Policy Improvement with Baseline Bootstrapping
- Shielding for safe exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# CVaR Visualization
# ============================================================

def demo_cvar():
    """Visualize VaR and CVaR for different return distributions."""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Risk-Sensitive Criteria: VaR and CVaR', fontsize=14, fontweight='bold')

    # Two policies with different return distributions
    n_samples = 10000
    # Policy A: High mean, heavy left tail (risky)
    returns_A = np.concatenate([
        np.random.normal(8, 2, int(n_samples * 0.7)),
        np.random.normal(-5, 3, int(n_samples * 0.3))
    ])
    # Policy B: Lower mean, tight distribution (safe)
    returns_B = np.random.normal(4, 1.5, n_samples)

    alpha_values = [0.05, 0.1, 0.25, 0.5]

    # Plot 1: Return distributions
    ax = axes[0, 0]
    ax.hist(returns_A, bins=80, alpha=0.6, color='red', density=True, label=f'Policy A (μ={returns_A.mean():.1f})')
    ax.hist(returns_B, bins=80, alpha=0.6, color='blue', density=True, label=f'Policy B (μ={returns_B.mean():.1f})')
    ax.axvline(returns_A.mean(), color='red', linestyle='--', linewidth=2)
    ax.axvline(returns_B.mean(), color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title('Return Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: CVaR for different alpha
    ax = axes[0, 1]
    cvar_A = []
    cvar_B = []
    for alpha in alpha_values:
        var_A = np.percentile(returns_A, alpha * 100)
        cvar_A.append(returns_A[returns_A <= var_A].mean())
        var_B = np.percentile(returns_B, alpha * 100)
        cvar_B.append(returns_B[returns_B <= var_B].mean())

    x = np.arange(len(alpha_values))
    width = 0.35
    ax.bar(x - width/2, cvar_A, width, color='red', alpha=0.7, label='Policy A (risky)')
    ax.bar(x + width/2, cvar_B, width, color='blue', alpha=0.7, label='Policy B (safe)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'α={a}' for a in alpha_values])
    ax.set_ylabel('CVaR_α')
    ax.set_title('CVaR: Expected Value of Worst α% Outcomes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 3: VaR vs CVaR illustration
    ax = axes[1, 0]
    alpha = 0.1
    sorted_A = np.sort(returns_A)
    var_idx = int(alpha * len(sorted_A))
    var_val = sorted_A[var_idx]
    cvar_val = sorted_A[:var_idx].mean()

    ax.hist(returns_A, bins=80, alpha=0.4, color='gray', density=True, label='Distribution')
    # Shade the tail
    ax.hist(sorted_A[:var_idx], bins=30, alpha=0.7, color='red', density=True, label=f'Worst {alpha*100:.0f}%')
    ax.axvline(var_val, color='orange', linewidth=2, linestyle='--', label=f'VaR_{alpha} = {var_val:.1f}')
    ax.axvline(cvar_val, color='darkred', linewidth=2, linestyle='-', label=f'CVaR_{alpha} = {cvar_val:.1f}')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title(f'VaR vs CVaR (α={alpha}): Policy A')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Decision boundary - when does risk-averse agent switch preference?
    ax = axes[1, 1]
    alphas = np.linspace(0.01, 1.0, 100)
    cvar_curve_A = []
    cvar_curve_B = []
    for a in alphas:
        var_A = np.percentile(returns_A, a * 100)
        tail_A = returns_A[returns_A <= var_A]
        cvar_curve_A.append(tail_A.mean() if len(tail_A) > 0 else var_A)

        var_B = np.percentile(returns_B, a * 100)
        tail_B = returns_B[returns_B <= var_B]
        cvar_curve_B.append(tail_B.mean() if len(tail_B) > 0 else var_B)

    ax.plot(alphas, cvar_curve_A, 'r-', linewidth=2, label='Policy A')
    ax.plot(alphas, cvar_curve_B, 'b-', linewidth=2, label='Policy B')

    # Find crossover
    diff = np.array(cvar_curve_A) - np.array(cvar_curve_B)
    cross_idx = np.where(np.diff(np.sign(diff)))[0]
    if len(cross_idx) > 0:
        cross_alpha = alphas[cross_idx[0]]
        ax.axvline(cross_alpha, color='green', linestyle='--',
                  label=f'Crossover α≈{cross_alpha:.2f}')

    ax.set_xlabel('Risk Level α')
    ax.set_ylabel('CVaR_α')
    ax.set_title('Risk Preference: Which Policy is Better?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate('Risk-averse\nprefers B', xy=(0.05, 0), fontsize=9, color='blue',
               xytext=(0.15, -6), arrowprops=dict(arrowstyle='->', color='blue'))

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/safe_rl/cvar_visualization.png', dpi=150)
    plt.show()


# ============================================================
# Robust MDP
# ============================================================

def demo_robust_mdp():
    """Visualize robust MDP: worst-case optimization over uncertainty set."""
    np.random.seed(42)

    # Simple 3-state MDP
    n_states = 3
    n_actions = 2
    gamma = 0.9

    # True transition probabilities
    T_true = np.array([
        # Action 0
        [[0.7, 0.2, 0.1],
         [0.1, 0.6, 0.3],
         [0.0, 0.1, 0.9]],
        # Action 1
        [[0.3, 0.5, 0.2],
         [0.4, 0.3, 0.3],
         [0.2, 0.3, 0.5]]
    ])  # Shape: (n_actions, n_states, n_states)

    R = np.array([[1.0, 0.5], [0.0, 2.0], [0.5, 0.0]])

    def solve_mdp(T, R, gamma, n_iters=100):
        V = np.zeros(n_states)
        for _ in range(n_iters):
            Q = np.zeros((n_states, n_actions))
            for s in range(n_states):
                for a in range(n_actions):
                    Q[s, a] = R[s, a] + gamma * T[a, s, :] @ V
            V = np.max(Q, axis=1)
        return V, np.argmax(Q, axis=1)

    # Vary uncertainty radius and solve robust MDP
    radii = np.linspace(0, 0.3, 20)
    n_perturbations = 50

    nominal_V, nominal_pi = solve_mdp(T_true, R, gamma)
    robust_values = []
    worst_case_values = []

    for radius in radii:
        worst_val = np.inf
        for _ in range(n_perturbations):
            T_perturbed = T_true.copy()
            for a in range(n_actions):
                for s in range(n_states):
                    noise = np.random.randn(n_states) * radius
                    T_perturbed[a, s, :] = np.maximum(T_true[a, s, :] + noise, 0.01)
                    T_perturbed[a, s, :] /= T_perturbed[a, s, :].sum()

            V_p, _ = solve_mdp(T_perturbed, R, gamma)
            if V_p[0] < worst_val:
                worst_val = V_p[0]

        worst_case_values.append(worst_val)
        robust_values.append(nominal_V[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Robust MDPs: Performance Under Model Uncertainty', fontsize=14, fontweight='bold')

    ax1.plot(radii, robust_values, 'b-', linewidth=2, label='Nominal V(s₀)')
    ax1.plot(radii, worst_case_values, 'r-', linewidth=2, label='Worst-case V(s₀)')
    ax1.fill_between(radii, worst_case_values, robust_values, alpha=0.2, color='orange',
                     label='Performance gap')
    ax1.set_xlabel('Uncertainty Radius ε')
    ax1.set_ylabel('V(s₀)')
    ax1.set_title('Nominal vs Worst-Case Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Uncertainty set visualization (2D simplex)
    ax2.set_title('Uncertainty Set in Transition Simplex\n(s₀, a₀ → 3 next states)')

    from matplotlib.patches import Circle
    # Draw the simplex triangle
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(triangle)

    # True probability as a point in the simplex
    p = T_true[0, 0, :]  # P(s'|s0, a0)
    point = p[0] * vertices[0] + p[1] * vertices[1] + p[2] * vertices[2]
    ax2.plot(point[0], point[1], 'bo', markersize=10, label='True P')

    # Uncertainty circles
    for r, color in [(0.05, 'green'), (0.15, 'orange'), (0.3, 'red')]:
        circle = Circle(point, r * 0.5, fill=False, edgecolor=color, linewidth=2,
                       linestyle='--', label=f'ε={r}')
        ax2.add_patch(circle)

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.set_xlabel('Simplex coordinate')
    ax2.grid(True, alpha=0.3)

    # Label vertices
    for i, name in enumerate(["s'=0", "s'=1", "s'=2"]):
        ax2.annotate(name, vertices[i], fontsize=10, fontweight='bold',
                    textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/safe_rl/robust_mdp.png', dpi=150)
    plt.show()


# ============================================================
# SPIBB: Safe Policy Improvement with Baseline Bootstrapping
# ============================================================

def demo_spibb():
    """Demonstrate SPIBB on a simple gridworld with offline data."""
    np.random.seed(42)

    n_states = 9  # 3x3 grid
    n_actions = 4
    gamma = 0.9

    # True MDP (3x3 grid, goal at state 8)
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    def grid_step(s, a):
        r, c = s // 3, s % 3
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[a]
        nr, nc = max(0, min(2, r+dr)), max(0, min(2, c+dc))
        ns = nr * 3 + nc
        reward = 10.0 if ns == 8 else -0.1
        return ns, reward

    # Behavior policy: mostly random with slight bias toward goal
    def behavior_policy(s):
        probs = np.ones(n_actions) / n_actions
        r, c = s // 3, s % 3
        if c < 2: probs[1] += 0.1  # bias right
        if r < 2: probs[2] += 0.1  # bias down
        probs /= probs.sum()
        return np.random.choice(n_actions, p=probs), probs

    # Collect offline dataset
    n_trajectories = 30
    dataset = []
    for _ in range(n_trajectories):
        s = 0
        for _ in range(50):
            a, _ = behavior_policy(s)
            ns, r = grid_step(s, a)
            dataset.append((s, a, r, ns))
            s = ns
            if s == 8:
                break

    # Count visits
    counts = np.zeros((n_states, n_actions))
    for s, a, r, ns in dataset:
        counts[s, a] += 1

    # Learn Q-values from data
    Q_hat = np.zeros((n_states, n_actions))
    for _ in range(100):
        Q_new = np.zeros_like(Q_hat)
        count_q = np.zeros_like(Q_hat)
        for s, a, r, ns in dataset:
            target = r + gamma * np.max(Q_hat[ns])
            Q_new[s, a] += target
            count_q[s, a] += 1
        for s in range(n_states):
            for a in range(n_actions):
                if count_q[s, a] > 0:
                    Q_hat[s, a] = Q_new[s, a] / count_q[s, a]

    # SPIBB with different thresholds
    thresholds = [1, 5, 10, 20]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SPIBB: Safe Policy Improvement with Baseline Bootstrapping', fontsize=14, fontweight='bold')

    action_names = ['↑', '→', '↓', '←']

    for ax, n_wedge in zip(axes.flat, thresholds):
        grid_actions = np.full((3, 3), '', dtype=object)
        grid_colors = np.zeros((3, 3))

        for s in range(n_states):
            r, c = s // 3, s % 3
            min_count = np.min(counts[s, :])

            if min_count < n_wedge:
                # Uncertain: use behavior policy
                _, probs = behavior_policy(s)
                best_a = np.argmax(probs)
                grid_actions[r, c] = action_names[best_a] + '*'
                grid_colors[r, c] = 0.3  # light (baseline)
            else:
                # Confident: use learned Q
                best_a = np.argmax(Q_hat[s])
                grid_actions[r, c] = action_names[best_a]
                grid_colors[r, c] = 1.0  # dark (improved)

        im = ax.imshow(grid_colors, cmap='RdYlGn', vmin=0, vmax=1)
        for r in range(3):
            for c in range(3):
                is_baseline = '*' in grid_actions[r, c]
                text = grid_actions[r, c].replace('*', '')
                color = 'red' if is_baseline else 'darkgreen'
                ax.text(c, r, text, ha='center', va='center', fontsize=16,
                       fontweight='bold', color=color)
                if is_baseline:
                    ax.text(c, r+0.35, 'baseline', ha='center', va='center',
                           fontsize=6, color='red', style='italic')

        ax.set_title(f'n∧ = {n_wedge} (threshold)')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.grid(True, alpha=0.3)

        # Count stats
        n_baseline = sum(1 for s in range(n_states) if np.min(counts[s, :]) < n_wedge)
        ax.set_xlabel(f'{n_baseline}/9 states use baseline')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/safe_rl/spibb.png', dpi=150)
    plt.show()

    # Visit count heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    count_grid = counts.sum(axis=1).reshape(3, 3)
    im = ax.imshow(count_grid, cmap='YlOrRd')
    plt.colorbar(im, ax=ax)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f'{int(count_grid[r, c])}', ha='center', va='center',
                   fontsize=14, fontweight='bold')
    ax.set_title('Offline Data: Visit Counts per State')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/safe_rl/visit_counts.png', dpi=150)
    plt.show()


# ============================================================
# Constrained MDP with Lagrangian Relaxation
# ============================================================

def demo_constrained_mdp():
    """Demonstrate Lagrangian approach to constrained MDPs."""
    np.random.seed(42)

    # Simple MDP: 2 actions, reward vs cost tradeoff
    # Action 0: safe (low reward, low cost)
    # Action 1: risky (high reward, high cost)
    n_episodes = 200
    lambdas = np.linspace(0, 5, 50)

    reward_a0, cost_a0 = 1.0, 0.1
    reward_a1, cost_a1 = 3.0, 0.8
    cost_threshold = 0.4

    expected_rewards = []
    expected_costs = []
    policy_probs = []

    for lam in lambdas:
        # Modified reward: r_tilde = r - lambda * c
        r_tilde_0 = reward_a0 - lam * cost_a0
        r_tilde_1 = reward_a1 - lam * cost_a1

        # Optimal policy for modified reward (softmax-like for visualization)
        if r_tilde_0 > r_tilde_1:
            p_risky = 0.0
        elif r_tilde_1 > r_tilde_0:
            p_risky = 1.0
        else:
            p_risky = 0.5

        exp_reward = (1 - p_risky) * reward_a0 + p_risky * reward_a1
        exp_cost = (1 - p_risky) * cost_a0 + p_risky * cost_a1

        expected_rewards.append(exp_reward)
        expected_costs.append(exp_cost)
        policy_probs.append(p_risky)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Constrained MDPs: Lagrangian Relaxation', fontsize=14, fontweight='bold')

    # Plot 1: Reward-cost Pareto front
    ax = axes[0]
    ax.plot(expected_costs, expected_rewards, 'bo-', markersize=3, linewidth=2)
    ax.axvline(cost_threshold, color='red', linestyle='--', linewidth=2, label=f'Cost limit = {cost_threshold}')
    ax.set_xlabel('Expected Cost')
    ax.set_ylabel('Expected Reward')
    ax.set_title('Reward-Cost Tradeoff (Pareto Front)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight feasible region
    feasible_mask = np.array(expected_costs) <= cost_threshold
    ax.fill_betweenx([0, 4], 0, cost_threshold, alpha=0.1, color='green')
    ax.text(0.15, 3.5, 'Feasible', fontsize=11, color='green', fontweight='bold')

    # Plot 2: Policy as function of lambda
    ax = axes[1]
    ax.plot(lambdas, policy_probs, 'g-', linewidth=2)
    ax.set_xlabel('Lagrange Multiplier λ')
    ax.set_ylabel('P(risky action)')
    ax.set_title('Policy vs λ: Higher λ → Safer')
    ax.grid(True, alpha=0.3)

    # Find optimal lambda
    optimal_idx = None
    for i, (c, r) in enumerate(zip(expected_costs, expected_rewards)):
        if c <= cost_threshold:
            if optimal_idx is None or expected_rewards[i] > expected_rewards[optimal_idx]:
                optimal_idx = i
    if optimal_idx is not None:
        ax.axvline(lambdas[optimal_idx], color='red', linestyle='--',
                  label=f'Optimal λ≈{lambdas[optimal_idx]:.1f}')
        ax.legend()

    # Plot 3: Dual objective
    ax = axes[2]
    dual_values = []
    for lam in lambdas:
        r_tilde_0 = reward_a0 - lam * cost_a0
        r_tilde_1 = reward_a1 - lam * cost_a1
        max_r_tilde = max(r_tilde_0, r_tilde_1)
        dual = max_r_tilde + lam * cost_threshold
        dual_values.append(dual)

    ax.plot(lambdas, dual_values, 'purple', linewidth=2)
    ax.set_xlabel('Lagrange Multiplier λ')
    ax.set_ylabel('Dual Objective L(λ)')
    ax.set_title('Dual Function: min_λ L(λ)')
    ax.grid(True, alpha=0.3)

    if optimal_idx is not None:
        ax.axvline(lambdas[optimal_idx], color='red', linestyle='--')
        ax.plot(lambdas[optimal_idx], dual_values[optimal_idx], 'r*', markersize=15,
               label='Optimal')
        ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/safe_rl/constrained_mdp.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: CVaR - Risk-Sensitive Optimization")
    print("=" * 60)
    demo_cvar()

    print("\n" + "=" * 60)
    print("Demo 2: Robust MDPs - Worst-Case Performance")
    print("=" * 60)
    demo_robust_mdp()

    print("\n" + "=" * 60)
    print("Demo 3: SPIBB - Safe Policy Improvement")
    print("=" * 60)
    demo_spibb()

    print("\n" + "=" * 60)
    print("Demo 4: Constrained MDPs - Lagrangian Relaxation")
    print("=" * 60)
    demo_constrained_mdp()
