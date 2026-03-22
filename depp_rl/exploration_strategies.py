"""
Exploration Strategies Comparison (Augments existing depp_rl/)
==============================================================
Visualizes and compares exploration strategies from Section 1:
- Epsilon-greedy with different epsilon schedules
- Softmax (Boltzmann) exploration with temperature
- UCB (Upper Confidence Bound)
- Optimistic initialization

Uses a simple gridworld to show how different exploration
strategies affect learning speed and state coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# Multi-Armed Bandit for Clean Comparison
# ============================================================

class GaussianBandit:
    """K-armed bandit with Gaussian rewards for exploration comparison."""
    def __init__(self, means, std=1.0):
        self.means = np.array(means)
        self.std = std
        self.k = len(means)
        self.best_arm = np.argmax(means)

    def pull(self, arm):
        return np.random.normal(self.means[arm], self.std)


# ============================================================
# Exploration Strategies
# ============================================================

class EpsilonGreedy:
    def __init__(self, k, epsilon=0.1, decay=1.0):
        self.k = k
        self.epsilon = epsilon
        self.decay = decay
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.name = f'ε-greedy (ε={epsilon}, decay={decay})'

    def select(self, t):
        eps = self.epsilon * (self.decay ** t)
        if np.random.random() < eps:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class Softmax:
    def __init__(self, k, tau=1.0, tau_decay=1.0):
        self.k = k
        self.tau = tau
        self.tau_decay = tau_decay
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.name = f'Softmax (τ={tau})'

    def select(self, t):
        tau = max(self.tau * (self.tau_decay ** t), 0.01)
        exp_Q = np.exp(self.Q / tau)
        probs = exp_Q / exp_Q.sum()
        return np.random.choice(self.k, p=probs)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class UCB:
    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.total = 0
        self.name = f'UCB (c={c})'

    def select(self, t):
        self.total += 1
        for i in range(self.k):
            if self.N[i] == 0:
                return i
        ucb = self.Q + self.c * np.sqrt(np.log(self.total) / self.N)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class OptimisticInit:
    def __init__(self, k, init_value=10.0):
        self.k = k
        self.Q = np.full(k, init_value)
        self.N = np.zeros(k)
        self.name = f'Optimistic (Q₀={init_value})'

    def select(self, t):
        return np.argmax(self.Q)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


# ============================================================
# Visualizations
# ============================================================

def demo_exploration_comparison():
    """Compare all exploration strategies on a 10-armed bandit."""
    np.random.seed(42)
    means = np.random.normal(0, 1, 10)
    bandit = GaussianBandit(means, std=1.0)

    T = 1000
    n_trials = 100

    strategies = [
        lambda: EpsilonGreedy(10, epsilon=0.1),
        lambda: EpsilonGreedy(10, epsilon=0.01),
        lambda: EpsilonGreedy(10, epsilon=0.1, decay=0.995),
        lambda: Softmax(10, tau=1.0),
        lambda: Softmax(10, tau=0.1),
        lambda: UCB(10, c=2.0),
        lambda: UCB(10, c=0.5),
        lambda: OptimisticInit(10, init_value=5.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploration Strategies Comparison (10-Armed Gaussian Bandit)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    all_results = {}
    for s_fn, color in zip(strategies, colors):
        agent = s_fn()
        name = agent.name
        cum_regrets = np.zeros((n_trials, T))
        optimal_pcts = np.zeros((n_trials, T))
        arm_counts = np.zeros((n_trials, bandit.k))

        for trial in range(n_trials):
            np.random.seed(trial * 1000)
            agent = s_fn()
            cum_regret = 0
            for t in range(T):
                arm = agent.select(t)
                reward = bandit.pull(arm)
                agent.update(arm, reward)
                cum_regret += bandit.means[bandit.best_arm] - bandit.means[arm]
                cum_regrets[trial, t] = cum_regret
                optimal_pcts[trial, t] = float(arm == bandit.best_arm)
            arm_counts[trial] = agent.N

        all_results[name] = {
            'regret': cum_regrets,
            'optimal': optimal_pcts,
            'arm_counts': arm_counts,
            'color': color
        }

    # Plot 1: Cumulative regret
    ax = axes[0, 0]
    for name, data in all_results.items():
        mean = data['regret'].mean(axis=0)
        ax.plot(range(T), mean, color=data['color'], linewidth=1.5, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: % optimal action (smoothed)
    ax = axes[0, 1]
    window = 50
    for name, data in all_results.items():
        mean = data['optimal'].mean(axis=0)
        smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, color=data['color'], linewidth=1.5, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Optimal Action Rate (smoothed)')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 3: Final regret comparison (bar chart)
    ax = axes[1, 0]
    final_regrets = []
    names = []
    bar_colors = []
    for name, data in all_results.items():
        final_regrets.append(data['regret'][:, -1].mean())
        names.append(name.split('(')[0].strip())
        bar_colors.append(data['color'])

    bars = ax.barh(range(len(names)), final_regrets, color=bar_colors, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f'Cumulative Regret at t={T}')
    ax.set_title('Final Regret Comparison')
    ax.grid(True, alpha=0.3)

    # Plot 4: Arm pull distribution for select strategies
    ax = axes[1, 1]
    select_strategies = ['UCB (c=2.0)', 'ε-greedy (ε=0.1, decay=1.0)', 'Softmax (τ=1.0)']
    x = np.arange(bandit.k)
    width = 0.25
    for i, name in enumerate(select_strategies):
        if name in all_results:
            counts = all_results[name]['arm_counts'].mean(axis=0)
            ax.bar(x + i * width, counts, width, label=name.split('(')[0].strip(),
                   alpha=0.7)

    # Highlight best arm
    ax.axvline(bandit.best_arm, color='gold', linewidth=2, linestyle='--', label='Best arm')
    ax.set_xlabel('Arm')
    ax.set_ylabel('Average Pull Count')
    ax.set_title('Arm Pull Distribution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/depp_rl/exploration_comparison.png', dpi=150)
    plt.show()


def demo_epsilon_schedule():
    """Show how different epsilon decay schedules affect exploration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Epsilon-Greedy: Exploration Schedules', fontsize=14, fontweight='bold')

    T = 1000
    t = np.arange(T)

    schedules = {
        'Constant ε=0.1': np.full(T, 0.1),
        'Constant ε=0.01': np.full(T, 0.01),
        'Linear decay': np.maximum(0.1 * (1 - t / T), 0.01),
        'Exponential decay': 0.1 * 0.995 ** t,
        '1/t decay': 0.1 / (1 + 0.01 * t),
    }

    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    # Plot 1: Epsilon over time
    for (name, eps), color in zip(schedules.items(), colors):
        ax1.plot(t, eps, color=color, linewidth=2, label=name)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('ε (exploration rate)')
    ax1.set_title('Exploration Rate Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.01, 0.15)

    # Plot 2: Cumulative exploration probability
    for (name, eps), color in zip(schedules.items(), colors):
        cum_explore = np.cumsum(eps)
        ax2.plot(t, cum_explore, color=color, linewidth=2, label=name)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative exploration probability')
    ax2.set_title('Total Exploration Budget')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/depp_rl/epsilon_schedules.png', dpi=150)
    plt.show()


def demo_ucb_confidence():
    """Visualize UCB confidence bounds shrinking with more samples."""
    np.random.seed(42)
    bandit = GaussianBandit([0.5, 0.8, 0.3, 0.6, 0.9], std=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('UCB: Confidence Bounds Drive Exploration', fontsize=14, fontweight='bold')

    snapshots = [10, 50, 200]
    agent = UCB(bandit.k, c=2.0)

    for ax, snap_t in zip(axes, snapshots):
        while agent.total < snap_t:
            arm = agent.select(agent.total)
            reward = bandit.pull(arm)
            agent.update(arm, reward)

        # Plot Q-values with confidence intervals
        x = np.arange(bandit.k)
        ucb_bonus = agent.c * np.sqrt(np.log(agent.total + 1) / (agent.N + 1e-8))
        ucb_values = agent.Q + ucb_bonus

        ax.bar(x, agent.Q, color='steelblue', alpha=0.7, label='Q̂(a)')
        ax.errorbar(x, agent.Q, yerr=ucb_bonus, fmt='none', ecolor='red',
                   capsize=5, linewidth=2, label='UCB bonus')
        ax.scatter(x, bandit.means, color='gold', marker='*', s=100, zorder=5,
                  label='True μ(a)')

        ax.set_xlabel('Arm')
        ax.set_ylabel('Value')
        ax.set_title(f't = {snap_t} (visits: {agent.N.astype(int)})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/depp_rl/ucb_confidence.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Exploration Strategies Comparison")
    print("=" * 60)
    demo_exploration_comparison()

    print("\n" + "=" * 60)
    print("Demo 2: Epsilon Decay Schedules")
    print("=" * 60)
    demo_epsilon_schedule()

    print("\n" + "=" * 60)
    print("Demo 3: UCB Confidence Bounds")
    print("=" * 60)
    demo_ucb_confidence()
