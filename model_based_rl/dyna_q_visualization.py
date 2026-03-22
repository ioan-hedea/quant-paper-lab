"""
Section 5: Model-Based Reinforcement Learning
===============================================
Visualizes core model-based RL concepts from the SDM notes:
- Dyna-Q: interleaving real experience with model-based planning
- Model learning with Hoeffding confidence bounds
- State abstraction via bisimulation metrics
- Model-free vs model-based sample efficiency comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# Gridworld Environment
# ============================================================

class GridWorld:
    """Simple deterministic gridworld for Dyna-Q demonstration."""
    def __init__(self, n=6, walls=None, goal=(5, 5), start=(0, 0)):
        self.n = n
        self.goal = goal
        self.start = start
        self.walls = walls or [(1, 2), (2, 2), (3, 2), (4, 2),  # vertical wall
                                (1, 4), (2, 4), (3, 4)]           # another wall
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
        self.action_names = ['↑', '→', '↓', '←']

    def step(self, state, action):
        if state == self.goal:
            return state, 0.0, True

        dr, dc = self.actions[action]
        nr, nc = state[0] + dr, state[1] + dc
        if (nr, nc) in self.walls or nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
            nr, nc = state
        new_state = (nr, nc)
        reward = 1.0 if new_state == self.goal else 0.0
        done = new_state == self.goal
        return new_state, reward, done

    def reset(self):
        return self.start


# ============================================================
# Dyna-Q Agent
# ============================================================

class DynaQAgent:
    """Dyna-Q: model-free Q-learning + model-based planning."""

    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning  # number of planning steps per real step

        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.model = {}  # (s, a) -> (r, s')
        self.visited_sa = []

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        # Direct RL update (Q-learning)
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

        # Update model
        self.model[(s, a)] = (r, s_next, done)
        if (s, a) not in self.visited_sa:
            self.visited_sa.append((s, a))

        # Planning: simulate from model
        for _ in range(self.n_planning):
            idx = np.random.randint(len(self.visited_sa))
            s_sim, a_sim = self.visited_sa[idx]
            r_sim, s_next_sim, done_sim = self.model[(s_sim, a_sim)]
            target_sim = r_sim if done_sim else r_sim + self.gamma * np.max(self.Q[s_next_sim])
            self.Q[s_sim][a_sim] += self.alpha * (target_sim - self.Q[s_sim][a_sim])


# ============================================================
# Training Loop
# ============================================================

def train_agent(env, agent, n_episodes=100, max_steps=200):
    """Train an agent and return episode lengths."""
    episode_steps = []
    for ep in range(n_episodes):
        s = env.reset()
        for step in range(max_steps):
            a = agent.select_action(s)
            s_next, r, done = env.step(s, a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            if done:
                break
        episode_steps.append(step + 1)
    return episode_steps


# ============================================================
# Visualizations
# ============================================================

def demo_dyna_q():
    """Compare Q-learning (n=0) vs Dyna-Q with different planning steps."""
    env = GridWorld(n=6)
    planning_steps = [0, 5, 25, 50]
    n_trials = 20
    n_episodes = 80

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dyna-Q: Model-Based Planning Accelerates Learning', fontsize=14, fontweight='bold')

    all_results = {}
    for n_plan in planning_steps:
        results = []
        for trial in range(n_trials):
            np.random.seed(trial)
            agent = DynaQAgent(n_planning=n_plan)
            steps = train_agent(env, agent, n_episodes)
            results.append(steps)
        all_results[n_plan] = np.array(results)

    # Plot 1: Learning curves
    ax = axes[0]
    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
    for (n_plan, results), color in zip(all_results.items(), colors):
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        label = f'n_planning={n_plan}' if n_plan > 0 else 'Q-learning (n=0)'
        ax.plot(range(1, n_episodes+1), mean, color=color, linewidth=2, label=label)
        ax.fill_between(range(1, n_episodes+1), mean-std, mean+std, color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Goal')
    ax.set_title('Learning Speed: More Planning = Faster Learning')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 210)

    # Plot 2: Steps to reach near-optimal performance
    ax = axes[1]
    threshold = 20  # steps within 20 of optimal
    episodes_to_threshold = []
    for n_plan in planning_steps:
        results = all_results[n_plan]
        mean = results.mean(axis=0)
        # Find first episode where mean < threshold
        idx = np.where(mean < threshold)[0]
        if len(idx) > 0:
            episodes_to_threshold.append(idx[0] + 1)
        else:
            episodes_to_threshold.append(n_episodes)

    ax.bar([str(n) for n in planning_steps], episodes_to_threshold,
           color=colors, edgecolor='black')
    ax.set_xlabel('Planning Steps per Real Step')
    ax.set_ylabel('Episodes to Near-Optimal')
    ax.set_title(f'Sample Efficiency (threshold: {threshold} steps)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/model_based_rl/dyna_q_comparison.png', dpi=150)
    plt.show()


def demo_model_learning():
    """Visualize model learning accuracy with Hoeffding bounds."""
    np.random.seed(42)
    # Simulate learning transition probabilities
    true_probs = np.array([0.7, 0.2, 0.1])  # True P(s'|s,a) for 3 next states
    n_states_next = len(true_probs)

    n_samples_range = range(1, 201)
    estimated_probs_history = []
    hoeffding_bounds = []
    delta = 0.05  # confidence 1-delta = 95%

    counts = np.zeros(n_states_next)
    for n in n_samples_range:
        # Sample one transition
        s_next = np.random.choice(n_states_next, p=true_probs)
        counts[s_next] += 1
        estimated = counts / n
        estimated_probs_history.append(estimated.copy())
        hoeffding_bounds.append(np.sqrt(np.log(2 * n_states_next / delta) / (2 * n)))

    estimated_probs_history = np.array(estimated_probs_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Learning: Transition Probability Estimation', fontsize=14, fontweight='bold')

    # Plot 1: Convergence of estimated probabilities
    colors = ['#2196F3', '#4CAF50', '#F44336']
    labels = ["P(s'=0|s,a)", "P(s'=1|s,a)", "P(s'=2|s,a)"]
    for i in range(n_states_next):
        ax1.plot(n_samples_range, estimated_probs_history[:, i], color=colors[i],
                linewidth=1.5, label=f'Est {labels[i]}')
        ax1.axhline(y=true_probs[i], color=colors[i], linestyle='--', alpha=0.5)

    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Estimated Probability')
    ax1.set_title('Adaptive DP: Model Converges to True Probabilities')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hoeffding error bounds
    errors = np.max(np.abs(estimated_probs_history - true_probs), axis=1)
    ax2.semilogy(n_samples_range, errors, 'b-', linewidth=1.5, label='Max |P̂ - P|')
    ax2.semilogy(n_samples_range, hoeffding_bounds, 'r--', linewidth=2,
                label=f'Hoeffding bound (δ={delta})')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_title('Error Bound: O(1/√n) Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/model_based_rl/model_learning.png', dpi=150)
    plt.show()


def demo_policy_from_learned_model():
    """Show how Dyna-Q builds an internal model and uses it for planning."""
    env = GridWorld(n=6)
    np.random.seed(42)
    agent = DynaQAgent(n_planning=50)
    train_agent(env, agent, n_episodes=50)

    # Visualize the learned Q-values as a policy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Dyna-Q: Learned Policy and Model Coverage', fontsize=14, fontweight='bold')

    n = env.n
    # Policy arrows
    arrow_map = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}

    V = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            s = (r, c)
            if s in agent.Q:
                V[r, c] = np.max(agent.Q[s])
                best_a = np.argmax(agent.Q[s])
                if s != env.goal and s not in env.walls:
                    dx, dy = arrow_map[best_a]
                    ax1.annotate('', xy=(c+dx, r+dy), xytext=(c, r),
                               arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))

    im = ax1.imshow(V, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Mark goal and walls
    gr, gc = env.goal
    ax1.add_patch(plt.Rectangle((gc-0.5, gr-0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3))
    for wr, wc in env.walls:
        ax1.add_patch(plt.Rectangle((wc-0.5, wr-0.5), 1, 1, fill=True,
                                     facecolor='gray', edgecolor='black'))

    ax1.set_title('Learned Value Function & Policy')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.grid(True, alpha=0.3)

    # Model coverage: how many (s,a) pairs have been visited
    coverage = np.zeros((n, n))
    for (s, a) in agent.visited_sa:
        r, c = s
        coverage[r, c] += 1

    im2 = ax2.imshow(coverage, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    for wr, wc in env.walls:
        ax2.add_patch(plt.Rectangle((wc-0.5, wr-0.5), 1, 1, fill=True,
                                     facecolor='gray', edgecolor='black'))
    ax2.set_title('Model Coverage (visits per state)')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.grid(True, alpha=0.3)

    # Annotate
    for r in range(n):
        for c in range(n):
            if coverage[r, c] > 0:
                ax2.text(c, r, f'{int(coverage[r, c])}', ha='center', va='center',
                        fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/model_based_rl/dyna_q_policy.png', dpi=150)
    plt.show()

    print(f"Model entries: {len(agent.model)}")
    print(f"Unique (s,a) visited: {len(agent.visited_sa)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Dyna-Q - Planning Steps Comparison")
    print("=" * 60)
    demo_dyna_q()

    print("\n" + "=" * 60)
    print("Demo 2: Model Learning with Hoeffding Bounds")
    print("=" * 60)
    demo_model_learning()

    print("\n" + "=" * 60)
    print("Demo 3: Dyna-Q Learned Policy & Model Coverage")
    print("=" * 60)
    demo_policy_from_learned_model()
