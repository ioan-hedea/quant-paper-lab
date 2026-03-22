"""
Section 3: Monte Carlo Tree Search (MCTS) with UCT
====================================================
Visualizes the core MCTS concepts from the SDM notes:
- Four phases: Selection, Expansion, Simulation, Backpropagation
- UCT action selection with exploration-exploitation tradeoff
- Tree growth visualization over iterations
- Comparison of rollout policies (random vs heuristic)
- Effect of exploration constant c on behavior

Complements the existing POMCP implementation in pomcp/.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# ============================================================
# Simple Game Environment: Grid Navigation MDP
# ============================================================

class GridNavigationMDP:
    """
    Simple grid navigation for MCTS demonstration.
    Agent starts at (0,0), goal at (n-1, n-1).
    Stochastic transitions. Reward +10 at goal, -0.1 per step.
    """
    def __init__(self, n=5, slip_prob=0.1):
        self.n = n
        self.slip_prob = slip_prob
        self.goal = (n-1, n-1)
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
        self.action_names = ['→', '↓', '←', '↑']

    def step(self, state, action_idx):
        if state == self.goal:
            return state, 0.0, True

        if np.random.random() < self.slip_prob:
            action_idx = np.random.randint(4)

        dr, dc = self.actions[action_idx]
        r, c = state
        nr, nc = max(0, min(self.n-1, r+dr)), max(0, min(self.n-1, c+dc))
        new_state = (nr, nc)
        reward = 10.0 if new_state == self.goal else -0.1
        done = new_state == self.goal
        return new_state, reward, done

    def get_actions(self, state):
        return list(range(4))


# ============================================================
# MCTS with UCT
# ============================================================

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # action that led here
        self.children = {}     # action -> MCTSNode
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None

    @property
    def q_value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self, actions):
        if self.untried_actions is None:
            self.untried_actions = list(actions)
        return len(self.untried_actions) == 0


class MCTS:
    def __init__(self, env, c=1.41, gamma=0.95, max_rollout_depth=20):
        self.env = env
        self.c = c
        self.gamma = gamma
        self.max_rollout_depth = max_rollout_depth
        self.tree_sizes = []

    def uct_select(self, node):
        """UCT selection: balance exploitation and exploration."""
        best_score = -np.inf
        best_child = None
        for action, child in node.children.items():
            exploit = child.q_value
            explore = self.c * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-8))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node):
        """Expand: add one untried action."""
        actions = self.env.get_actions(node.state)
        if node.untried_actions is None:
            node.untried_actions = list(actions)
        action = node.untried_actions.pop()
        next_state, reward, done = self.env.step(node.state, action)
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        return child, reward, done

    def rollout(self, state, rollout_policy='random'):
        """Simulate from state to estimate value."""
        total_return = 0.0
        discount = 1.0
        for _ in range(self.max_rollout_depth):
            if state == self.env.goal:
                break
            if rollout_policy == 'random':
                action = np.random.randint(4)
            elif rollout_policy == 'heuristic':
                # Greedy: move toward goal
                r, c = state
                gr, gc = self.env.goal
                if abs(gr - r) > abs(gc - c):
                    action = 1 if gr > r else 3  # down or up
                else:
                    action = 0 if gc > c else 2  # right or left
            else:
                action = np.random.randint(4)

            state, reward, done = self.env.step(state, action)
            total_return += discount * reward
            discount *= self.gamma
            if done:
                break
        return total_return

    def backpropagate(self, node, value):
        """Backpropagate return up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def search(self, root_state, n_simulations, rollout_policy='random'):
        """Run MCTS from root_state for n_simulations."""
        root = MCTSNode(root_state)
        actions = self.env.get_actions(root_state)

        sim_values = []
        for sim in range(n_simulations):
            node = root
            cumulative_reward = 0.0
            discount = 1.0

            # Selection: traverse tree using UCT
            while node.is_fully_expanded(actions) and node.children:
                node = self.uct_select(node)
                # Simulate transition (for tree traversal)
                _, reward, done = self.env.step(node.parent.state if node.parent else root_state,
                                                 node.action)
                cumulative_reward += discount * reward
                discount *= self.gamma
                if done:
                    break

            # Expansion
            if not node.is_fully_expanded(actions) and node.state != self.env.goal:
                child, reward, done = self.expand(node)
                cumulative_reward += discount * reward
                discount *= self.gamma
                node = child

            # Simulation (rollout)
            if node.state != self.env.goal:
                rollout_value = self.rollout(node.state, rollout_policy)
                cumulative_reward += discount * rollout_value

            # Backpropagation
            self.backpropagate(node, cumulative_reward)
            sim_values.append(cumulative_reward)

            # Track tree size
            self.tree_sizes.append(self._count_nodes(root))

        return root, sim_values

    def _count_nodes(self, node):
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def get_best_action(self, root):
        """Return action with highest visit count (robust child)."""
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action


# ============================================================
# Visualizations
# ============================================================

def demo_mcts_phases():
    """Visualize MCTS tree growth over iterations."""
    env = GridNavigationMDP(n=5, slip_prob=0.1)
    np.random.seed(42)

    mcts = MCTS(env, c=1.41, gamma=0.95)
    root, sim_values = mcts.search((0, 0), n_simulations=500)

    # Plot 1: Tree size growth
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MCTS: Tree Growth and Value Estimation', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(range(1, len(mcts.tree_sizes)+1), mcts.tree_sizes, 'b-', linewidth=1.5)
    ax.set_xlabel('Simulation')
    ax.set_ylabel('Tree Size (nodes)')
    ax.set_title('Tree Growth Over Simulations')
    ax.grid(True, alpha=0.3)

    # Plot 2: Running average of simulation returns
    ax = axes[1]
    running_avg = np.cumsum(sim_values) / np.arange(1, len(sim_values)+1)
    ax.plot(range(1, len(sim_values)+1), sim_values, 'gray', alpha=0.3, label='Individual')
    ax.plot(range(1, len(running_avg)+1), running_avg, 'r-', linewidth=2, label='Running avg')
    ax.set_xlabel('Simulation')
    ax.set_ylabel('Return')
    ax.set_title('Simulation Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Visit counts for root actions
    ax = axes[2]
    action_visits = {}
    action_values = {}
    for action, child in root.children.items():
        action_visits[env.action_names[action]] = child.visits
        action_values[env.action_names[action]] = child.q_value

    bars = ax.bar(action_visits.keys(), action_visits.values(), color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
    ax.set_ylabel('Visit Count')
    ax.set_title('Root Action Statistics')
    ax2 = ax.twinx()
    ax2.plot(list(action_values.keys()), list(action_values.values()), 'ko-', markersize=8)
    ax2.set_ylabel('Q-value')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mcts/mcts_phases.png', dpi=150)
    plt.show()

    best_action = mcts.get_best_action(root)
    print(f"Best action from (0,0): {env.action_names[best_action]}")
    print(f"Visit counts: {action_visits}")
    print(f"Q-values: {action_values}")


def demo_exploration_constant():
    """Show effect of UCT exploration constant c on behavior."""
    env = GridNavigationMDP(n=5, slip_prob=0.1)
    c_values = [0.1, 0.5, 1.0, 1.41, 2.0, 5.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Effect of Exploration Constant c on UCT Behavior', fontsize=14, fontweight='bold')

    for ax, c in zip(axes.flat, c_values):
        np.random.seed(42)
        mcts = MCTS(env, c=c, gamma=0.95)
        root, sim_values = mcts.search((0, 0), n_simulations=300)

        # Show visit distribution across root actions
        visits = []
        names = []
        for a in range(4):
            if a in root.children:
                visits.append(root.children[a].visits)
            else:
                visits.append(0)
            names.append(env.action_names[a])

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        ax.bar(names, visits, color=colors)
        ax.set_title(f'c = {c}')
        ax.set_ylabel('Visit Count')

        # Compute entropy of visit distribution
        v = np.array(visits, dtype=float)
        v = v / (v.sum() + 1e-10)
        entropy = -np.sum(v * np.log(v + 1e-10))
        ax.text(0.95, 0.95, f'H={entropy:.2f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=10, style='italic')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mcts/exploration_constant.png', dpi=150)
    plt.show()


def demo_rollout_comparison():
    """Compare random vs heuristic rollout policies."""
    env = GridNavigationMDP(n=6, slip_prob=0.1)
    n_sims_range = [10, 25, 50, 100, 200, 500]
    n_trials = 20

    results = {'random': [], 'heuristic': []}

    for policy in ['random', 'heuristic']:
        for n_sims in n_sims_range:
            values = []
            for trial in range(n_trials):
                np.random.seed(trial)
                mcts = MCTS(env, c=1.41, gamma=0.95)
                root, sim_values = mcts.search((0, 0), n_sims, rollout_policy=policy)
                best_a = mcts.get_best_action(root)
                if best_a is not None:
                    values.append(root.children[best_a].q_value)
                else:
                    values.append(0)
            results[policy].append((np.mean(values), np.std(values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Rollout Policy Comparison: Random vs Heuristic', fontsize=14, fontweight='bold')

    for policy, color, marker in [('random', 'blue', 'o'), ('heuristic', 'green', 's')]:
        means = [r[0] for r in results[policy]]
        stds = [r[1] for r in results[policy]]
        ax1.errorbar(n_sims_range, means, yerr=stds, color=color, marker=marker,
                    capsize=4, linewidth=2, label=policy.capitalize())

    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Best Action Q-value')
    ax1.set_title('Q-value Estimate vs Simulation Budget')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Convergence speed comparison
    np.random.seed(42)
    mcts_random = MCTS(env, c=1.41)
    _, vals_random = mcts_random.search((0, 0), 500, rollout_policy='random')

    np.random.seed(42)
    mcts_heuristic = MCTS(env, c=1.41)
    _, vals_heuristic = mcts_heuristic.search((0, 0), 500, rollout_policy='heuristic')

    window = 20
    avg_r = np.convolve(vals_random, np.ones(window)/window, mode='valid')
    avg_h = np.convolve(vals_heuristic, np.ones(window)/window, mode='valid')

    ax2.plot(range(len(avg_r)), avg_r, 'b-', linewidth=2, label='Random rollout')
    ax2.plot(range(len(avg_h)), avg_h, 'g-', linewidth=2, label='Heuristic rollout')
    ax2.set_xlabel('Simulation')
    ax2.set_ylabel(f'Return (moving avg, window={window})')
    ax2.set_title('Convergence Speed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mcts/rollout_comparison.png', dpi=150)
    plt.show()


def demo_sparse_sampling():
    """Demonstrate that MCTS performance is independent of state space size."""
    slip = 0.1
    n_sims = 200
    grid_sizes = [4, 6, 8, 10, 15, 20]
    n_trials = 15

    mean_vals = []
    std_vals = []

    for n in grid_sizes:
        env = GridNavigationMDP(n=n, slip_prob=slip)
        values = []
        for trial in range(n_trials):
            np.random.seed(trial)
            mcts = MCTS(env, c=1.41, gamma=0.95, max_rollout_depth=30)
            root, _ = mcts.search((0, 0), n_sims)
            best_a = mcts.get_best_action(root)
            if best_a is not None and best_a in root.children:
                values.append(root.children[best_a].q_value)
            else:
                values.append(0)
        mean_vals.append(np.mean(values))
        std_vals.append(np.std(values))

    state_space_sizes = [n**2 for n in grid_sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(state_space_sizes, mean_vals, yerr=std_vals, 
                capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel('State Space Size |S|')
    ax.set_ylabel('Best Action Q-value (200 sims)')
    ax.set_title('MCTS Quality vs State Space Size\n(Sparse Sampling: performance independent of |S|)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/mcts/sparse_sampling.png', dpi=150)
    plt.show()

    print("State space sizes:", state_space_sizes)
    print("Mean Q-values:", [f"{v:.2f}" for v in mean_vals])


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: MCTS Four Phases & Tree Growth")
    print("=" * 60)
    demo_mcts_phases()

    print("\n" + "=" * 60)
    print("Demo 2: Effect of Exploration Constant c")
    print("=" * 60)
    demo_exploration_constant()

    print("\n" + "=" * 60)
    print("Demo 3: Random vs Heuristic Rollout")
    print("=" * 60)
    demo_rollout_comparison()

    print("\n" + "=" * 60)
    print("Demo 4: Sparse Sampling (State Space Independence)")
    print("=" * 60)
    demo_sparse_sampling()
