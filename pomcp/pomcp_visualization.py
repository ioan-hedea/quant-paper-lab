"""
POMCP Visualization — Partially Observable Monte Carlo Planning
================================================================
Visualizes key POMCP concepts using the existing generic POMCP planner:
- Tree growth & Q-value convergence with increasing simulations
- Belief particle evolution (rejection sampling)
- UCB exploration constant effect on action selection
- Online planning on Tiger and Rocksample-like problems

Uses the POMCP implementation from pomcp.py.
"""

import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# Add parent so we can import pomcp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pomcp import POMDP, POMCP, TreeNode


# ============================================================
# Problem Definitions
# ============================================================

class TigerPOMDP(POMDP):
    """Classic Tiger problem for POMCP."""
    N_ACTIONS = 3  # 0=listen, 1=open-left, 2=open-right
    N_OBS = 2      # 0=hear-left, 1=hear-right

    def __init__(self, listen_cost=-1, correct_reward=10, wrong_penalty=-100,
                 listen_accuracy=0.85):
        self.listen_cost = listen_cost
        self.correct_reward = correct_reward
        self.wrong_penalty = wrong_penalty
        self.listen_accuracy = listen_accuracy

    def sample_initial_state(self):
        return random.choice([0, 1])  # 0=tiger-left, 1=tiger-right

    def step(self, state, action):
        if action == 0:  # listen
            if random.random() < self.listen_accuracy:
                obs = state  # correct observation
            else:
                obs = 1 - state
            return state, obs, self.listen_cost
        else:
            # open a door
            new_state = random.choice([0, 1])  # reset
            obs = random.choice([0, 1])
            if action == 1:  # open left
                reward = self.wrong_penalty if state == 0 else self.correct_reward
            else:  # open right
                reward = self.wrong_penalty if state == 1 else self.correct_reward
            return new_state, obs, reward

    ACTION_NAMES = {0: 'Listen', 1: 'Open Left', 2: 'Open Right'}


class ChainPOMDP(POMDP):
    """
    Chain navigation under partial observability.
    Agent on a 1D chain of N states, goal is rightmost state.
    Observations are noisy position readings (left-half vs right-half).
    """
    N_ACTIONS = 2  # 0=move-left, 1=move-right
    N_OBS = 2      # 0=seems-left, 1=seems-right

    def __init__(self, n_states=5, obs_accuracy=0.75):
        self.n_states = n_states
        self.obs_accuracy = obs_accuracy

    def sample_initial_state(self):
        return 0  # start at leftmost

    def step(self, state, action):
        if action == 0:
            next_s = max(0, state - 1)
        else:
            next_s = min(self.n_states - 1, state + 1)

        # Noisy observation: left-half or right-half
        true_side = 1 if next_s >= self.n_states // 2 else 0
        if random.random() < self.obs_accuracy:
            obs = true_side
        else:
            obs = 1 - true_side

        reward = 10.0 if next_s == self.n_states - 1 else -0.1
        return next_s, obs, reward


# ============================================================
# Demo 1: Tree Growth & Q-Value Convergence
# ============================================================

def demo_tree_growth():
    """Show how POMCP tree and Q-values evolve with more simulations."""
    random.seed(42)
    np.random.seed(42)

    tiger = TigerPOMDP()
    sim_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('POMCP: Tree Growth & Q-Value Convergence (Tiger Problem)',
                 fontsize=14, fontweight='bold')

    # --- Run POMCP with increasing simulations and track Q-values ---
    q_trajectories = {a: [] for a in range(tiger.N_ACTIONS)}
    visit_trajectories = {a: [] for a in range(tiger.N_ACTIONS)}
    tree_sizes = []
    tree_depths = []

    for n_sims in sim_counts:
        random.seed(42)
        planner = POMCP(tiger, n_sims=n_sims, max_depth=10, ucb_c=50.0,
                        gamma=0.95, n_particles=200)
        planner.plan()

        for a in range(tiger.N_ACTIONS):
            q_trajectories[a].append(planner.root.action_values[a])
            visit_trajectories[a].append(planner.root.action_counts[a])

        # Count tree size and depth
        size, depth = _tree_stats(planner.root)
        tree_sizes.append(size)
        tree_depths.append(depth)

    # Plot 1: Q-values vs simulations
    ax = axes[0, 0]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    for a in range(tiger.N_ACTIONS):
        ax.plot(sim_counts, q_trajectories[a], 'o-', color=colors[a],
                linewidth=2, markersize=5, label=tiger.ACTION_NAMES[a])
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Q-value at Root')
    ax.set_title('Q-Value Convergence')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Visit counts per action
    ax = axes[0, 1]
    x = np.arange(len(sim_counts))
    width = 0.25
    for a in range(tiger.N_ACTIONS):
        ax.bar(x + a * width, visit_trajectories[a], width, color=colors[a],
               label=tiger.ACTION_NAMES[a], alpha=0.8)
    ax.set_xlabel('Simulation Budget')
    ax.set_ylabel('Visit Count N(h,a)')
    ax.set_title('Root Action Visits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(sim_counts, rotation=45, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Tree size growth
    ax = axes[1, 0]
    ax.plot(sim_counts, tree_sizes, 's-', color='#9C27B0', linewidth=2,
            markersize=6, label='Tree nodes')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Tree Size (nodes)')
    ax.set_title('Search Tree Growth')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Tree depth
    ax = axes[1, 1]
    ax.plot(sim_counts, tree_depths, 'D-', color='#FF9800', linewidth=2,
            markersize=6, label='Max depth')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Max Tree Depth')
    ax.set_title('Search Tree Depth')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_tree_growth.png'), dpi=150)
    plt.show()


def _tree_stats(node, depth=0):
    """Count nodes and max depth in a POMCP tree."""
    if not node.children:
        return 1, depth
    total = 1
    max_d = depth
    for child in node.children.values():
        s, d = _tree_stats(child, depth + 1)
        total += s
        max_d = max(max_d, d)
    return total, max_d


# ============================================================
# Demo 2: Belief Particle Evolution
# ============================================================

def demo_belief_particles():
    """Visualize how POMCP belief particles evolve with observations."""
    random.seed(42)
    np.random.seed(42)

    tiger = TigerPOMDP()
    planner = POMCP(tiger, n_sims=500, max_depth=10, ucb_c=50.0,
                    gamma=0.95, n_particles=1000)

    # True state: tiger is on the left (state=0)
    true_state = 0
    n_steps = 8
    beliefs = []
    actions_taken = []
    obs_received = []

    # Record initial belief
    belief_left = sum(1 for s in planner.belief_particles if s == 0) / len(planner.belief_particles)
    beliefs.append(belief_left)

    for step in range(n_steps):
        action = planner.plan()
        actions_taken.append(action)

        # Generate observation from true state
        _, obs, _ = tiger.step(true_state, action)
        obs_received.append(obs)

        # If opened a door, reset true state
        if action != 0:
            true_state = random.choice([0, 1])

        planner.update_belief(action, obs)
        belief_left = sum(1 for s in planner.belief_particles if s == 0) / len(planner.belief_particles)
        beliefs.append(belief_left)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('POMCP: Belief Particle Evolution (Tiger Problem)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Belief trajectory
    ax = axes[0]
    steps = range(len(beliefs))
    ax.plot(steps, beliefs, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Uniform belief')
    ax.fill_between(steps, 0.5, beliefs, alpha=0.2, color='#2196F3')
    ax.set_xlabel('Step')
    ax.set_ylabel('P(tiger=left)')
    ax.set_title('Belief Over Time')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate actions/observations
    for i, (a, o) in enumerate(zip(actions_taken, obs_received)):
        a_name = tiger.ACTION_NAMES[a][:1]
        o_name = 'L' if o == 0 else 'R'
        ax.annotate(f'{a_name},{o_name}', (i + 0.5, beliefs[i + 1]),
                    fontsize=6, ha='center', va='bottom')

    # Plot 2: Particle histograms at select steps
    ax = axes[1]
    random.seed(42)
    np.random.seed(42)
    planner2 = POMCP(tiger, n_sims=500, max_depth=10, ucb_c=50.0,
                     gamma=0.95, n_particles=1000)

    snapshot_steps = [0, 2, 4, 6]
    snapshot_beliefs = []
    true_state2 = 0
    step_idx = 0
    snapshot_beliefs.append(
        sum(1 for s in planner2.belief_particles if s == 0) / len(planner2.belief_particles)
    )

    for step in range(7):
        action = planner2.plan()
        _, obs, _ = tiger.step(true_state2, action)
        if action != 0:
            true_state2 = random.choice([0, 1])
        planner2.update_belief(action, obs)
        if step + 1 in snapshot_steps:
            bl = sum(1 for s in planner2.belief_particles if s == 0) / len(planner2.belief_particles)
            snapshot_beliefs.append(bl)

    all_beliefs_snap = [beliefs[s] for s in snapshot_steps]
    bar_width = 0.35
    x = np.arange(len(snapshot_steps))
    ax.bar(x - bar_width/2, all_beliefs_snap, bar_width, color='#F44336',
           label='P(tiger=left)', alpha=0.8)
    ax.bar(x + bar_width/2, [1 - b for b in all_beliefs_snap], bar_width,
           color='#4CAF50', label='P(tiger=right)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}' for s in snapshot_steps])
    ax.set_ylabel('Particle Proportion')
    ax.set_title('Belief Snapshots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Particle count over time (checking for depletion)
    ax = axes[2]
    random.seed(42)
    np.random.seed(42)
    planner3 = POMCP(tiger, n_sims=200, max_depth=10, ucb_c=50.0,
                     gamma=0.95, n_particles=500)

    particle_counts = [len(planner3.belief_particles)]
    unique_counts = [len(set(planner3.belief_particles))]
    true_state3 = 0

    for step in range(12):
        action = planner3.plan()
        _, obs, _ = tiger.step(true_state3, action)
        if action != 0:
            true_state3 = random.choice([0, 1])
        planner3.update_belief(action, obs)
        particle_counts.append(len(planner3.belief_particles))
        unique_counts.append(len(set(planner3.belief_particles)))

    ax.plot(range(len(particle_counts)), particle_counts, 'o-', color='#2196F3',
            linewidth=2, label='Total particles')
    ax.plot(range(len(unique_counts)), unique_counts, 's--', color='#FF9800',
            linewidth=2, label='Unique particles')
    ax.set_xlabel('Step')
    ax.set_ylabel('Particle Count')
    ax.set_title('Particle Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_belief_particles.png'), dpi=150)
    plt.show()


# ============================================================
# Demo 3: UCB Exploration Constant Effect
# ============================================================

def demo_ucb_effect():
    """Show how the UCB constant c affects POMCP action selection."""
    random.seed(42)
    np.random.seed(42)

    tiger = TigerPOMDP()
    c_values = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    n_sims = 1000
    n_trials = 20

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('POMCP: Effect of UCB Exploration Constant c',
                 fontsize=14, fontweight='bold')

    # Track Q-values, visit distributions, and chosen actions
    q_means = {a: [] for a in range(tiger.N_ACTIONS)}
    q_stds = {a: [] for a in range(tiger.N_ACTIONS)}
    visit_entropies = []
    chosen_actions = []

    for c in c_values:
        q_trials = {a: [] for a in range(tiger.N_ACTIONS)}
        entropy_trials = []
        action_counts = defaultdict(int)

        for trial in range(n_trials):
            random.seed(trial * 100)
            planner = POMCP(tiger, n_sims=n_sims, max_depth=10, ucb_c=c,
                            gamma=0.95, n_particles=300)
            best_a = planner.plan()
            action_counts[best_a] += 1

            for a in range(tiger.N_ACTIONS):
                q_trials[a].append(planner.root.action_values[a])

            # Visit entropy
            visits = np.array([planner.root.action_counts[a] for a in range(tiger.N_ACTIONS)])
            visits = visits / visits.sum()
            ent = -np.sum(v * np.log(v + 1e-10) for v in visits)
            entropy_trials.append(ent)

        for a in range(tiger.N_ACTIONS):
            q_means[a].append(np.mean(q_trials[a]))
            q_stds[a].append(np.std(q_trials[a]))
        visit_entropies.append(np.mean(entropy_trials))
        chosen_actions.append(dict(action_counts))

    # Plot 1: Q-values vs c
    ax = axes[0]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    for a in range(tiger.N_ACTIONS):
        ax.errorbar(c_values, q_means[a], yerr=q_stds[a], fmt='o-',
                    color=colors[a], linewidth=2, capsize=4,
                    label=tiger.ACTION_NAMES[a])
    ax.set_xlabel('UCB constant c')
    ax.set_ylabel('Q-value at Root')
    ax.set_title('Q-Values vs Exploration')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Visit entropy vs c
    ax = axes[1]
    ax.plot(c_values, visit_entropies, 'D-', color='#9C27B0', linewidth=2,
            markersize=8)
    ax.set_xlabel('UCB constant c')
    ax.set_ylabel('Visit Entropy (nats)')
    ax.set_title('Exploration Uniformity')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Chosen action distribution
    ax = axes[2]
    x = np.arange(len(c_values))
    bottom = np.zeros(len(c_values))
    for a in range(tiger.N_ACTIONS):
        vals = [ca.get(a, 0) / n_trials for ca in chosen_actions]
        ax.bar(x, vals, bottom=bottom, color=colors[a],
               label=tiger.ACTION_NAMES[a], alpha=0.8)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f'c={c}' for c in c_values], fontsize=7, rotation=45)
    ax.set_ylabel('Fraction of Trials')
    ax.set_title('Action Selection Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_ucb_effect.png'), dpi=150)
    plt.show()


# ============================================================
# Demo 4: Online Planning Episode
# ============================================================

def demo_online_planning():
    """Run a full online planning episode and visualize decisions."""
    random.seed(42)
    np.random.seed(42)

    tiger = TigerPOMDP()
    planner = POMCP(tiger, n_sims=1000, max_depth=15, ucb_c=50.0,
                    gamma=0.95, n_particles=500)

    n_episodes = 50
    ep_length = 15
    episode_rewards = []
    all_actions = []
    all_beliefs = []

    for ep in range(n_episodes):
        random.seed(ep * 1000)
        planner_ep = POMCP(tiger, n_sims=500, max_depth=10, ucb_c=50.0,
                           gamma=0.95, n_particles=500)
        true_state = random.choice([0, 1])
        total_reward = 0
        ep_actions = []
        ep_beliefs = []

        for t in range(ep_length):
            bl = sum(1 for s in planner_ep.belief_particles if s == 0) / len(planner_ep.belief_particles)
            ep_beliefs.append(bl)

            action = planner_ep.plan()
            ep_actions.append(action)

            s_next, obs, reward = tiger.step(true_state, action)
            total_reward += reward

            if action != 0:
                true_state = random.choice([0, 1])
            else:
                true_state = s_next

            planner_ep.update_belief(action, obs)

        episode_rewards.append(total_reward)
        all_actions.append(ep_actions)
        all_beliefs.append(ep_beliefs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('POMCP: Online Planning Episodes (Tiger Problem)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Episode reward distribution
    ax = axes[0, 0]
    ax.hist(episode_rewards, bins=25, color='#2196F3', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(episode_rewards), color='red', linewidth=2, linestyle='--',
               label=f'Mean={np.mean(episode_rewards):.1f}')
    ax.set_xlabel('Episode Total Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Action frequency over time
    ax = axes[0, 1]
    action_freq = np.zeros((ep_length, tiger.N_ACTIONS))
    for ep_acts in all_actions:
        for t, a in enumerate(ep_acts):
            action_freq[t, a] += 1
    action_freq /= n_episodes

    colors = ['#F44336', '#4CAF50', '#2196F3']
    for a in range(tiger.N_ACTIONS):
        ax.plot(range(ep_length), action_freq[:, a], 'o-', color=colors[a],
                linewidth=2, label=tiger.ACTION_NAMES[a])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Frequency')
    ax.set_title('Action Selection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Belief at decision points (when door is opened)
    ax = axes[1, 0]
    open_beliefs = []
    open_actions = []
    for ep_acts, ep_bels in zip(all_actions, all_beliefs):
        for t, (a, b) in enumerate(zip(ep_acts, ep_bels)):
            if a != 0:  # door opened
                open_beliefs.append(b)
                open_actions.append(a)

    if open_beliefs:
        correct = []
        incorrect = []
        for b, a in zip(open_beliefs, open_actions):
            # action=1 is open-left. Correct if tiger is right (b < 0.5)
            # action=2 is open-right. Correct if tiger is left (b > 0.5)
            if (a == 1 and b < 0.5) or (a == 2 and b > 0.5):
                correct.append(b)
            else:
                incorrect.append(b)

        if correct:
            ax.hist(correct, bins=20, color='#4CAF50', alpha=0.7,
                    label=f'Correct ({len(correct)})', edgecolor='black')
        if incorrect:
            ax.hist(incorrect, bins=20, color='#F44336', alpha=0.7,
                    label=f'Incorrect ({len(incorrect)})', edgecolor='black')
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('P(tiger=left) at Decision')
        ax.set_ylabel('Count')
        ax.set_title('Belief When Opening Door')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative reward trajectory for a few episodes
    ax = axes[1, 1]
    for ep in range(min(10, n_episodes)):
        random.seed(ep * 1000)
        planner_show = POMCP(tiger, n_sims=500, max_depth=10, ucb_c=50.0,
                             gamma=0.95, n_particles=500)
        true_state = random.choice([0, 1])
        cum_rewards = []
        cum = 0
        for t in range(ep_length):
            action = planner_show.plan()
            s_next, obs, reward = tiger.step(true_state, action)
            cum += reward
            cum_rewards.append(cum)
            if action != 0:
                true_state = random.choice([0, 1])
            else:
                true_state = s_next
            planner_show.update_belief(action, obs)
        ax.plot(range(ep_length), cum_rewards, alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Sample Episode Trajectories')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_online_planning.png'), dpi=150)
    plt.show()


# ============================================================
# Demo 5: POMCP vs Random on Chain Problem
# ============================================================

def demo_pomcp_vs_random():
    """Compare POMCP to random policy on a partially observable chain."""
    random.seed(42)
    np.random.seed(42)

    chain = ChainPOMDP(n_states=5, obs_accuracy=0.75)
    n_episodes = 30
    ep_length = 20

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('POMCP vs Random Policy (Chain Navigation POMDP)',
                 fontsize=14, fontweight='bold')

    # Run POMCP episodes
    pomcp_rewards = []
    pomcp_goal_times = []
    for ep in range(n_episodes):
        random.seed(ep * 100)
        planner = POMCP(chain, n_sims=200, max_depth=15, ucb_c=5.0,
                        gamma=0.95, n_particles=200)
        state = 0
        total = 0
        goal_time = ep_length  # default if never reached
        for t in range(ep_length):
            action = planner.plan()
            state, obs, reward = chain.step(state, action)
            total += reward
            planner.update_belief(action, obs)
            if state == chain.n_states - 1 and goal_time == ep_length:
                goal_time = t
        pomcp_rewards.append(total)
        pomcp_goal_times.append(goal_time)

    # Run Random episodes
    random_rewards = []
    random_goal_times = []
    for ep in range(n_episodes):
        random.seed(ep * 100)
        state = 0
        total = 0
        goal_time = ep_length
        for t in range(ep_length):
            action = random.randint(0, chain.N_ACTIONS - 1)
            state, obs, reward = chain.step(state, action)
            total += reward
            if state == chain.n_states - 1 and goal_time == ep_length:
                goal_time = t
        random_rewards.append(total)
        random_goal_times.append(goal_time)

    # Plot 1: Reward comparison
    ax = axes[0]
    bp = ax.boxplot([pomcp_rewards, random_rewards],
                    labels=['POMCP', 'Random'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#FF9800')
    ax.set_ylabel('Total Episode Reward')
    ax.set_title('Reward Comparison')
    ax.grid(True, alpha=0.3)

    # Plot 2: Time to goal
    ax = axes[1]
    bp = ax.boxplot([pomcp_goal_times, random_goal_times],
                    labels=['POMCP', 'Random'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#FF9800')
    ax.set_ylabel('First Time Reaching Goal')
    ax.set_title('Goal Achievement Speed')
    ax.grid(True, alpha=0.3)

    # Plot 3: Single episode visualization
    ax = axes[2]
    random.seed(7)
    planner = POMCP(chain, n_sims=200, max_depth=15, ucb_c=5.0,
                    gamma=0.95, n_particles=200)
    state = 0
    positions = [state]
    for t in range(ep_length):
        action = planner.plan()
        state, obs, reward = chain.step(state, action)
        positions.append(state)
        planner.update_belief(action, obs)

    ax.plot(range(len(positions)), positions, 'o-', color='#2196F3',
            linewidth=2, markersize=6, label='POMCP')

    random.seed(7)
    state = 0
    positions_rand = [state]
    for t in range(ep_length):
        action = random.randint(0, chain.N_ACTIONS - 1)
        state, obs, reward = chain.step(state, action)
        positions_rand.append(state)

    ax.plot(range(len(positions_rand)), positions_rand, 's--', color='#FF9800',
            linewidth=2, markersize=6, label='Random', alpha=0.7)

    ax.axhline(chain.n_states - 1, color='gold', linewidth=2, linestyle=':',
               label='Goal')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.set_title('Sample Trajectory')
    ax.set_yticks(range(chain.n_states))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_vs_random.png'), dpi=150)
    plt.show()


# ============================================================
# Demo 6: Simulation Budget vs Performance
# ============================================================

def demo_sim_budget():
    """Show how increasing simulation budget improves POMCP performance."""
    random.seed(42)
    np.random.seed(42)

    tiger = TigerPOMDP()
    budgets = [10, 25, 50, 100, 250, 500, 1000, 2000]
    n_episodes = 30
    ep_length = 10

    mean_rewards = []
    std_rewards = []

    for n_sims in budgets:
        rewards = []
        for ep in range(n_episodes):
            random.seed(ep * 200)
            planner = POMCP(tiger, n_sims=n_sims, max_depth=10, ucb_c=50.0,
                            gamma=0.95, n_particles=300)
            true_state = random.choice([0, 1])
            total = 0
            for t in range(ep_length):
                action = planner.plan()
                s_next, obs, reward = tiger.step(true_state, action)
                total += reward
                if action != 0:
                    true_state = random.choice([0, 1])
                else:
                    true_state = s_next
                planner.update_belief(action, obs)
            rewards.append(total)
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('POMCP: Simulation Budget vs Performance',
                 fontsize=14, fontweight='bold')

    # Plot 1: Mean reward vs budget
    ax1.errorbar(budgets, mean_rewards, yerr=std_rewards, fmt='o-',
                 color='#2196F3', linewidth=2, capsize=5, markersize=7)
    ax1.set_xlabel('Simulation Budget (per step)')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Performance vs Computation')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Marginal improvement
    marginal = [0] + [mean_rewards[i] - mean_rewards[i-1]
                      for i in range(1, len(mean_rewards))]
    ax2.bar(range(len(budgets)), marginal, color='#4CAF50', alpha=0.8,
            edgecolor='black')
    ax2.set_xticks(range(len(budgets)))
    ax2.set_xticklabels(budgets, rotation=45, fontsize=8)
    ax2.set_xlabel('Simulation Budget')
    ax2.set_ylabel('Marginal Reward Gain')
    ax2.set_title('Diminishing Returns')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'pomcp_sim_budget.png'), dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Tree Growth & Q-Value Convergence")
    print("=" * 60)
    demo_tree_growth()

    print("\n" + "=" * 60)
    print("Demo 2: Belief Particle Evolution")
    print("=" * 60)
    demo_belief_particles()

    print("\n" + "=" * 60)
    print("Demo 3: UCB Exploration Constant Effect")
    print("=" * 60)
    demo_ucb_effect()

    print("\n" + "=" * 60)
    print("Demo 4: Online Planning Episodes")
    print("=" * 60)
    demo_online_planning()

    print("\n" + "=" * 60)
    print("Demo 5: POMCP vs Random (Chain POMDP)")
    print("=" * 60)
    demo_pomcp_vs_random()

    print("\n" + "=" * 60)
    print("Demo 6: Simulation Budget vs Performance")
    print("=" * 60)
    demo_sim_budget()
