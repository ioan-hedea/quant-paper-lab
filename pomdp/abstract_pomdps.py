"""
Abstract POMDP Problems: Beyond the Tiger
==========================================
More general and realistic POMDP scenarios that illustrate
key concepts: belief tracking, information gathering, and
the value of observations in different domains.

Problems:
1. Medical Diagnosis POMDP - diagnose disease from noisy tests
2. Robot Navigation POMDP - navigate a grid with noisy sensors
3. Machine Maintenance POMDP - monitor/repair degrading machine
4. Search & Rescue POMDP - find target in unknown environment
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def belief_update(T, Obs, b, a, o, n_states):
    """Generic Bayesian belief update."""
    b_new = np.zeros(n_states)
    for sp in range(n_states):
        pred = sum(T[a, s, sp] * b[s] for s in range(n_states))
        b_new[sp] = Obs[a, sp, o] * pred
    total = b_new.sum()
    if total > 0:
        b_new /= total
    return b_new


# ============================================================
# 1. Medical Diagnosis POMDP
# ============================================================

class MedicalDiagnosisPOMDP:
    """
    A doctor must diagnose a patient's condition (hidden state)
    using noisy medical tests (observations) and decide on treatment.

    States: Healthy, MildDisease, SevereDisease
    Actions: WaitAndWatch, RunBloodTest, RunImaging, TreatMild, TreatAggressive
    Observations: Normal, Abnormal, HighlyAbnormal
    """
    state_names = ['Healthy', 'Mild Disease', 'Severe Disease']
    action_names = ['Wait', 'Blood Test', 'Imaging', 'Treat Mild', 'Treat Aggressive']
    obs_names = ['Normal', 'Abnormal', 'Highly Abnormal']

    n_states = 3
    n_actions = 5
    n_obs = 3
    gamma = 0.95

    def __init__(self):
        self._build_model()

    def _build_model(self):
        # Transition: T[a, s, s']
        self.T = np.zeros((self.n_actions, self.n_states, self.n_states))

        # Wait: disease can progress
        self.T[0] = np.array([
            [0.90, 0.08, 0.02],  # healthy -> mostly stays
            [0.05, 0.80, 0.15],  # mild -> can worsen
            [0.01, 0.09, 0.90],  # severe -> mostly stays
        ])
        # Blood test: no effect on disease
        self.T[1] = self.T[0].copy()
        # Imaging: no effect on disease
        self.T[2] = self.T[0].copy()
        # Treat mild: helps if mild, slight help if severe
        self.T[3] = np.array([
            [0.95, 0.04, 0.01],
            [0.40, 0.55, 0.05],  # good chance of recovery
            [0.05, 0.30, 0.65],  # some improvement
        ])
        # Treat aggressive: strong effect but risky for healthy
        self.T[4] = np.array([
            [0.70, 0.20, 0.10],  # side effects for healthy!
            [0.60, 0.35, 0.05],  # strong recovery
            [0.30, 0.50, 0.20],  # good improvement
        ])

        # Observation: Obs[a, s', o]
        self.Obs = np.zeros((self.n_actions, self.n_states, self.n_obs))

        # Wait: no diagnostic info
        self.Obs[0, :, :] = np.array([
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.3, 0.6],
        ])
        # Blood test: moderate accuracy
        self.Obs[1, :, :] = np.array([
            [0.80, 0.15, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.20, 0.75],
        ])
        # Imaging: high accuracy
        self.Obs[2, :, :] = np.array([
            [0.90, 0.08, 0.02],
            [0.05, 0.85, 0.10],
            [0.02, 0.08, 0.90],
        ])
        # Treatments: weak observation
        self.Obs[3] = self.Obs[0].copy()
        self.Obs[4] = self.Obs[0].copy()

        # Rewards: R[s, a]
        self.R = np.array([
            #  Wait,  Blood, Imaging, TreatM, TreatA
            [  0.0,  -1.0,   -3.0,   -5.0,  -15.0],  # Healthy (treatment is costly)
            [ -2.0,  -1.0,   -3.0,    5.0,    3.0],   # Mild (treat mild is good)
            [-10.0,  -1.0,   -3.0,    0.0,   10.0],   # Severe (aggressive needed)
        ])


def demo_medical_diagnosis():
    """Simulate a medical diagnosis scenario with belief tracking."""
    pomdp = MedicalDiagnosisPOMDP()
    np.random.seed(42)

    # True state: patient has mild disease (index 1)
    true_state = 1

    # Strategy: run tests first, then decide on treatment
    action_sequence = [0, 1, 1, 2, 2]  # wait, blood, blood, imaging, imaging
    b = np.array([0.5, 0.3, 0.2])  # prior: probably healthy

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Medical Diagnosis POMDP: Noisy Tests → Belief → Treatment Decision',
                 fontsize=14, fontweight='bold')

    beliefs_history = [b.copy()]
    obs_history = []
    action_history = []

    for a in action_sequence:
        # Sample observation from true state
        o = np.random.choice(pomdp.n_obs, p=pomdp.Obs[a, true_state, :])
        b = belief_update(pomdp.T, pomdp.Obs, b, a, o, pomdp.n_states)

        # Disease progression
        true_state = np.random.choice(pomdp.n_states, p=pomdp.T[a, true_state, :])

        beliefs_history.append(b.copy())
        obs_history.append(pomdp.obs_names[o])
        action_history.append(pomdp.action_names[a])

    beliefs_arr = np.array(beliefs_history)

    # Plot 1: Belief evolution
    ax = axes[0, 0]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    for i, (name, color) in enumerate(zip(pomdp.state_names, colors)):
        ax.plot(range(len(beliefs_arr)), beliefs_arr[:, i], 'o-',
                color=color, linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Belief Probability')
    ax.set_title('Belief Evolution: Tests Reveal True Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Annotate actions/observations
    for i, (act, obs) in enumerate(zip(action_history, obs_history)):
        ax.annotate(f'{act}\n→{obs}', xy=(i+1, 0.02), fontsize=6,
                   ha='center', rotation=0, color='gray')

    # Plot 2: Information value of different tests
    ax = axes[0, 1]
    test_actions = [0, 1, 2]  # wait, blood, imaging
    p_range = np.linspace(0.01, 0.99, 100)

    for a, color, name in zip(test_actions, ['gray', 'blue', 'green'],
                               ['Wait', 'Blood Test', 'Imaging']):
        info_gains = []
        for p in p_range:
            b_test = np.array([p, (1-p)*0.6, (1-p)*0.4])
            b_test /= b_test.sum()

            # Expected info gain
            eig = 0.0
            for o in range(pomdp.n_obs):
                b_new = belief_update(pomdp.T, pomdp.Obs, b_test, a, o, pomdp.n_states)
                p_o = sum(pomdp.Obs[a, sp, o] * sum(pomdp.T[a, s, sp] * b_test[s]
                          for s in range(pomdp.n_states))
                          for sp in range(pomdp.n_states))
                if p_o > 1e-10:
                    kl = sum(b_new[s] * np.log(b_new[s] / (b_test[s] + 1e-10) + 1e-10)
                             for s in range(pomdp.n_states) if b_new[s] > 1e-10)
                    eig += p_o * kl
            info_gains.append(eig)

        ax.plot(p_range, info_gains, color=color, linewidth=2, label=name)

    ax.set_xlabel('P(Healthy)')
    ax.set_ylabel('Expected Information Gain')
    ax.set_title('Value of Information: Which Test to Run?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Expected reward for each treatment given belief
    ax = axes[1, 0]
    p_healthy = np.linspace(0, 1, 100)
    for a in [0, 3, 4]:  # wait, treat mild, treat aggressive
        rewards = []
        for ph in p_healthy:
            # Assume rest split 60/40 between mild and severe
            b = np.array([ph, (1-ph)*0.6, (1-ph)*0.4])
            expected_r = b @ pomdp.R[:, a]
            rewards.append(expected_r)
        ax.plot(p_healthy, rewards, linewidth=2, label=pomdp.action_names[a])

    ax.set_xlabel('P(Healthy)')
    ax.set_ylabel('Expected Immediate Reward')
    ax.set_title('Treatment Value Depends on Belief')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Plot 4: Decision regions
    ax = axes[1, 1]
    # For each belief point on the simplex (parameterized by p_healthy, p_mild)
    n_grid = 50
    from matplotlib.colors import ListedColormap
    action_cmap = ListedColormap(['#E0E0E0', '#2196F3', '#4CAF50', '#FF9800', '#F44336'])

    img = np.zeros((n_grid, n_grid))
    for i, ph in enumerate(np.linspace(0, 1, n_grid)):
        for j, pm in enumerate(np.linspace(0, 1-ph, n_grid)):
            ps = max(0, 1 - ph - pm)
            b = np.array([ph, pm, ps])
            if b.sum() < 0.99:
                img[j, i] = -1
                continue
            expected_rewards = [b @ pomdp.R[:, a] for a in range(pomdp.n_actions)]
            img[j, i] = np.argmax(expected_rewards)

    img = np.ma.masked_where(img < 0, img)
    ax.imshow(img, origin='lower', cmap=action_cmap, aspect='auto',
             extent=[0, 1, 0, 1])
    ax.set_xlabel('P(Healthy)')
    ax.set_ylabel('P(Mild Disease)')
    ax.set_title('Optimal Immediate Action by Belief')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n)
                       for c, n in zip(['#E0E0E0', '#2196F3', '#4CAF50', '#FF9800', '#F44336'],
                                        pomdp.action_names)]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/medical_diagnosis.png', dpi=150)
    plt.show()


# ============================================================
# 2. Machine Maintenance POMDP
# ============================================================

class MachineMaintPOMDP:
    """
    A factory machine degrades over time. The operator can:
    - Run production (reward but accelerates wear)
    - Inspect (costly but informative)
    - Perform maintenance (costly but resets condition)

    States: Good, Degraded, Failed
    Actions: Produce, Inspect, Maintain
    Observations: Quiet, Noisy, Alarm
    """
    state_names = ['Good', 'Degraded', 'Failed']
    action_names = ['Produce', 'Inspect', 'Maintain']
    obs_names = ['Quiet', 'Noisy', 'Alarm']

    n_states = 3
    n_actions = 3
    n_obs = 3
    gamma = 0.95

    def __init__(self):
        self._build()

    def _build(self):
        self.T = np.zeros((3, 3, 3))
        # Produce: wear increases
        self.T[0] = np.array([
            [0.7, 0.25, 0.05],
            [0.0, 0.6,  0.4],
            [0.0, 0.0,  1.0],
        ])
        # Inspect: same degradation as produce but info
        self.T[1] = self.T[0].copy()
        # Maintain: restores machine
        self.T[2] = np.array([
            [0.95, 0.05, 0.0],
            [0.80, 0.20, 0.0],
            [0.50, 0.40, 0.10],
        ])

        self.Obs = np.zeros((3, 3, 3))
        # Produce: poor observation
        self.Obs[0] = np.array([
            [0.7, 0.25, 0.05],
            [0.2, 0.6,  0.2],
            [0.05, 0.15, 0.8],
        ])
        # Inspect: good observation
        self.Obs[1] = np.array([
            [0.90, 0.08, 0.02],
            [0.05, 0.85, 0.10],
            [0.01, 0.04, 0.95],
        ])
        # Maintain: poor observation
        self.Obs[2] = self.Obs[0].copy()

        self.R = np.array([
            # Produce, Inspect, Maintain
            [10.0,  -2.0,  -8.0],   # Good
            [ 5.0,  -2.0,  -8.0],   # Degraded
            [-20.0, -2.0,  -8.0],   # Failed (production on failed = disaster)
        ])


def demo_machine_maintenance():
    """Simulate predictive maintenance with belief-based decisions."""
    pomdp = MachineMaintPOMDP()
    np.random.seed(42)

    n_episodes = 200
    episode_length = 30

    # Strategy 1: Always produce (no maintenance)
    # Strategy 2: Periodic maintenance every 10 steps
    # Strategy 3: Belief-based (inspect when uncertain, maintain when P(failed) high)

    strategies = {
        'Always Produce': lambda b, t: 0,
        'Periodic Maint (every 10)': lambda b, t: 2 if t % 10 == 9 else 0,
        'Belief-Based': lambda b, t: (
            2 if b[2] > 0.3 else  # maintain if P(failed) > 0.3
            1 if b[1] + b[2] > 0.5 and t % 3 == 0 else  # inspect periodically
            0  # produce
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Machine Maintenance POMDP: Predictive Maintenance via Belief Tracking',
                 fontsize=14, fontweight='bold')

    all_returns = {}
    colors = ['#F44336', '#FF9800', '#4CAF50']

    for (name, policy), color in zip(strategies.items(), colors):
        episode_returns = []
        for ep in range(n_episodes):
            np.random.seed(ep * 100)
            true_state = 0  # start good
            b = np.array([0.8, 0.15, 0.05])
            total_reward = 0

            for t in range(episode_length):
                a = policy(b, t)
                reward = pomdp.R[true_state, a]
                total_reward += reward

                true_state = np.random.choice(3, p=pomdp.T[a, true_state, :])
                o = np.random.choice(3, p=pomdp.Obs[a, true_state, :])
                b = belief_update(pomdp.T, pomdp.Obs, b, a, o, 3)

            episode_returns.append(total_reward)
        all_returns[name] = episode_returns

    # Plot 1: Return distributions
    ax = axes[0, 0]
    positions = range(len(strategies))
    bp = ax.boxplot([all_returns[name] for name in strategies],
                    labels=[n.replace(' ', '\n') for n in strategies.keys()],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Episode Return')
    ax.set_title('Return Distribution by Strategy')
    ax.grid(True, alpha=0.3)

    # Plot 2: Single episode belief trajectory (belief-based strategy)
    ax = axes[0, 1]
    np.random.seed(42)
    true_state = 0
    b = np.array([0.8, 0.15, 0.05])
    belief_traj = [b.copy()]
    state_traj = [true_state]
    action_traj = []

    for t in range(episode_length):
        a = strategies['Belief-Based'](b, t)
        action_traj.append(a)
        true_state = np.random.choice(3, p=pomdp.T[a, true_state, :])
        o = np.random.choice(3, p=pomdp.Obs[a, true_state, :])
        b = belief_update(pomdp.T, pomdp.Obs, b, a, o, 3)
        belief_traj.append(b.copy())
        state_traj.append(true_state)

    belief_arr = np.array(belief_traj)
    state_colors = ['#4CAF50', '#FF9800', '#F44336']
    for i, (name, color) in enumerate(zip(pomdp.state_names, state_colors)):
        ax.plot(belief_arr[:, i], color=color, linewidth=2, label=f'P({name})')

    # Mark true state
    for t, s in enumerate(state_traj):
        ax.scatter(t, -0.05, c=state_colors[s], s=20, zorder=5)

    # Mark actions
    action_colors_map = {0: '#E0E0E0', 1: '#2196F3', 2: '#F44336'}
    for t, a in enumerate(action_traj):
        ax.axvspan(t, t+1, alpha=0.1, color=action_colors_map[a])

    ax.set_xlabel('Step')
    ax.set_ylabel('Belief')
    ax.set_title('Belief-Based Strategy: One Episode')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # Plot 3: Belief simplex with decision regions
    ax = axes[1, 0]
    n_grid = 60
    img = np.zeros((n_grid, n_grid))
    for i, pg in enumerate(np.linspace(0, 1, n_grid)):
        for j, pd in enumerate(np.linspace(0, 1 - pg, n_grid)):
            pf = max(0, 1 - pg - pd)
            b = np.array([pg, pd, pf])
            # Use belief-based policy
            if pf > 0.3:
                img[j, i] = 2  # maintain
            elif pd + pf > 0.5:
                img[j, i] = 1  # inspect
            else:
                img[j, i] = 0  # produce

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(['#4CAF50', '#2196F3', '#F44336'])
    img_masked = np.ma.masked_where(np.isnan(img), img)
    ax.imshow(img_masked, origin='lower', cmap=cmap, aspect='auto', extent=[0, 1, 0, 1])
    ax.set_xlabel('P(Good)')
    ax.set_ylabel('P(Degraded)')
    ax.set_title('Decision Regions in Belief Space')
    legend_elements = [Patch(facecolor=c, label=n)
                       for c, n in zip(['#4CAF50', '#2196F3', '#F44336'],
                                        pomdp.action_names)]
    ax.legend(handles=legend_elements, fontsize=9)

    # Plot 4: Failure rate comparison
    ax = axes[1, 1]
    failure_rates = {}
    for name, policy in strategies.items():
        n_failures = 0
        total_steps = 0
        for ep in range(n_episodes):
            np.random.seed(ep * 100)
            true_state = 0
            b = np.array([0.8, 0.15, 0.05])
            for t in range(episode_length):
                a = policy(b, t)
                true_state = np.random.choice(3, p=pomdp.T[a, true_state, :])
                o = np.random.choice(3, p=pomdp.Obs[a, true_state, :])
                b = belief_update(pomdp.T, pomdp.Obs, b, a, o, 3)
                total_steps += 1
                if true_state == 2 and a == 0:
                    n_failures += 1
        failure_rates[name] = n_failures / total_steps * 100

    ax.bar([n.replace(' ', '\n') for n in failure_rates.keys()],
           failure_rates.values(), color=colors, edgecolor='black')
    ax.set_ylabel('Production-While-Failed Rate (%)')
    ax.set_title('Safety: Producing on Failed Machine')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/machine_maintenance.png', dpi=150)
    plt.show()


# ============================================================
# 3. Search & Rescue POMDP (Grid-Based)
# ============================================================

def demo_search_rescue():
    """
    A rescue drone searches for a lost person on a 4x4 grid.
    The drone has a noisy sensor that detects presence within range.

    Hidden state: person's location (16 states)
    Actions: move N/S/E/W, scan (stay and gather sensor data)
    Observations: signal_strength (none, weak, strong)
    """
    np.random.seed(42)
    grid_size = 4
    n_cells = grid_size ** 2

    # Person's true location
    person_loc = (3, 2)  # row, col
    person_idx = person_loc[0] * grid_size + person_loc[1]

    # Drone starts with uniform belief
    belief = np.ones(n_cells) / n_cells
    drone_pos = [0, 0]

    # Sensor model: detection probability based on distance
    def sensor_prob(drone_r, drone_c, person_r, person_c):
        dist = abs(drone_r - person_r) + abs(drone_c - person_c)
        if dist == 0:
            return np.array([0.05, 0.10, 0.85])  # strong signal
        elif dist == 1:
            return np.array([0.10, 0.60, 0.30])  # weak signal
        elif dist == 2:
            return np.array([0.30, 0.50, 0.20])
        else:
            return np.array([0.70, 0.25, 0.05])  # likely nothing

    obs_names = ['None', 'Weak', 'Strong']

    # Simulate search trajectory
    trajectory = [(drone_pos[0], drone_pos[1])]
    belief_entropy = [-(belief * np.log(belief + 1e-10)).sum()]
    beliefs = [belief.copy()]

    # Simple search strategy: move toward highest-belief cell, scan periodically
    for step in range(20):
        # Decide action: move toward max-belief cell
        best_cell = np.argmax(belief)
        target_r, target_c = best_cell // grid_size, best_cell % grid_size

        if step % 3 == 2:
            # Scan (stay in place)
            pass
        else:
            # Move toward target
            if target_r > drone_pos[0]:
                drone_pos[0] = min(drone_pos[0] + 1, grid_size - 1)
            elif target_r < drone_pos[0]:
                drone_pos[0] = max(drone_pos[0] - 1, 0)
            elif target_c > drone_pos[1]:
                drone_pos[1] = min(drone_pos[1] + 1, grid_size - 1)
            elif target_c < drone_pos[1]:
                drone_pos[1] = max(drone_pos[1] - 1, 0)

        # Get observation from true person location
        obs_probs = sensor_prob(drone_pos[0], drone_pos[1], person_loc[0], person_loc[1])
        obs = np.random.choice(3, p=obs_probs)

        # Update belief for ALL possible person locations
        new_belief = np.zeros(n_cells)
        for idx in range(n_cells):
            pr, pc = idx // grid_size, idx % grid_size
            obs_p = sensor_prob(drone_pos[0], drone_pos[1], pr, pc)
            new_belief[idx] = obs_p[obs] * belief[idx]

        if new_belief.sum() > 0:
            new_belief /= new_belief.sum()
        belief = new_belief

        trajectory.append((drone_pos[0], drone_pos[1]))
        belief_entropy.append(-(belief * np.log(belief + 1e-10)).sum())
        beliefs.append(belief.copy())

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Search & Rescue POMDP: Drone Searching for Lost Person',
                 fontsize=14, fontweight='bold')

    # Plot beliefs at different timesteps
    timesteps = [0, 4, 8, 12, 16, len(beliefs)-1]
    for ax, t in zip(axes.flat, timesteps):
        belief_grid = beliefs[t].reshape(grid_size, grid_size)
        im = ax.imshow(belief_grid, cmap='YlOrRd', vmin=0, vmax=max(0.3, belief_grid.max()))
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Mark person location
        ax.plot(person_loc[1], person_loc[0], 'g*', markersize=20, label='Person')

        # Mark drone trajectory up to this point
        traj_up_to = trajectory[:min(t+1, len(trajectory))]
        if traj_up_to:
            rows, cols = zip(*traj_up_to)
            ax.plot(cols, rows, 'b.-', markersize=8, linewidth=2, label='Drone path')
            ax.plot(cols[-1], rows[-1], 'bs', markersize=12)

        ax.set_title(f'Step {t} (entropy: {belief_entropy[t]:.2f})')
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.3)
        if t == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/search_rescue.png', dpi=150)
    plt.show()

    # Entropy plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(belief_entropy, 'b-o', markersize=4, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Belief Entropy (bits)')
    ax.set_title('Search & Rescue: Uncertainty Decreases as Drone Gathers Information')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/search_rescue_entropy.png', dpi=150)
    plt.show()


# ============================================================
# 4. Generic N-State POMDP Visualization
# ============================================================

def demo_belief_simplex_3state():
    """
    Visualize the belief simplex for a generic 3-state POMDP.
    Shows how belief updates trace paths through the simplex.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Belief Simplex: Paths Through Probability Space (3-State POMDP)',
                 fontsize=14, fontweight='bold')

    # Convert barycentric coordinates to 2D cartesian for plotting
    def bary_to_cart(p):
        """Convert barycentric coords (p1, p2, p3) to 2D cartesian."""
        x = 0.5 * (2 * p[1] + p[2])
        y = (np.sqrt(3) / 2) * p[2]
        return x, y

    def draw_simplex(ax):
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        # Label vertices
        for i, (name, offset) in enumerate(zip(['S₁', 'S₂', 'S₃'],
                                                [(-0.08, -0.05), (1.05, -0.05), (0.5, 0.9)])):
            ax.text(offset[0], offset[1], name, fontsize=12, fontweight='bold')

    # Plot 1: Random belief trajectories
    ax = axes[0]
    draw_simplex(ax)

    # Simulate multiple belief trajectories from a simple 3-state system
    T = np.array([  # single action for simplicity
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.1, 0.7]
    ])
    Obs = np.array([  # 3 observations
        [0.8, 0.15, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8]
    ])

    for trial in range(5):
        true_s = np.random.randint(3)
        b = np.array([1/3, 1/3, 1/3])
        path_x, path_y = [], []

        for _ in range(15):
            x, y = bary_to_cart(b)
            path_x.append(x)
            path_y.append(y)

            # Transition
            true_s = np.random.choice(3, p=T[true_s])
            # Observation
            o = np.random.choice(3, p=Obs[true_s])
            # Belief update
            b_new = np.zeros(3)
            for sp in range(3):
                pred = sum(T[s, sp] * b[s] for s in range(3))
                b_new[sp] = Obs[sp, o] * pred
            if b_new.sum() > 0:
                b_new /= b_new.sum()
            b = b_new

        ax.plot(path_x, path_y, 'o-', markersize=3, linewidth=1.5, alpha=0.7)
        ax.plot(path_x[0], path_y[0], 'ko', markersize=6)
        ax.plot(path_x[-1], path_y[-1], 'r^', markersize=8)

    ax.set_title('Belief Trajectories\n(different true states)')

    # Plot 2: Information gain heatmap over simplex
    ax = axes[1]
    draw_simplex(ax)

    n_pts = 30
    for i in range(n_pts):
        for j in range(n_pts - i):
            p1 = i / n_pts
            p2 = j / n_pts
            p3 = 1 - p1 - p2
            if p3 < 0:
                continue
            b = np.array([p1, p2, p3])
            if b.min() < 0.01:
                continue

            # Expected info gain from one observation
            eig = 0.0
            for o in range(3):
                b_new = np.zeros(3)
                for sp in range(3):
                    pred = sum(T[s, sp] * b[s] for s in range(3))
                    b_new[sp] = Obs[sp, o] * pred
                p_o = b_new.sum()
                if p_o > 1e-10:
                    b_new /= p_o
                    kl = sum(b_new[s] * np.log(b_new[s] / (b[s] + 1e-10) + 1e-10)
                             for s in range(3) if b_new[s] > 1e-10)
                    eig += p_o * kl

            x, y = bary_to_cart(b)
            ax.scatter(x, y, c=eig, cmap='YlOrRd', s=50, vmin=0, vmax=0.5,
                      edgecolors='none')

    ax.set_title('Expected Info Gain\n(max at center = max uncertainty)')

    # Plot 3: Value function regions (if we had alpha-vectors)
    ax = axes[2]
    draw_simplex(ax)

    # Fake alpha-vectors for illustration
    alphas = [
        np.array([2, 0, 0]),   # best when in S1
        np.array([0, 3, 0]),   # best when in S2
        np.array([0, 0, 1.5]), # best when in S3
    ]
    alpha_colors = ['#F44336', '#2196F3', '#4CAF50']

    for i in range(n_pts + 1):
        for j in range(n_pts - i + 1):
            p1 = i / n_pts
            p2 = j / n_pts
            p3 = 1 - p1 - p2
            if p3 < -0.01:
                continue
            p3 = max(0, p3)
            b = np.array([p1, p2, p3])

            vals = [b @ alpha for alpha in alphas]
            best = np.argmax(vals)

            x, y = bary_to_cart(b)
            ax.scatter(x, y, c=alpha_colors[best], s=30, edgecolors='none', alpha=0.6)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=f'α-vector {i+1}')
                  for i, c in enumerate(alpha_colors)]
    ax.legend(handles=legend_els, fontsize=8, loc='upper right')
    ax.set_title('Alpha-Vector Regions\n(PWLC value function)')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/belief_simplex.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Medical Diagnosis POMDP")
    print("=" * 60)
    demo_medical_diagnosis()

    print("\n" + "=" * 60)
    print("Demo 2: Machine Maintenance POMDP")
    print("=" * 60)
    demo_machine_maintenance()

    print("\n" + "=" * 60)
    print("Demo 3: Search & Rescue POMDP")
    print("=" * 60)
    demo_search_rescue()

    print("\n" + "=" * 60)
    print("Demo 4: Belief Simplex Visualization (3-State)")
    print("=" * 60)
    demo_belief_simplex_3state()
