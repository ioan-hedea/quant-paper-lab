"""
Section 2: POMDP - Belief Updates, Alpha-Vectors, and Point-Based Solvers
==========================================================================
Visualizes core POMDP concepts from the SDM notes:
- Belief update (Bayesian filtering) on the Tiger problem
- Alpha-vectors and PWLC value functions (2-state example)
- Exact solver: Monahan enumeration with pruning
- Point-Based Value Iteration (PBVI) on Tiger problem
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Tiger POMDP
# ============================================================

class TigerPOMDP:
    """
    Classic Tiger problem (Kaelbling et al., 1998).
    States: LEFT (tiger behind left), RIGHT (tiger behind right)
    Actions: LISTEN, OPEN_LEFT, OPEN_RIGHT
    Observations: HEAR_LEFT, HEAR_RIGHT
    """
    S = ['tiger-left', 'tiger-right']
    A = ['listen', 'open-left', 'open-right']
    O = ['hear-left', 'hear-right']

    n_states = 2
    n_actions = 3
    n_obs = 2

    listen_accuracy = 0.85
    gamma = 0.95

    # Rewards: R[s, a]
    R = np.array([
        # listen, open-left, open-right
        [-1.0, -100.0, +10.0],   # tiger-left
        [-1.0, +10.0, -100.0],   # tiger-right
    ])

    # Transition: T[s, a, s']
    # Listen: stays. Open: resets uniformly.
    T = np.zeros((2, 3, 2))
    T[0, 0, 0] = 1.0  # listen from left -> left
    T[1, 0, 1] = 1.0  # listen from right -> right
    for a in [1, 2]:   # open actions reset
        T[:, a, 0] = 0.5
        T[:, a, 1] = 0.5

    # Observation: O[s', a, o]
    Obs = np.zeros((2, 3, 2))
    # Listen: 85% accurate
    Obs[0, 0, 0] = 0.85   # tiger-left, listen, hear-left
    Obs[0, 0, 1] = 0.15
    Obs[1, 0, 0] = 0.15   # tiger-right, listen, hear-left
    Obs[1, 0, 1] = 0.85
    # Open: uniform observation
    for a in [1, 2]:
        Obs[:, a, :] = 0.5


def belief_update(pomdp, b, a, o):
    """Bayesian belief update: b' = eta * O(o|s',a) * sum_s T(s'|s,a) * b(s)"""
    b_new = np.zeros(pomdp.n_states)
    for sp in range(pomdp.n_states):
        pred = sum(pomdp.T[s, a, sp] * b[s] for s in range(pomdp.n_states))
        b_new[sp] = pomdp.Obs[sp, a, o] * pred
    if b_new.sum() > 0:
        b_new /= b_new.sum()
    return b_new


def obs_probability(pomdp, b, a, o):
    """P(o | b, a) = sum_{s'} O(o|s',a) sum_s T(s'|s,a) b(s)"""
    prob = 0.0
    for sp in range(pomdp.n_states):
        pred = sum(pomdp.T[s, a, sp] * b[s] for s in range(pomdp.n_states))
        prob += pomdp.Obs[sp, a, o] * pred
    return prob


# ============================================================
# Demo 1: Belief Update Visualization
# ============================================================

def demo_belief_updates():
    """Visualize sequential belief updates in the Tiger problem."""
    pomdp = TigerPOMDP()
    b = np.array([0.5, 0.5])  # uniform prior

    # Simulate a sequence of listens with observations
    np.random.seed(42)
    true_state = 0  # tiger is on the left

    beliefs = [b.copy()]
    observations = []

    for step in range(10):
        a = 0  # listen
        # Sample observation from true state
        o = np.random.choice(pomdp.n_obs, p=pomdp.Obs[true_state, a, :])
        b = belief_update(pomdp, b, a, o)
        beliefs.append(b.copy())
        observations.append(pomdp.O[o])

    beliefs = np.array(beliefs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Tiger POMDP: Bayesian Belief Updates', fontsize=14, fontweight='bold')

    # Belief trajectory
    ax1.plot(range(len(beliefs)), beliefs[:, 0], 'ro-', label='P(tiger-left)', markersize=8, linewidth=2)
    ax1.plot(range(len(beliefs)), beliefs[:, 1], 'bs-', label='P(tiger-right)', markersize=8, linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Belief probability')
    ax1.set_xlabel('Step')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'True state: tiger-left | Observations shape belief over time')

    # Observation annotations
    for i, obs in enumerate(observations):
        color = 'red' if obs == 'hear-left' else 'blue'
        ax2.barh(0, 1, left=i+0.5, height=0.6, color=color, alpha=0.7)
        ax2.text(i+1, 0, obs.replace('hear-', 'H-'), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    ax2.set_xlim(0.5, len(observations)+0.5)
    ax2.set_yticks([])
    ax2.set_xlabel('Step')
    ax2.set_title('Observations received')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/belief_updates.png', dpi=150)
    plt.show()

    print("Belief after 10 listens:", beliefs[-1])
    print("True state: tiger-left")


def demo_belief_update_grid():
    """Show how belief evolves for different observation sequences."""
    pomdp = TigerPOMDP()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Belief Evolution Under Different Observation Sequences', fontsize=14, fontweight='bold')

    scenarios = [
        ("All hear-left", [0]*6),
        ("All hear-right", [1]*6),
        ("Alternating", [0, 1, 0, 1, 0, 1]),
        ("Mostly left", [0, 0, 0, 1, 0, 0]),
        ("Mostly right", [1, 1, 1, 0, 1, 1]),
        ("Random mix", [0, 1, 1, 0, 0, 1]),
    ]

    for ax, (title, obs_seq) in zip(axes.flat, scenarios):
        b = np.array([0.5, 0.5])
        p_left = [b[0]]
        for o in obs_seq:
            b = belief_update(pomdp, b, 0, o)  # action=listen
            p_left.append(b[0])

        ax.plot(range(len(p_left)), p_left, 'ko-', markersize=6, linewidth=2)
        ax.fill_between(range(len(p_left)), p_left, 0.5, alpha=0.3,
                        color='red' if p_left[-1] > 0.5 else 'blue')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(tiger-left)')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/belief_scenarios.png', dpi=150)
    plt.show()


# ============================================================
# Demo 2: Alpha-Vectors and PWLC Value Functions
# ============================================================

def demo_alpha_vectors():
    """
    Visualize alpha-vectors for a 2-state POMDP (Tiger).
    The belief simplex is 1D: p = P(tiger-left), 1-p = P(tiger-right).
    """
    pomdp = TigerPOMDP()

    # Compute alpha-vectors for increasing horizons using exact backup
    p_range = np.linspace(0, 1, 500)

    def immediate_alphas():
        """Horizon 0: one alpha per action = R[:, a]"""
        alphas = {}
        for a in range(pomdp.n_actions):
            alphas[pomdp.A[a]] = pomdp.R[:, a].copy()
        return alphas

    def compute_value(alphas_list, p):
        """V(b) = max_alpha alpha . b"""
        b = np.array([p, 1 - p])
        return max(alpha @ b for alpha in alphas_list)

    def bellman_backup(prev_alphas):
        """One-step exact Bellman backup for 2-state Tiger POMDP."""
        new_alphas = []
        new_actions = []

        # For each action
        for a in range(pomdp.n_actions):
            # For each combination of alpha-vectors (one per observation)
            for combo in product(range(len(prev_alphas)), repeat=pomdp.n_obs):
                alpha_new = np.zeros(pomdp.n_states)
                for s in range(pomdp.n_states):
                    val = pomdp.R[s, a]
                    for o in range(pomdp.n_obs):
                        k = combo[o]
                        future = 0.0
                        for sp in range(pomdp.n_states):
                            future += pomdp.T[s, a, sp] * pomdp.Obs[sp, a, o] * prev_alphas[k][sp]
                        val += pomdp.gamma * future
                    alpha_new[s] = val
                new_alphas.append(alpha_new)
                new_actions.append(a)

        return new_alphas, new_actions

    def prune_alphas(alphas, actions):
        """Remove dominated alpha-vectors by sampling beliefs."""
        p_samples = np.linspace(0, 1, 200)
        kept = set()
        for p in p_samples:
            b = np.array([p, 1 - p])
            vals = [alpha @ b for alpha in alphas]
            kept.add(np.argmax(vals))
        kept = sorted(kept)
        return [alphas[i] for i in kept], [actions[i] for i in kept]

    # Build value functions for horizons 0, 1, 2, 3
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Alpha-Vectors: PWLC Value Function Over Belief Simplex (Tiger POMDP)',
                 fontsize=14, fontweight='bold')

    colors_action = {'listen': 'green', 'open-left': 'red', 'open-right': 'blue'}

    # Horizon 0
    ax = axes[0, 0]
    imm = immediate_alphas()
    for name, alpha in imm.items():
        vals = [np.array([p, 1-p]) @ alpha for p in p_range]
        ax.plot(p_range, vals, color=colors_action[name], linewidth=2, label=name)
    # Upper envelope
    all_alphas_0 = list(imm.values())
    envelope = [compute_value(all_alphas_0, p) for p in p_range]
    ax.plot(p_range, envelope, 'k--', linewidth=1.5, alpha=0.5, label='V*(b)')
    ax.set_title('Horizon 0 (immediate rewards)')
    ax.set_xlabel('P(tiger-left)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Horizons 1, 2, 3
    prev_alphas = list(imm.values())
    prev_actions = [0, 1, 2]

    for h, ax in zip([1, 2, 3], [axes[0, 1], axes[1, 0], axes[1, 1]]):
        new_alphas, new_actions = bellman_backup(prev_alphas)
        new_alphas, new_actions = prune_alphas(new_alphas, new_actions)

        for alpha, a_idx in zip(new_alphas, new_actions):
            vals = [np.array([p, 1-p]) @ alpha for p in p_range]
            ax.plot(p_range, vals, color=colors_action[pomdp.A[a_idx]],
                    linewidth=1.5, alpha=0.7)

        envelope = [compute_value(new_alphas, p) for p in p_range]
        ax.plot(p_range, envelope, 'k--', linewidth=2, alpha=0.5, label='V*(b)')

        ax.set_title(f'Horizon {h} ({len(new_alphas)} alpha-vectors)')
        ax.set_xlabel('P(tiger-left)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        prev_alphas = new_alphas
        prev_actions = new_actions

    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=2, label=n)
                       for n, c in colors_action.items()]
    legend_elements.append(Line2D([0], [0], color='k', lw=2, ls='--', label='V*(b)'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/alpha_vectors.png', dpi=150)
    plt.show()

    print(f"Final horizon alpha-vector count: {len(new_alphas)}")


# ============================================================
# Demo 3: Point-Based Value Iteration (PBVI)
# ============================================================

def demo_pbvi():
    """
    Implement and visualize Point-Based Value Iteration on Tiger POMDP.
    Shows how value function improves as we iterate.
    """
    pomdp = TigerPOMDP()

    # Collect belief points via forward simulation
    def collect_beliefs(n_points=50, n_steps=10):
        beliefs = [np.array([0.5, 0.5])]
        b = np.array([0.5, 0.5])
        np.random.seed(123)
        for _ in range(n_points):
            b = np.array([0.5, 0.5])
            for _ in range(np.random.randint(1, n_steps)):
                a = np.random.randint(pomdp.n_actions)
                o = np.random.randint(pomdp.n_obs)
                b = belief_update(pomdp, b, a, o)
            beliefs.append(b.copy())
        return beliefs

    belief_points = collect_beliefs(100)
    B = np.array(belief_points)

    # Initialize: alpha-vectors (one per belief point, set to immediate max reward)
    alphas = []
    for b in belief_points:
        best_val = -np.inf
        best_alpha = None
        for a in range(pomdp.n_actions):
            val = b @ pomdp.R[:, a]
            if val > best_val:
                best_val = val
                best_alpha = pomdp.R[:, a].copy()
        alphas.append(best_alpha)

    def value_at_belief(b, alpha_set):
        return max(b @ alpha for alpha in alpha_set)

    def pbvi_backup(belief_points, alpha_set):
        """One PBVI backup iteration."""
        new_alphas = []
        for b in belief_points:
            best_val = -np.inf
            best_alpha = None

            for a in range(pomdp.n_actions):
                alpha_a = pomdp.R[:, a].copy()

                for o in range(pomdp.n_obs):
                    # Compute updated belief
                    b_ao = belief_update(pomdp, b, a, o)
                    p_o = obs_probability(pomdp, b, a, o)

                    if p_o < 1e-10:
                        continue

                    # Find best alpha for b_ao
                    best_k = max(range(len(alpha_set)),
                                key=lambda k: b_ao @ alpha_set[k])

                    # Add contribution
                    for s in range(pomdp.n_states):
                        contrib = 0.0
                        for sp in range(pomdp.n_states):
                            contrib += pomdp.T[s, a, sp] * pomdp.Obs[sp, a, o] * alpha_set[best_k][sp]
                        alpha_a[s] += pomdp.gamma * contrib

                val = b @ alpha_a
                if val > best_val:
                    best_val = val
                    best_alpha = alpha_a.copy()

            new_alphas.append(best_alpha)
        return new_alphas

    # Run PBVI
    n_iters = 20
    value_history = []

    p_range = np.linspace(0, 1, 200)
    for it in range(n_iters):
        vals = [value_at_belief(np.array([p, 1-p]), alphas) for p in p_range]
        value_history.append(vals)
        alphas = pbvi_backup(belief_points, alphas)

    # Final values
    vals = [value_at_belief(np.array([p, 1-p]), alphas) for p in p_range]
    value_history.append(vals)

    # Plot: PBVI convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Point-Based Value Iteration (PBVI) on Tiger POMDP', fontsize=14, fontweight='bold')

    # Value function at selected iterations
    iters_show = [0, 2, 5, 10, n_iters]
    cmap = plt.cm.viridis
    for i, it in enumerate(iters_show):
        color = cmap(i / len(iters_show))
        ax1.plot(p_range, value_history[it], color=color, linewidth=2,
                label=f'Iter {it}')

    ax1.set_xlabel('P(tiger-left)')
    ax1.set_ylabel('V(b)')
    ax1.set_title('Value Function Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Belief points scatter
    ax2.scatter(B[:, 0], np.zeros(len(B)), alpha=0.5, s=20, c='blue')
    # Overlay final value
    ax2_twin = ax2.twinx()
    ax2_twin.plot(p_range, value_history[-1], 'r-', linewidth=2, label='Final V(b)')
    ax2.set_xlabel('P(tiger-left)')
    ax2.set_ylabel('Belief points (projected)')
    ax2_twin.set_ylabel('V(b)', color='red')
    ax2.set_title(f'Belief Point Distribution ({len(B)} points)')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/pbvi_convergence.png', dpi=150)
    plt.show()

    # Policy visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('PBVI Optimal Policy Over Belief Space', fontsize=14, fontweight='bold')

    action_colors = ['green', 'red', 'blue']
    action_names = ['listen', 'open-left', 'open-right']

    for p in p_range:
        b = np.array([p, 1-p])
        best_a = -1
        best_val = -np.inf
        for a in range(pomdp.n_actions):
            # Compute Q(b, a) using final alphas
            alpha_a = pomdp.R[:, a].copy()
            for o in range(pomdp.n_obs):
                b_ao = belief_update(pomdp, b, a, o)
                p_o = obs_probability(pomdp, b, a, o)
                if p_o < 1e-10:
                    continue
                best_k = max(range(len(alphas)), key=lambda k: b_ao @ alphas[k])
                for s in range(pomdp.n_states):
                    contrib = 0.0
                    for sp in range(pomdp.n_states):
                        contrib += pomdp.T[s, a, sp] * pomdp.Obs[sp, a, o] * alphas[best_k][sp]
                    alpha_a[s] += pomdp.gamma * contrib
            val = b @ alpha_a
            if val > best_val:
                best_val = val
                best_a = a
        ax.axvspan(p - 0.001, p + 0.001, color=action_colors[best_a], alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(action_colors, action_names)]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper center')
    ax.set_xlabel('P(tiger-left)', fontsize=12)
    ax.set_ylabel('Optimal action region')
    ax.set_title('When uncertain: listen. When confident: open the safe door.')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/pbvi_policy.png', dpi=150)
    plt.show()


# ============================================================
# Demo 4: Information Gain Visualization
# ============================================================

def demo_information_gain():
    """Visualize information gain (KL divergence) from observations."""
    pomdp = TigerPOMDP()

    p_range = np.linspace(0.01, 0.99, 200)

    # For each prior belief, compute expected information gain from listening
    expected_ig = []
    ig_hear_left = []
    ig_hear_right = []

    for p in p_range:
        b = np.array([p, 1 - p])
        eig = 0.0
        igs = []
        for o in range(pomdp.n_obs):
            b_new = belief_update(pomdp, b, 0, o)  # listen
            p_o = obs_probability(pomdp, b, 0, o)

            # KL divergence: D_KL(b_new || b)
            kl = 0.0
            for s in range(pomdp.n_states):
                if b_new[s] > 1e-10 and b[s] > 1e-10:
                    kl += b_new[s] * np.log(b_new[s] / b[s])
            igs.append(kl)
            eig += p_o * kl

        expected_ig.append(eig)
        ig_hear_left.append(igs[0])
        ig_hear_right.append(igs[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Information Gain from Listening (Tiger POMDP)', fontsize=14, fontweight='bold')

    ax1.plot(p_range, expected_ig, 'k-', linewidth=2, label='Expected IG')
    ax1.plot(p_range, ig_hear_left, 'r--', linewidth=1.5, label='IG if hear-left')
    ax1.plot(p_range, ig_hear_right, 'b--', linewidth=1.5, label='IG if hear-right')
    ax1.set_xlabel('P(tiger-left)')
    ax1.set_ylabel('Information Gain (KL divergence)')
    ax1.set_title('Information gain is highest when uncertain (p=0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Entropy of belief
    entropy = [-p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10) for p in p_range]
    ax2.plot(p_range, entropy, 'purple', linewidth=2)
    ax2.set_xlabel('P(tiger-left)')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Belief Entropy: Uncertainty About Hidden State')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/pomdp/information_gain.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Belief Updates in Tiger POMDP")
    print("=" * 60)
    demo_belief_updates()
    demo_belief_update_grid()

    print("\n" + "=" * 60)
    print("Demo 2: Alpha-Vectors and PWLC Value Functions")
    print("=" * 60)
    demo_alpha_vectors()

    print("\n" + "=" * 60)
    print("Demo 3: Point-Based Value Iteration (PBVI)")
    print("=" * 60)
    demo_pbvi()

    print("\n" + "=" * 60)
    print("Demo 4: Information Gain from Observations")
    print("=" * 60)
    demo_information_gain()
