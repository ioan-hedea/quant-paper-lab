"""
Section 7: Bayesian Reinforcement Learning
============================================
Visualizes core Bayesian RL concepts from the SDM notes:
- Normal-Gamma conjugate prior for Bayesian Q-learning
- Thompson Sampling vs UCB vs epsilon-greedy (multi-armed bandit)
- Bayes-Adaptive MDP: exploration as planning in augmented state
- Posterior shrinkage and sample efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# ============================================================
# Multi-Armed Bandit Environment
# ============================================================

class MultiArmedBandit:
    """K-armed bandit with Bernoulli rewards."""
    def __init__(self, means):
        self.means = np.array(means)
        self.k = len(means)
        self.best_arm = np.argmax(means)
        self.best_mean = np.max(means)

    def pull(self, arm):
        return float(np.random.random() < self.means[arm])


# ============================================================
# Thompson Sampling Agent
# ============================================================

class ThompsonSamplingAgent:
    """Thompson Sampling with Beta priors for Bernoulli bandits."""
    def __init__(self, k):
        self.k = k
        self.alpha = np.ones(k)  # successes + 1
        self.beta = np.ones(k)   # failures + 1

    def select_arm(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.k)]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class UCBAgent:
    """UCB1 agent."""
    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total = 0

    def select_arm(self):
        self.total += 1
        for i in range(self.k):
            if self.counts[i] == 0:
                return i
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class EpsilonGreedyAgent:
    """Epsilon-greedy agent."""
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


# ============================================================
# Visualizations
# ============================================================

def demo_thompson_sampling():
    """Compare Thompson Sampling vs UCB vs epsilon-greedy on a bandit."""
    bandit = MultiArmedBandit([0.3, 0.5, 0.7, 0.4, 0.6])
    T = 1000
    n_trials = 50

    agents_config = [
        ('Thompson Sampling', lambda: ThompsonSamplingAgent(bandit.k), '#4CAF50'),
        ('UCB (c=2)', lambda: UCBAgent(bandit.k, c=2.0), '#2196F3'),
        ('ε-greedy (ε=0.1)', lambda: EpsilonGreedyAgent(bandit.k, 0.1), '#F44336'),
        ('ε-greedy (ε=0.01)', lambda: EpsilonGreedyAgent(bandit.k, 0.01), '#FF9800'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploration Strategies: Thompson Sampling vs UCB vs ε-greedy',
                 fontsize=14, fontweight='bold')

    all_regrets = {}
    all_optimal = {}

    for name, agent_fn, color in agents_config:
        cumulative_regrets = np.zeros((n_trials, T))
        optimal_actions = np.zeros((n_trials, T))

        for trial in range(n_trials):
            np.random.seed(trial)
            agent = agent_fn()
            cum_regret = 0.0
            for t in range(T):
                arm = agent.select_arm()
                reward = bandit.pull(arm)
                agent.update(arm, reward)
                cum_regret += bandit.best_mean - bandit.means[arm]
                cumulative_regrets[trial, t] = cum_regret
                optimal_actions[trial, t] = float(arm == bandit.best_arm)

        all_regrets[name] = cumulative_regrets
        all_optimal[name] = optimal_actions

    # Plot 1: Cumulative regret
    ax = axes[0, 0]
    for name, _, color in agents_config:
        mean = all_regrets[name].mean(axis=0)
        std = all_regrets[name].std(axis=0)
        ax.plot(range(T), mean, color=color, linewidth=2, label=name)
        ax.fill_between(range(T), mean-std, mean+std, color=color, alpha=0.1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: % optimal action
    ax = axes[0, 1]
    window = 50
    for name, _, color in agents_config:
        mean = all_optimal[name].mean(axis=0)
        smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Optimal Action Selection Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 3: Thompson Sampling posterior evolution
    ax = axes[1, 0]
    np.random.seed(42)
    ts = ThompsonSamplingAgent(bandit.k)
    x = np.linspace(0, 1, 200)

    # Show posteriors at different time steps
    snapshots = [0, 10, 50, 200]
    colors_arms = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    t_step = 0
    posterior_data = {}

    for snap_t in snapshots:
        while t_step < snap_t:
            arm = ts.select_arm()
            reward = bandit.pull(arm)
            ts.update(arm, reward)
            t_step += 1
        posterior_data[snap_t] = (ts.alpha.copy(), ts.beta.copy())

    # Show final posteriors
    for i in range(bandit.k):
        a, b = posterior_data[200]
        y = stats.beta.pdf(x, a[i], b[i])
        ax.plot(x, y, color=colors_arms[i], linewidth=2,
               label=f'Arm {i} (true={bandit.means[i]:.1f})')
        ax.axvline(bandit.means[i], color=colors_arms[i], linestyle='--', alpha=0.5)

    ax.set_xlabel('θ (success probability)')
    ax.set_ylabel('Posterior density')
    ax.set_title('Thompson Sampling: Posteriors after 200 pulls')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Posterior evolution for best arm
    ax = axes[1, 1]
    for snap_t in snapshots:
        a, b = posterior_data.get(snap_t, (np.ones(bandit.k), np.ones(bandit.k)))
        best = bandit.best_arm
        y = stats.beta.pdf(x, a[best], b[best])
        ax.plot(x, y, linewidth=2, label=f't={snap_t} (α={a[best]:.0f}, β={b[best]:.0f})')

    ax.axvline(bandit.means[bandit.best_arm], color='black', linestyle='--',
              label=f'True θ*={bandit.means[bandit.best_arm]}')
    ax.set_xlabel('θ (success probability)')
    ax.set_ylabel('Posterior density')
    ax.set_title(f'Posterior Shrinkage: Best Arm ({bandit.best_arm})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/bayesian_rl/thompson_sampling.png', dpi=150)
    plt.show()


def demo_normal_gamma():
    """Visualize Normal-Gamma conjugate prior updates for Bayesian Q-learning."""
    np.random.seed(42)

    # True Q-value parameters
    true_mu = 5.0
    true_sigma = 1.5

    # Prior hyperparameters
    mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 1.0, 1.0

    # Generate observed returns
    n_observations = 50
    returns = np.random.normal(true_mu, true_sigma, n_observations)

    # Track posterior evolution
    snapshots = [0, 1, 3, 10, 25, 50]
    posterior_params = []

    mu_n, kappa_n, alpha_n, beta_n = mu_0, kappa_0, alpha_0, beta_0
    posterior_params.append((mu_n, kappa_n, alpha_n, beta_n))

    for i in range(n_observations):
        G = returns[i]
        kappa_new = kappa_n + 1
        mu_new = (kappa_n * mu_n + G) / kappa_new
        alpha_new = alpha_n + 0.5
        beta_new = beta_n + 0.5 * kappa_n * (G - mu_n)**2 / kappa_new
        mu_n, kappa_n, alpha_n, beta_n = mu_new, kappa_new, alpha_new, beta_new

        if (i + 1) in snapshots:
            posterior_params.append((mu_n, kappa_n, alpha_n, beta_n))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Bayesian Q-Learning: Normal-Gamma Posterior Updates', fontsize=14, fontweight='bold')

    x = np.linspace(-3, 10, 300)

    for ax, (snap, params) in zip(axes.flat, zip(snapshots, posterior_params)):
        mu, kappa, alpha, beta = params
        # Marginal posterior for mu is Student-t
        df = 2 * alpha
        scale = np.sqrt(beta / (alpha * kappa))
        y = stats.t.pdf(x, df=df, loc=mu, scale=scale)

        ax.plot(x, y, 'b-', linewidth=2)
        ax.fill_between(x, y, alpha=0.3, color='blue')
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2, label=f'True μ={true_mu}')
        ax.axvline(mu, color='blue', linestyle=':', linewidth=2, label=f'Post. μ={mu:.2f}')

        # 95% credible interval
        ci_low = stats.t.ppf(0.025, df=df, loc=mu, scale=scale)
        ci_high = stats.t.ppf(0.975, df=df, loc=mu, scale=scale)
        ax.axvspan(ci_low, ci_high, alpha=0.1, color='green')

        ax.set_title(f'n = {snap} observations\nCI=[{ci_low:.1f}, {ci_high:.1f}]')
        ax.set_xlabel('Q-value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/bayesian_rl/normal_gamma.png', dpi=150)
    plt.show()


def demo_bayes_adaptive_mdp():
    """
    Demonstrate the Bayes-Adaptive MDP concept:
    exploration emerges from planning in the augmented state space.
    """
    np.random.seed(42)

    # Simple 2-state, 2-action MDP
    # State 0: safe (known), State 1: unknown (needs exploration)
    # Action 0: stay in current state, Action 1: move

    # True transitions (unknown to agent)
    T_true = np.array([
        # Action 0 (stay)
        [[0.9, 0.1],
         [0.1, 0.9]],
        # Action 1 (move)
        [[0.2, 0.8],
         [0.8, 0.2]]
    ])

    R = np.array([[0.5, 0.5],   # state 0 rewards
                   [2.0, 2.0]])  # state 1 rewards (unknown, actually high)

    gamma = 0.9
    n_episodes = 100
    episode_length = 20

    # Bayesian agent: tracks Dirichlet counts
    class BayesAdaptiveAgent:
        def __init__(self, n_states, n_actions, epsilon_explore=0.0):
            self.n_s = n_states
            self.n_a = n_actions
            # Dirichlet prior: uniform (all 1s)
            self.counts = np.ones((n_actions, n_states, n_states))
            self.epsilon = epsilon_explore

        def get_posterior_predictive(self):
            T_hat = np.zeros_like(self.counts)
            for a in range(self.n_a):
                for s in range(self.n_s):
                    T_hat[a, s, :] = self.counts[a, s, :] / self.counts[a, s, :].sum()
            return T_hat

        def thompson_sample(self):
            T_sample = np.zeros_like(self.counts)
            for a in range(self.n_a):
                for s in range(self.n_s):
                    T_sample[a, s, :] = np.random.dirichlet(self.counts[a, s, :])
            return T_sample

        def select_action(self, s, method='thompson'):
            if method == 'thompson':
                T = self.thompson_sample()
            else:
                T = self.get_posterior_predictive()

            # Solve for Q given sampled model
            V = np.zeros(self.n_s)
            for _ in range(50):
                Q = np.zeros((self.n_s, self.n_a))
                for si in range(self.n_s):
                    for a in range(self.n_a):
                        Q[si, a] = R[si, a] + gamma * T[a, si, :] @ V
                V = np.max(Q, axis=1)
            return np.argmax(Q[s])

        def update(self, s, a, s_next):
            self.counts[a, s, s_next] += 1

    # Compare Thompson Sampling BA-MDP vs epsilon-greedy
    methods = {
        'Thompson Sampling (BA-MDP)': 'thompson',
        'Greedy (expected model)': 'greedy',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Bayes-Adaptive MDP: Exploration via Planning in Augmented Space',
                 fontsize=14, fontweight='bold')

    n_trials = 20
    all_rewards = {}

    for name, method in methods.items():
        trial_rewards = np.zeros((n_trials, n_episodes))
        for trial in range(n_trials):
            np.random.seed(trial * 100)
            agent = BayesAdaptiveAgent(2, 2)
            for ep in range(n_episodes):
                s = 0
                ep_reward = 0
                for t in range(episode_length):
                    a = agent.select_action(s, method)
                    s_next = np.random.choice(2, p=T_true[a, s, :])
                    r = R[s, a]
                    agent.update(s, a, s_next)
                    ep_reward += r
                    s = s_next
                trial_rewards[trial, ep] = ep_reward
        all_rewards[name] = trial_rewards

    # Plot 1: Learning curves
    ax = axes[0]
    for (name, rewards), color in zip(all_rewards.items(), ['#4CAF50', '#F44336']):
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        ax.plot(range(n_episodes), mean, color=color, linewidth=2, label=name)
        ax.fill_between(range(n_episodes), mean-std, mean+std, color=color, alpha=0.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Return')
    ax.set_title('Learning Performance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Posterior convergence (transition estimation error)
    ax = axes[1]
    np.random.seed(42)
    agent = BayesAdaptiveAgent(2, 2)
    errors = []
    for ep in range(n_episodes):
        s = 0
        for t in range(episode_length):
            a = agent.select_action(s, 'thompson')
            s_next = np.random.choice(2, p=T_true[a, s, :])
            agent.update(s, a, s_next)
            s = s_next
        T_hat = agent.get_posterior_predictive()
        error = np.max(np.abs(T_hat - T_true))
        errors.append(error)

    ax.plot(range(n_episodes), errors, 'b-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max |T̂ - T_true|')
    ax.set_title('Model Estimation Error (Posterior Convergence)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Dirichlet posterior for one (s,a) pair
    ax = axes[2]
    # Show final counts
    counts = agent.counts
    bar_labels = []
    bar_vals = []
    bar_true = []
    for a in range(2):
        for s in range(2):
            total = counts[a, s, :].sum()
            for sp in range(2):
                label = f'T(s\'={sp}|s={s},a={a})'
                bar_labels.append(label)
                bar_vals.append(counts[a, s, sp] / total)
                bar_true.append(T_true[a, s, sp])

    x = np.arange(len(bar_labels))
    width = 0.35
    ax.bar(x - width/2, bar_vals, width, color='blue', alpha=0.7, label='Posterior mean')
    ax.bar(x + width/2, bar_true, width, color='red', alpha=0.7, label='True')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Probability')
    ax.set_title('Final Posterior vs True Transitions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/bayesian_rl/bayes_adaptive_mdp.png', dpi=150)
    plt.show()


def demo_posterior_shrinkage():
    """Visualize how posterior variance shrinks with data (1/sqrt(n) rate)."""
    np.random.seed(42)
    true_mean = 3.0
    n_max = 200

    samples = np.random.normal(true_mean, 2.0, n_max)

    # Track posterior variance (Normal model with known variance for simplicity)
    prior_var = 10.0
    sigma_sq = 4.0  # known observation variance

    posterior_means = []
    posterior_vars = []
    ci_lows = []
    ci_highs = []

    for n in range(1, n_max + 1):
        obs = samples[:n]
        post_var = 1.0 / (1.0 / prior_var + n / sigma_sq)
        post_mean = post_var * (0.0 / prior_var + obs.sum() / sigma_sq)
        posterior_means.append(post_mean)
        posterior_vars.append(post_var)
        ci_lows.append(post_mean - 1.96 * np.sqrt(post_var))
        ci_highs.append(post_mean + 1.96 * np.sqrt(post_var))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bayesian Posterior Shrinkage: Uncertainty Decreases with Data',
                 fontsize=14, fontweight='bold')

    ns = range(1, n_max + 1)
    ax1.plot(ns, posterior_means, 'b-', linewidth=2, label='Posterior mean')
    ax1.fill_between(ns, ci_lows, ci_highs, alpha=0.3, color='blue', label='95% CI')
    ax1.axhline(true_mean, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mean}')
    ax1.set_xlabel('Number of Observations')
    ax1.set_ylabel('Q-value estimate')
    ax1.set_title('Posterior Mean Converges to True Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ns, posterior_vars, 'b-', linewidth=2, label='Posterior variance')
    ax2.semilogy(ns, [sigma_sq / n for n in ns], 'r--', linewidth=2, label='σ²/n (theoretical)')
    ax2.set_xlabel('Number of Observations')
    ax2.set_ylabel('Posterior Variance (log scale)')
    ax2.set_title('Variance Shrinkage: O(1/n) Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/bayesian_rl/posterior_shrinkage.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Thompson Sampling vs UCB vs ε-greedy")
    print("=" * 60)
    demo_thompson_sampling()

    print("\n" + "=" * 60)
    print("Demo 2: Normal-Gamma Conjugate Prior")
    print("=" * 60)
    demo_normal_gamma()

    print("\n" + "=" * 60)
    print("Demo 3: Bayes-Adaptive MDP")
    print("=" * 60)
    demo_bayes_adaptive_mdp()

    print("\n" + "=" * 60)
    print("Demo 4: Posterior Shrinkage")
    print("=" * 60)
    demo_posterior_shrinkage()
