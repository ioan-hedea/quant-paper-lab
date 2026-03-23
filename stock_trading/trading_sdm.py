"""
Stock Trading & Portfolio Management — Sequential Decision Making
==================================================================
A unified scenario that combines concepts from all course sections:

1. MDP formulation    — Value iteration for a discrete trading MDP
2. POMDP / Belief     — Hidden market regime with noisy observations
3. MCTS               — Online lookahead planning for trade decisions
4. MPC                — Rolling-horizon portfolio rebalancing
5. Model-Based RL     — Learning market transition dynamics (Dyna-Q)
6. Safe RL            — CVaR-constrained portfolio, risk budgets
7. Bayesian RL        — Thompson Sampling for unknown return distributions
8. Exploration        — Epsilon-greedy vs UCB for asset allocation

Each demo is self-contained with matplotlib visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import random


# ============================================================
# Shared Market Simulator
# ============================================================

class MarketSimulator:
    """
    Simulates a market with hidden regimes (bull/bear/sideways).
    Produces daily returns for N assets.
    """
    REGIMES = ['Bull', 'Bear', 'Sideways']

    def __init__(self, n_assets=3, seed=42):
        self.n_assets = n_assets
        self.rng = np.random.RandomState(seed)

        # Regime transition matrix
        self.regime_T = np.array([
            [0.90, 0.05, 0.05],   # bull -> ...
            [0.05, 0.85, 0.10],   # bear -> ...
            [0.10, 0.10, 0.80],   # sideways -> ...
        ])

        # Mean returns per regime per asset
        self.regime_means = {
            0: np.array([0.002, 0.001, 0.0005]),   # bull
            1: np.array([-0.002, -0.001, 0.0003]),  # bear
            2: np.array([0.0003, 0.0002, 0.0004]),  # sideways
        }
        self.regime_stds = {
            0: np.array([0.015, 0.010, 0.003]),
            1: np.array([0.025, 0.015, 0.005]),
            2: np.array([0.008, 0.006, 0.002]),
        }
        self.regime = 0  # start bull

    def step(self):
        """Advance one day: regime transition + generate returns."""
        self.regime = self.rng.choice(3, p=self.regime_T[self.regime])
        means = self.regime_means[self.regime]
        stds = self.regime_stds[self.regime]
        returns = self.rng.normal(means, stds)
        return returns, self.regime

    def noisy_indicator(self):
        """Observable market indicator (noisy signal of regime)."""
        # 0 = positive signal, 1 = negative signal, 2 = neutral
        if self.rng.random() < 0.7:
            if self.regime == 0:
                return 0
            elif self.regime == 1:
                return 1
            else:
                return 2
        return self.rng.choice(3)


# ============================================================
# Demo 1: Trading as an MDP — Value Iteration
# ============================================================

def demo_trading_mdp():
    """
    Formulate trading as a finite MDP.
    States: (position, price_level). Actions: buy/hold/sell.
    Solve with value iteration.
    """
    np.random.seed(42)

    # Discretize: 3 positions (short/flat/long), 5 price levels
    n_positions = 3   # 0=short, 1=flat, 2=long
    n_prices = 5      # 0=very_low ... 4=very_high
    n_states = n_positions * n_prices
    n_actions = 3     # 0=sell, 1=hold, 2=buy
    gamma = 0.95

    pos_names = ['Short', 'Flat', 'Long']
    price_names = ['VLow', 'Low', 'Mid', 'High', 'VHigh']

    def state_idx(pos, price):
        return pos * n_prices + price

    def state_from_idx(s):
        return s // n_prices, s % n_prices

    # Transition and reward
    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # Price transition: mean-reverting random walk
    price_T = np.array([
        [0.3, 0.4, 0.2, 0.1, 0.0],
        [0.1, 0.3, 0.3, 0.2, 0.1],
        [0.05, 0.15, 0.4, 0.25, 0.15],
        [0.0, 0.1, 0.2, 0.4, 0.3],
        [0.0, 0.05, 0.15, 0.3, 0.5],
    ])

    for pos in range(n_positions):
        for price in range(n_prices):
            s = state_idx(pos, price)
            for action in range(n_actions):
                # New position
                if action == 0:   # sell
                    new_pos = max(0, pos - 1)
                elif action == 2:  # buy
                    new_pos = min(2, pos + 1)
                else:
                    new_pos = pos

                # Transaction cost
                tx_cost = 0.001 if action != 1 else 0.0

                for new_price in range(n_prices):
                    s_next = state_idx(new_pos, new_price)
                    T[s, action, s_next] = price_T[price, new_price]

                # Reward: position * price change expectation - tx cost
                expected_return = (new_pos - 1) * (price - 2) * 0.01
                R[s, action] = expected_return - tx_cost

    # Value iteration
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    n_iters = 100
    v_history = []

    for it in range(n_iters):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            q_vals = np.array([R[s, a] + gamma * T[s, a] @ V for a in range(n_actions)])
            V_new[s] = q_vals.max()
            policy[s] = q_vals.argmax()
        v_history.append(np.max(np.abs(V_new - V)))
        V = V_new

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Trading as an MDP: Value Iteration',
                 fontsize=14, fontweight='bold')

    # Plot 1: Value function heatmap
    ax = axes[0, 0]
    V_grid = V.reshape(n_positions, n_prices)
    im = ax.imshow(V_grid, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_prices))
    ax.set_xticklabels(price_names)
    ax.set_yticks(range(n_positions))
    ax.set_yticklabels(pos_names)
    ax.set_xlabel('Price Level')
    ax.set_ylabel('Position')
    ax.set_title('Optimal Value Function V*(s)')
    plt.colorbar(im, ax=ax)
    for i in range(n_positions):
        for j in range(n_prices):
            ax.text(j, i, f'{V_grid[i,j]:.3f}', ha='center', va='center', fontsize=8)

    # Plot 2: Optimal policy
    ax = axes[0, 1]
    action_labels = ['Sell', 'Hold', 'Buy']
    policy_grid = policy.reshape(n_positions, n_prices)
    colors_map = {0: '#F44336', 1: '#9E9E9E', 2: '#4CAF50'}
    for i in range(n_positions):
        for j in range(n_prices):
            a = policy_grid[i, j]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       color=colors_map[a], alpha=0.6))
            ax.text(j, i, action_labels[a], ha='center', va='center',
                    fontsize=9, fontweight='bold')
    ax.set_xlim(-0.5, n_prices - 0.5)
    ax.set_ylim(-0.5, n_positions - 0.5)
    ax.set_xticks(range(n_prices))
    ax.set_xticklabels(price_names)
    ax.set_yticks(range(n_positions))
    ax.set_yticklabels(pos_names)
    ax.set_xlabel('Price Level')
    ax.set_ylabel('Position')
    ax.set_title('Optimal Policy pi*(s)')
    ax.invert_yaxis()

    # Plot 3: Bellman error convergence
    ax = axes[1, 0]
    ax.semilogy(v_history, color='#2196F3', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('max |V_new - V_old|')
    ax.set_title('Value Iteration Convergence')
    ax.grid(True, alpha=0.3)

    # Plot 4: Simulate policy
    ax = axes[1, 1]
    np.random.seed(0)
    n_sim_steps = 100
    wealth = [1.0]
    position = 1  # start flat
    price = 2     # start mid

    for t in range(n_sim_steps):
        s = state_idx(position, price)
        action = policy[s]
        if action == 0:
            position = max(0, position - 1)
        elif action == 2:
            position = min(2, position + 1)
        price = np.random.choice(n_prices, p=price_T[price])
        daily_return = (position - 1) * (price - 2) * 0.005
        wealth.append(wealth[-1] * (1 + daily_return))

    ax.plot(wealth, color='#2196F3', linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Policy Simulation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/trading_mdp.png', dpi=150)
    plt.show()


# ============================================================
# Demo 2: Hidden Regime POMDP with Belief Tracking
# ============================================================

def demo_regime_pomdp():
    """
    Market has hidden regime (bull/bear/sideways).
    Agent observes noisy indicators and maintains belief.
    """
    np.random.seed(42)

    regime_T = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.85, 0.10],
        [0.10, 0.10, 0.80],
    ])

    # Observation model: P(obs | regime)
    obs_model = np.array([
        [0.70, 0.10, 0.20],   # bull: mostly positive signal
        [0.10, 0.70, 0.20],   # bear: mostly negative signal
        [0.20, 0.20, 0.60],   # sideways: mostly neutral
    ])
    obs_names = ['Positive', 'Negative', 'Neutral']
    regime_names = ['Bull', 'Bear', 'Sideways']

    n_days = 100
    belief = np.array([1/3, 1/3, 1/3])
    true_regime = 0

    beliefs_history = [belief.copy()]
    true_regimes = [true_regime]
    observations = []

    for t in range(n_days):
        # Regime transition
        true_regime = np.random.choice(3, p=regime_T[true_regime])
        true_regimes.append(true_regime)

        # Observation
        obs = np.random.choice(3, p=obs_model[true_regime])
        observations.append(obs)

        # Bayesian belief update
        # Predict: b' = T^T @ b
        predicted = regime_T.T @ belief
        # Update: b'' = P(o|s) * b' / normalizer
        updated = obs_model[:, obs] * predicted
        belief = updated / updated.sum()
        beliefs_history.append(belief.copy())

    beliefs_history = np.array(beliefs_history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hidden Market Regime: POMDP Belief Tracking',
                 fontsize=14, fontweight='bold')

    # Plot 1: Belief over time
    ax = axes[0, 0]
    colors = ['#4CAF50', '#F44336', '#FF9800']
    for r in range(3):
        ax.plot(beliefs_history[:, r], color=colors[r], linewidth=1.5,
                label=regime_names[r], alpha=0.8)
    # Shade true regime
    for t in range(n_days):
        ax.axvspan(t, t+1, color=colors[true_regimes[t+1]], alpha=0.05)
    ax.set_xlabel('Day')
    ax.set_ylabel('Belief Probability')
    ax.set_title('Belief Evolution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Belief on simplex (2D projection)
    ax = axes[0, 1]
    # Simplex corners in 2D
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    for i, name in enumerate(regime_names):
        ax.plot(*corners[i], 'o', color=colors[i], markersize=12)
        offset = [(-0.08, -0.05), (0.03, -0.05), (0.03, 0.03)]
        ax.annotate(name, corners[i] + offset[i], fontsize=10,
                    fontweight='bold', color=colors[i])

    # Draw simplex edges
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]],
                'k-', alpha=0.3)

    # Project beliefs onto simplex
    xy = beliefs_history @ corners
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=range(len(xy)),
                        cmap='viridis', s=10, alpha=0.6)
    ax.plot(xy[0, 0], xy[0, 1], 'r*', markersize=15, label='Start')
    plt.colorbar(scatter, ax=ax, label='Day')
    ax.set_title('Belief Trajectory on Simplex')
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')

    # Plot 3: Observation accuracy
    ax = axes[1, 0]
    correct = 0
    map_regime = np.argmax(beliefs_history[1:], axis=1)
    accuracy_running = []
    for t in range(n_days):
        correct += int(map_regime[t] == true_regimes[t+1])
        accuracy_running.append(correct / (t + 1))
    ax.plot(accuracy_running, color='#2196F3', linewidth=2)
    ax.axhline(1/3, color='gray', linestyle='--', label='Random guess', alpha=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Accuracy')
    ax.set_title('MAP Regime Estimation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Belief entropy
    ax = axes[1, 1]
    entropy = -np.sum(beliefs_history[1:] * np.log(beliefs_history[1:] + 1e-10), axis=1)
    ax.plot(entropy, color='#9C27B0', linewidth=1.5, alpha=0.7)
    ax.axhline(np.log(3), color='gray', linestyle='--', label='Max entropy', alpha=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Belief Entropy (nats)')
    ax.set_title('Uncertainty Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/regime_pomdp.png', dpi=150)
    plt.show()


# ============================================================
# Demo 3: MCTS for Trade Planning
# ============================================================

def demo_mcts_trading():
    """
    Use MCTS to plan trades with lookahead.
    Compare to greedy and random baselines.
    """
    np.random.seed(42)

    class TradingMCTSNode:
        def __init__(self, wealth, position, parent=None):
            self.wealth = wealth
            self.position = position  # -1, 0, 1
            self.parent = parent
            self.children = {}
            self.visits = 0
            self.value = 0.0

    def mcts_simulate(wealth, position, depth, max_depth, rng):
        """Random rollout from state."""
        total = 0
        gamma = 0.95
        factor = 1.0
        for d in range(depth, max_depth):
            action = rng.choice(3)  # sell/hold/buy
            new_pos = position + (action - 1)
            new_pos = max(-1, min(1, new_pos))
            daily_ret = rng.normal(0.001 * new_pos, 0.02)
            wealth *= (1 + daily_ret)
            total += factor * daily_ret
            factor *= gamma
            position = new_pos
        return total

    def mcts_plan(wealth, position, n_sims=200, max_depth=10, c=1.4):
        """Simple MCTS planner for trading."""
        rng = np.random.RandomState()
        root = TradingMCTSNode(wealth, position)
        root.visits = 1

        for _ in range(n_sims):
            node = root
            # Selection: UCB at root
            if root.children:
                best_a = max(root.children.keys(),
                             key=lambda a: (root.children[a].value / (root.children[a].visits + 1e-8)
                                           + c * math.sqrt(math.log(root.visits) / (root.children[a].visits + 1e-8))))
                node = root.children[best_a]
                action = best_a
            else:
                action = rng.choice(3)

            # Expansion
            if action not in root.children:
                new_pos = position + (action - 1)
                new_pos = max(-1, min(1, new_pos))
                root.children[action] = TradingMCTSNode(wealth, new_pos, root)

            child = root.children[action]
            new_pos = position + (action - 1)
            new_pos = max(-1, min(1, new_pos))

            # Rollout
            val = mcts_simulate(wealth, new_pos, 1, max_depth, rng)

            # Backprop
            child.visits += 1
            child.value += val
            root.visits += 1

        # Best action
        best_a = max(root.children.keys(),
                     key=lambda a: root.children[a].visits)
        return best_a

    # Simulate episodes
    n_days = 100
    n_episodes = 20

    strategies = {
        'MCTS (200 sims)': lambda w, p: mcts_plan(w, p, n_sims=200),
        'Greedy (always long)': lambda w, p: 2,  # always buy
        'Random': lambda w, p: np.random.choice(3),
        'Hold': lambda w, p: 1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MCTS for Trade Planning: Lookahead vs Baselines',
                 fontsize=14, fontweight='bold')

    all_final_wealth = {}
    colors = {'MCTS (200 sims)': '#2196F3', 'Greedy (always long)': '#4CAF50',
              'Random': '#FF9800', 'Hold': '#9E9E9E'}

    for name, strategy in strategies.items():
        final_wealths = []
        sample_trajectory = None

        for ep in range(n_episodes):
            np.random.seed(ep * 100)
            wealth = 1.0
            position = 0
            trajectory = [wealth]

            for day in range(n_days):
                action = strategy(wealth, position)
                new_pos = position + (action - 1)
                new_pos = max(-1, min(1, new_pos))
                daily_ret = np.random.normal(0.0005, 0.015)
                wealth *= (1 + daily_ret * new_pos)
                position = new_pos
                trajectory.append(wealth)

            final_wealths.append(wealth)
            if ep == 0:
                sample_trajectory = trajectory

        all_final_wealth[name] = final_wealths

        # Plot sample trajectory
        axes[0].plot(sample_trajectory, color=colors[name], linewidth=1.5,
                     label=name, alpha=0.8)

    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].set_title('Sample Trajectories')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1.0, color='black', linestyle=':', alpha=0.3)

    # Box plot comparison
    ax = axes[1]
    data = [all_final_wealth[name] for name in strategies]
    bp = ax.boxplot(data, labels=[n.split('(')[0].strip() for n in strategies],
                    patch_artist=True)
    for patch, name in zip(bp['boxes'], strategies):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Wealth')
    ax.set_title('Final Wealth Distribution')
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.3)

    # Sharpe ratio comparison
    ax = axes[2]
    sharpe_ratios = {}
    for name, wealths in all_final_wealth.items():
        returns = np.array(wealths) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        sharpe_ratios[name] = sharpe

    bars = ax.barh(range(len(sharpe_ratios)),
                   list(sharpe_ratios.values()),
                   color=[colors[n] for n in sharpe_ratios],
                   edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(sharpe_ratios)))
    ax.set_yticklabels([n.split('(')[0].strip() for n in sharpe_ratios], fontsize=8)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Risk-Adjusted Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/mcts_trading.png', dpi=150)
    plt.show()


# ============================================================
# Demo 4: MPC Rolling-Horizon Portfolio Rebalancing
# ============================================================

def demo_mpc_portfolio():
    """
    Model Predictive Control for portfolio rebalancing.
    At each step, solve a short-horizon optimization and apply first action.
    """
    np.random.seed(42)

    n_assets = 3
    asset_names = ['Stocks', 'Bonds', 'Cash']
    n_days = 200
    horizon = 10  # MPC planning horizon

    # Expected returns and covariance (assumed known)
    mu = np.array([0.0008, 0.0003, 0.0001])
    cov = np.array([
        [0.0004, 0.0001, 0.0],
        [0.0001, 0.0001, 0.0],
        [0.0,    0.0,    0.00001],
    ])

    def mpc_rebalance(weights, mu, cov, horizon, risk_aversion=2.0):
        """Simple mean-variance MPC: solve for target weights."""
        # Greedy mean-variance: w* = argmax mu^T w - (lambda/2) w^T Sigma w
        # Analytical solution for unconstrained: w* = (1/lambda) Sigma^{-1} mu
        inv_cov = np.linalg.inv(cov * horizon)
        w_star = inv_cov @ (mu * horizon) / risk_aversion
        # Project to simplex (normalize to sum=1, clip negatives)
        w_star = np.maximum(w_star, 0)
        if w_star.sum() > 0:
            w_star /= w_star.sum()
        else:
            w_star = np.ones(len(mu)) / len(mu)
        return w_star

    # Compare MPC vs fixed allocation vs equal weight
    strategies = {
        'MPC (h=10, lambda=2)': {'risk_aversion': 2.0, 'horizon': 10},
        'MPC (h=10, lambda=5)': {'risk_aversion': 5.0, 'horizon': 10},
        'MPC (h=5, lambda=2)':  {'risk_aversion': 2.0, 'horizon': 5},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MPC: Rolling-Horizon Portfolio Rebalancing',
                 fontsize=14, fontweight='bold')

    colors_strat = ['#2196F3', '#F44336', '#4CAF50']
    all_weights = {}

    for idx, (name, params) in enumerate(strategies.items()):
        np.random.seed(42)
        wealth = 1.0
        weights = np.ones(n_assets) / n_assets
        wealth_history = [wealth]
        weight_history = [weights.copy()]

        for day in range(n_days):
            # MPC: rebalance
            target = mpc_rebalance(weights, mu, cov, params['horizon'],
                                   params['risk_aversion'])
            # Smooth transition (don't fully rebalance in one step)
            weights = 0.3 * target + 0.7 * weights
            weights /= weights.sum()

            # Simulate market
            returns = np.random.multivariate_normal(mu, cov)
            wealth *= (1 + weights @ returns)
            wealth_history.append(wealth)
            weight_history.append(weights.copy())

        all_weights[name] = np.array(weight_history)

        axes[0, 0].plot(wealth_history, color=colors_strat[idx],
                        linewidth=1.5, label=name)

    # Equal weight benchmark
    np.random.seed(42)
    wealth = 1.0
    eq_hist = [wealth]
    for day in range(n_days):
        returns = np.random.multivariate_normal(mu, cov)
        wealth *= (1 + np.ones(n_assets) / n_assets @ returns)
        eq_hist.append(wealth)
    axes[0, 0].plot(eq_hist, color='gray', linewidth=1.5, linestyle='--',
                    label='Equal Weight')

    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].set_title('Wealth Trajectories')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Weight evolution for first MPC strategy
    ax = axes[0, 1]
    first_name = list(strategies.keys())[0]
    w = all_weights[first_name]
    asset_colors = ['#2196F3', '#FF9800', '#4CAF50']
    ax.stackplot(range(len(w)), w[:, 0], w[:, 1], w[:, 2],
                 labels=asset_names, colors=asset_colors, alpha=0.7)
    ax.set_xlabel('Day')
    ax.set_ylabel('Weight')
    ax.set_title(f'Portfolio Weights ({first_name})')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Turnover (weight changes)
    ax = axes[1, 0]
    for idx, (name, w_hist) in enumerate(all_weights.items()):
        turnover = np.sum(np.abs(np.diff(w_hist, axis=0)), axis=1)
        cum_turnover = np.cumsum(turnover)
        ax.plot(cum_turnover, color=colors_strat[idx], linewidth=1.5, label=name)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Turnover')
    ax.set_title('Trading Activity')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Risk-return scatter
    ax = axes[1, 1]
    for idx, (name, params) in enumerate(strategies.items()):
        np.random.seed(42)
        wealth = 1.0
        weights = np.ones(n_assets) / n_assets
        daily_returns = []
        for day in range(n_days):
            target = mpc_rebalance(weights, mu, cov, params['horizon'],
                                   params['risk_aversion'])
            weights = 0.3 * target + 0.7 * weights
            weights /= weights.sum()
            returns = np.random.multivariate_normal(mu, cov)
            dr = weights @ returns
            daily_returns.append(dr)
            wealth *= (1 + dr)

        ann_ret = np.mean(daily_returns) * 252
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        ax.scatter(ann_vol, ann_ret, color=colors_strat[idx], s=100,
                   zorder=5, edgecolors='black', label=name)

    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Risk-Return Tradeoff')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/mpc_portfolio.png', dpi=150)
    plt.show()


# ============================================================
# Demo 5: Model-Based RL — Learning Market Dynamics (Dyna-Q)
# ============================================================

def demo_dyna_trading():
    """
    Dyna-Q for trading: learn transition model from experience,
    then plan using the learned model.
    """
    np.random.seed(42)

    # Discretized market: 5 price levels, 3 positions
    n_prices = 5
    n_positions = 3  # short/flat/long
    n_states = n_prices * n_positions
    n_actions = 3    # sell/hold/buy

    # True transition probabilities (unknown to agent)
    true_price_T = np.array([
        [0.3, 0.4, 0.2, 0.1, 0.0],
        [0.1, 0.3, 0.3, 0.2, 0.1],
        [0.05, 0.15, 0.4, 0.25, 0.15],
        [0.0, 0.1, 0.2, 0.4, 0.3],
        [0.0, 0.05, 0.15, 0.3, 0.5],
    ])

    def get_state(pos, price):
        return pos * n_prices + price

    def from_state(s):
        return s // n_prices, s % n_prices

    def env_step(state, action):
        pos, price = from_state(state)
        new_pos = max(0, min(2, pos + action - 1))
        new_price = np.random.choice(n_prices, p=true_price_T[price])
        reward = (new_pos - 1) * (new_price - 2) * 0.01
        if action != 1:
            reward -= 0.001
        return get_state(new_pos, new_price), reward

    # Dyna-Q agent
    class DynaQTrader:
        def __init__(self, n_states, n_actions, n_planning=0, alpha=0.1,
                     gamma=0.95, epsilon=0.1):
            self.Q = np.zeros((n_states, n_actions))
            self.model = {}  # (s, a) -> [(s', r)]
            self.n_planning = n_planning
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon

        def act(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(n_actions)
            return np.argmax(self.Q[state])

        def learn(self, s, a, r, s_next):
            # Direct RL
            td = r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a]
            self.Q[s, a] += self.alpha * td

            # Store in model
            if (s, a) not in self.model:
                self.model[(s, a)] = []
            self.model[(s, a)].append((s_next, r))

            # Planning (Dyna)
            if self.n_planning > 0 and self.model:
                keys = list(self.model.keys())
                for _ in range(self.n_planning):
                    s_m, a_m = keys[np.random.randint(len(keys))]
                    s_next_m, r_m = self.model[(s_m, a_m)][
                        np.random.randint(len(self.model[(s_m, a_m)]))]
                    td_m = r_m + self.gamma * np.max(self.Q[s_next_m]) - self.Q[s_m, a_m]
                    self.Q[s_m, a_m] += self.alpha * td_m

    # Compare different planning steps
    planning_steps = [0, 5, 20, 50]
    n_episodes = 300
    ep_length = 50

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dyna-Q for Trading: Model-Based RL',
                 fontsize=14, fontweight='bold')

    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']

    for idx, n_plan in enumerate(planning_steps):
        np.random.seed(42)
        agent = DynaQTrader(n_states, n_actions, n_planning=n_plan)
        episode_rewards = []

        for ep in range(n_episodes):
            state = get_state(1, 2)  # flat, mid price
            total_reward = 0
            for t in range(ep_length):
                action = agent.act(state)
                next_state, reward = env_step(state, action)
                agent.learn(state, action, reward, next_state)
                total_reward += reward
                state = next_state
            episode_rewards.append(total_reward)

        # Smooth rewards
        window = 20
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(smoothed, color=colors[idx], linewidth=1.5,
                        label=f'n_plan={n_plan}')

        # Model coverage
        coverage = len(agent.model) / (n_states * n_actions)
        axes[0, 1].bar(idx, coverage, color=colors[idx], alpha=0.8,
                       edgecolor='black')

    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Smoothed Episode Reward')
    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xticks(range(len(planning_steps)))
    axes[0, 1].set_xticklabels([f'n={n}' for n in planning_steps])
    axes[0, 1].set_ylabel('Model Coverage')
    axes[0, 1].set_title('State-Action Pairs Visited')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot learned vs true model for best agent
    np.random.seed(42)
    best_agent = DynaQTrader(n_states, n_actions, n_planning=50)
    for ep in range(n_episodes):
        state = get_state(1, 2)
        for t in range(ep_length):
            action = best_agent.act(state)
            next_state, reward = env_step(state, action)
            best_agent.learn(state, action, reward, next_state)
            state = next_state

    # Learned Q-values heatmap
    ax = axes[1, 0]
    Q_best = best_agent.Q.reshape(n_positions, n_prices, n_actions)
    # Show max Q across actions
    V_learned = Q_best.max(axis=2)
    im = ax.imshow(V_learned, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_prices))
    ax.set_xticklabels(['VLow', 'Low', 'Mid', 'High', 'VHigh'])
    ax.set_yticks(range(n_positions))
    ax.set_yticklabels(['Short', 'Flat', 'Long'])
    ax.set_xlabel('Price Level')
    ax.set_ylabel('Position')
    ax.set_title('Learned Value Function')
    plt.colorbar(im, ax=ax)

    # Sample efficiency
    ax = axes[1, 1]
    for idx, n_plan in enumerate(planning_steps):
        np.random.seed(42)
        agent = DynaQTrader(n_states, n_actions, n_planning=n_plan)
        total_steps = 0
        rewards_per_step = []
        cum_reward = 0
        for ep in range(n_episodes):
            state = get_state(1, 2)
            for t in range(ep_length):
                action = agent.act(state)
                next_state, reward = env_step(state, action)
                agent.learn(state, action, reward, next_state)
                total_steps += 1
                cum_reward += reward
                if total_steps % 100 == 0:
                    rewards_per_step.append(cum_reward / total_steps)
                state = next_state

        ax.plot(range(len(rewards_per_step)), rewards_per_step,
                color=colors[idx], linewidth=1.5, label=f'n_plan={n_plan}')

    ax.set_xlabel('Steps (x100)')
    ax.set_ylabel('Avg Reward per Step')
    ax.set_title('Sample Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/dyna_trading.png', dpi=150)
    plt.show()


# ============================================================
# Demo 6: Safe RL — CVaR-Constrained Portfolio
# ============================================================

def demo_safe_portfolio():
    """
    Compare risk-neutral vs CVaR-constrained portfolio allocation.
    Shows VaR, CVaR, tail risk, and the efficient frontier.
    """
    np.random.seed(42)

    n_assets = 2
    n_scenarios = 10000

    # Asset parameters
    mu = np.array([0.08, 0.03])  # annual returns
    sigma = np.array([0.20, 0.05])  # annual volatility
    rho = 0.3
    cov = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2],
    ])

    # Generate scenarios
    returns = np.random.multivariate_normal(mu / 252, cov / 252, n_scenarios)

    # Compute portfolio metrics for different weights
    weights_range = np.linspace(0, 1, 101)  # weight in stocks
    port_means = []
    port_stds = []
    port_vars = []
    port_cvars = []

    alpha = 0.05  # 5% CVaR

    for w in weights_range:
        weights = np.array([w, 1 - w])
        port_ret = returns @ weights
        port_means.append(port_ret.mean() * 252)
        port_stds.append(port_ret.std() * np.sqrt(252))

        sorted_ret = np.sort(port_ret)
        var_idx = int(alpha * n_scenarios)
        var_val = sorted_ret[var_idx]
        cvar_val = sorted_ret[:var_idx].mean()
        port_vars.append(-var_val * np.sqrt(252))
        port_cvars.append(-cvar_val * np.sqrt(252))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Safe RL: CVaR-Constrained Portfolio Allocation',
                 fontsize=14, fontweight='bold')

    # Plot 1: Efficient frontier with CVaR overlay
    ax = axes[0, 0]
    sc = ax.scatter(port_stds, port_means, c=port_cvars, cmap='RdYlGn_r',
                    s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='CVaR (annual)')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Efficient Frontier (colored by CVaR)')
    ax.grid(True, alpha=0.3)

    # Mark optimal points
    max_sharpe_idx = np.argmax(np.array(port_means) / (np.array(port_stds) + 1e-8))
    min_cvar_idx = np.argmin(port_cvars)
    ax.scatter(port_stds[max_sharpe_idx], port_means[max_sharpe_idx],
               marker='*', s=200, color='gold', zorder=5, edgecolors='black',
               label=f'Max Sharpe (w={weights_range[max_sharpe_idx]:.0%})')
    ax.scatter(port_stds[min_cvar_idx], port_means[min_cvar_idx],
               marker='D', s=100, color='red', zorder=5, edgecolors='black',
               label=f'Min CVaR (w={weights_range[min_cvar_idx]:.0%})')
    ax.legend(fontsize=8)

    # Plot 2: VaR and CVaR vs stock weight
    ax = axes[0, 1]
    ax.plot(weights_range, port_vars, color='#FF9800', linewidth=2, label='VaR (5%)')
    ax.plot(weights_range, port_cvars, color='#F44336', linewidth=2, label='CVaR (5%)')
    ax.fill_between(weights_range, port_vars, port_cvars, alpha=0.2, color='#F44336')
    ax.set_xlabel('Stock Weight')
    ax.set_ylabel('Risk Measure (annualized)')
    ax.set_title('VaR vs CVaR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Return distribution for two portfolios
    ax = axes[1, 0]
    risky_w = np.array([0.9, 0.1])
    safe_w = np.array([0.3, 0.7])
    risky_ret = returns @ risky_w * 252
    safe_ret = returns @ safe_w * 252

    ax.hist(risky_ret, bins=80, alpha=0.5, color='#F44336', density=True,
            label='90% Stocks')
    ax.hist(safe_ret, bins=80, alpha=0.5, color='#4CAF50', density=True,
            label='30% Stocks')

    # Mark CVaR regions
    risky_sorted = np.sort(risky_ret)
    risky_var = risky_sorted[int(0.05 * len(risky_sorted))]
    ax.axvline(risky_var, color='#F44336', linestyle='--', linewidth=2)
    safe_sorted = np.sort(safe_ret)
    safe_var = safe_sorted[int(0.05 * len(safe_sorted))]
    ax.axvline(safe_var, color='#4CAF50', linestyle='--', linewidth=2)

    ax.set_xlabel('Annualized Return')
    ax.set_ylabel('Density')
    ax.set_title('Return Distributions & 5% VaR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: CVaR-constrained allocation over time
    ax = axes[1, 1]
    np.random.seed(42)
    n_days = 200
    cvar_budget = 0.15  # max allowed CVaR

    wealth_unconstrained = [1.0]
    wealth_constrained = [1.0]
    stock_weight_history = []

    w_stock = 0.5
    for day in range(n_days):
        day_ret = np.random.multivariate_normal(mu / 252, cov / 252)

        # Unconstrained: fixed 80% stocks
        wealth_unconstrained.append(
            wealth_unconstrained[-1] * (1 + np.array([0.8, 0.2]) @ day_ret))

        # CVaR-constrained: reduce stock weight if rolling CVaR exceeds budget
        w = np.array([w_stock, 1 - w_stock])
        wealth_constrained.append(wealth_constrained[-1] * (1 + w @ day_ret))

        # Estimate rolling CVaR and adjust
        if day > 20:
            window_ret = np.diff(np.log(wealth_constrained[-21:]))
            sorted_w = np.sort(window_ret)
            rolling_cvar = -sorted_w[:max(1, int(0.05 * len(sorted_w)))].mean() * np.sqrt(252)
            if rolling_cvar > cvar_budget:
                w_stock = max(0.1, w_stock - 0.05)
            elif rolling_cvar < cvar_budget * 0.8:
                w_stock = min(0.9, w_stock + 0.02)

        stock_weight_history.append(w_stock)

    ax.plot(wealth_unconstrained, color='#F44336', linewidth=1.5,
            label='Unconstrained (80% stocks)')
    ax.plot(wealth_constrained, color='#4CAF50', linewidth=1.5,
            label='CVaR-constrained')

    ax2 = ax.twinx()
    ax2.plot(stock_weight_history, color='gray', linewidth=1, alpha=0.5,
             linestyle='--', label='Stock weight')
    ax2.set_ylabel('Stock Weight', color='gray')
    ax2.set_ylim(0, 1)

    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Dynamic Risk Management')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/safe_portfolio.png', dpi=150)
    plt.show()


# ============================================================
# Demo 7: Bayesian RL — Thompson Sampling for Asset Selection
# ============================================================

def demo_bayesian_trading():
    """
    Thompson Sampling vs UCB vs epsilon-greedy for choosing
    which asset to allocate capital to, with unknown return distributions.
    """
    np.random.seed(42)

    # True asset returns (unknown to agent)
    true_means = [0.05, 0.08, 0.03, 0.06, 0.10]
    true_stds = [0.15, 0.25, 0.05, 0.10, 0.30]
    n_assets = len(true_means)
    asset_names = [f'Asset {i+1}' for i in range(n_assets)]

    T = 500
    n_trials = 50

    class ThompsonSampler:
        """Normal-Gamma posterior for Gaussian rewards."""
        def __init__(self, k):
            self.k = k
            self.mu0 = np.zeros(k)
            self.lambda0 = np.ones(k)
            self.alpha0 = np.ones(k)
            self.beta0 = np.ones(k)
            self.n = np.zeros(k)
            self.sum_x = np.zeros(k)
            self.sum_x2 = np.zeros(k)

        def select(self):
            samples = np.zeros(self.k)
            for i in range(self.k):
                # Posterior parameters
                n = self.n[i]
                lam = self.lambda0[i] + n
                mu = (self.lambda0[i] * self.mu0[i] + self.sum_x[i]) / lam
                alpha = self.alpha0[i] + n / 2
                beta = (self.beta0[i] + 0.5 * (self.sum_x2[i] - self.sum_x[i]**2 / max(n, 1))
                        + self.lambda0[i] * n * (self.sum_x[i] / max(n, 1) - self.mu0[i])**2 / (2 * lam))

                # Sample precision then mean
                tau = np.random.gamma(alpha, 1.0 / max(beta, 1e-8))
                samples[i] = np.random.normal(mu, 1.0 / np.sqrt(max(lam * tau, 1e-8)))
            return np.argmax(samples)

        def update(self, arm, reward):
            self.n[arm] += 1
            self.sum_x[arm] += reward
            self.sum_x2[arm] += reward ** 2

    class UCBAgent:
        def __init__(self, k, c=2.0):
            self.k = k
            self.c = c
            self.Q = np.zeros(k)
            self.N = np.zeros(k)
            self.total = 0

        def select(self):
            self.total += 1
            for i in range(self.k):
                if self.N[i] == 0:
                    return i
            return np.argmax(self.Q + self.c * np.sqrt(np.log(self.total) / self.N))

        def update(self, arm, reward):
            self.N[arm] += 1
            self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    class EpsGreedy:
        def __init__(self, k, eps=0.1):
            self.k = k
            self.eps = eps
            self.Q = np.zeros(k)
            self.N = np.zeros(k)

        def select(self):
            if np.random.random() < self.eps:
                return np.random.randint(self.k)
            return np.argmax(self.Q)

        def update(self, arm, reward):
            self.N[arm] += 1
            self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    agents = {
        'Thompson Sampling': lambda: ThompsonSampler(n_assets),
        'UCB (c=2)': lambda: UCBAgent(n_assets, c=2.0),
        'Epsilon-Greedy (0.1)': lambda: EpsGreedy(n_assets, eps=0.1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bayesian RL: Thompson Sampling for Asset Selection',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    best_arm = np.argmax(true_means)

    for idx, (name, agent_fn) in enumerate(agents.items()):
        cum_regrets = np.zeros((n_trials, T))
        optimal_pcts = np.zeros((n_trials, T))
        final_allocations = np.zeros((n_trials, n_assets))

        for trial in range(n_trials):
            np.random.seed(trial * 100)
            agent = agent_fn()
            cum_regret = 0

            for t in range(T):
                arm = agent.select()
                reward = np.random.normal(true_means[arm], true_stds[arm])
                agent.update(arm, reward)
                cum_regret += true_means[best_arm] - true_means[arm]
                cum_regrets[trial, t] = cum_regret
                optimal_pcts[trial, t] = float(arm == best_arm)

            if hasattr(agent, 'N'):
                final_allocations[trial] = agent.N / agent.N.sum()
            else:
                final_allocations[trial] = agent.n / agent.n.sum()

        # Plot 1: Cumulative regret
        mean_regret = cum_regrets.mean(axis=0)
        axes[0, 0].plot(mean_regret, color=colors[idx], linewidth=2, label=name)

        # Plot 2: Optimal action rate
        window = 30
        mean_opt = optimal_pcts.mean(axis=0)
        smoothed = np.convolve(mean_opt, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed, color=colors[idx], linewidth=2, label=name)

        # Final allocation
        axes[1, 1].bar(np.arange(n_assets) + idx * 0.25,
                       final_allocations.mean(axis=0), 0.25,
                       color=colors[idx], alpha=0.8, label=name)

    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].set_title('Regret Comparison')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('% Optimal Asset')
    axes[0, 1].set_title('Optimal Selection Rate')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # Plot 3: Posterior evolution for Thompson Sampling
    ax = axes[1, 0]
    np.random.seed(42)
    ts = ThompsonSampler(n_assets)
    snapshots = [0, 10, 50, 200]
    snap_idx = 0

    for t in range(201):
        if t in snapshots:
            x_range = np.linspace(-0.3, 0.3, 200)
            for arm in range(n_assets):
                n = max(ts.n[arm], 1)
                lam = ts.lambda0[arm] + n
                mu = (ts.lambda0[arm] * ts.mu0[arm] + ts.sum_x[arm]) / lam
                std = 1.0 / np.sqrt(lam * max(n, 1))
                pdf = np.exp(-0.5 * ((x_range - mu) / max(std, 1e-6))**2) / (max(std, 1e-6) * np.sqrt(2 * np.pi))
                ax.plot(x_range, pdf + snap_idx * 15, alpha=0.5, linewidth=1.5,
                        label=f'A{arm+1}' if snap_idx == 0 else None)
            ax.axhline(snap_idx * 15, color='gray', linestyle=':', alpha=0.3)
            ax.text(-0.28, snap_idx * 15 + 12, f't={t}', fontsize=8,
                    fontweight='bold')
            snap_idx += 1

        arm = ts.select()
        reward = np.random.normal(true_means[arm], true_stds[arm])
        ts.update(arm, reward)

    ax.set_xlabel('Return')
    ax.set_ylabel('Density (stacked)')
    ax.set_title('Posterior Evolution (Thompson)')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Final allocation bar chart
    axes[1, 1].set_xticks(np.arange(n_assets) + 0.25)
    axes[1, 1].set_xticklabels(asset_names, fontsize=8)
    axes[1, 1].axvline(best_arm, color='gold', linewidth=2, linestyle='--',
                       alpha=0.5, label='Best asset')
    axes[1, 1].set_ylabel('Allocation Fraction')
    axes[1, 1].set_title('Final Capital Allocation')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/bayesian_trading.png', dpi=150)
    plt.show()


# ============================================================
# Demo 8: Exploration Strategies for Portfolio Diversification
# ============================================================

def demo_exploration_portfolio():
    """
    Compare exploration strategies for discovering profitable assets
    in a non-stationary market environment.
    """
    np.random.seed(42)

    # Non-stationary: asset means shift halfway
    n_assets = 6
    T = 1000
    n_trials = 40

    def get_mean(arm, t):
        """Non-stationary means: regime shift at t=500."""
        base = [0.02, 0.05, 0.08, 0.03, 0.06, 0.04]
        if t < 500:
            return base[arm]
        else:
            shifted = [0.07, 0.02, 0.03, 0.08, 0.01, 0.09]
            return shifted[arm]

    class SlidingWindowEpsGreedy:
        def __init__(self, k, eps=0.1, window=100):
            self.k = k
            self.eps = eps
            self.window = window
            self.history = []  # (arm, reward) pairs

        def select(self):
            if np.random.random() < self.eps or not self.history:
                return np.random.randint(self.k)
            recent = self.history[-self.window:]
            means = np.zeros(self.k)
            counts = np.zeros(self.k)
            for a, r in recent:
                means[a] += r
                counts[a] += 1
            means = np.divide(means, counts, where=counts > 0)
            return np.argmax(means)

        def update(self, arm, reward):
            self.history.append((arm, reward))

    class SlidingWindowUCB:
        def __init__(self, k, c=2.0, window=100):
            self.k = k
            self.c = c
            self.window = window
            self.history = []
            self.total = 0

        def select(self):
            self.total += 1
            recent = self.history[-self.window:]
            counts = np.zeros(self.k)
            means = np.zeros(self.k)
            for a, r in recent:
                means[a] += r
                counts[a] += 1

            for i in range(self.k):
                if counts[i] == 0:
                    return i
            means = np.divide(means, counts, where=counts > 0)
            n_total = sum(counts)
            ucb = means + self.c * np.sqrt(np.log(n_total + 1) / (counts + 1e-8))
            return np.argmax(ucb)

        def update(self, arm, reward):
            self.history.append((arm, reward))

    class DecayingEpsGreedy:
        def __init__(self, k, eps0=0.3, decay=0.995):
            self.k = k
            self.eps0 = eps0
            self.decay = decay
            self.Q = np.zeros(k)
            self.N = np.zeros(k)
            self.t = 0

        def select(self):
            self.t += 1
            if np.random.random() < self.eps0 * (self.decay ** self.t):
                return np.random.randint(self.k)
            return np.argmax(self.Q)

        def update(self, arm, reward):
            self.N[arm] += 1
            alpha = 0.1  # constant step size for non-stationary
            self.Q[arm] += alpha * (reward - self.Q[arm])

    strategies = {
        'SW eps-greedy': lambda: SlidingWindowEpsGreedy(n_assets, eps=0.15, window=100),
        'SW UCB': lambda: SlidingWindowUCB(n_assets, c=1.5, window=100),
        'Decaying eps': lambda: DecayingEpsGreedy(n_assets, eps0=0.3, decay=0.995),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploration Strategies for Non-Stationary Asset Selection',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#F44336', '#4CAF50']

    # Plot 1: True means over time
    ax = axes[0, 0]
    for arm in range(n_assets):
        means_over_time = [get_mean(arm, t) for t in range(T)]
        ax.plot(means_over_time, linewidth=1.5, alpha=0.7, label=f'Asset {arm+1}')
    ax.axvline(500, color='black', linestyle='--', linewidth=2, alpha=0.5,
               label='Regime shift')
    ax.set_xlabel('Step')
    ax.set_ylabel('True Mean Return')
    ax.set_title('Non-Stationary Asset Returns')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    for idx, (name, agent_fn) in enumerate(strategies.items()):
        cum_regrets = np.zeros((n_trials, T))
        arm_selections = np.zeros((n_trials, T), dtype=int)

        for trial in range(n_trials):
            np.random.seed(trial * 100)
            agent = agent_fn()
            cum_regret = 0
            for t in range(T):
                arm = agent.select()
                arm_selections[trial, t] = arm
                best_mean = max(get_mean(a, t) for a in range(n_assets))
                reward = np.random.normal(get_mean(arm, t), 0.1)
                agent.update(arm, reward)
                cum_regret += best_mean - get_mean(arm, t)
                cum_regrets[trial, t] = cum_regret

        # Plot 2: Cumulative regret
        mean_regret = cum_regrets.mean(axis=0)
        axes[0, 1].plot(mean_regret, color=colors[idx], linewidth=2, label=name)

        # Plot 3: Arm selection heatmap (use first trial)
        if idx == 0:
            ax = axes[1, 0]
            window = 20
            selection_freq = np.zeros((n_assets, T - window))
            for t in range(T - window):
                for a in range(n_assets):
                    selection_freq[a, t] = np.mean(arm_selections[0, t:t+window] == a)
            im = ax.imshow(selection_freq, aspect='auto', cmap='hot',
                           interpolation='nearest')
            ax.set_yticks(range(n_assets))
            ax.set_yticklabels([f'Asset {i+1}' for i in range(n_assets)])
            ax.set_xlabel('Step')
            ax.set_title(f'Arm Selection Heatmap ({name})')
            ax.axvline(500, color='cyan', linewidth=2, linestyle='--')
            plt.colorbar(im, ax=ax, label='Selection freq')

    axes[0, 1].axvline(500, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cumulative Regret')
    axes[0, 1].set_title('Regret Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 4: Adaptation speed after regime shift
    ax = axes[1, 1]
    for idx, (name, agent_fn) in enumerate(strategies.items()):
        # Track optimal arm selection rate after shift
        post_shift_optimal = np.zeros(T - 500)
        for trial in range(n_trials):
            np.random.seed(trial * 100)
            agent = agent_fn()
            for t in range(T):
                arm = agent.select()
                reward = np.random.normal(get_mean(arm, t), 0.1)
                agent.update(arm, reward)
                if t >= 500:
                    best_arm_t = max(range(n_assets), key=lambda a: get_mean(a, t))
                    post_shift_optimal[t - 500] += float(arm == best_arm_t)
        post_shift_optimal /= n_trials
        window = 20
        smoothed = np.convolve(post_shift_optimal, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color=colors[idx], linewidth=2, label=name)

    ax.set_xlabel('Steps After Regime Shift')
    ax.set_ylabel('Optimal Selection Rate')
    ax.set_title('Adaptation After Regime Change')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/exploration_portfolio.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs('stock_trading', exist_ok=True)

    print("=" * 60)
    print("Demo 1: Trading as an MDP (Value Iteration)")
    print("=" * 60)
    demo_trading_mdp()

    print("\n" + "=" * 60)
    print("Demo 2: Hidden Regime POMDP (Belief Tracking)")
    print("=" * 60)
    demo_regime_pomdp()

    print("\n" + "=" * 60)
    print("Demo 3: MCTS for Trade Planning")
    print("=" * 60)
    demo_mcts_trading()

    print("\n" + "=" * 60)
    print("Demo 4: MPC Rolling-Horizon Portfolio")
    print("=" * 60)
    demo_mpc_portfolio()

    print("\n" + "=" * 60)
    print("Demo 5: Dyna-Q for Trading (Model-Based RL)")
    print("=" * 60)
    demo_dyna_trading()

    print("\n" + "=" * 60)
    print("Demo 6: CVaR-Constrained Portfolio (Safe RL)")
    print("=" * 60)
    demo_safe_portfolio()

    print("\n" + "=" * 60)
    print("Demo 7: Thompson Sampling for Asset Selection")
    print("=" * 60)
    demo_bayesian_trading()

    print("\n" + "=" * 60)
    print("Demo 8: Exploration for Non-Stationary Markets")
    print("=" * 60)
    demo_exploration_portfolio()
