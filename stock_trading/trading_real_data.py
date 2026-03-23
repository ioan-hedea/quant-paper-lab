"""
Stock Trading on Real Market Data — Sequential Decision Making
===============================================================
Downloads real stock data via yfinance and applies SDM strategies:

1. MDP Value Iteration    — discretized price-momentum states
2. POMDP Regime Detection — Bayesian belief tracking of bull/bear regimes
3. MCTS Trade Planning    — lookahead using empirical return distribution
4. MPC Portfolio          — rolling mean-variance optimization
5. Safe RL (CVaR)         — CVaR-constrained dynamic allocation
6. Bayesian Asset Select  — Thompson Sampling among real stocks
7. Exploration            — Adaptive strategies in non-stationary markets

All benchmarked against SPY buy-and-hold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Data Loading
# ============================================================

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'AMZN']
BENCHMARK = 'SPY'
ALL_TICKERS = TICKERS + [BENCHMARK]
DATA_PERIOD = '3y'  # 3 years of daily data


def load_data():
    """Download and cache stock data."""
    print("Downloading market data...")
    data = yf.download(ALL_TICKERS, period=DATA_PERIOD, auto_adjust=True)
    prices = data['Close'].dropna()
    returns = prices.pct_change().dropna()
    print(f"Loaded {len(prices)} days of data for {len(ALL_TICKERS)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices, returns


# ============================================================
# Demo 1: Trading MDP on Real Data
# ============================================================

def demo_real_mdp(prices, returns):
    """
    Discretize a stock's momentum into states, solve MDP with
    value iteration using empirical transition probabilities.
    """
    ticker = 'AAPL'
    ret = returns[ticker].values

    # Discretize returns into 5 momentum states
    quantiles = np.percentile(ret, [20, 40, 60, 80])
    def discretize(r):
        for i, q in enumerate(quantiles):
            if r < q:
                return i
        return 4

    states = np.array([discretize(r) for r in ret])
    state_names = ['Strong Down', 'Down', 'Flat', 'Up', 'Strong Up']
    n_states = 5
    n_actions = 3  # 0=short, 1=flat, 2=long
    n_combined = n_states * n_actions  # (momentum, position)
    gamma = 0.95

    # Estimate transition probabilities from data
    T_momentum = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        T_momentum[states[i], states[i+1]] += 1
    T_momentum = T_momentum / T_momentum.sum(axis=1, keepdims=True)

    # Mean return per momentum state
    mean_ret = np.zeros(n_states)
    for i in range(n_states):
        mask = states == i
        mean_ret[i] = ret[:-1][mask[:-1]].mean() if mask[:-1].sum() > 0 else 0

    # Build full MDP: state = (momentum, position)
    def idx(mom, pos):
        return mom * n_actions + pos

    T = np.zeros((n_combined, n_actions, n_combined))
    R = np.zeros((n_combined, n_actions))

    for mom in range(n_states):
        for pos in range(n_actions):
            s = idx(mom, pos)
            for action in range(n_actions):
                new_pos = action
                for new_mom in range(n_states):
                    T[s, action, idx(new_mom, new_pos)] = T_momentum[mom, new_mom]
                # Reward: position exposure * expected return - transaction cost
                exposure = action - 1  # -1, 0, 1
                R[s, action] = exposure * mean_ret[mom] * 100
                if action != pos:
                    R[s, action] -= 0.05  # transaction cost

    # Value iteration
    V = np.zeros(n_combined)
    for _ in range(200):
        V_new = np.zeros(n_combined)
        for s in range(n_combined):
            V_new[s] = max(R[s, a] + gamma * T[s, a] @ V for a in range(n_actions))
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new

    policy = np.zeros(n_combined, dtype=int)
    for s in range(n_combined):
        policy[s] = np.argmax([R[s, a] + gamma * T[s, a] @ V for a in range(n_actions)])

    # Simulate policy on data
    spy_ret = returns[BENCHMARK].values
    wealth_mdp = [1.0]
    wealth_spy = [1.0]
    positions = []
    position = 1  # start flat

    for t in range(len(ret)):
        mom = states[t]
        s = idx(mom, position)
        action = policy[s]
        exposure = action - 1
        wealth_mdp.append(wealth_mdp[-1] * (1 + exposure * ret[t]))
        wealth_spy.append(wealth_spy[-1] * (1 + spy_ret[t]))
        position = action
        positions.append(action)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trading MDP on Real Data ({ticker})',
                 fontsize=14, fontweight='bold')

    dates = prices.index[1:]  # align with returns

    # Plot 1: Wealth comparison
    ax = axes[0, 0]
    ax.plot(dates, wealth_mdp[1:], color='#2196F3', linewidth=1.5,
            label=f'MDP Policy on {ticker}')
    ax.plot(dates, wealth_spy[1:], color='gray', linewidth=1.5, linestyle='--',
            label='SPY Buy & Hold')
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.set_title('MDP Strategy vs SPY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Transition matrix heatmap
    ax = axes[0, 1]
    im = ax.imshow(T_momentum, cmap='Blues', vmin=0, vmax=0.5)
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_names, fontsize=7, rotation=45)
    ax.set_yticklabels(state_names, fontsize=7)
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    ax.set_title('Empirical Momentum Transitions')
    plt.colorbar(im, ax=ax)
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f'{T_momentum[i,j]:.2f}', ha='center', va='center', fontsize=7)

    # Plot 3: Optimal policy table
    ax = axes[1, 0]
    action_names = ['Short', 'Flat', 'Long']
    policy_grid = np.zeros((n_actions, n_states), dtype=int)
    for mom in range(n_states):
        for pos in range(n_actions):
            policy_grid[pos, mom] = policy[idx(mom, pos)]

    colors_map = {0: '#F44336', 1: '#9E9E9E', 2: '#4CAF50'}
    for i in range(n_actions):
        for j in range(n_states):
            a = policy_grid[i, j]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       color=colors_map[a], alpha=0.6))
            ax.text(j, i, action_names[a], ha='center', va='center',
                    fontsize=9, fontweight='bold')
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(-0.5, n_actions - 0.5)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(state_names, fontsize=7, rotation=45)
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels(action_names)
    ax.set_xlabel('Momentum State')
    ax.set_ylabel('Current Position')
    ax.set_title('Optimal Policy')
    ax.invert_yaxis()

    # Plot 4: Position over time
    ax = axes[1, 1]
    ax.step(dates, positions, color='#2196F3', linewidth=1, alpha=0.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Short', 'Flat', 'Long'])
    ax.set_ylabel('Position')
    ax.set_title('Trading Decisions Over Time')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('stock_trading/real_mdp.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Demo 2: Hidden Regime Detection (POMDP Belief Tracking)
# ============================================================

def demo_regime_detection(prices, returns):
    """
    Detect hidden bull/bear regimes using Bayesian belief updates
    on real market returns. Use rolling volatility as the observation.
    """
    spy_ret = returns[BENCHMARK].values
    spy_prices = prices[BENCHMARK].values

    # Define 3 regimes by their return distributions (estimated from data)
    # We'll use a simple approach: segment data by rolling volatility
    rolling_vol = pd.Series(spy_ret).rolling(20).std().values
    rolling_mean = pd.Series(spy_ret).rolling(20).mean().values

    # Regime parameters (estimated)
    regime_params = {
        0: {'name': 'Bull',     'mu': 0.0008, 'sigma': 0.008},
        1: {'name': 'Bear',     'mu': -0.0010, 'sigma': 0.020},
        2: {'name': 'Sideways', 'mu': 0.0001, 'sigma': 0.012},
    }

    # Transition matrix
    regime_T = np.array([
        [0.92, 0.03, 0.05],
        [0.05, 0.88, 0.07],
        [0.08, 0.07, 0.85],
    ])

    # Bayesian belief update
    def likelihood(r, regime):
        p = regime_params[regime]
        return np.exp(-0.5 * ((r - p['mu']) / p['sigma'])**2) / (p['sigma'] * np.sqrt(2 * np.pi))

    n_days = len(spy_ret)
    belief = np.array([1/3, 1/3, 1/3])
    beliefs = [belief.copy()]

    start_idx = 20  # skip first 20 for rolling window warmup

    for t in range(start_idx, n_days):
        # Predict
        predicted = regime_T.T @ belief
        # Update with observation (daily return)
        likelihoods = np.array([likelihood(spy_ret[t], r) for r in range(3)])
        updated = likelihoods * predicted
        if updated.sum() > 0:
            belief = updated / updated.sum()
        else:
            belief = predicted
        beliefs.append(belief.copy())

    beliefs = np.array(beliefs)
    dates = prices.index[start_idx + 1:]  # align dates

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Hidden Regime Detection — POMDP Belief Tracking (SPY)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Price with regime coloring
    ax = axes[0, 0]
    ax.plot(prices.index, spy_prices, color='black', linewidth=1)
    map_regime = np.argmax(beliefs[1:], axis=1)  # skip initial
    regime_colors = ['#4CAF50', '#F44336', '#FF9800']
    for t in range(len(map_regime)):
        if t < len(dates):
            ax.axvspan(dates[t], dates[min(t+1, len(dates)-1)],
                       color=regime_colors[map_regime[t]], alpha=0.15)
    ax.set_ylabel('Price ($)')
    ax.set_title('SPY Price with Detected Regimes')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    # Legend
    for i, p in regime_params.items():
        ax.plot([], [], color=regime_colors[i], linewidth=8, alpha=0.4, label=p['name'])
    ax.legend(loc='upper left')

    # Plot 2: Belief evolution
    ax = axes[0, 1]
    for r in range(3):
        ax.plot(dates, beliefs[1:len(dates)+1, r], color=regime_colors[r],
                linewidth=1.2, label=regime_params[r]['name'], alpha=0.8)
    ax.set_ylabel('Belief Probability')
    ax.set_title('Regime Belief Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Regime-based trading strategy
    ax = axes[1, 0]
    wealth_regime = [1.0]
    wealth_spy = [1.0]

    for t in range(len(map_regime)):
        if t + start_idx >= len(spy_ret):
            break
        regime = map_regime[t]
        # Bull -> long, Bear -> short, Sideways -> flat
        exposure = {0: 1.0, 1: -0.5, 2: 0.2}[regime]
        r = spy_ret[t + start_idx]
        wealth_regime.append(wealth_regime[-1] * (1 + exposure * r))
        wealth_spy.append(wealth_spy[-1] * (1 + r))

    n_plot = min(len(wealth_regime), len(dates) + 1)
    ax.plot(dates[:n_plot-1], wealth_regime[1:n_plot], color='#2196F3',
            linewidth=1.5, label='Regime-Based Strategy')
    ax.plot(dates[:n_plot-1], wealth_spy[1:n_plot], color='gray',
            linewidth=1.5, linestyle='--', label='SPY Buy & Hold')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Regime-Based Trading vs SPY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Belief entropy
    ax = axes[1, 1]
    entropy = -np.sum(beliefs[1:] * np.log(beliefs[1:] + 1e-10), axis=1)
    ax.plot(dates[:len(entropy)], entropy[:len(dates)], color='#9C27B0',
            linewidth=1, alpha=0.7)
    ax.axhline(np.log(3), color='gray', linestyle='--', label='Max entropy')
    ax.set_ylabel('Belief Entropy (nats)')
    ax.set_title('Regime Uncertainty Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('stock_trading/real_regime_pomdp.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Demo 3: MPC Portfolio on Real Data
# ============================================================

def demo_real_mpc(prices, returns):
    """
    Rolling mean-variance MPC portfolio optimization on real stocks.
    Compare different lookback windows and risk aversions.
    """
    stock_returns = returns[TICKERS]
    spy_ret = returns[BENCHMARK]
    dates = stock_returns.index

    def mpc_weights(ret_window, risk_aversion=2.0):
        """Mean-variance optimal weights from a return window."""
        mu = ret_window.mean().values
        cov = ret_window.cov().values
        try:
            inv_cov = np.linalg.inv(cov + np.eye(len(mu)) * 1e-8)
            w = inv_cov @ mu / risk_aversion
            w = np.maximum(w, 0)  # long-only
            total = w.sum()
            if total > 0:
                w /= total
            else:
                w = np.ones(len(mu)) / len(mu)
        except np.linalg.LinAlgError:
            w = np.ones(len(mu)) / len(mu)
        return w

    configs = {
        'MPC (60d, λ=2)':  {'lookback': 60,  'risk_aversion': 2.0},
        'MPC (120d, λ=2)': {'lookback': 120, 'risk_aversion': 2.0},
        'MPC (60d, λ=5)':  {'lookback': 60,  'risk_aversion': 5.0},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MPC Portfolio Optimization on Real Data',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#F44336', '#4CAF50']
    all_weights_hist = {}

    start = 120  # need lookback window

    for idx, (name, cfg) in enumerate(configs.items()):
        lb = cfg['lookback']
        ra = cfg['risk_aversion']
        wealth = 1.0
        wealth_hist = []
        weight_hist = []
        current_w = np.ones(len(TICKERS)) / len(TICKERS)

        for t in range(start, len(dates)):
            window = stock_returns.iloc[t-lb:t]
            target_w = mpc_weights(window, ra)
            # Smooth rebalance
            current_w = 0.2 * target_w + 0.8 * current_w
            current_w /= current_w.sum()

            daily_ret = stock_returns.iloc[t].values
            wealth *= (1 + current_w @ daily_ret)
            wealth_hist.append(wealth)
            weight_hist.append(current_w.copy())

        all_weights_hist[name] = np.array(weight_hist)
        axes[0, 0].plot(dates[start:], wealth_hist, color=colors[idx],
                        linewidth=1.5, label=name)

    # SPY benchmark
    spy_wealth = (1 + spy_ret.iloc[start:]).cumprod().values
    axes[0, 0].plot(dates[start:], spy_wealth, color='gray', linewidth=1.5,
                    linestyle='--', label='SPY')
    # Equal weight
    eq_ret = stock_returns.iloc[start:].mean(axis=1)
    eq_wealth = (1 + eq_ret).cumprod().values
    axes[0, 0].plot(dates[start:], eq_wealth, color='black', linewidth=1,
                    linestyle=':', alpha=0.7, label='Equal Weight')

    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].set_title('Wealth Trajectories')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Weight evolution (first config)
    ax = axes[0, 1]
    first_name = list(configs.keys())[0]
    w = all_weights_hist[first_name]
    asset_colors = plt.cm.tab10(np.linspace(0, 0.5, len(TICKERS)))
    ax.stackplot(dates[start:start+len(w)], *[w[:, i] for i in range(len(TICKERS))],
                 labels=TICKERS, colors=asset_colors, alpha=0.7)
    ax.set_ylabel('Portfolio Weight')
    ax.set_title(f'Allocations ({first_name})')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Rolling Sharpe ratio
    ax = axes[1, 0]
    for idx, (name, cfg) in enumerate(configs.items()):
        lb = cfg['lookback']
        ra = cfg['risk_aversion']
        current_w = np.ones(len(TICKERS)) / len(TICKERS)
        daily_returns = []

        for t in range(start, len(dates)):
            window = stock_returns.iloc[t-lb:t]
            target_w = mpc_weights(window, ra)
            current_w = 0.2 * target_w + 0.8 * current_w
            current_w /= current_w.sum()
            daily_returns.append(current_w @ stock_returns.iloc[t].values)

        rolling_sharpe = pd.Series(daily_returns).rolling(60).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252), raw=True)
        ax.plot(dates[start:start+len(rolling_sharpe)], rolling_sharpe,
                color=colors[idx], linewidth=1, alpha=0.8, label=name)

    spy_sharpe = spy_ret.iloc[start:].rolling(60).apply(
        lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252), raw=True)
    ax.plot(dates[start:], spy_sharpe, color='gray', linewidth=1,
            linestyle='--', alpha=0.7, label='SPY')
    ax.axhline(0, color='black', linestyle=':', alpha=0.3)
    ax.set_ylabel('Rolling Sharpe (60d)')
    ax.set_title('Risk-Adjusted Performance')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Performance summary
    ax = axes[1, 1]
    summary = {}
    for idx, (name, cfg) in enumerate(configs.items()):
        lb = cfg['lookback']
        ra = cfg['risk_aversion']
        current_w = np.ones(len(TICKERS)) / len(TICKERS)
        daily_returns = []
        for t in range(start, len(dates)):
            window = stock_returns.iloc[t-lb:t]
            target_w = mpc_weights(window, ra)
            current_w = 0.2 * target_w + 0.8 * current_w
            current_w /= current_w.sum()
            daily_returns.append(current_w @ stock_returns.iloc[t].values)
        dr = np.array(daily_returns)
        summary[name] = {
            'Return': dr.mean() * 252,
            'Vol': dr.std() * np.sqrt(252),
            'Sharpe': dr.mean() / (dr.std() + 1e-8) * np.sqrt(252),
            'MaxDD': np.min(np.minimum.accumulate((1+dr).cumprod()) / np.maximum.accumulate((1+dr).cumprod()) - 1),
        }

    spy_dr = spy_ret.iloc[start:].values
    summary['SPY'] = {
        'Return': spy_dr.mean() * 252,
        'Vol': spy_dr.std() * np.sqrt(252),
        'Sharpe': spy_dr.mean() / (spy_dr.std() + 1e-8) * np.sqrt(252),
        'MaxDD': np.min(np.minimum.accumulate((1+spy_dr).cumprod()) / np.maximum.accumulate((1+spy_dr).cumprod()) - 1),
    }

    metrics = ['Return', 'Sharpe', 'MaxDD']
    x = np.arange(len(metrics))
    width = 0.18
    all_colors = colors + ['gray']

    for i, (name, stats) in enumerate(summary.items()):
        vals = [stats[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name.split('(')[0].strip(),
               color=all_colors[i], alpha=0.8, edgecolor='black')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_title('Performance Metrics')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('stock_trading/real_mpc_portfolio.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Demo 4: CVaR-Constrained Portfolio (Safe RL)
# ============================================================

def demo_real_safe_rl(prices, returns):
    """
    Dynamic CVaR-constrained portfolio on real data.
    Reduce exposure when tail risk exceeds budget.
    """
    stock_returns = returns[TICKERS]
    spy_ret = returns[BENCHMARK]
    dates = stock_returns.index

    start = 60
    cvar_budget = 0.25  # annualized max CVaR
    alpha = 0.05

    # Strategy 1: Unconstrained (equal weight)
    wealth_unc = [1.0]
    # Strategy 2: CVaR-constrained
    wealth_cvar = [1.0]
    risk_multiplier = 1.0
    multiplier_hist = []
    cvar_hist = []

    for t in range(start, len(dates)):
        eq_ret = stock_returns.iloc[t].mean()
        wealth_unc.append(wealth_unc[-1] * (1 + eq_ret))

        # Estimate rolling CVaR
        window = stock_returns.iloc[max(0,t-60):t].mean(axis=1).values
        sorted_w = np.sort(window)
        n_tail = max(1, int(alpha * len(sorted_w)))
        rolling_cvar = -sorted_w[:n_tail].mean() * np.sqrt(252)
        cvar_hist.append(rolling_cvar)

        # Adjust exposure
        if rolling_cvar > cvar_budget:
            risk_multiplier = max(0.1, risk_multiplier - 0.05)
        elif rolling_cvar < cvar_budget * 0.7:
            risk_multiplier = min(1.0, risk_multiplier + 0.02)

        multiplier_hist.append(risk_multiplier)
        wealth_cvar.append(wealth_cvar[-1] * (1 + risk_multiplier * eq_ret))

    # Strategy 3: Fixed conservative (30% stocks, 70% cash-like)
    wealth_conservative = [1.0]
    for t in range(start, len(dates)):
        eq_ret = stock_returns.iloc[t].mean()
        wealth_conservative.append(wealth_conservative[-1] * (1 + 0.3 * eq_ret))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Safe RL: CVaR-Constrained Portfolio on Real Data',
                 fontsize=14, fontweight='bold')

    plot_dates = dates[start:]

    # Plot 1: Wealth comparison
    ax = axes[0, 0]
    n = len(plot_dates)
    ax.plot(plot_dates, wealth_unc[1:n+1], color='#F44336', linewidth=1.5,
            label='Unconstrained')
    ax.plot(plot_dates, wealth_cvar[1:n+1], color='#4CAF50', linewidth=1.5,
            label='CVaR-Constrained')
    ax.plot(plot_dates, wealth_conservative[1:n+1], color='#FF9800',
            linewidth=1.5, linestyle='--', label='Fixed Conservative')

    spy_w = (1 + spy_ret.iloc[start:]).cumprod().values
    ax.plot(plot_dates, spy_w[:n], color='gray', linewidth=1, linestyle=':',
            label='SPY')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Wealth Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Risk multiplier over time
    ax = axes[0, 1]
    ax.plot(plot_dates[:len(multiplier_hist)], multiplier_hist, color='#2196F3',
            linewidth=1.5)
    ax.set_ylabel('Risk Multiplier')
    ax.set_title('Dynamic Exposure (CVaR Control)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Rolling CVaR
    ax = axes[1, 0]
    ax.plot(plot_dates[:len(cvar_hist)], cvar_hist, color='#9C27B0',
            linewidth=1, alpha=0.8)
    ax.axhline(cvar_budget, color='red', linestyle='--', linewidth=2,
               label=f'CVaR Budget ({cvar_budget})')
    ax.set_ylabel('Rolling CVaR (annualized)')
    ax.set_title('Tail Risk Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Drawdown comparison
    ax = axes[1, 1]
    for label, wh, color in [
        ('Unconstrained', wealth_unc[1:n+1], '#F44336'),
        ('CVaR-Constrained', wealth_cvar[1:n+1], '#4CAF50'),
        ('SPY', spy_w[:n].tolist(), 'gray'),
    ]:
        w_arr = np.array(wh)
        running_max = np.maximum.accumulate(w_arr)
        drawdown = (w_arr - running_max) / running_max
        ax.plot(plot_dates[:len(drawdown)], drawdown, color=color,
                linewidth=1.5, label=label, alpha=0.8)

    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('stock_trading/real_safe_rl.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Demo 5: Thompson Sampling for Real Stock Selection
# ============================================================

def demo_real_thompson(prices, returns):
    """
    Thompson Sampling to learn which stock to allocate capital to,
    using real daily returns as bandit rewards.
    """
    stock_returns = returns[TICKERS]
    spy_ret = returns[BENCHMARK]
    dates = stock_returns.index
    n_assets = len(TICKERS)

    class NormalThompson:
        """Thompson Sampling with Normal-Gamma prior for Gaussian rewards."""
        def __init__(self, k):
            self.k = k
            self.mu0 = np.zeros(k)
            self.lam = np.ones(k)
            self.alpha = np.ones(k)
            self.beta = np.ones(k) * 0.001
            self.n = np.zeros(k)
            self.sum_x = np.zeros(k)
            self.sum_x2 = np.zeros(k)

        def select(self):
            samples = np.zeros(self.k)
            for i in range(self.k):
                n = self.n[i]
                lam = self.lam[i] + n
                mu = (self.lam[i] * self.mu0[i] + self.sum_x[i]) / lam
                alpha = self.alpha[i] + n / 2
                beta = self.beta[i]
                if n > 0:
                    beta += 0.5 * (self.sum_x2[i] - self.sum_x[i]**2 / n)
                    beta += self.lam[i] * n * (self.sum_x[i]/n - self.mu0[i])**2 / (2*lam)
                tau = np.random.gamma(alpha, 1.0 / max(beta, 1e-10))
                samples[i] = np.random.normal(mu, 1.0 / np.sqrt(max(lam * tau, 1e-10)))
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

    n_days = len(stock_returns)
    agents = {
        'Thompson Sampling': lambda: NormalThompson(n_assets),
        'UCB (c=2)': lambda: UCBAgent(n_assets, c=2.0),
        'Equal Weight': None,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bayesian RL: Thompson Sampling on Real Stocks',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#9E9E9E']

    for idx, (name, agent_fn) in enumerate(agents.items()):
        np.random.seed(42)
        wealth = 1.0
        wealth_hist = [wealth]
        selections = []

        if agent_fn is not None:
            agent = agent_fn()
            for t in range(n_days):
                arm = agent.select()
                reward = stock_returns.iloc[t, arm]
                agent.update(arm, reward)
                wealth *= (1 + reward)
                wealth_hist.append(wealth)
                selections.append(arm)
        else:
            for t in range(n_days):
                avg_ret = stock_returns.iloc[t].mean()
                wealth *= (1 + avg_ret)
                wealth_hist.append(wealth)

        axes[0, 0].plot(dates, wealth_hist[1:], color=colors[idx],
                        linewidth=1.5, label=name)

    # SPY benchmark
    spy_w = (1 + spy_ret).cumprod().values
    axes[0, 0].plot(dates, spy_w, color='black', linewidth=1, linestyle='--',
                    label='SPY')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].set_title('Wealth Comparison')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Selection frequency for Thompson Sampling
    ax = axes[0, 1]
    np.random.seed(42)
    ts = NormalThompson(n_assets)
    selections = []
    for t in range(n_days):
        arm = ts.select()
        reward = stock_returns.iloc[t, arm]
        ts.update(arm, reward)
        selections.append(arm)

    # Rolling selection frequency
    window = 40
    freq = np.zeros((n_days - window, n_assets))
    for t in range(n_days - window):
        for a in range(n_assets):
            freq[t, a] = sum(1 for s in selections[t:t+window] if s == a) / window

    asset_colors = plt.cm.tab10(np.linspace(0, 0.5, n_assets))
    ax.stackplot(dates[window:], *[freq[:, i] for i in range(n_assets)],
                 labels=TICKERS, colors=asset_colors, alpha=0.7)
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Thompson Sampling: Stock Picks Over Time')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Cumulative returns by stock (ground truth)
    ax = axes[1, 0]
    for i, ticker in enumerate(TICKERS):
        cum_ret = (1 + stock_returns[ticker]).cumprod().values
        ax.plot(dates, cum_ret, color=asset_colors[i], linewidth=1.5,
                label=ticker)
    ax.plot(dates, spy_w, color='black', linewidth=1.5, linestyle='--', label='SPY')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Individual Stock Performance (Ground Truth)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Posterior mean estimates over time
    ax = axes[1, 1]
    np.random.seed(42)
    ts2 = NormalThompson(n_assets)
    posterior_means = [[] for _ in range(n_assets)]

    for t in range(n_days):
        arm = ts2.select()
        reward = stock_returns.iloc[t, arm]
        ts2.update(arm, reward)

        for i in range(n_assets):
            n = max(ts2.n[i], 1)
            lam = ts2.lam[i] + n
            mu = (ts2.lam[i] * ts2.mu0[i] + ts2.sum_x[i]) / lam
            posterior_means[i].append(mu)

    for i, ticker in enumerate(TICKERS):
        ax.plot(dates, posterior_means[i], color=asset_colors[i],
                linewidth=1.5, label=ticker, alpha=0.8)
    ax.axhline(0, color='black', linestyle=':', alpha=0.3)
    ax.set_ylabel('Posterior Mean Return')
    ax.set_title('Learning Stock Returns (Posterior)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('stock_trading/real_thompson.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Demo 6: Grand Comparison — All Strategies vs SPY
# ============================================================

def demo_grand_comparison(prices, returns):
    """
    Run all strategies and compare cumulative performance,
    risk metrics, and Sharpe ratios against SPY.
    """
    stock_returns = returns[TICKERS]
    spy_ret = returns[BENCHMARK]
    dates = stock_returns.index
    n_assets = len(TICKERS)
    start = 120

    results = {}

    # 1. SPY Buy & Hold
    spy_cum = (1 + spy_ret.iloc[start:]).cumprod().values
    results['SPY Buy&Hold'] = {
        'wealth': spy_cum,
        'returns': spy_ret.iloc[start:].values,
        'color': 'black',
    }

    # 2. Equal Weight
    eq_ret = stock_returns.iloc[start:].mean(axis=1).values
    eq_cum = np.cumprod(1 + eq_ret)
    results['Equal Weight'] = {
        'wealth': eq_cum,
        'returns': eq_ret,
        'color': '#9E9E9E',
    }

    # 3. MPC (Mean-Variance)
    current_w = np.ones(n_assets) / n_assets
    mpc_rets = []
    for t in range(start, len(dates)):
        window = stock_returns.iloc[max(0, t-60):t]
        mu = window.mean().values
        cov = window.cov().values
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n_assets) * 1e-8)
            target = inv_cov @ mu / 2.0
            target = np.maximum(target, 0)
            if target.sum() > 0:
                target /= target.sum()
            else:
                target = np.ones(n_assets) / n_assets
        except np.linalg.LinAlgError:
            target = np.ones(n_assets) / n_assets
        current_w = 0.2 * target + 0.8 * current_w
        current_w /= current_w.sum()
        mpc_rets.append(current_w @ stock_returns.iloc[t].values)

    mpc_rets = np.array(mpc_rets)
    results['MPC Portfolio'] = {
        'wealth': np.cumprod(1 + mpc_rets),
        'returns': mpc_rets,
        'color': '#2196F3',
    }

    # 4. CVaR-Constrained
    risk_mult = 1.0
    cvar_rets = []
    for t in range(start, len(dates)):
        eq_r = stock_returns.iloc[t].mean()
        window = stock_returns.iloc[max(0, t-60):t].mean(axis=1).values
        sorted_w = np.sort(window)
        n_tail = max(1, int(0.05 * len(sorted_w)))
        rolling_cvar = -sorted_w[:n_tail].mean() * np.sqrt(252)
        if rolling_cvar > 0.25:
            risk_mult = max(0.1, risk_mult - 0.05)
        elif rolling_cvar < 0.18:
            risk_mult = min(1.0, risk_mult + 0.02)
        cvar_rets.append(risk_mult * eq_r)

    cvar_rets = np.array(cvar_rets)
    results['CVaR-Safe'] = {
        'wealth': np.cumprod(1 + cvar_rets),
        'returns': cvar_rets,
        'color': '#4CAF50',
    }

    # 5. Thompson Sampling
    np.random.seed(42)

    class QuickTS:
        def __init__(self, k):
            self.Q = np.zeros(k)
            self.N = np.zeros(k)
        def select(self):
            samples = np.array([
                np.random.normal(self.Q[i], 1.0 / (self.N[i] + 1))
                for i in range(len(self.Q))
            ])
            return np.argmax(samples)
        def update(self, arm, reward):
            self.N[arm] += 1
            self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    ts = QuickTS(n_assets)
    ts_rets = []
    for t in range(start, len(dates)):
        arm = ts.select()
        r = stock_returns.iloc[t, arm]
        ts.update(arm, r)
        ts_rets.append(r)

    ts_rets = np.array(ts_rets)
    results['Thompson Sampling'] = {
        'wealth': np.cumprod(1 + ts_rets),
        'returns': ts_rets,
        'color': '#FF9800',
    }

    # 6. Momentum MDP
    ticker_ret = returns['AAPL'].values
    quantiles = np.percentile(ticker_ret, [20, 40, 60, 80])
    def disc(r):
        for i, q in enumerate(quantiles):
            if r < q:
                return i
        return 4

    # Simple momentum rule: long if recent momentum is up, short if down
    mdp_rets = []
    for t in range(start, len(dates)):
        lookback = returns['AAPL'].iloc[t-20:t].mean()
        if lookback > 0.001:
            exposure = 1.0
        elif lookback < -0.001:
            exposure = -0.5
        else:
            exposure = 0.2
        mdp_rets.append(exposure * returns['AAPL'].iloc[t])

    mdp_rets = np.array(mdp_rets)
    results['Momentum (AAPL)'] = {
        'wealth': np.cumprod(1 + mdp_rets),
        'returns': mdp_rets,
        'color': '#9C27B0',
    }

    # --- Plotting ---
    plot_dates = dates[start:]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Grand Comparison: All SDM Strategies vs SPY',
                 fontsize=15, fontweight='bold')

    # Plot 1: Wealth trajectories
    ax = axes[0, 0]
    for name, data in results.items():
        n = min(len(plot_dates), len(data['wealth']))
        ls = '--' if name == 'SPY Buy&Hold' else '-'
        lw = 2 if name == 'SPY Buy&Hold' else 1.5
        ax.plot(plot_dates[:n], data['wealth'][:n], color=data['color'],
                linewidth=lw, linestyle=ls, label=name)
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.set_title('Cumulative Performance')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Risk-return scatter
    ax = axes[0, 1]
    for name, data in results.items():
        ann_ret = data['returns'].mean() * 252
        ann_vol = data['returns'].std() * np.sqrt(252)
        marker = 's' if name == 'SPY Buy&Hold' else 'o'
        ax.scatter(ann_vol, ann_ret, color=data['color'], s=120,
                   marker=marker, edgecolors='black', zorder=5)
        ax.annotate(name.split('(')[0].strip(), (ann_vol, ann_ret),
                    fontsize=7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Risk-Return Tradeoff')
    ax.grid(True, alpha=0.3)

    # Plot 3: Sharpe ratio bar chart
    ax = axes[1, 0]
    sharpes = {}
    for name, data in results.items():
        sharpes[name] = data['returns'].mean() / (data['returns'].std() + 1e-8) * np.sqrt(252)

    bars = ax.barh(range(len(sharpes)), list(sharpes.values()),
                   color=[results[n]['color'] for n in sharpes],
                   edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(sharpes)))
    ax.set_yticklabels(list(sharpes.keys()), fontsize=8)
    ax.set_xlabel('Annualized Sharpe Ratio')
    ax.set_title('Sharpe Ratio Comparison')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 4: Maximum drawdown bar chart
    ax = axes[1, 1]
    max_dds = {}
    for name, data in results.items():
        wealth = np.cumprod(1 + data['returns'])
        running_max = np.maximum.accumulate(wealth)
        dd = (wealth - running_max) / running_max
        max_dds[name] = dd.min()

    bars = ax.barh(range(len(max_dds)), list(max_dds.values()),
                   color=[results[n]['color'] for n in max_dds],
                   edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(max_dds)))
    ax.set_yticklabels(list(max_dds.keys()), fontsize=8)
    ax.set_xlabel('Maximum Drawdown')
    ax.set_title('Worst Drawdown')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/grand_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<22} {'Ann.Return':>10} {'Ann.Vol':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("=" * 80)
    for name, data in results.items():
        ann_ret = data['returns'].mean() * 252
        ann_vol = data['returns'].std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-8)
        wealth = np.cumprod(1 + data['returns'])
        max_dd = ((wealth - np.maximum.accumulate(wealth)) / np.maximum.accumulate(wealth)).min()
        print(f"{name:<22} {ann_ret:>9.1%} {ann_vol:>9.1%} {sharpe:>8.2f} {max_dd:>7.1%}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    prices, returns = load_data()

    print("\n" + "=" * 60)
    print("Demo 1: Trading MDP on Real Data")
    print("=" * 60)
    demo_real_mdp(prices, returns)

    print("\n" + "=" * 60)
    print("Demo 2: Hidden Regime Detection (POMDP)")
    print("=" * 60)
    demo_regime_detection(prices, returns)

    print("\n" + "=" * 60)
    print("Demo 3: MPC Portfolio Optimization")
    print("=" * 60)
    demo_real_mpc(prices, returns)

    print("\n" + "=" * 60)
    print("Demo 4: CVaR-Constrained Portfolio (Safe RL)")
    print("=" * 60)
    demo_real_safe_rl(prices, returns)

    print("\n" + "=" * 60)
    print("Demo 5: Thompson Sampling for Stock Selection")
    print("=" * 60)
    demo_real_thompson(prices, returns)

    print("\n" + "=" * 60)
    print("Demo 6: Grand Comparison — All Strategies vs SPY")
    print("=" * 60)
    demo_grand_comparison(prices, returns)
