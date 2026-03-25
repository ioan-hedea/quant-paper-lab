"""Visualization helpers for the quant trading pipeline."""

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE
from .rl import ExecutionRL

def plot_alpha_models(results, prices, returns):
    """Visualize all alpha model outputs."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Alpha Model Decomposition', fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    dates = results['dates']
    tickers = results['tickers']

    # 1. Factor scores over time
    ax = fig.add_subplot(gs[0, 0])
    factor_arr = np.array(results['factor_scores_hist'])
    for i, t in enumerate(tickers[:5]):
        ax.plot(dates, factor_arr[:, i], alpha=0.7, linewidth=0.8, label=t)
    ax.set_title('Fama-French Composite Factor Scores')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 2. GARCH volatility forecasts
    ax = fig.add_subplot(gs[0, 1])
    garch_arr = np.array(results['garch_vols'])
    for i, t in enumerate(tickers[:5]):
        if i < garch_arr.shape[1]:
            ax.plot(dates, garch_arr[:, i] * np.sqrt(252) * 100,
                    alpha=0.7, linewidth=0.8, label=t)
    ax.set_title('GARCH(1,1) Vol Forecasts (ann. %)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 3. HMM regime beliefs
    ax = fig.add_subplot(gs[0, 2])
    beliefs = np.array(results['regime_beliefs'])
    ax.fill_between(dates, beliefs, 0.5, where=beliefs >= 0.5,
                    color='#4CAF50', alpha=0.5, label='Bull')
    ax.fill_between(dates, beliefs, 0.5, where=beliefs < 0.5,
                    color='#F44336', alpha=0.5, label='Bear')
    ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('HMM Regime Detection: P(Bull)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 4. Adaptive alpha-source weights
    ax = fig.add_subplot(gs[1, 0])
    source_weights = np.array(results['source_weights_hist'])
    if len(source_weights) > 0:
        ax.plot(dates, source_weights[:, 0], color='#2196F3', linewidth=1, label='Factor')
        ax.plot(dates, source_weights[:, 1], color='#9C27B0', linewidth=1, label='Pairs')
        ax.plot(dates, source_weights[:, 2], color='#FF9800', linewidth=1, label='LSTM')
        ax.fill_between(dates, source_weights[:, 0], 0, color='#2196F3', alpha=0.08)
    ax.set_title('Adaptive Alpha Source Weights')
    ax.set_ylabel('Weight')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 5. Portfolio weights over time
    ax = fig.add_subplot(gs[1, 1])
    weights_arr = np.array(results['portfolio_weights'])
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    ax.stackplot(dates, *[weights_arr[:, i] for i in range(min(len(tickers), weights_arr.shape[1]))],
                 labels=tickers[:weights_arr.shape[1]], colors=colors, alpha=0.8)
    ax.set_title('Portfolio Weights (RL-Determined)')
    ax.legend(fontsize=5, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 6. RL risk level over time
    ax = fig.add_subplot(gs[1, 2])
    actions = np.array(results['actions'])
    risk_names = results['portfolio_rl'].get_action_labels()
    act_colors = ['#9E9E9E', '#4CAF50', '#2196F3', '#FF9800', '#F44336']
    window = 20
    if len(actions) > window:
        for a in range(5):
            freq = pd.Series((actions == a).astype(float)).rolling(window).mean()
            ax.plot(dates[:len(freq)], freq.values, color=act_colors[a],
                    linewidth=1, alpha=0.7, label=risk_names[a])
    ax.set_title('RL Risk Budget Allocation')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 7. Hedging actions
    ax = fig.add_subplot(gs[2, 0])
    hedge_actions = np.array(results['hedge_actions'])
    hedge_ratios = results['hedging_rl'].hedge_ratios
    actual_hedge = np.array([hedge_ratios[min(a, len(hedge_ratios)-1)] for a in hedge_actions])
    ax.fill_between(dates, actual_hedge, 0, color='#F44336', alpha=0.4)
    ax.plot(dates, actual_hedge, color='#F44336', linewidth=0.8)
    ax.set_title('Dynamic Hedge Ratio (RL)')
    ax.set_ylabel('Hedge Fraction')
    ax.set_ylim(0, max(0.2, max(hedge_ratios) * 1.2))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 8. Realized turnover after no-trade band
    ax = fig.add_subplot(gs[2, 1])
    turnover = pd.Series(results['turnover'], index=dates)
    ax.plot(dates, turnover.values, color='#FF9800', linewidth=1)
    ax.fill_between(dates, turnover.values, 0, color='#FF9800', alpha=0.2)
    ax.set_title('Turnover After No-Trade Band')
    ax.set_ylabel('One-Way Turnover')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 9. Correlation matrix (recent)
    ax = fig.add_subplot(gs[2, 2])
    recent_corr = returns[tickers[-10:]].iloc[-60:].corr()
    im = ax.imshow(recent_corr.values, cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(recent_corr)))
    ax.set_xticklabels(recent_corr.columns, fontsize=6, rotation=45)
    ax.set_yticks(range(len(recent_corr)))
    ax.set_yticklabels(recent_corr.columns, fontsize=6)
    ax.set_title('Recent Correlation Matrix (60d)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.savefig('stock_trading/pipeline_alpha_models.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_performance(results):
    """Performance comparison: pipeline vs benchmarks."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Quant Pipeline Performance: Alpha + RL vs Benchmarks',
                 fontsize=15, fontweight='bold')

    dates = results['dates']
    wealth = np.array(results['wealth'][1:])
    spy = np.array(results['spy'][1:])
    equal = np.array(results['equal'][1:])
    factor = np.array(results['factor'][1:])
    voltarget = np.array(results.get('voltarget', results['factor'])[1:])
    ddlever = np.array(results.get('ddlever', results['factor'])[1:])
    e2e_rl = np.array(results.get('e2e_rl', results['factor'])[1:])

    # 1. Cumulative returns
    ax = axes[0, 0]
    ax.plot(dates, wealth, color='#2196F3', linewidth=2, label='Full Pipeline (Alpha + RL)')
    ax.plot(dates, spy, color='black', linewidth=1.5, linestyle='--', label='SPY Buy & Hold')
    ax.plot(dates, equal, color='gray', linewidth=1, alpha=0.7, label='Equal Weight')
    ax.plot(dates, factor, color='#FF9800', linewidth=1, alpha=0.7, label='Factor-Only (no RL)')
    ax.plot(dates, voltarget, color='#4CAF50', linewidth=1, alpha=0.7, linestyle='-.', label='Vol-Target')
    ax.plot(dates, ddlever, color='#9C27B0', linewidth=1, alpha=0.7, linestyle=':', label='DD-Delever')
    if len(e2e_rl) == len(dates):
        ax.plot(dates, e2e_rl, color='#F44336', linewidth=1, alpha=0.7, linestyle='--', label='E2E RL (PPO)')
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.set_title('Cumulative Performance')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 2. Drawdowns
    ax = axes[0, 1]
    dd = np.array(results['drawdowns'])
    spy_dd = np.array(results['spy_drawdowns'])
    ax.fill_between(dates, dd, 0, color='#2196F3', alpha=0.4, label='Pipeline')
    ax.fill_between(dates, spy_dd, 0, color='gray', alpha=0.3, label='SPY')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 3. Rolling Sharpe
    ax = axes[0, 2]
    pipe_rets = np.diff(results['wealth']) / np.array(results['wealth'][:-1])
    spy_rets_arr = np.diff(results['spy']) / np.array(results['spy'][:-1])
    window = 60

    if len(pipe_rets) > window:
        rolling_sharpe = pd.Series(pipe_rets).rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252))
        spy_rolling_sharpe = pd.Series(spy_rets_arr).rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252))
        ax.plot(dates[:len(rolling_sharpe)], rolling_sharpe.values,
                color='#2196F3', linewidth=1, label='Pipeline')
        ax.plot(dates[:len(spy_rolling_sharpe)], spy_rolling_sharpe.values,
                color='black', linewidth=1, linestyle='--', label='SPY')
        ax.axhline(0, color='red', linewidth=0.5, linestyle=':')
    ax.set_title('Rolling 60d Sharpe Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 4. Return distribution
    ax = axes[1, 0]
    pipe_pct = pipe_rets * 100
    spy_pct = spy_rets_arr * 100
    ax.hist(pipe_pct, bins=60, alpha=0.5, color='#2196F3', density=True, label='Pipeline')
    ax.hist(spy_pct, bins=60, alpha=0.4, color='gray', density=True, label='SPY')

    pipe_var5 = np.percentile(pipe_pct, 5)
    spy_var5 = np.percentile(spy_pct, 5)
    ax.axvline(pipe_var5, color='#2196F3', linestyle='--', linewidth=2, label=f'Pipeline VaR5={pipe_var5:.2f}%')
    ax.axvline(spy_var5, color='gray', linestyle='--', linewidth=2, label=f'SPY VaR5={spy_var5:.2f}%')
    ax.set_xlabel('Daily Return (%)')
    ax.set_title('Return Distribution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Risk-return scatter
    ax = axes[1, 1]
    strategies = {
        'Full Pipeline': pipe_rets,
        'SPY': spy_rets_arr,
        'Equal Weight': np.diff(results['equal']) / np.array(results['equal'][:-1]),
        'Factor Only': np.diff(results['factor']) / np.array(results['factor'][:-1]),
        'Vol-Target': np.diff(results.get('voltarget', results['factor'])) / np.array(results.get('voltarget', results['factor'])[:-1]),
        'DD-Delever': np.diff(results.get('ddlever', results['factor'])) / np.array(results.get('ddlever', results['factor'])[:-1]),
        'E2E RL (PPO)': np.diff(results.get('e2e_rl', results['factor'])) / np.array(results.get('e2e_rl', results['factor'])[:-1]),
    }
    colors = ['#2196F3', 'black', 'gray', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    for (name, rets), color, marker in zip(strategies.items(), colors, markers):
        ann_ret = np.mean(rets) * 252
        ann_vol = np.std(rets) * np.sqrt(252)
        ax.scatter(ann_vol, ann_ret, color=color, marker=marker, s=100,
                   edgecolors='black', zorder=5)
        ax.annotate(name, (ann_vol, ann_ret), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Risk-Return Tradeoff')
    ax.grid(True, alpha=0.3)

    # 6. Metrics table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = []
    for name, rets in strategies.items():
        ann_ret = np.mean(rets) * 252
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / (ann_vol + 1e-8)
        w = np.cumprod(1 + rets)
        max_dd = ((w - np.maximum.accumulate(w)) / np.maximum.accumulate(w)).min()
        calmar = ann_ret / (abs(max_dd) + 1e-8)
        sortino_down = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 1e-8
        sortino = (ann_ret - RISK_FREE_RATE) / sortino_down
        table_data.append([name, f'{ann_ret:.1%}', f'{ann_vol:.1%}',
                          f'{sharpe:.2f}', f'{max_dd:.1%}',
                          f'{calmar:.2f}', f'{sortino:.2f}'])

    table = ax.table(cellText=table_data,
                     colLabels=['Strategy', 'Return', 'Vol', 'Sharpe', 'MaxDD', 'Calmar', 'Sortino'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)

    row_colors = ['#E3F2FD', '#F5F5F5', '#EEEEEE', '#FFF3E0', '#E8F5E9', '#F3E5F5', '#FFEBEE']
    for i in range(len(table_data)):
        for j in range(7):
            table[i + 1, j].set_facecolor(row_colors[i % len(row_colors)])

    ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)

    plt.savefig('stock_trading/pipeline_performance.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rl_analysis(results):
    """Deep dive into RL component behavior."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('RL Component Analysis: How the Agent Decides',
                 fontsize=15, fontweight='bold')

    dates = results['dates']
    actions = np.array(results['actions'])
    hedge_actions = np.array(results['hedge_actions'])
    beliefs = np.array(results['regime_beliefs'])
    dd = np.array(results['drawdowns'])

    # 1. RL action vs regime
    ax = axes[0, 0]
    ax2 = ax.twinx()
    risk_names = results['portfolio_rl'].get_action_labels()
    ax.plot(dates, actions, '.', color='#2196F3', alpha=0.3, markersize=1)
    rolling_action = pd.Series(actions).rolling(20).mean()
    ax.plot(dates[:len(rolling_action)], rolling_action.values,
            color='#2196F3', linewidth=2, label='Avg Risk Level')
    ax2.plot(dates, beliefs, color='#F44336', linewidth=1, alpha=0.5, label='P(Bull)')
    ax.set_ylabel('Risk Level', color='#2196F3')
    ax2.set_ylabel('P(Bull)', color='#F44336')
    ax.set_title('Portfolio RL: Risk Level vs Market Regime')
    ax.set_yticks(range(5))
    ax.set_yticklabels(risk_names, fontsize=7)
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 2. Hedge ratio vs drawdown
    ax = axes[0, 1]
    hedge_ratios_vals = results['hedging_rl'].hedge_ratios
    actual_hedge = np.array([hedge_ratios_vals[min(a, len(hedge_ratios_vals)-1)] for a in hedge_actions])
    ax.plot(dates, dd, color='#F44336', linewidth=1, alpha=0.6, label='Drawdown')
    ax2 = ax.twinx()
    ax2.fill_between(dates, actual_hedge, 0, color='#4CAF50', alpha=0.3, label='Hedge')
    ax2.plot(dates, pd.Series(actual_hedge).rolling(10).mean(),
             color='#4CAF50', linewidth=1.5)
    ax.set_ylabel('Drawdown', color='#F44336')
    ax2.set_ylabel('Hedge Ratio', color='#4CAF50')
    ax.set_title('Hedging RL: Responds to Drawdowns')
    ax.legend(loc='lower left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 3. Q-value heatmap for portfolio RL
    ax = axes[0, 2]
    portfolio_rl = results['portfolio_rl']
    if portfolio_rl.Q:
        states = list(portfolio_rl.Q.keys())[:30]
        q_matrix = np.array([portfolio_rl.Q[s] for s in states])
        im = ax.imshow(q_matrix.T, aspect='auto', cmap='RdYlGn')
        ax.set_xlabel('State Index')
        ax.set_ylabel('Action (Risk Level)')
        ax.set_yticks(range(5))
        ax.set_yticklabels(risk_names, fontsize=7)
        ax.set_title('Portfolio RL: Learned Q-Values')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'No Q-values learned', ha='center', va='center')

    # 4. Action distribution by regime
    ax = axes[1, 0]
    bull_mask = beliefs > 0.6
    bear_mask = beliefs < 0.4
    neutral_mask = ~bull_mask & ~bear_mask

    x = np.arange(5)
    width = 0.25
    for i, (mask, name, color) in enumerate([
        (bull_mask, 'Bull', '#4CAF50'),
        (neutral_mask, 'Neutral', '#FF9800'),
        (bear_mask, 'Bear', '#F44336'),
    ]):
        if mask.sum() > 0:
            counts = np.bincount(actions[mask], minlength=5) / mask.sum()
            ax.bar(x + i * width, counts, width, color=color,
                   alpha=0.8, label=name, edgecolor='black')

    ax.set_xticks(x + width)
    ax.set_xticklabels(risk_names, fontsize=7)
    ax.set_ylabel('Frequency')
    ax.set_title('Action Distribution by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Hedge action distribution by vol regime
    ax = axes[1, 1]
    garch_arr = np.array(results['garch_vols'])
    avg_vol = garch_arr.mean(axis=1) if garch_arr.ndim > 1 else garch_arr
    vol_med = np.median(avg_vol)
    high_vol = avg_vol > vol_med
    low_vol = ~high_vol

    x = np.arange(4)
    hedge_names = [f'{h:.0%}' for h in results['hedging_rl'].hedge_ratios]
    for i, (mask, name, color) in enumerate([
        (low_vol, 'Low Vol', '#4CAF50'),
        (high_vol, 'High Vol', '#F44336'),
    ]):
        if mask.sum() > 0:
            counts = np.bincount(hedge_actions[mask], minlength=4)[:4] / mask.sum()
            ax.bar(x + i * 0.35, counts, 0.35, color=color,
                   alpha=0.8, label=name, edgecolor='black')

    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(hedge_names)
    ax.set_ylabel('Frequency')
    ax.set_title('Hedge Distribution by Volatility Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary: where each component adds value
    ax = axes[1, 2]
    ax.axis('off')

    wealth_arr = np.array(results['wealth'])
    pipe_rets = np.diff(wealth_arr) / wealth_arr[:-1]
    spy_arr = np.array(results['spy'])
    spy_rets = np.diff(spy_arr) / spy_arr[:-1]

    pipe_sharpe = (np.mean(pipe_rets) * 252 - RISK_FREE_RATE) / (np.std(pipe_rets) * np.sqrt(252) + 1e-8)
    spy_sharpe = (np.mean(spy_rets) * 252 - RISK_FREE_RATE) / (np.std(spy_rets) * np.sqrt(252) + 1e-8)

    pipe_maxdd = min(dd)
    spy_maxdd = min(results['spy_drawdowns'])

    pipe_vol = np.std(pipe_rets) * np.sqrt(252)
    spy_vol = np.std(spy_rets) * np.sqrt(252)

    lines = [
        "PIPELINE COMPONENT VALUE-ADD:",
        "",
        f"  Sharpe: {pipe_sharpe:.2f} vs SPY {spy_sharpe:.2f}",
        f"    {'>' if pipe_sharpe > spy_sharpe else '<'} {'Better' if pipe_sharpe > spy_sharpe else 'Worse'} risk-adjusted returns",
        "",
        f"  Max DD: {pipe_maxdd:.1%} vs SPY {spy_maxdd:.1%}",
        f"    {'>' if abs(pipe_maxdd) < abs(spy_maxdd) else '<'} {'Better' if abs(pipe_maxdd) < abs(spy_maxdd) else 'Worse'} drawdown control",
        "",
        f"  Volatility: {pipe_vol:.1%} vs SPY {spy_vol:.1%}",
        f"    {'>' if pipe_vol < spy_vol else '<'} {'Lower' if pipe_vol < spy_vol else 'Higher'} risk taken",
        "",
        "WHERE EACH COMPONENT HELPS:",
        f"  Factors: stock selection signal",
        f"  GARCH: position sizing by vol",
        f"  Adaptive Weights: shift toward working alphas",
        f"  Rebalance Band: reduce turnover drag",
        f"  HMM: regime-aware allocation",
        f"  RL Portfolio: dynamic risk budget",
        f"  RL Hedging: stress-profiting protection",
    ]

    for i, line in enumerate(lines):
        weight = 'bold' if line.startswith(('PIPELINE', 'WHERE')) else 'normal'
        color = '#2196F3' if line.startswith(('PIPELINE', 'WHERE')) else 'black'
        ax.text(0.05, 0.95 - i * 0.055, line, transform=ax.transAxes,
                fontsize=9, fontweight=weight, color=color,
                family='monospace', va='top')

    plt.savefig('stock_trading/pipeline_rl_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_execution_demo():
    """Standalone demo of execution RL — order splitting."""
    print("\n" + "=" * 60)
    print("STAGE 4: Execution RL Demo")
    print("=" * 60)

    np.random.seed(42)

    # Simulate volume profile (U-shaped, typical trading day)
    n_periods = 20
    t = np.linspace(0, 1, n_periods)
    volume_profile = 1e6 * (1.5 - 2 * (t - 0.5) ** 2 + 0.3 * np.random.randn(n_periods))
    volume_profile = np.maximum(volume_profile, 1e5)

    total_shares = 50000

    # Train execution RL
    exec_rl = ExecutionRL(n_time_slices=5, alpha=0.1, epsilon=0.3)
    training_costs = []

    for episode in range(500):
        vol = volume_profile * (1 + 0.2 * np.random.randn(n_periods))
        vol = np.maximum(vol, 1e4)
        cost, _ = exec_rl.execute_order(total_shares, n_periods, vol)
        training_costs.append(cost)

    exec_rl.epsilon = 0.0  # pure exploitation

    # Compare strategies
    strategies = {}

    # 1. TWAP (equal slices)
    twap_cost = 0
    twap_log = []
    remaining = total_shares
    for ti in range(n_periods):
        shares = total_shares / n_periods
        impact = 0.001 * np.sqrt(shares / (volume_profile[ti] + 1e-8))
        twap_cost += shares * impact
        remaining -= shares
        twap_log.append({'executed': shares, 'remaining': remaining, 'impact': impact})
    strategies['TWAP'] = {'cost': twap_cost, 'log': twap_log}

    # 2. VWAP (proportional to volume)
    total_vol = volume_profile.sum()
    vwap_cost = 0
    vwap_log = []
    remaining = total_shares
    for ti in range(n_periods):
        shares = total_shares * volume_profile[ti] / total_vol
        impact = 0.001 * np.sqrt(shares / (volume_profile[ti] + 1e-8))
        vwap_cost += shares * impact
        remaining -= shares
        vwap_log.append({'executed': shares, 'remaining': remaining, 'impact': impact})
    strategies['VWAP'] = {'cost': vwap_cost, 'log': vwap_log}

    # 3. RL execution
    rl_cost, rl_log = exec_rl.execute_order(total_shares, n_periods, volume_profile)
    strategies['RL'] = {'cost': rl_cost, 'log': rl_log}

    # 4. Naive (execute all at once)
    naive_impact = 0.001 * np.sqrt(total_shares / (volume_profile[0] + 1e-8))
    naive_cost = total_shares * naive_impact
    strategies['Naive (all at once)'] = {'cost': naive_cost, 'log': []}

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Execution RL: Optimal Order Splitting\n'
                 '(Minimizing market impact via Almgren-Chriss model)',
                 fontsize=14, fontweight='bold')

    # 1. Execution schedule comparison
    ax = axes[0, 0]
    for name, color in [('TWAP', 'gray'), ('VWAP', '#FF9800'), ('RL', '#2196F3')]:
        log = strategies[name]['log']
        executed = [e['executed'] for e in log]
        ax.bar(np.arange(n_periods) + ({'TWAP': -0.25, 'VWAP': 0, 'RL': 0.25}[name]),
               executed, 0.25, color=color, alpha=0.7, label=name, edgecolor='black')

    ax2 = ax.twinx()
    ax2.plot(range(n_periods), volume_profile / 1e6, 'k--', alpha=0.4, label='Volume')
    ax.set_xlabel('Time Slice')
    ax.set_ylabel('Shares Executed')
    ax.set_title('Execution Schedule')
    ax.legend(fontsize=8, loc='upper left')
    ax2.set_ylabel('Market Volume (M)', color='gray')
    ax.grid(True, alpha=0.3)

    # 2. Market impact per slice
    ax = axes[0, 1]
    for name, color in [('TWAP', 'gray'), ('VWAP', '#FF9800'), ('RL', '#2196F3')]:
        log = strategies[name]['log']
        if log:
            impacts = [e['impact'] * 10000 for e in log]  # bps
            ax.plot(range(len(impacts)), impacts, 'o-', color=color,
                    linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel('Time Slice')
    ax.set_ylabel('Market Impact (bps)')
    ax.set_title('Impact Per Slice')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Total cost comparison
    ax = axes[1, 0]
    names = list(strategies.keys())
    costs = [strategies[n]['cost'] for n in names]
    colors = ['gray', '#FF9800', '#2196F3', '#F44336']
    bars = ax.bar(range(len(names)), costs, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Total Execution Cost ($)')
    ax.set_title('Total Cost Comparison')
    ax.grid(True, alpha=0.3)

    # Annotate savings
    if costs[2] < costs[0]:
        savings = (costs[0] - costs[2]) / costs[0] * 100
        ax.annotate(f'RL saves {savings:.0f}% vs TWAP',
                    xy=(2, costs[2]), xytext=(2.5, costs[0]),
                    fontsize=9, fontweight='bold', color='#2196F3',
                    arrowprops=dict(arrowstyle='->', color='#2196F3'))

    # 4. Training convergence
    ax = axes[1, 1]
    window = 20
    rolling_cost = pd.Series(training_costs).rolling(window).mean()
    ax.plot(range(len(training_costs)), training_costs, alpha=0.2, color='#2196F3')
    ax.plot(range(len(rolling_cost)), rolling_cost.values,
            color='#2196F3', linewidth=2, label='Rolling avg cost')
    ax.axhline(twap_cost, color='gray', linestyle='--', label='TWAP cost')
    ax.axhline(vwap_cost, color='#FF9800', linestyle='--', label='VWAP cost')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Execution Cost')
    ax.set_title('RL Learning Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/pipeline_execution_rl.png', dpi=150, bbox_inches='tight')
    plt.show()

