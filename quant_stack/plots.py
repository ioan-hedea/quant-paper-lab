"""Visualization helpers for the quant trading pipeline."""

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from .config import RISK_FREE_RATE
from .evaluation_helpers import (
    _build_ablation_summary,
    _build_control_comparison_summary,
    _control_color,
    _display_label,
    _pareto_frontier_points,
)
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
        ax.plot(dates, source_weights[:, 1], color='#9C27B0', linewidth=1, label='GARCH')
        ax.plot(dates, source_weights[:, 2], color='#FF9800', linewidth=1, label='HMM')
        ax.fill_between(dates, source_weights[:, 0], 0, color='#2196F3', alpha=0.08)
    ax.set_title('Adaptive Alpha Sleeve Weights')
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

    # 6. Portfolio control over time
    ax = fig.add_subplot(gs[1, 2])
    invested = np.array(results.get('invested_fractions', []), dtype=float)
    overlay = np.array(results.get('overlay_sizes', []), dtype=float)
    if len(invested) == len(dates):
        ax.plot(dates, invested, color='#2196F3', linewidth=1.5, label='Invested fraction')
        ax.fill_between(dates, invested, 0, color='#2196F3', alpha=0.15)
    ax2 = ax.twinx()
    if len(overlay) == len(dates):
        ax2.plot(dates, overlay, color='#FF9800', linewidth=1.2, label='Active overlay size')
    ax.set_title('Portfolio RL Control Outputs')
    ax.set_ylabel('Invested fraction', color='#2196F3')
    ax2.set_ylabel('Overlay size', color='#FF9800')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 7. Overlay size history
    ax = fig.add_subplot(gs[2, 0])
    overlay = np.array(results.get('overlay_sizes', []), dtype=float)
    ax.fill_between(dates, overlay, 0, color='#F44336', alpha=0.3)
    ax.plot(dates, overlay, color='#F44336', linewidth=0.8)
    ax.set_title('Active Overlay Size')
    ax.set_ylabel('Overlay Fraction')
    ymax = max(0.2, overlay.max() * 1.2) if len(overlay) > 0 else 0.2
    ax.set_ylim(0, ymax)
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

    # 9. Option / IV feature stack
    ax = fig.add_subplot(gs[2, 2])
    iv_ann = np.array(results.get('iv_annualized', []), dtype=float)
    realized_vol = np.array(results.get('garch_vol_uncertainty', []), dtype=float)
    iv_pct = np.array(results.get('iv_percentile', []), dtype=float)
    if len(iv_ann) == len(dates) and len(iv_ann) > 0 and np.nanmax(np.abs(iv_ann)) > 1e-8:
        ax.plot(dates, iv_ann * 100, color='#6A1B9A', linewidth=1.5, label='Implied vol proxy')
        ax2 = ax.twinx()
        ax2.plot(dates, iv_pct, color='#009688', linewidth=1.0, alpha=0.8, label='IV percentile')
        ax.set_title('Option Feature Stack: IV Level and Percentile')
        ax.set_ylabel('IV annualized (%)', color='#6A1B9A')
        ax2.set_ylabel('IV percentile', color='#009688')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
    else:
        recent_corr = returns[tickers[-10:]].iloc[-60:].corr()
        im = ax.imshow(recent_corr.values, cmap='RdYlGn_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(recent_corr)))
        ax.set_xticklabels(recent_corr.columns, fontsize=6, rotation=45)
        ax.set_yticks(range(len(recent_corr)))
        ax.set_yticklabels(recent_corr.columns, fontsize=6)
        ax.set_title('Recent Correlation Matrix (60d)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.savefig('pipeline_alpha_models.png', dpi=150, bbox_inches='tight')
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

    plt.savefig('pipeline_performance.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rl_analysis(results):
    """Deep dive into RL component behavior."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('RL Component Analysis: How the Agent Decides',
                 fontsize=15, fontweight='bold')

    dates = results['dates']
    actions = np.array(results['actions'])
    invested = np.array(results.get('invested_fractions', []), dtype=float)
    overlay = np.array(results.get('overlay_sizes', []), dtype=float)
    hedge_actions = np.array(results['hedge_actions'])
    hedge_type_actions = np.array(results.get('hedge_type_actions', []))
    beliefs = np.array(results['regime_beliefs'])
    dd = np.array(results['drawdowns'])
    hedge_active = results.get('hedging_rl') is not None

    # 1. RL exposure vs regime
    ax = axes[0, 0]
    ax2 = ax.twinx()
    risk_names = results['portfolio_rl'].get_action_labels()
    overlay_names = results['portfolio_rl'].get_overlay_labels()
    invested_series = invested if len(invested) == len(dates) else np.array([results['portfolio_rl'].decode_action(a)[0] for a in actions], dtype=float)
    overlay_series = overlay if len(overlay) == len(dates) else np.array([results['portfolio_rl'].decode_action(a)[1] for a in actions], dtype=float)
    ax.plot(dates, invested_series, color='#2196F3', linewidth=1.5, alpha=0.9, label='Invested fraction')
    ax.fill_between(dates, invested_series, 0, color='#2196F3', alpha=0.15)
    ax2.plot(dates, beliefs, color='#F44336', linewidth=1, alpha=0.5, label='P(Bull)')
    if len(overlay_series) == len(dates):
        scaled_overlay = overlay_series / max(float(overlay_series.max()), 1e-8)
        ax2.plot(dates, scaled_overlay, color='#FF9800', linewidth=1, alpha=0.7,
                 linestyle='--', label='Overlay size (scaled)')
    ax.set_ylabel('Invested fraction', color='#2196F3')
    ax2.set_ylabel('Regime / scaled overlay', color='#F44336')
    ax.set_title('Portfolio RL: Exposure and Overlay vs Market Regime')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # 2. Hedge ratio vs drawdown
    ax = axes[0, 1]
    actual_hedge = np.array(results.get('effective_hedge_ratios', results.get('hedge_ratios', [])))
    if hedge_active:
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
    else:
        ax.plot(dates, dd, color='#F44336', linewidth=1, alpha=0.7)
        ax.set_title('No Hedge Controller in Current Architecture')
        ax.set_ylabel('Drawdown')
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
        ax.set_ylabel('Action (Invested / Overlay)')
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'{risk_names[i]} / {overlay_names[i]}' for i in range(5)], fontsize=7)
        ax.set_title('Portfolio RL: Learned Q-Values')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'No Q-values learned', ha='center', va='center')

    # 4. Invested-fraction distribution by regime
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
    ax.set_title('Invested-Fraction Ladder by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Overlay-size distribution by volatility regime
    ax = axes[1, 1]
    iv_pct = np.array(results.get('iv_percentile', []), dtype=float)
    if len(iv_pct) == 0 or np.nanmax(np.abs(iv_pct)) < 1e-8:
        garch_arr = np.array(results['garch_vols'])
        avg_vol = garch_arr.mean(axis=1) if garch_arr.ndim > 1 else garch_arr
        iv_pct = (avg_vol > np.median(avg_vol)).astype(float)
    high_iv = iv_pct > np.median(iv_pct)
    low_iv = ~high_iv

    if hedge_active:
        type_names = results['hedging_rl'].get_type_labels()
        x = np.arange(len(type_names))
        for i, (mask, name, color) in enumerate([
            (low_iv, 'Low IV', '#4CAF50'),
            (high_iv, 'High IV', '#F44336'),
        ]):
            if mask.sum() > 0:
                actions_for_mask = hedge_type_actions[mask] if len(hedge_type_actions) == len(mask) else np.zeros(mask.sum(), dtype=int)
                counts = np.bincount(actions_for_mask, minlength=len(type_names))[:len(type_names)] / mask.sum()
                ax.bar(x + i * 0.35, counts, 0.35, color=color,
                       alpha=0.8, label=name, edgecolor='black')

        ax.set_xticks(x + 0.175)
        ax.set_xticklabels(type_names, rotation=15, fontsize=8)
        ax.set_ylabel('Frequency')
        ax.set_title('Option Hedge Type by IV Regime')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        x = np.arange(len(overlay_names))
        for i, (mask, name, color) in enumerate([
            (low_iv, 'Lower-vol regime', '#4CAF50'),
            (high_iv, 'Higher-vol regime', '#F44336'),
        ]):
            if mask.sum() > 0:
                counts = np.bincount(actions[mask], minlength=len(overlay_names))[:len(overlay_names)] / mask.sum()
                ax.bar(x + i * 0.35, counts, 0.35, color=color,
                       alpha=0.8, label=name, edgecolor='black')

        ax.set_xticks(x + 0.175)
        ax.set_xticklabels(overlay_names, fontsize=8)
        ax.set_ylabel('Frequency')
        ax.set_title('Overlay-Size Ladder by Volatility Regime')
        ax.legend(fontsize=8)
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
        f"  RL Portfolio: invested fraction + active overlay control",
        f"  Hedge Sleeve: removed in current architecture",
    ]

    for i, line in enumerate(lines):
        weight = 'bold' if line.startswith(('PIPELINE', 'WHERE')) else 'normal'
        color = '#2196F3' if line.startswith(('PIPELINE', 'WHERE')) else 'black'
        ax.text(0.05, 0.95 - i * 0.055, line, transform=ax.transAxes,
                fontsize=9, fontweight=weight, color=color,
                family='monospace', va='top')

    plt.savefig('pipeline_rl_analysis.png', dpi=150, bbox_inches='tight')
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
    ax.set_title('Illustrative Cost Comparison')
    ax.grid(True, alpha=0.3)

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
    ax.set_title('Archived Execution-Submodule Training Trace')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pipeline_execution_rl.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_windows(
    metrics_df: pd.DataFrame,
    rolling_references_df: pd.DataFrame | None = None,
    output_path: str | Path = 'pipeline_rolling_windows.png',
) -> None:
    """Figure 4: Rolling-window robustness — Sharpe and Calmar distributions across windows."""
    rolling = metrics_df[metrics_df['suite'] == 'rolling_window'].copy()
    if rolling.empty:
        print("plot_rolling_windows: no rolling_window rows found; skipping.")
        return

    _ref_display = {
        'SPY': 'SPY',
        'factor_benchmark': 'Factor',
        'vol_target': 'Vol-Target',
        'dd_delever': 'DD-Delever',
    }
    _ref_colors = {
        'SPY': 'black',
        'factor_benchmark': '#FF9800',
        'vol_target': '#4CAF50',
        'dd_delever': '#9C27B0',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 4: Rolling-Window Robustness — Sharpe and Calmar Distributions',
                 fontsize=14, fontweight='bold')

    for ax, metric, title, ylabel in [
        (axes[0], 'sharpe', 'Sharpe Ratio across Rolling Windows', 'Sharpe Ratio'),
        (axes[1], 'calmar', 'Calmar Ratio across Rolling Windows', 'Calmar Ratio'),
    ]:
        groups: dict[str, np.ndarray] = {}
        group_colors: dict[str, str] = {}

        groups['Full Pipeline'] = rolling[metric].values.astype(float)
        group_colors['Full Pipeline'] = '#2196F3'

        if rolling_references_df is not None and not rolling_references_df.empty:
            for ref_label, display in _ref_display.items():
                ref_group = rolling_references_df[rolling_references_df['label'] == ref_label]
                if not ref_group.empty and metric in ref_group.columns:
                    groups[display] = ref_group[metric].values.astype(float)
                    group_colors[display] = _ref_colors[ref_label]

        labels = list(groups.keys())
        data = [groups[k] for k in labels]
        colors = [group_colors[k] for k in labels]

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops=dict(color='black', linewidth=2),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Overlay individual points
        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.default_rng(seed=42 + i).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color=color, s=30, zorder=5, alpha=0.8, edgecolors='black', linewidths=0.5,
            )

        ax.axhline(0, color='red', linewidth=0.7, linestyle=':', alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_reward_ablation(
    metrics_df: pd.DataFrame,
    output_path: str | Path = 'pipeline_reward_ablation.png',
) -> None:
    """Figure 6: Reward function ablation — pipeline performance across 4 reward modes."""
    reward_data = metrics_df[
        (metrics_df['suite'] == 'reward_ablation') &
        (metrics_df['label'].str.startswith('full_pipeline_reward_'))
    ].copy()
    if reward_data.empty:
        print("plot_reward_ablation: no reward_ablation rows found; skipping.")
        return

    e2e_data = metrics_df[
        (metrics_df['suite'] == 'reward_ablation') &
        (metrics_df['label'].str.startswith('e2e_reward_'))
    ].copy()

    _mode_display = {
        'full_pipeline_reward_differential_sharpe': 'Diff. Sharpe',
        'full_pipeline_reward_return': 'Return',
        'full_pipeline_reward_sortino': 'Sortino',
        'full_pipeline_reward_mean_variance': 'Mean-Var.',
    }
    _e2e_display = {
        'e2e_reward_differential_sharpe': 'E2E Diff. Sharpe',
        'e2e_reward_return': 'E2E Return',
        'e2e_reward_sortino': 'E2E Sortino',
        'e2e_reward_mean_variance': 'E2E Mean-Var.',
    }
    _metrics_info = [
        ('sharpe', 'Sharpe Ratio'),
        ('ann_return', 'Annualized Return'),
        ('calmar', 'Calmar Ratio'),
        ('max_drawdown', 'Max Drawdown'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Figure 6: Reward Function Ablation — Full Pipeline vs E2E RL Across Reward Modes',
                 fontsize=13, fontweight='bold')

    pipe_labels = [_mode_display.get(lbl, lbl) for lbl in reward_data['label']]
    pipe_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    for ax, (metric, metric_title) in zip(axes, _metrics_info):
        if metric not in reward_data.columns:
            ax.set_title(metric_title)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        pipe_vals = reward_data[metric].values.astype(float)
        x = np.arange(len(pipe_labels))
        bars = ax.bar(
            x - 0.2, pipe_vals,
            width=0.35,
            color=pipe_colors[:len(pipe_vals)],
            alpha=0.85,
            edgecolor='black',
            label='Full Pipeline',
        )

        if not e2e_data.empty and metric in e2e_data.columns:
            e2e_vals = e2e_data[metric].values.astype(float)
            n = min(len(pipe_vals), len(e2e_vals))
            ax.bar(
                x[:n] + 0.2, e2e_vals[:n],
                width=0.35,
                color='#F44336',
                alpha=0.55,
                edgecolor='black',
                label='E2E RL (PPO)',
                hatch='//',
            )

        # Annotate bar values
        for rect, val in zip(bars, pipe_vals):
            fmt = f'{val:.2f}' if metric != 'ann_return' else f'{val:.1%}'
            if metric == 'max_drawdown':
                fmt = f'{val:.1%}'
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + abs(pipe_vals.max()) * 0.02,
                fmt, ha='center', va='bottom', fontsize=7, rotation=0,
            )

        ax.set_title(metric_title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pipe_labels, fontsize=8, rotation=15)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.grid(True, alpha=0.3, axis='y')
        if ax is axes[0]:
            ax.legend(fontsize=7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_research_evaluation(
    metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    baseline_results: dict[str, object] | None = None,
    rolling_references: pd.DataFrame | None = None,
    output_path: Path = Path('pipeline_research_eval.png'),
    frontier_output_path: Path = Path('control_pareto_frontier.png'),
) -> None:
    """Create the paper-facing research summary and frontier figures."""
    del regime_summary

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        'Research Story: Control Comparison, Legacy Ablation, and Robustness',
        fontsize=15,
        fontweight='bold',
    )

    control_summary = _build_control_comparison_summary(metrics)
    ablation_summary = _build_ablation_summary(metrics)

    ax = axes[0, 0]
    if baseline_results is not None:
        dates = baseline_results.get('dates', [])
        wealth = np.asarray(baseline_results.get('wealth', [1.0])[1:], dtype=float)
        spy = np.asarray(baseline_results.get('spy', [1.0])[1:], dtype=float)
        factor = np.asarray(baseline_results.get('factor', [1.0])[1:], dtype=float)
        voltarget = np.asarray(baseline_results.get('voltarget', [1.0])[1:], dtype=float)
        ddlever = np.asarray(baseline_results.get('ddlever', [1.0])[1:], dtype=float)
        e2e_rl = np.asarray(baseline_results.get('e2e_rl', [1.0])[1:], dtype=float)
        if len(dates) == len(wealth):
            ax.plot(dates, wealth, color='#1f77b4', linewidth=2, label='Full Pipeline')
            ax.plot(dates, factor, color='#ff7f0e', linewidth=1.5, label='Factor Benchmark')
            ax.plot(dates, voltarget, color='#2ca02c', linewidth=1.3, linestyle='-.', label='Vol-Target')
            ax.plot(dates, ddlever, color='#9467bd', linewidth=1.3, linestyle=':', label='DD-Delever')
            if len(e2e_rl) == len(dates):
                ax.plot(dates, e2e_rl, color='#d62728', linewidth=1.3, linestyle='--', label='E2E RL (PPO)')
            ax.plot(dates, spy, color='black', linewidth=1.2, linestyle='--', label='SPY')
            ax.legend(fontsize=6, ncol=2)
    ax.set_title('Legacy Full-Pipeline Reference Split')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if not control_summary.empty:
        ranked = control_summary.sort_values('mean_sharpe', ascending=True)
        labels = [_display_label(label).replace('\n', ' ') for label in ranked['component_label']]
        colors = [_control_color(label) for label in ranked['component_label']]
        ax.barh(labels, ranked['mean_sharpe'], color=colors, alpha=0.9)
        for y, val in enumerate(ranked['mean_sharpe']):
            ax.text(val + 0.01, y, f'{val:.2f}', va='center', fontsize=8)
    ax.set_title('Control Comparison: Mean Sharpe')
    ax.set_xlabel('Mean Sharpe')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if not control_summary.empty:
        for _, row in control_summary.iterrows():
            label = str(row['component_label'])
            x = float(row['mean_vol'])
            y = float(row['mean_return'])
            ax.scatter(
                x, y, s=95, color=_control_color(label),
                edgecolors='black', linewidth=0.6, alpha=0.9, zorder=4,
            )
            ax.annotate(_display_label(label).replace('\n', ' '), (x, y), fontsize=7, xytext=(5, 4), textcoords='offset points')
    ax.set_title('Control Comparison: Return vs Volatility')
    ax.set_xlabel('Mean annualized volatility')
    ax.set_ylabel('Mean annualized return')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not ablation_summary.empty:
        labels = [_display_label(label).replace('\n', ' ') for label in ablation_summary['component_label']]
        colors = ['#bbbbbb', '#9ecae1', '#6baed6', '#3182bd', '#08519c', '#08306b']
        ax.barh(labels, ablation_summary['mean_sharpe'], color=colors[:len(labels)], alpha=0.9)
        for y, val in enumerate(ablation_summary['mean_sharpe']):
            ax.text(val + 0.01, y, f'{val:.2f}', va='center', fontsize=8)
    ax.set_title('Legacy Ablation: Mean Sharpe by Stack')
    ax.set_xlabel('Mean Sharpe')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    rolling = metrics[metrics['suite'] == 'rolling_window']
    if not rolling.empty:
        ax.plot(rolling['window_id'], rolling['sharpe'], marker='o', color='#1f77b4', linewidth=2, label='Full Pipeline')
    if rolling_references is not None and not rolling_references.empty:
        ref_styles = [
            ('factor_benchmark', '#ff7f0e', '-'),
            ('vol_target', '#2ca02c', '-.'),
            ('dd_delever', '#9467bd', ':'),
            ('SPY', 'black', '--'),
        ]
        for label, color, linestyle in ref_styles:
            group = rolling_references[rolling_references['label'] == label]
            if not group.empty:
                ax.plot(group['window_id'], group['sharpe'], marker='o', color=color, linestyle=linestyle, label=_display_label(label))
        ax.legend(fontsize=7, ncol=2)
    ax.set_title('Robustness: Rolling-Window Sharpe')
    ax.set_xlabel('Window')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if not control_summary.empty:
        ranked = control_summary.copy()
        ranked['drawdown_abs'] = ranked['mean_max_drawdown'].abs()
        for _, row in ranked.iterrows():
            label = str(row['component_label'])
            x = float(row['drawdown_abs'])
            y = float(row['mean_return'])
            ax.scatter(
                x, y, s=95, color=_control_color(label),
                edgecolors='black', linewidth=0.6, alpha=0.9, zorder=4,
            )
            ax.annotate(_display_label(label).replace('\n', ' '), (x, y), fontsize=7, xytext=(5, 4), textcoords='offset points')
        frontier = _pareto_frontier_points(ranked)
        if not frontier.empty and len(frontier) >= 2:
            ax.plot(frontier['drawdown_abs'], frontier['mean_return'], color='black', linewidth=1.4, linestyle='--', alpha=0.8, label='Pareto frontier')
            ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Control Comparison: Return vs Drawdown')
    ax.set_xlabel('Absolute mean max drawdown')
    ax.set_ylabel('Mean annualized return')
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if not control_summary.empty:
        frontier_output_path.parent.mkdir(parents=True, exist_ok=True)
        frontier_fig, frontier_ax = plt.subplots(figsize=(8.5, 6.0))
        ranked = control_summary.copy()
        ranked['drawdown_abs'] = ranked['mean_max_drawdown'].abs()
        for _, row in ranked.iterrows():
            label = str(row['component_label'])
            x = float(row['drawdown_abs'])
            y = float(row['mean_return'])
            frontier_ax.scatter(x, y, s=110, color=_control_color(label), edgecolors='black', linewidth=0.6, alpha=0.9)
            frontier_ax.annotate(_display_label(label).replace('\n', ' '), (x, y), fontsize=8, xytext=(5, 4), textcoords='offset points')
        frontier = _pareto_frontier_points(ranked)
        if not frontier.empty:
            frontier_ax.plot(frontier['drawdown_abs'], frontier['mean_return'], color='black', linewidth=1.5, linestyle='--')
        frontier_ax.set_title('Control-Method Pareto Frontier')
        frontier_ax.set_xlabel('Absolute mean max drawdown')
        frontier_ax.set_ylabel('Mean annualized return')
        frontier_ax.grid(True, alpha=0.3)
        frontier_fig.tight_layout()
        frontier_fig.savefig(frontier_output_path, dpi=150, bbox_inches='tight')
        plt.close(frontier_fig)
