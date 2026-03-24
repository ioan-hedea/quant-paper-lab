"""Research evaluation engine for ablations, robustness, and regime analysis."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EvaluationConfig, ExperimentConfig, PipelineConfig, RISK_FREE_RATE
from .pipeline import run_full_pipeline


def _daily_returns_from_path(path: list[float]) -> np.ndarray:
    wealth = np.asarray(path, dtype=float)
    if len(wealth) < 2:
        return np.array([], dtype=float)
    return np.diff(wealth) / np.clip(wealth[:-1], 1e-8, None)


def _path_metric_summary(path: list[float] | np.ndarray, label: str) -> dict[str, float | str]:
    rets = _daily_returns_from_path(path)
    if len(rets) == 0:
        return {'label': label}

    ann_ret = float(np.mean(rets) * 252)
    ann_vol = float(np.std(rets) * np.sqrt(252))
    sharpe = float((ann_ret - RISK_FREE_RATE) / (ann_vol + 1e-8))
    wealth = np.asarray(path, dtype=float)
    dd = (wealth - np.maximum.accumulate(wealth)) / np.maximum.accumulate(wealth)
    max_dd = float(dd.min())
    calmar = float(ann_ret / (abs(max_dd) + 1e-8))
    var5 = float(np.percentile(rets * 100, 5))
    return {
        'label': label,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'var5_pct': var5,
    }


def _metric_summary(results: dict[str, object]) -> dict[str, float | str]:
    summary = _path_metric_summary(
        results['wealth'],
        str(results.get('experiment_label', 'unknown')),
    )
    turnover = float(np.mean(results.get('turnover', [0.0])))
    tx_cost = float(np.mean(results.get('transaction_costs', [0.0])))
    summary.update({
        'avg_turnover': turnover,
        'avg_transaction_cost': tx_cost,
    })
    return summary


def _regime_summary(results: dict[str, object]) -> list[dict[str, float | str]]:
    beliefs = np.asarray(results.get('regime_beliefs', []), dtype=float)
    actions = np.asarray(results.get('actions', []), dtype=int)
    hedge_actions = np.asarray(results.get('hedge_actions', []), dtype=int)
    turnover = np.asarray(results.get('turnover', []), dtype=float)
    hedge_ratios = np.asarray(results.get('hedge_ratios', []), dtype=float)
    cash_weights = np.asarray(results.get('cash_weights', []), dtype=float)
    tx_costs = np.asarray(results.get('transaction_costs', []), dtype=float)
    wealth_rets = _daily_returns_from_path(results.get('wealth', []))

    if len(beliefs) == 0:
        return []

    n_obs = min(
        len(beliefs),
        len(actions),
        len(hedge_actions),
        len(turnover),
        len(wealth_rets),
        len(hedge_ratios),
        len(cash_weights),
        len(tx_costs),
    )
    beliefs = beliefs[:n_obs]
    actions = actions[:n_obs]
    hedge_actions = hedge_actions[:n_obs]
    turnover = turnover[:n_obs]
    hedge_ratios = hedge_ratios[:n_obs]
    cash_weights = cash_weights[:n_obs]
    tx_costs = tx_costs[:n_obs]
    wealth_rets = wealth_rets[:n_obs]

    regime_masks = {
        'bull': beliefs > 0.60,
        'neutral': (beliefs >= 0.40) & (beliefs <= 0.60),
        'bear': beliefs < 0.40,
    }

    rows: list[dict[str, float | str]] = []
    for regime, mask in regime_masks.items():
        if mask.sum() == 0:
            continue
        rows.append({
            'label': results.get('experiment_label', 'unknown'),
            'regime': regime,
            'avg_action': float(actions[mask].mean()),
            'avg_hedge_action': float(hedge_actions[mask].mean()),
            'avg_hedge_ratio': float(hedge_ratios[mask].mean()),
            'avg_cash_weight': float(cash_weights[mask].mean()),
            'avg_turnover': float(turnover[mask].mean()),
            'avg_transaction_cost': float(tx_costs[mask].mean()),
            'ann_return': float(wealth_rets[mask].mean() * 252),
            'ann_vol': float(wealth_rets[mask].std() * np.sqrt(252)),
        })
    return rows


def _slice_inputs(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    macro_data: pd.DataFrame,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = returns.index[start:end]
    return (
        prices.loc[idx],
        volumes.loc[idx],
        returns.loc[idx],
        macro_data.loc[idx] if not macro_data.empty else pd.DataFrame(index=idx),
    )


def build_ablation_suite(base_config: PipelineConfig) -> list[PipelineConfig]:
    configs: list[PipelineConfig] = []

    factor_only = copy.deepcopy(base_config)
    factor_only.experiment = ExperimentConfig(
        label='factor_only',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    factor_only.optimizer.use_optimizer = False
    factor_only.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(factor_only)

    alpha_stack = copy.deepcopy(base_config)
    alpha_stack.experiment = ExperimentConfig(
        label='alpha_stack_no_rl',
        use_factor=True,
        use_pairs=True,
        use_lstm=True,
        adaptive_combiner=True,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    configs.append(alpha_stack)

    portfolio_only = copy.deepcopy(base_config)
    portfolio_only.experiment = ExperimentConfig(
        label='alpha_plus_portfolio_rl',
        use_factor=True,
        use_pairs=True,
        use_lstm=True,
        adaptive_combiner=True,
        use_portfolio_rl=True,
        use_hedge_rl=False,
    )
    portfolio_only.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(portfolio_only)

    hedge_only = copy.deepcopy(base_config)
    hedge_only.experiment = ExperimentConfig(
        label='alpha_plus_hedge_rl',
        use_factor=True,
        use_pairs=True,
        use_lstm=True,
        adaptive_combiner=True,
        use_portfolio_rl=False,
        use_hedge_rl=True,
    )
    configs.append(hedge_only)

    full_pipeline = copy.deepcopy(base_config)
    full_pipeline.experiment = ExperimentConfig(label='full_pipeline')
    configs.append(full_pipeline)

    return configs


def _component_label(label: str) -> str:
    return label.rsplit('_tf', 1)[0] if '_tf' in label else label


def _display_label(label: str) -> str:
    mapping = {
        'factor_only': 'Factor Only',
        'alpha_stack_no_rl': 'Alpha Stack\nNo RL',
        'alpha_plus_portfolio_rl': 'Alpha +\nPortfolio RL',
        'alpha_plus_hedge_rl': 'Alpha +\nHedge RL',
        'full_pipeline': 'Full\nPipeline',
        'SPY': 'SPY',
        'factor_benchmark': 'Factor Bench',
    }
    return mapping.get(label, label.replace('_', ' ').title())


def _table_label(label: str) -> str:
    return _display_label(label).replace('\n', ' ')


def _latex_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}\\%"


def _build_ablation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    ablation = metrics[metrics['suite'] == 'ablation'].copy()
    if ablation.empty:
        return pd.DataFrame()

    ablation['component_label'] = ablation['label'].map(_component_label)
    summary = (
        ablation.groupby('component_label')
        .agg(
            mean_return=('ann_return', 'mean'),
            mean_vol=('ann_vol', 'mean'),
            mean_sharpe=('sharpe', 'mean'),
            mean_max_drawdown=('max_drawdown', 'mean'),
            mean_calmar=('calmar', 'mean'),
        )
        .reset_index()
    )
    ordering = ['factor_only', 'alpha_stack_no_rl', 'alpha_plus_portfolio_rl', 'alpha_plus_hedge_rl', 'full_pipeline']
    summary['order'] = summary['component_label'].apply(lambda x: ordering.index(x) if x in ordering else len(ordering))
    summary = summary.sort_values('order').drop(columns='order')
    return summary


def _build_robustness_summary(
    metrics: pd.DataFrame,
    rolling_references: pd.DataFrame,
) -> pd.DataFrame:
    rolling_full = metrics[metrics['suite'] == 'rolling_window'].copy()
    if rolling_full.empty:
        return pd.DataFrame()

    summary: dict[str, float] = {
        'rolling_window_count': float(len(rolling_full)),
        'median_full_sharpe': float(rolling_full['sharpe'].median()),
        'median_full_calmar': float(rolling_full['calmar'].median()),
        'median_full_max_drawdown': float(rolling_full['max_drawdown'].median()),
    }

    if not rolling_references.empty:
        spy = rolling_references[rolling_references['label'] == 'SPY'].copy()
        factor = rolling_references[rolling_references['label'] == 'factor_benchmark'].copy()

        if not spy.empty:
            merged_spy = rolling_full.merge(
                spy[['window_id', 'sharpe', 'calmar']],
                on='window_id',
                suffixes=('_full', '_spy'),
            )
            if not merged_spy.empty:
                summary['frac_full_beats_spy_sharpe'] = float((merged_spy['sharpe_full'] > merged_spy['sharpe_spy']).mean())
                summary['frac_full_beats_spy_calmar'] = float((merged_spy['calmar_full'] > merged_spy['calmar_spy']).mean())

        if not factor.empty:
            merged_factor = rolling_full.merge(
                factor[['window_id', 'max_drawdown', 'calmar']],
                on='window_id',
                suffixes=('_full', '_factor'),
            )
            if not merged_factor.empty:
                summary['frac_full_beats_factor_drawdown'] = float(
                    (merged_factor['max_drawdown_full'] > merged_factor['max_drawdown_factor']).mean()
                )
                summary['frac_full_beats_factor_calmar'] = float(
                    (merged_factor['calmar_full'] > merged_factor['calmar_factor']).mean()
                )

    return pd.DataFrame([summary])


def _write_research_tables(
    ablation_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []

    if not ablation_summary.empty:
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Ablation summary across train/test split settings.}",
            r"\label{tab:ablation_summary}",
            r"\begin{tabular}{@{} l r r r r r @{}}",
            r"\toprule",
            r"Component & Mean Return & Mean Vol. & Mean Sharpe & Mean Max DD & Mean Calmar \\",
            r"\midrule",
        ])
        for _, row in ablation_summary.iterrows():
            lines.append(
                f"{_table_label(str(row['component_label']))} & "
                f"{_latex_pct(float(row['mean_return']))} & "
                f"{_latex_pct(float(row['mean_vol']))} & "
                f"{row['mean_sharpe']:.2f} & "
                f"${_latex_pct(float(row['mean_max_drawdown']))}$ & "
                f"{row['mean_calmar']:.2f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    if not robustness_summary.empty:
        row = robustness_summary.iloc[0]
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Rolling-window robustness summary for the full pipeline.}",
            r"\label{tab:robustness_summary}",
            r"\begin{tabular}{@{} l r @{}}",
            r"\toprule",
            r"Statistic & Value \\",
            r"\midrule",
            f"Rolling windows & {int(row['rolling_window_count'])} \\\\",
            f"Median full-pipeline Sharpe & {row['median_full_sharpe']:.2f} \\\\",
            f"Median full-pipeline Calmar & {row['median_full_calmar']:.2f} \\\\",
            f"Median full-pipeline Max DD & ${_latex_pct(float(row['median_full_max_drawdown']))}$ \\\\",
        ])
        if 'frac_full_beats_factor_drawdown' in row:
            lines.append(f"Frac. windows full beats factor on drawdown & {_latex_pct(float(row['frac_full_beats_factor_drawdown']), digits=0)} \\\\")
        if 'frac_full_beats_factor_calmar' in row:
            lines.append(f"Frac. windows full beats factor on Calmar & {_latex_pct(float(row['frac_full_beats_factor_calmar']), digits=0)} \\\\")
        if 'frac_full_beats_spy_sharpe' in row:
            lines.append(f"Frac. windows full beats SPY on Sharpe & {_latex_pct(float(row['frac_full_beats_spy_sharpe']), digits=0)} \\\\")
        if 'frac_full_beats_spy_calmar' in row:
            lines.append(f"Frac. windows full beats SPY on Calmar & {_latex_pct(float(row['frac_full_beats_spy_calmar']), digits=0)} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open('stock_trading/research_paper_tables.tex', 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def plot_research_evaluation(
    metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    baseline_results: dict[str, object] | None = None,
    rolling_references: pd.DataFrame | None = None,
) -> None:
    """Create a compact workshop-style summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Research Story: Main Result, Ablation, Robustness, and Control Behavior', fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    if baseline_results is not None:
        dates = baseline_results.get('dates', [])
        wealth = np.asarray(baseline_results.get('wealth', [1.0])[1:], dtype=float)
        spy = np.asarray(baseline_results.get('spy', [1.0])[1:], dtype=float)
        factor = np.asarray(baseline_results.get('factor', [1.0])[1:], dtype=float)
        if len(dates) == len(wealth):
            ax.plot(dates, wealth, color='#1f77b4', linewidth=2, label='Full Pipeline')
            ax.plot(dates, factor, color='#ff7f0e', linewidth=1.5, label='Factor Benchmark')
            ax.plot(dates, spy, color='black', linewidth=1.2, linestyle='--', label='SPY')
            ax.legend(fontsize=8)
    ax.set_title('Main Result: Equity Curves')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ablation_summary = _build_ablation_summary(metrics)
    if not ablation_summary.empty:
        ax.bar(
            [_display_label(label) for label in ablation_summary['component_label']],
            ablation_summary['mean_sharpe'],
            color=['#bbbbbb', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],
            alpha=0.9,
        )
    ax.set_title('Ablation: Mean Sharpe by Component Stack')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    rolling = metrics[metrics['suite'] == 'rolling_window']
    if not rolling.empty:
        ax.plot(rolling['window_id'], rolling['sharpe'], marker='o', color='#1f77b4', linewidth=2, label='Full Pipeline')
    if rolling_references is not None and not rolling_references.empty:
        for label, color, linestyle in [('factor_benchmark', '#ff7f0e', '-'), ('SPY', 'black', '--')]:
            group = rolling_references[rolling_references['label'] == label]
            if not group.empty:
                ax.plot(group['window_id'], group['sharpe'], marker='o', color=color, linestyle=linestyle, label=_display_label(label))
        ax.legend(fontsize=8)
    ax.set_title('Robustness: Rolling-Window Sharpe')
    ax.set_xlabel('Window')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if not regime_summary.empty:
        pivot = (
            regime_summary.pivot_table(
                index='regime',
                values=['avg_cash_weight', 'avg_hedge_ratio'],
                aggfunc='mean',
            )
            .reindex(['bull', 'neutral', 'bear'])
            .fillna(0.0)
        )
        invested = 1.0 - pivot['avg_cash_weight']
        x = np.arange(len(pivot.index))
        width = 0.35
        ax.bar(x - width / 2, invested, width=width, color='#2ca02c', alpha=0.85, label='Avg invested fraction')
        ax.bar(x + width / 2, pivot['avg_hedge_ratio'], width=width, color='#d62728', alpha=0.85, label='Avg hedge ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.legend(fontsize=8)
    ax.set_title('Behavior: Regime-Conditional Controller Response')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_trading/pipeline_research_eval.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_research_evaluation(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    sec_quality_scores: pd.Series | None = None,
    base_config: PipelineConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """
    Run the ablation and robustness engine.

    Outputs:
    - stock_trading/research_metrics.csv
    - stock_trading/research_regime_summary.csv
    - stock_trading/research_summary.json
    - stock_trading/pipeline_research_eval.png
    """
    base_config = copy.deepcopy(base_config or PipelineConfig())
    evaluation_config = evaluation_config or EvaluationConfig()
    macro_data = macro_data if macro_data is not None else pd.DataFrame(index=returns.index)
    sec_quality_scores = sec_quality_scores if sec_quality_scores is not None else pd.Series(dtype=float)

    metric_rows: list[dict[str, float | str]] = []
    regime_rows: list[dict[str, float | str]] = []
    rolling_reference_rows: list[dict[str, float | str]] = []
    baseline_results: dict[str, object] | None = None

    for config in build_ablation_suite(base_config):
        base_label = config.experiment.label
        for train_frac in evaluation_config.train_fracs:
            run_config = copy.deepcopy(config)
            run_config.train_frac = train_frac
            run_config.experiment.label = f'{base_label}_tf{train_frac:.2f}'
            results = run_full_pipeline(
                prices,
                volumes,
                returns,
                macro_data=macro_data,
                sec_quality_scores=sec_quality_scores,
                config=run_config,
            )
            row = _metric_summary(results)
            row.update({'suite': 'ablation', 'window_id': 'full_sample', 'param_name': 'train_frac', 'param_value': train_frac})
            metric_rows.append(row)
            regime_rows.extend(_regime_summary(results))
            if base_label == 'full_pipeline' and abs(train_frac - base_config.train_frac) < 1e-12:
                baseline_results = results

    if baseline_results is None:
        baseline_config = copy.deepcopy(base_config)
        baseline_config.experiment.label = 'full_pipeline_baseline'
        baseline_results = run_full_pipeline(
            prices,
            volumes,
            returns,
            macro_data=macro_data,
            sec_quality_scores=sec_quality_scores,
            config=baseline_config,
        )

    strict_timing = copy.deepcopy(base_config)
    strict_timing.feature_availability.macro_lag_days = max(evaluation_config.macro_lag_grid)
    strict_timing.feature_availability.allow_static_sec_quality = False
    strict_timing.experiment.label = 'full_pipeline_strict_timing'
    strict_results = run_full_pipeline(
        prices,
        volumes,
        returns,
        macro_data=macro_data,
        sec_quality_scores=sec_quality_scores,
        config=strict_timing,
    )
    strict_row = _metric_summary(strict_results)
    strict_row.update({'suite': 'timing_discipline', 'window_id': 'full_sample', 'param_name': 'strict_timing', 'param_value': 1})
    metric_rows.append(strict_row)
    regime_rows.extend(_regime_summary(strict_results))

    starts = list(range(0, max(1, len(returns) - evaluation_config.rolling_window_days + 1), evaluation_config.rolling_step_days))
    starts = starts[-evaluation_config.max_rolling_windows:]
    for window_id, start in enumerate(starts):
        end = start + evaluation_config.rolling_window_days
        if end > len(returns):
            continue
        p_slice, v_slice, r_slice, m_slice = _slice_inputs(prices, volumes, returns, macro_data, start, end)
        rolling_config = copy.deepcopy(base_config)
        rolling_config.train_frac = evaluation_config.rolling_train_frac
        rolling_config.experiment.label = 'full_pipeline'
        results = run_full_pipeline(
            p_slice,
            v_slice,
            r_slice,
            macro_data=m_slice,
            sec_quality_scores=sec_quality_scores,
            config=rolling_config,
        )
        row = _metric_summary(results)
        row.update({'suite': 'rolling_window', 'window_id': window_id, 'param_name': 'window_start', 'param_value': str(r_slice.index[0].date())})
        metric_rows.append(row)
        regime_rows.extend(_regime_summary(results))
        spy_row = _path_metric_summary(results['spy'], 'SPY')
        spy_row.update({'suite': 'rolling_reference', 'window_id': window_id, 'param_name': 'benchmark', 'param_value': 'SPY'})
        rolling_reference_rows.append(spy_row)
        factor_row = _path_metric_summary(results['factor'], 'factor_benchmark')
        factor_row.update({'suite': 'rolling_reference', 'window_id': window_id, 'param_name': 'benchmark', 'param_value': 'factor_benchmark'})
        rolling_reference_rows.append(factor_row)

    for base_cost_bps in evaluation_config.cost_bps_grid:
        cost_config = copy.deepcopy(base_config)
        cost_config.cost_model.base_cost_bps = base_cost_bps
        cost_config.experiment.label = 'full_pipeline'
        results = run_full_pipeline(
            prices,
            volumes,
            returns,
            macro_data=macro_data,
            sec_quality_scores=sec_quality_scores,
            config=cost_config,
        )
        row = _metric_summary(results)
        row.update({'suite': 'cost_sensitivity', 'window_id': 'full_sample', 'param_name': 'base_cost_bps', 'param_value': base_cost_bps})
        metric_rows.append(row)

    for rebalance_band in evaluation_config.rebalance_band_grid:
        band_config = copy.deepcopy(base_config)
        band_config.rebalance_band = rebalance_band
        band_config.experiment.label = 'full_pipeline'
        results = run_full_pipeline(
            prices,
            volumes,
            returns,
            macro_data=macro_data,
            sec_quality_scores=sec_quality_scores,
            config=band_config,
        )
        row = _metric_summary(results)
        row.update({'suite': 'rebalance_sensitivity', 'window_id': 'full_sample', 'param_name': 'rebalance_band', 'param_value': rebalance_band})
        metric_rows.append(row)

    for hedge_scale in evaluation_config.hedge_scale_grid:
        hedge_config = copy.deepcopy(base_config)
        hedge_config.hedge_ratios = tuple(round(h * hedge_scale, 4) for h in hedge_config.hedge_ratios)
        hedge_config.experiment.label = 'full_pipeline'
        results = run_full_pipeline(
            prices,
            volumes,
            returns,
            macro_data=macro_data,
            sec_quality_scores=sec_quality_scores,
            config=hedge_config,
        )
        row = _metric_summary(results)
        row.update({'suite': 'hedge_sensitivity', 'window_id': 'full_sample', 'param_name': 'hedge_scale', 'param_value': hedge_scale})
        metric_rows.append(row)

    for macro_lag in evaluation_config.macro_lag_grid:
        lag_config = copy.deepcopy(base_config)
        lag_config.feature_availability.macro_lag_days = macro_lag
        lag_config.experiment.label = 'full_pipeline'
        results = run_full_pipeline(
            prices,
            volumes,
            returns,
            macro_data=macro_data,
            sec_quality_scores=sec_quality_scores,
            config=lag_config,
        )
        row = _metric_summary(results)
        row.update({'suite': 'macro_lag_sensitivity', 'window_id': 'full_sample', 'param_name': 'macro_lag_days', 'param_value': macro_lag})
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    regime_summary = pd.DataFrame(regime_rows)
    rolling_references = pd.DataFrame(rolling_reference_rows)
    ablation_summary = _build_ablation_summary(metrics)
    robustness_summary = _build_robustness_summary(metrics, rolling_references)

    metrics.to_csv('stock_trading/research_metrics.csv', index=False)
    regime_summary.to_csv('stock_trading/research_regime_summary.csv', index=False)
    rolling_references.to_csv('stock_trading/research_rolling_references.csv', index=False)
    ablation_summary.to_csv('stock_trading/research_ablation_summary.csv', index=False)
    robustness_summary.to_csv('stock_trading/research_robustness_summary.csv', index=False)
    _write_research_tables(ablation_summary, robustness_summary)
    plot_research_evaluation(metrics, regime_summary, baseline_results=baseline_results, rolling_references=rolling_references)

    summary = {
        'base_config': asdict(base_config),
        'evaluation_config': asdict(evaluation_config),
        'n_metric_rows': len(metrics),
        'n_regime_rows': len(regime_summary),
        'ablation_summary': ablation_summary.to_dict(orient='records'),
        'robustness_summary': robustness_summary.to_dict(orient='records'),
        'best_sharpe_by_suite': {
            suite: (
                group.sort_values('sharpe', ascending=False)
                .head(1)[['label', 'sharpe', 'ann_return', 'max_drawdown']]
                .to_dict(orient='records')[0]
            )
            for suite, group in metrics.groupby('suite')
        },
    }
    with open('stock_trading/research_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    return {
        'metrics': metrics,
        'regime_summary': regime_summary,
        'ablation_summary': ablation_summary,
        'robustness_summary': robustness_summary,
        'summary': summary,
    }
