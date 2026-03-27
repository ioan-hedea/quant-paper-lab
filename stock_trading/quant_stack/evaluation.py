"""Research evaluation engine for ablations, robustness, and regime analysis."""

from __future__ import annotations

import copy
from datetime import datetime
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import EvaluationConfig, ExperimentConfig, PipelineConfig, RISK_FREE_RATE
from .pipeline import run_full_pipeline
from .plots import plot_rolling_windows, plot_reward_ablation

CHECKPOINT_SCHEMA_VERSION = 1


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


def _returns_metric_summary(rets: np.ndarray, label: str) -> dict[str, float | str]:
    if len(rets) == 0:
        return {'label': label}
    ann_ret = float(np.mean(rets) * 252)
    ann_vol = float(np.std(rets) * np.sqrt(252))
    sharpe = float((ann_ret - RISK_FREE_RATE) / (ann_vol + 1e-8))
    wealth = np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    dd = (wealth - np.maximum.accumulate(wealth)) / np.maximum.accumulate(wealth)
    max_dd = float(dd.min())
    calmar = float(ann_ret / (abs(max_dd) + 1e-8))
    return {
        'label': label,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
    }


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = int(np.clip(block_size, 1, n))
    indices: list[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        end = start + b
        if end <= n:
            block = list(range(start, end))
        else:
            block = list(range(start, n)) + list(range(0, end - n))
        indices.extend(block)
    return np.asarray(indices[:n], dtype=int)


def _compute_bootstrap_cis(
    path_map: dict[str, list[float] | np.ndarray],
    n_samples: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(seed)

    for label, path in path_map.items():
        rets = _daily_returns_from_path(path)
        if len(rets) < 30:
            continue
        point = _returns_metric_summary(rets, label)
        sampled_ann_return: list[float] = []
        sampled_sharpe: list[float] = []
        sampled_calmar: list[float] = []
        sampled_max_dd: list[float] = []
        for _ in range(n_samples):
            idx = _block_bootstrap_indices(len(rets), block_size, rng)
            sample = rets[idx]
            sample_summary = _returns_metric_summary(sample, label)
            sampled_ann_return.append(float(sample_summary['ann_return']))
            sampled_sharpe.append(float(sample_summary['sharpe']))
            sampled_calmar.append(float(sample_summary['calmar']))
            sampled_max_dd.append(float(sample_summary['max_drawdown']))

        rows.append({
            'label': label,
            'ann_return_point': float(point['ann_return']),
            'ann_return_ci_low': float(np.percentile(sampled_ann_return, 2.5)),
            'ann_return_ci_high': float(np.percentile(sampled_ann_return, 97.5)),
            'sharpe_point': float(point['sharpe']),
            'sharpe_ci_low': float(np.percentile(sampled_sharpe, 2.5)),
            'sharpe_ci_high': float(np.percentile(sampled_sharpe, 97.5)),
            'calmar_point': float(point['calmar']),
            'calmar_ci_low': float(np.percentile(sampled_calmar, 2.5)),
            'calmar_ci_high': float(np.percentile(sampled_calmar, 97.5)),
            'max_dd_point': float(point['max_drawdown']),
            'max_dd_ci_low': float(np.percentile(sampled_max_dd, 2.5)),
            'max_dd_ci_high': float(np.percentile(sampled_max_dd, 97.5)),
            'bootstrap_samples': int(n_samples),
            'bootstrap_block_size': int(block_size),
        })
    return pd.DataFrame(rows)


def _compute_bootstrap_pairwise_significance(
    path_map: dict[str, list[float] | np.ndarray],
    base_label: str,
    compare_labels: list[str],
    n_samples: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if base_label not in path_map:
        return pd.DataFrame(rows)
    base_rets = _daily_returns_from_path(path_map[base_label])
    if len(base_rets) < 30:
        return pd.DataFrame(rows)

    rng = np.random.default_rng(seed + 17)
    for compare_label in compare_labels:
        if compare_label not in path_map:
            continue
        cmp_rets = _daily_returns_from_path(path_map[compare_label])
        n = min(len(base_rets), len(cmp_rets))
        if n < 30:
            continue
        base = base_rets[-n:]
        comp = cmp_rets[-n:]
        sharpe_deltas: list[float] = []
        ann_ret_deltas: list[float] = []
        calmar_deltas: list[float] = []
        for _ in range(n_samples):
            idx = _block_bootstrap_indices(n, block_size, rng)
            base_s = _returns_metric_summary(base[idx], base_label)
            comp_s = _returns_metric_summary(comp[idx], compare_label)
            sharpe_deltas.append(float(base_s['sharpe']) - float(comp_s['sharpe']))
            ann_ret_deltas.append(float(base_s['ann_return']) - float(comp_s['ann_return']))
            calmar_deltas.append(float(base_s['calmar']) - float(comp_s['calmar']))

        def _pvalue_from_deltas(values: list[float]) -> float:
            arr = np.asarray(values, dtype=float)
            p_left = float((arr <= 0).mean())
            p_right = float((arr >= 0).mean())
            return float(min(1.0, 2.0 * min(p_left, p_right)))

        rows.append({
            'base_label': base_label,
            'compare_label': compare_label,
            'delta_sharpe_point': float(_returns_metric_summary(base, base_label)['sharpe'] - _returns_metric_summary(comp, compare_label)['sharpe']),
            'delta_sharpe_ci_low': float(np.percentile(sharpe_deltas, 2.5)),
            'delta_sharpe_ci_high': float(np.percentile(sharpe_deltas, 97.5)),
            'delta_sharpe_pvalue_two_sided': _pvalue_from_deltas(sharpe_deltas),
            'delta_ann_return_point': float(_returns_metric_summary(base, base_label)['ann_return'] - _returns_metric_summary(comp, compare_label)['ann_return']),
            'delta_ann_return_ci_low': float(np.percentile(ann_ret_deltas, 2.5)),
            'delta_ann_return_ci_high': float(np.percentile(ann_ret_deltas, 97.5)),
            'delta_ann_return_pvalue_two_sided': _pvalue_from_deltas(ann_ret_deltas),
            'delta_calmar_point': float(_returns_metric_summary(base, base_label)['calmar'] - _returns_metric_summary(comp, compare_label)['calmar']),
            'delta_calmar_ci_low': float(np.percentile(calmar_deltas, 2.5)),
            'delta_calmar_ci_high': float(np.percentile(calmar_deltas, 97.5)),
            'delta_calmar_pvalue_two_sided': _pvalue_from_deltas(calmar_deltas),
            'bootstrap_samples': int(n_samples),
            'bootstrap_block_size': int(block_size),
        })
    return pd.DataFrame(rows)


def _jobson_korkie_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
) -> dict[str, float]:
    """
    Test whether two Sharpe ratios differ significantly.
    Jobson-Korkie (1981) with Memmel (2003) correction for correlation.
    Returns test statistic and two-sided p-value.
    """
    n = min(len(returns_a), len(returns_b))
    if n < 30:
        return {'jk_stat': 0.0, 'jk_pvalue': 1.0}
    ra, rb = returns_a[-n:], returns_b[-n:]
    mu_a, mu_b = float(ra.mean()), float(rb.mean())
    sig_a, sig_b = float(ra.std()), float(rb.std())
    rho = float(np.corrcoef(ra, rb)[0, 1])
    sharpe_a = mu_a / (sig_a + 1e-12)
    sharpe_b = mu_b / (sig_b + 1e-12)
    # Memmel (2003) asymptotic variance
    v = (2.0 * (1.0 - rho)
         + 0.5 * (sharpe_a ** 2 + sharpe_b ** 2
                   - 2.0 * sharpe_a * sharpe_b * rho ** 2))
    theta = (sharpe_a - sharpe_b) * np.sqrt(n) / (np.sqrt(v) + 1e-12)
    pvalue = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(theta))))
    return {'jk_stat': float(theta), 'jk_pvalue': pvalue}


def _compute_jobson_korkie_table(
    path_map: dict[str, list[float] | np.ndarray],
    base_label: str,
    compare_labels: list[str],
) -> pd.DataFrame:
    """Run Jobson-Korkie for base vs each comparison strategy."""
    if base_label not in path_map:
        return pd.DataFrame()
    base_rets = _daily_returns_from_path(path_map[base_label])
    rows: list[dict] = []
    for compare_label in compare_labels:
        if compare_label not in path_map:
            continue
        cmp_rets = _daily_returns_from_path(path_map[compare_label])
        result = _jobson_korkie_test(base_rets, cmp_rets)
        rows.append({
            'base_label': base_label,
            'compare_label': compare_label,
            **result,
        })
    return pd.DataFrame(rows)


def _run_time_series_cv(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    macro_data: pd.DataFrame,
    sec_quality_scores: pd.Series,
    base_config: PipelineConfig,
    n_folds: int,
) -> pd.DataFrame:
    """Blocked time-series cross-validation: n_folds expanding-window splits."""
    n = len(returns)
    fold_rows: list[dict] = []
    min_train = max(504, int(n * 0.3))
    test_size = (n - min_train) // n_folds
    if test_size < 126:
        print(f"  ts_cv: not enough data for {n_folds} folds (n={n}), skipping.")
        return pd.DataFrame()

    for fold in range(n_folds):
        train_end_idx = min_train + fold * test_size
        test_end_idx = min(train_end_idx + test_size, n)
        if test_end_idx <= train_end_idx:
            continue
        fold_returns = returns.iloc[:test_end_idx]
        fold_prices = prices.loc[fold_returns.index]
        fold_volumes = volumes.loc[fold_returns.index]
        fold_macro = (macro_data.loc[fold_returns.index]
                      if not macro_data.empty else pd.DataFrame(index=fold_returns.index))

        fold_config = copy.deepcopy(base_config)
        fold_config.train_frac = train_end_idx / test_end_idx
        fold_config.experiment.label = f'ts_cv_fold_{fold}'
        fold_config.enable_e2e_baseline = False

        print(f"\n  ts_cv fold {fold + 1}/{n_folds}: "
              f"train {fold_returns.index[0].date()}–{fold_returns.index[train_end_idx - 1].date()}, "
              f"test {fold_returns.index[train_end_idx].date()}–{fold_returns.index[test_end_idx - 1].date()}")
        try:
            results = run_full_pipeline(
                fold_prices, fold_volumes, fold_returns,
                macro_data=fold_macro,
                sec_quality_scores=sec_quality_scores,
                config=fold_config,
            )
            row = _path_metric_summary(results['wealth'], f'ts_cv_fold_{fold}')
            row.update({
                'suite': 'ts_cv',
                'fold': fold,
                'train_end': str(fold_returns.index[train_end_idx - 1].date()),
                'test_end': str(fold_returns.index[test_end_idx - 1].date()),
            })
            fold_rows.append(row)
        except Exception as exc:
            print(f"  ts_cv fold {fold} failed: {exc}")

    return pd.DataFrame(fold_rows)


def _rolling_starts(
    n_obs: int,
    window_days: int,
    step_days: int,
    min_windows: int,
    max_windows: int,
) -> list[int]:
    if n_obs <= 0:
        return []
    if n_obs <= window_days:
        return [0]

    starts = list(range(0, n_obs - window_days + 1, step_days))
    if len(starts) < max(1, min_windows):
        dense = np.linspace(0, n_obs - window_days, num=max(1, min_windows))
        starts = sorted({int(round(x)) for x in dense})

    if len(starts) > max(1, max_windows):
        keep = np.linspace(0, len(starts) - 1, num=max_windows)
        starts = sorted({starts[int(round(i))] for i in keep})
    return starts


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
    hedge_type_actions = np.asarray(results.get('hedge_type_actions', []), dtype=int)
    turnover = np.asarray(results.get('turnover', []), dtype=float)
    hedge_ratios = np.asarray(results.get('hedge_ratios', []), dtype=float)
    hedge_costs = np.asarray(results.get('hedge_costs', []), dtype=float)
    hedge_benefits = np.asarray(results.get('hedge_benefits', []), dtype=float)
    cash_weights = np.asarray(results.get('cash_weights', []), dtype=float)
    tx_costs = np.asarray(results.get('transaction_costs', []), dtype=float)
    uncertainty = np.asarray(results.get('uncertainty_score', []), dtype=float)
    hedge_types = np.asarray(results.get('hedge_types', []), dtype=object)
    wealth_rets = _daily_returns_from_path(results.get('wealth', []))

    if len(beliefs) == 0:
        return []

    n_obs = min(
        len(beliefs),
        len(actions),
        len(hedge_actions),
        len(hedge_type_actions) if len(hedge_type_actions) > 0 else len(beliefs),
        len(turnover),
        len(wealth_rets),
        len(hedge_ratios),
        len(hedge_costs) if len(hedge_costs) > 0 else len(beliefs),
        len(hedge_benefits) if len(hedge_benefits) > 0 else len(beliefs),
        len(cash_weights),
        len(tx_costs),
        len(uncertainty),
        len(hedge_types) if len(hedge_types) > 0 else len(beliefs),
    )
    beliefs = beliefs[:n_obs]
    actions = actions[:n_obs]
    hedge_actions = hedge_actions[:n_obs]
    hedge_type_actions = hedge_type_actions[:n_obs] if len(hedge_type_actions) > 0 else np.zeros(n_obs, dtype=int)
    turnover = turnover[:n_obs]
    hedge_ratios = hedge_ratios[:n_obs]
    hedge_costs = hedge_costs[:n_obs] if len(hedge_costs) > 0 else np.zeros(n_obs, dtype=float)
    hedge_benefits = hedge_benefits[:n_obs] if len(hedge_benefits) > 0 else np.zeros(n_obs, dtype=float)
    cash_weights = cash_weights[:n_obs]
    tx_costs = tx_costs[:n_obs]
    uncertainty = uncertainty[:n_obs]
    hedge_types = hedge_types[:n_obs] if len(hedge_types) > 0 else np.array(['none'] * n_obs, dtype=object)
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
        regime_hedge_types = hedge_types[mask]
        dominant_hedge_type = str(pd.Series(regime_hedge_types).mode().iloc[0]) if len(regime_hedge_types) > 0 else 'none'
        rows.append({
            'label': results.get('experiment_label', 'unknown'),
            'regime': regime,
            'avg_action': float(actions[mask].mean()),
            'avg_hedge_action': float(hedge_actions[mask].mean()),
            'avg_hedge_type_action': float(hedge_type_actions[mask].mean()),
            'dominant_hedge_type': dominant_hedge_type,
            'avg_hedge_ratio': float(hedge_ratios[mask].mean()),
            'avg_hedge_cost': float(hedge_costs[mask].mean()),
            'avg_hedge_benefit': float(hedge_benefits[mask].mean()),
            'avg_cash_weight': float(cash_weights[mask].mean()),
            'avg_turnover': float(turnover[mask].mean()),
            'avg_transaction_cost': float(tx_costs[mask].mean()),
            'avg_uncertainty_score': float(uncertainty[mask].mean()),
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

    factor_plus_pairs = copy.deepcopy(base_config)
    factor_plus_pairs.experiment = ExperimentConfig(
        label='factor_plus_pairs',
        use_factor=True,
        use_pairs=True,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    factor_plus_pairs.optimizer.use_optimizer = False
    factor_plus_pairs.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(factor_plus_pairs)

    factor_plus_lstm = copy.deepcopy(base_config)
    factor_plus_lstm.experiment = ExperimentConfig(
        label='factor_plus_lstm',
        use_factor=True,
        use_pairs=False,
        use_lstm=True,
        adaptive_combiner=False,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    factor_plus_lstm.optimizer.use_optimizer = False
    factor_plus_lstm.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(factor_plus_lstm)

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

    # ---- State-feature ablations (RQ: which state info does RL actually use?) ----
    no_uncertainty = copy.deepcopy(base_config)
    no_uncertainty.experiment = ExperimentConfig(
        label='full_no_uncertainty',
        use_uncertainty_state=False,
    )
    configs.append(no_uncertainty)

    no_regime = copy.deepcopy(base_config)
    no_regime.experiment = ExperimentConfig(
        label='full_no_regime',
        use_regime_state=False,
    )
    configs.append(no_regime)

    no_vol = copy.deepcopy(base_config)
    no_vol.experiment = ExperimentConfig(
        label='full_no_vol',
        use_vol_state=False,
    )
    configs.append(no_vol)

    return configs


def _component_label(label: str) -> str:
    return label.rsplit('_tf', 1)[0] if '_tf' in label else label


def _display_label(label: str) -> str:
    mapping = {
        'factor_only': 'Factor Only',
        'factor_plus_pairs': 'Factor +\nPairs',
        'factor_plus_lstm': 'Factor +\nLSTM',
        'alpha_stack_no_rl': 'Alpha Stack\nNo RL',
        'alpha_plus_portfolio_rl': 'Alpha +\nPortfolio RL',
        'alpha_plus_hedge_rl': 'Alpha +\nHedge RL',
        'full_pipeline': 'Full\nPipeline',
        'full_no_uncertainty': 'Full\n(no uncert.)',
        'full_no_regime': 'Full\n(no regime)',
        'full_no_vol': 'Full\n(no vol)',
        'SPY': 'SPY',
        'factor_benchmark': 'Factor Bench',
        'vol_target': 'Vol-Target',
        'dd_delever': 'DD-Delever',
        'e2e_rl': 'E2E RL\n(PPO)',
        'risk_parity': 'Risk Parity',
    }
    if label.startswith('full_pipeline_reward_'):
        reward_name = label.replace('full_pipeline_reward_', '').replace('_', ' ').title()
        return f'Full Pipeline\nReward={reward_name}'
    if label.startswith('e2e_reward_'):
        reward_name = label.replace('e2e_reward_', '').replace('_', ' ').title()
        return f'E2E RL\nReward={reward_name}'
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
    ordering = ['factor_only', 'factor_plus_pairs', 'factor_plus_lstm', 'alpha_stack_no_rl', 'alpha_plus_portfolio_rl', 'alpha_plus_hedge_rl', 'full_pipeline']
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
    bootstrap_significance: pd.DataFrame | None = None,
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

    if bootstrap_significance is not None and not bootstrap_significance.empty:
        lines.extend([
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Block-bootstrap significance for full pipeline versus baselines (Sharpe deltas).}",
            r"\label{tab:bootstrap_significance}",
            r"\begin{tabular}{@{} l r r r @{}}",
            r"\toprule",
            r"Comparison & Delta Sharpe & 95\% CI & p-value \\",
            r"\midrule",
        ])
        for _, row in bootstrap_significance.iterrows():
            delta = float(row['delta_sharpe_point'])
            low = float(row['delta_sharpe_ci_low'])
            high = float(row['delta_sharpe_ci_high'])
            pval = float(row['delta_sharpe_pvalue_two_sided'])
            comp = f"Full - {_table_label(str(row['compare_label']))}"
            lines.append(f"{comp} & {delta:.2f} & [{low:.2f}, {high:.2f}] & {pval:.3f} \\\\")
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


def _checkpoint_path(checkpoint_dir: Path, run_key: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in run_key)
    return checkpoint_dir / f"{safe_name}.pkl"


def _progress_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "research_progress.json"


def _frame_signature(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {'rows': 0, 'columns': [], 'start': None, 'end': None}
    return {
        'rows': int(len(frame)),
        'columns': [str(col) for col in frame.columns],
        'start': str(frame.index[0]),
        'end': str(frame.index[-1]),
    }


def _series_signature(series: pd.Series) -> dict[str, object]:
    if series.empty:
        return {'rows': 0, 'sum': 0.0}
    numeric = pd.to_numeric(series, errors='coerce').fillna(0.0)
    return {
        'rows': int(len(series)),
        'sum': round(float(numeric.sum()), 10),
    }


def _checkpoint_metadata(
    run_prices: pd.DataFrame,
    run_volumes: pd.DataFrame,
    run_returns: pd.DataFrame,
    run_macro: pd.DataFrame,
    sec_quality_scores: pd.Series,
    run_config: PipelineConfig,
    *,
    suite: str,
    include_e2e: bool,
    run_key: str,
) -> dict[str, object]:
    return {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'run_key': run_key,
        'suite': suite,
        'include_e2e': include_e2e,
        'config': asdict(run_config),
        'prices': _frame_signature(run_prices),
        'volumes': _frame_signature(run_volumes),
        'returns': _frame_signature(run_returns),
        'macro': _frame_signature(run_macro),
        'sec_quality': _series_signature(sec_quality_scores),
    }


def _load_checkpoint_results(
    checkpoint_path: Path,
    expected_metadata: dict[str, object],
) -> dict[str, object] | None:
    try:
        with checkpoint_path.open('rb') as handle:
            payload = pickle.load(handle)
    except Exception:
        print(f"  Ignoring unreadable checkpoint at {checkpoint_path}; recomputing.")
        return None

    if not isinstance(payload, dict) or 'results' not in payload or 'metadata' not in payload:
        print(f"  Ignoring legacy checkpoint at {checkpoint_path}; recomputing.")
        return None

    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        print(f"  Ignoring incompatible checkpoint at {checkpoint_path}; recomputing.")
        return None

    if payload.get('metadata') != expected_metadata:
        print(f"  Ignoring mismatched checkpoint at {checkpoint_path}; recomputing.")
        return None

    return payload['results']


def _sanitize_for_checkpoint(value: object) -> object:
    try:
        pickle.dumps(value)
        return value
    except Exception:
        pass

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_for_checkpoint(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_checkpoint(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_checkpoint(item) for item in value)
    if isinstance(value, Path):
        return str(value)

    return {
        '__checkpoint_repr__': repr(value),
        '__checkpoint_type__': type(value).__name__,
    }


def _write_progress_manifest(
    checkpoint_dir: Path,
    *,
    total_runs: int,
    completed_run_keys: list[str],
    status: str,
    current_run: dict[str, object] | None,
    last_completed_run: dict[str, object] | None,
) -> None:
    payload = {
        'updated_at': datetime.now().isoformat(),
        'status': status,
        'total_runs': total_runs,
        'completed_runs': len(completed_run_keys),
        'completed_run_keys': completed_run_keys,
        'current_run': current_run,
        'last_completed_run': last_completed_run,
    }
    _progress_path(checkpoint_dir).write_text(json.dumps(payload, indent=2), encoding='utf-8')


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
    checkpoint_dir = Path(evaluation_config.checkpoint_dir)
    if evaluation_config.enable_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Research checkpoints enabled: {checkpoint_dir}")

    metric_rows: list[dict[str, float | str]] = []
    regime_rows: list[dict[str, float | str]] = []
    rolling_reference_rows: list[dict[str, float | str]] = []
    baseline_results: dict[str, object] | None = None
    deferred_baseline_config: PipelineConfig | None = None
    first_ppo_logs_shown = False
    total_runs = (
        len(build_ablation_suite(base_config)) * len(evaluation_config.train_fracs)
        + (
            0
            if any(abs(train_frac - base_config.train_frac) < 1e-12 for train_frac in evaluation_config.train_fracs)
            else 1
        )
        + len(_rolling_starts(
            n_obs=len(returns),
            window_days=evaluation_config.rolling_window_days,
            step_days=evaluation_config.rolling_step_days,
            min_windows=evaluation_config.min_rolling_windows,
            max_windows=evaluation_config.max_rolling_windows,
        ))
        + len(evaluation_config.cost_bps_grid)
        + len(evaluation_config.rebalance_band_grid)
        + len(evaluation_config.hedge_scale_grid)
        + len(evaluation_config.macro_lag_grid)
        + len(evaluation_config.reward_mode_grid)
    )
    run_counter = 0
    completed_run_keys: list[str] = []
    last_completed_run: dict[str, object] | None = None

    if evaluation_config.enable_checkpoints:
        _write_progress_manifest(
            checkpoint_dir,
            total_runs=total_runs,
            completed_run_keys=completed_run_keys,
            status='running',
            current_run=None,
            last_completed_run=last_completed_run,
        )

    def _run_with_research_logging(
        run_prices: pd.DataFrame,
        run_volumes: pd.DataFrame,
        run_returns: pd.DataFrame,
        run_macro: pd.DataFrame,
        run_config: PipelineConfig,
        *,
        suite: str,
        include_e2e: bool,
        run_key: str,
    ) -> dict[str, object]:
        nonlocal first_ppo_logs_shown, run_counter, last_completed_run
        run_counter += 1
        run_config.enable_e2e_baseline = include_e2e
        print(
            f"\nResearch run {run_counter}/{total_runs}: "
            f"{suite} [{run_config.experiment.label}]"
        )
        checkpoint_path = _checkpoint_path(checkpoint_dir, run_key)
        checkpoint_metadata = _checkpoint_metadata(
            run_prices,
            run_volumes,
            run_returns,
            run_macro,
            sec_quality_scores,
            run_config,
            suite=suite,
            include_e2e=include_e2e,
            run_key=run_key,
        )
        current_run = {
            'ordinal': run_counter,
            'total_runs': total_runs,
            'suite': suite,
            'label': run_config.experiment.label,
            'run_key': run_key,
            'include_e2e': include_e2e,
        }
        if evaluation_config.enable_checkpoints:
            _write_progress_manifest(
                checkpoint_dir,
                total_runs=total_runs,
                completed_run_keys=completed_run_keys,
                status='running',
                current_run=current_run,
                last_completed_run=last_completed_run,
            )
        if evaluation_config.enable_checkpoints and checkpoint_path.exists():
            cached_results = _load_checkpoint_results(checkpoint_path, checkpoint_metadata)
            if cached_results is not None:
                print(f"  Loading cached result from {checkpoint_path}")
                if include_e2e:
                    first_ppo_logs_shown = True
                completed_run_keys.append(run_key)
                last_completed_run = {
                    **current_run,
                    'source': 'cache',
                    'completed_at': datetime.now().isoformat(),
                    'checkpoint_path': str(checkpoint_path),
                }
                _write_progress_manifest(
                    checkpoint_dir,
                    total_runs=total_runs,
                    completed_run_keys=completed_run_keys,
                    status='running',
                    current_run=None,
                    last_completed_run=last_completed_run,
                )
                return cached_results
        if include_e2e and not first_ppo_logs_shown:
            run_config.e2e_ppo_verbose = 1
            run_config.e2e_ppo_log_interval = 10
            print("  Research note: showing PPO training logs for this first run only.")
            first_ppo_logs_shown = True
        else:
            run_config.e2e_ppo_verbose = 0
        results = run_full_pipeline(
            run_prices,
            run_volumes,
            run_returns,
            macro_data=run_macro,
            sec_quality_scores=sec_quality_scores,
            config=run_config,
        )
        if evaluation_config.enable_checkpoints:
            checkpoint_payload = {
                'schema_version': CHECKPOINT_SCHEMA_VERSION,
                'saved_at': datetime.now().isoformat(),
                'metadata': checkpoint_metadata,
                'results': _sanitize_for_checkpoint(results),
            }
            temp_checkpoint_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
            with temp_checkpoint_path.open('wb') as handle:
                pickle.dump(checkpoint_payload, handle)
            temp_checkpoint_path.replace(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            completed_run_keys.append(run_key)
            last_completed_run = {
                **current_run,
                'source': 'fresh',
                'completed_at': datetime.now().isoformat(),
                'checkpoint_path': str(checkpoint_path),
            }
            _write_progress_manifest(
                checkpoint_dir,
                total_runs=total_runs,
                completed_run_keys=completed_run_keys,
                status='running',
                current_run=None,
                last_completed_run=last_completed_run,
            )
        return results

    for config in build_ablation_suite(base_config):
        base_label = config.experiment.label
        for train_frac in evaluation_config.train_fracs:
            run_config = copy.deepcopy(config)
            run_config.train_frac = train_frac
            run_config.experiment.label = f'{base_label}_tf{train_frac:.2f}'
            should_defer_baseline_e2e = (
                evaluation_config.research_e2e_scope == 'baseline_only'
                and base_label == 'full_pipeline'
                and abs(train_frac - base_config.train_frac) < 1e-12
            )
            if should_defer_baseline_e2e:
                deferred_baseline_config = copy.deepcopy(run_config)
                continue
            results = _run_with_research_logging(
                prices,
                volumes,
                returns,
                macro_data,
                run_config,
                suite='ablation',
                include_e2e=evaluation_config.research_e2e_scope == 'all',
                run_key=f"ablation_{run_config.experiment.label}",
            )
            row = _metric_summary(results)
            row.update({'suite': 'ablation', 'window_id': 'full_sample', 'param_name': 'train_frac', 'param_value': train_frac})
            metric_rows.append(row)
            regime_rows.extend(_regime_summary(results))
            if base_label == 'full_pipeline' and abs(train_frac - base_config.train_frac) < 1e-12:
                baseline_results = results

    strict_timing = copy.deepcopy(base_config)
    strict_timing.feature_availability.macro_lag_days = max(evaluation_config.macro_lag_grid)
    strict_timing.feature_availability.allow_static_sec_quality = False
    strict_timing.experiment.label = 'full_pipeline_strict_timing'
    strict_results = _run_with_research_logging(
        prices,
        volumes,
        returns,
        macro_data,
        strict_timing,
        suite='timing_discipline',
        include_e2e=evaluation_config.research_e2e_scope == 'all',
        run_key='timing_discipline_full_pipeline_strict_timing',
    )
    strict_row = _metric_summary(strict_results)
    strict_row.update({'suite': 'timing_discipline', 'window_id': 'full_sample', 'param_name': 'strict_timing', 'param_value': 1})
    metric_rows.append(strict_row)
    regime_rows.extend(_regime_summary(strict_results))

    starts = _rolling_starts(
        n_obs=len(returns),
        window_days=evaluation_config.rolling_window_days,
        step_days=evaluation_config.rolling_step_days,
        min_windows=evaluation_config.min_rolling_windows,
        max_windows=evaluation_config.max_rolling_windows,
    )
    for window_id, start in enumerate(starts):
        end = start + evaluation_config.rolling_window_days
        if end > len(returns):
            continue
        p_slice, v_slice, r_slice, m_slice = _slice_inputs(prices, volumes, returns, macro_data, start, end)
        rolling_config = copy.deepcopy(base_config)
        rolling_config.train_frac = evaluation_config.rolling_train_frac
        rolling_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            p_slice,
            v_slice,
            r_slice,
            m_slice,
            rolling_config,
            suite='rolling_window',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"rolling_window_{window_id}",
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
        voltarget_row = _path_metric_summary(results['voltarget'], 'vol_target')
        voltarget_row.update({'suite': 'rolling_reference', 'window_id': window_id, 'param_name': 'benchmark', 'param_value': 'vol_target'})
        rolling_reference_rows.append(voltarget_row)
        ddlever_row = _path_metric_summary(results['ddlever'], 'dd_delever')
        ddlever_row.update({'suite': 'rolling_reference', 'window_id': window_id, 'param_name': 'benchmark', 'param_value': 'dd_delever'})
        rolling_reference_rows.append(ddlever_row)

    for base_cost_bps in evaluation_config.cost_bps_grid:
        cost_config = copy.deepcopy(base_config)
        cost_config.cost_model.base_cost_bps = base_cost_bps
        cost_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            cost_config,
            suite='cost_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"cost_sensitivity_{base_cost_bps}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'cost_sensitivity', 'window_id': 'full_sample', 'param_name': 'base_cost_bps', 'param_value': base_cost_bps})
        metric_rows.append(row)

    for rebalance_band in evaluation_config.rebalance_band_grid:
        band_config = copy.deepcopy(base_config)
        band_config.rebalance_band = rebalance_band
        band_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            band_config,
            suite='rebalance_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"rebalance_sensitivity_{rebalance_band}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'rebalance_sensitivity', 'window_id': 'full_sample', 'param_name': 'rebalance_band', 'param_value': rebalance_band})
        metric_rows.append(row)

    for hedge_scale in evaluation_config.hedge_scale_grid:
        hedge_config = copy.deepcopy(base_config)
        hedge_config.hedge_ratios = tuple(round(h * hedge_scale, 4) for h in hedge_config.hedge_ratios)
        hedge_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            hedge_config,
            suite='hedge_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"hedge_sensitivity_{hedge_scale}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'hedge_sensitivity', 'window_id': 'full_sample', 'param_name': 'hedge_scale', 'param_value': hedge_scale})
        metric_rows.append(row)

    for macro_lag in evaluation_config.macro_lag_grid:
        lag_config = copy.deepcopy(base_config)
        lag_config.feature_availability.macro_lag_days = macro_lag
        lag_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            lag_config,
            suite='macro_lag_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"macro_lag_sensitivity_{macro_lag}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'macro_lag_sensitivity', 'window_id': 'full_sample', 'param_name': 'macro_lag_days', 'param_value': macro_lag})
        metric_rows.append(row)

    for reward_mode in evaluation_config.reward_mode_grid:
        reward_config = copy.deepcopy(base_config)
        reward_config.portfolio_reward_mode = reward_mode
        reward_config.hedge_reward_mode = reward_mode if reward_mode != 'differential_sharpe' else 'asymmetric_return'
        reward_config.e2e_reward_mode = reward_mode
        reward_config.experiment.label = f'full_pipeline_reward_{reward_mode}'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            reward_config,
            suite='reward_ablation',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"reward_ablation_{reward_mode}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'reward_ablation', 'window_id': 'full_sample', 'param_name': 'reward_mode', 'param_value': reward_mode})
        metric_rows.append(row)

        e2e_row = _path_metric_summary(results.get('e2e_rl', [1.0]), f'e2e_reward_{reward_mode}')
        if 'ann_return' in e2e_row:
            e2e_row.update({'suite': 'reward_ablation', 'window_id': 'full_sample', 'param_name': 'reward_mode', 'param_value': f'e2e_{reward_mode}'})
            metric_rows.append(e2e_row)

    if deferred_baseline_config is not None:
        baseline_results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            deferred_baseline_config,
            suite='ablation',
            include_e2e=True,
            run_key=f"ablation_{deferred_baseline_config.experiment.label}",
        )
        row = _metric_summary(baseline_results)
        row.update({'suite': 'ablation', 'window_id': 'full_sample', 'param_name': 'train_frac', 'param_value': deferred_baseline_config.train_frac})
        metric_rows.append(row)
        regime_rows.extend(_regime_summary(baseline_results))

    if baseline_results is None:
        baseline_config = copy.deepcopy(base_config)
        baseline_config.experiment.label = 'full_pipeline_baseline'
        baseline_results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            baseline_config,
            suite='baseline_backfill',
            include_e2e=evaluation_config.research_e2e_scope in {'all', 'baseline_only'},
            run_key='baseline_backfill_full_pipeline_baseline',
        )

    metrics = pd.DataFrame(metric_rows)
    regime_summary = pd.DataFrame(regime_rows)
    rolling_references = pd.DataFrame(rolling_reference_rows)
    ablation_summary = _build_ablation_summary(metrics)
    robustness_summary = _build_robustness_summary(metrics, rolling_references)
    bootstrap_cis = pd.DataFrame()
    bootstrap_significance = pd.DataFrame()

    if baseline_results is not None:
        bootstrap_paths = {
            'full_pipeline': baseline_results.get('wealth', [1.0]),
            'SPY': baseline_results.get('spy', [1.0]),
            'factor_benchmark': baseline_results.get('factor', [1.0]),
            'vol_target': baseline_results.get('voltarget', [1.0]),
            'dd_delever': baseline_results.get('ddlever', [1.0]),
            'e2e_rl': baseline_results.get('e2e_rl', [1.0]),
        }
        bootstrap_cis = _compute_bootstrap_cis(
            path_map=bootstrap_paths,
            n_samples=evaluation_config.bootstrap_samples,
            block_size=evaluation_config.bootstrap_block_size,
            seed=evaluation_config.bootstrap_seed,
        )
        bootstrap_significance = _compute_bootstrap_pairwise_significance(
            path_map=bootstrap_paths,
            base_label='full_pipeline',
            compare_labels=['SPY', 'factor_benchmark', 'vol_target', 'dd_delever', 'e2e_rl'],
            n_samples=evaluation_config.bootstrap_samples,
            block_size=evaluation_config.bootstrap_block_size,
            seed=evaluation_config.bootstrap_seed,
        )

    # Jobson-Korkie Sharpe ratio equality test
    jk_table = pd.DataFrame()
    if baseline_results is not None:
        jk_table = _compute_jobson_korkie_table(
            path_map=bootstrap_paths,
            base_label='full_pipeline',
            compare_labels=['SPY', 'factor_benchmark', 'vol_target', 'dd_delever', 'e2e_rl'],
        )
        if not jk_table.empty:
            jk_table.to_csv('stock_trading/research_jobson_korkie.csv', index=False)
            print("\nJobson-Korkie Sharpe ratio tests:")
            for _, row in jk_table.iterrows():
                print(f"  Full Pipeline vs {row['compare_label']}: "
                      f"JK stat={row['jk_stat']:.3f}, p={row['jk_pvalue']:.4f}")

    # Time-series cross-validation
    ts_cv_results = pd.DataFrame()
    if evaluation_config.enable_ts_cv:
        print("\n--- Time-Series Cross-Validation ---")
        ts_cv_results = _run_time_series_cv(
            prices, volumes, returns, macro_data, sec_quality_scores,
            base_config, evaluation_config.ts_cv_folds,
        )
        if not ts_cv_results.empty:
            ts_cv_results.to_csv('stock_trading/research_ts_cv.csv', index=False)
            print(f"\n  TS-CV Summary ({len(ts_cv_results)} folds):")
            print(f"    Sharpe: {ts_cv_results['sharpe'].mean():.2f} "
                  f"± {ts_cv_results['sharpe'].std():.2f}")
            print(f"    Calmar: {ts_cv_results['calmar'].mean():.2f} "
                  f"± {ts_cv_results['calmar'].std():.2f}")
            print(f"    MaxDD:  {ts_cv_results['max_drawdown'].mean():.1%} "
                  f"± {ts_cv_results['max_drawdown'].std():.1%}")
            # Add CV rows to main metrics
            for _, row in ts_cv_results.iterrows():
                metric_rows.append(dict(row))
            metrics = pd.DataFrame(metric_rows)

    metrics.to_csv('stock_trading/research_metrics.csv', index=False)
    regime_summary.to_csv('stock_trading/research_regime_summary.csv', index=False)
    rolling_references.to_csv('stock_trading/research_rolling_references.csv', index=False)
    ablation_summary.to_csv('stock_trading/research_ablation_summary.csv', index=False)
    robustness_summary.to_csv('stock_trading/research_robustness_summary.csv', index=False)
    if not bootstrap_cis.empty:
        bootstrap_cis.to_csv('stock_trading/research_bootstrap_cis.csv', index=False)
    if not bootstrap_significance.empty:
        bootstrap_significance.to_csv('stock_trading/research_bootstrap_significance.csv', index=False)
    _write_research_tables(ablation_summary, robustness_summary, bootstrap_significance=bootstrap_significance)
    plot_research_evaluation(metrics, regime_summary, baseline_results=baseline_results, rolling_references=rolling_references)
    plot_rolling_windows(metrics, rolling_references_df=rolling_references if not rolling_references.empty else None)
    plot_reward_ablation(metrics)

    summary = {
        'base_config': asdict(base_config),
        'evaluation_config': asdict(evaluation_config),
        'n_metric_rows': len(metrics),
        'n_regime_rows': len(regime_summary),
        'ablation_summary': ablation_summary.to_dict(orient='records'),
        'robustness_summary': robustness_summary.to_dict(orient='records'),
        'bootstrap_cis': bootstrap_cis.to_dict(orient='records'),
        'bootstrap_significance': bootstrap_significance.to_dict(orient='records'),
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

    if evaluation_config.enable_checkpoints:
        _write_progress_manifest(
            checkpoint_dir,
            total_runs=total_runs,
            completed_run_keys=completed_run_keys,
            status='completed',
            current_run=None,
            last_completed_run=last_completed_run,
        )

    return {
        'metrics': metrics,
        'regime_summary': regime_summary,
        'ablation_summary': ablation_summary,
        'robustness_summary': robustness_summary,
        'bootstrap_cis': bootstrap_cis,
        'bootstrap_significance': bootstrap_significance,
        'summary': summary,
    }
