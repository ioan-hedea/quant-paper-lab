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

from .config import (
    CONTROL_METHODS,
    ControlConfig,
    EvaluationConfig,
    ExperimentConfig,
    PipelineConfig,
    RISK_FREE_RATE,
)
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
    invested_fractions = np.asarray(results.get('invested_fractions', []), dtype=float)
    overlay_sizes = np.asarray(results.get('overlay_sizes', []), dtype=float)
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
    convexity_modes = np.asarray(results.get('convexity_modes', []), dtype=int)
    convexity_mode_names = np.asarray(results.get('convexity_mode_names', []), dtype=object)
    convexity_carries = np.asarray(results.get('convexity_carries', []), dtype=float)
    convexity_benefits = np.asarray(results.get('convexity_benefits', []), dtype=float)
    council_weight_regime = np.asarray(results.get('council_weight_regime_rules', []), dtype=float)
    council_weight_linucb = np.asarray(results.get('council_weight_linucb', []), dtype=float)
    council_weight_cvar = np.asarray(results.get('council_weight_cvar_robust', []), dtype=float)
    council_dominant = np.asarray(results.get('council_dominant_expert', []), dtype=object)
    council_best = np.asarray(results.get('council_best_expert', []), dtype=object)
    council_entropy = np.asarray(results.get('council_gate_entropy', []), dtype=float)
    wealth_rets = _daily_returns_from_path(results.get('wealth', []))

    if len(beliefs) == 0:
        return []

    n_obs = min(
        len(beliefs),
        len(actions),
        len(invested_fractions) if len(invested_fractions) > 0 else len(beliefs),
        len(overlay_sizes) if len(overlay_sizes) > 0 else len(beliefs),
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
        len(convexity_modes) if len(convexity_modes) > 0 else len(beliefs),
        len(convexity_mode_names) if len(convexity_mode_names) > 0 else len(beliefs),
        len(convexity_carries) if len(convexity_carries) > 0 else len(beliefs),
        len(convexity_benefits) if len(convexity_benefits) > 0 else len(beliefs),
        len(council_weight_regime) if len(council_weight_regime) > 0 else len(beliefs),
        len(council_weight_linucb) if len(council_weight_linucb) > 0 else len(beliefs),
        len(council_weight_cvar) if len(council_weight_cvar) > 0 else len(beliefs),
        len(council_dominant) if len(council_dominant) > 0 else len(beliefs),
        len(council_best) if len(council_best) > 0 else len(beliefs),
        len(council_entropy) if len(council_entropy) > 0 else len(beliefs),
    )
    beliefs = beliefs[:n_obs]
    actions = actions[:n_obs]
    invested_fractions = invested_fractions[:n_obs] if len(invested_fractions) > 0 else np.zeros(n_obs, dtype=float)
    overlay_sizes = overlay_sizes[:n_obs] if len(overlay_sizes) > 0 else np.zeros(n_obs, dtype=float)
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
    convexity_modes = convexity_modes[:n_obs] if len(convexity_modes) > 0 else np.zeros(n_obs, dtype=int)
    convexity_mode_names = convexity_mode_names[:n_obs] if len(convexity_mode_names) > 0 else np.array(['none'] * n_obs, dtype=object)
    convexity_carries = convexity_carries[:n_obs] if len(convexity_carries) > 0 else np.zeros(n_obs, dtype=float)
    convexity_benefits = convexity_benefits[:n_obs] if len(convexity_benefits) > 0 else np.zeros(n_obs, dtype=float)
    council_weight_regime = council_weight_regime[:n_obs] if len(council_weight_regime) > 0 else np.zeros(n_obs, dtype=float)
    council_weight_linucb = council_weight_linucb[:n_obs] if len(council_weight_linucb) > 0 else np.zeros(n_obs, dtype=float)
    council_weight_cvar = council_weight_cvar[:n_obs] if len(council_weight_cvar) > 0 else np.zeros(n_obs, dtype=float)
    council_dominant = council_dominant[:n_obs] if len(council_dominant) > 0 else np.array(['none'] * n_obs, dtype=object)
    council_best = council_best[:n_obs] if len(council_best) > 0 else np.array(['none'] * n_obs, dtype=object)
    council_entropy = council_entropy[:n_obs] if len(council_entropy) > 0 else np.zeros(n_obs, dtype=float)
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
        dominant_convexity_mode = str(pd.Series(convexity_mode_names[mask]).mode().iloc[0]) if mask.sum() > 0 else 'none'
        dominant_council_expert = str(pd.Series(council_dominant[mask]).mode().iloc[0]) if mask.sum() > 0 else 'none'
        best_council_expert = str(pd.Series(council_best[mask]).mode().iloc[0]) if mask.sum() > 0 else 'none'
        rows.append({
            'label': results.get('experiment_label', 'unknown'),
            'regime': regime,
            'avg_action': float(actions[mask].mean()),
            'avg_invested_fraction': float(invested_fractions[mask].mean()),
            'avg_overlay_size': float(overlay_sizes[mask].mean()),
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
            'avg_convexity_mode': float(convexity_modes[mask].mean()),
            'dominant_convexity_mode': dominant_convexity_mode,
            'avg_convexity_carry': float(convexity_carries[mask].mean()),
            'avg_convexity_benefit': float(convexity_benefits[mask].mean()),
            'avg_council_weight_regime_rules': float(council_weight_regime[mask].mean()),
            'avg_council_weight_linucb': float(council_weight_linucb[mask].mean()),
            'avg_council_weight_cvar_robust': float(council_weight_cvar[mask].mean()),
            'avg_council_gate_entropy': float(council_entropy[mask].mean()),
            'dominant_council_expert': dominant_council_expert,
            'best_council_expert': best_council_expert,
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


def build_control_comparison_suite(base_config: PipelineConfig) -> list[PipelineConfig]:
    """Build configs for comparing all control candidates from architecture v2.

    Each candidate shares the same alpha layer (factor + GARCH + HMM + adaptive combiner)
    and constrained allocator. Only the control layer differs.
    """
    configs: list[PipelineConfig] = []

    # Shared experiment settings for all control candidates
    shared_experiment = ExperimentConfig(
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=True,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )

    # 0. Factor-only (no control)
    factor_only = copy.deepcopy(base_config)
    factor_only.experiment = copy.deepcopy(shared_experiment)
    factor_only.experiment.label = 'factor_only'
    factor_only.control = ControlConfig(method='none')
    factor_only.enable_e2e_baseline = False
    configs.append(factor_only)

    # A1: Fixed allocator (95% invested)
    fixed = copy.deepcopy(base_config)
    fixed.experiment = copy.deepcopy(shared_experiment)
    fixed.experiment.label = 'A1_fixed'
    fixed.experiment.control_method = 'fixed'
    fixed.control = ControlConfig(method='fixed', fixed_invested_fraction=0.95)
    fixed.enable_e2e_baseline = False
    configs.append(fixed)

    # A2: Vol-target
    vol_target = copy.deepcopy(base_config)
    vol_target.experiment = copy.deepcopy(shared_experiment)
    vol_target.experiment.label = 'A2_vol_target'
    vol_target.experiment.control_method = 'vol_target'
    vol_target.control = ControlConfig(method='vol_target', vol_target_annual=0.12)
    vol_target.enable_e2e_baseline = False
    configs.append(vol_target)

    # A3: DD-delever
    dd_delever = copy.deepcopy(base_config)
    dd_delever.experiment = copy.deepcopy(shared_experiment)
    dd_delever.experiment.label = 'A3_dd_delever'
    dd_delever.experiment.control_method = 'dd_delever'
    dd_delever.control = ControlConfig(method='dd_delever')
    dd_delever.enable_e2e_baseline = False
    configs.append(dd_delever)

    # A4: Regime rules
    regime = copy.deepcopy(base_config)
    regime.experiment = copy.deepcopy(shared_experiment)
    regime.experiment.label = 'A4_regime_rules'
    regime.experiment.control_method = 'regime_rules'
    regime.control = ControlConfig(method='regime_rules')
    regime.enable_e2e_baseline = False
    configs.append(regime)

    # A5: Ensemble (mean)
    ensemble_mean = copy.deepcopy(base_config)
    ensemble_mean.experiment = copy.deepcopy(shared_experiment)
    ensemble_mean.experiment.label = 'A5_ensemble_mean'
    ensemble_mean.experiment.control_method = 'ensemble_rules'
    ensemble_mean.control = ControlConfig(method='ensemble_rules', ensemble_mode='mean')
    ensemble_mean.enable_e2e_baseline = False
    configs.append(ensemble_mean)

    # A5b: Ensemble (min)
    ensemble_min = copy.deepcopy(base_config)
    ensemble_min.experiment = copy.deepcopy(shared_experiment)
    ensemble_min.experiment.label = 'A5_ensemble_min'
    ensemble_min.experiment.control_method = 'ensemble_rules'
    ensemble_min.control = ControlConfig(method='ensemble_rules', ensemble_mode='min')
    ensemble_min.enable_e2e_baseline = False
    configs.append(ensemble_min)

    # B1: LinUCB
    linucb = copy.deepcopy(base_config)
    linucb.experiment = copy.deepcopy(shared_experiment)
    linucb.experiment.label = 'B1_linucb'
    linucb.experiment.control_method = 'linucb'
    linucb.control = ControlConfig(method='linucb')
    linucb.enable_e2e_baseline = False
    configs.append(linucb)

    # B2: Thompson Sampling
    thompson = copy.deepcopy(base_config)
    thompson.experiment = copy.deepcopy(shared_experiment)
    thompson.experiment.label = 'B2_thompson'
    thompson.experiment.control_method = 'thompson'
    thompson.control = ControlConfig(method='thompson')
    thompson.enable_e2e_baseline = False
    configs.append(thompson)

    # B3: Epsilon-greedy
    eps_greedy = copy.deepcopy(base_config)
    eps_greedy.experiment = copy.deepcopy(shared_experiment)
    eps_greedy.experiment.label = 'B3_epsilon_greedy'
    eps_greedy.experiment.control_method = 'epsilon_greedy'
    eps_greedy.control = ControlConfig(method='epsilon_greedy')
    eps_greedy.enable_e2e_baseline = False
    configs.append(eps_greedy)

    # C: Supervised controller (logistic)
    supervised = copy.deepcopy(base_config)
    supervised.experiment = copy.deepcopy(shared_experiment)
    supervised.experiment.label = 'C_supervised'
    supervised.experiment.control_method = 'supervised'
    supervised.control = ControlConfig(method='supervised', supervised_model='logistic')
    supervised.enable_e2e_baseline = False
    configs.append(supervised)

    # D: CVaR-robust optimizer
    cvar = copy.deepcopy(base_config)
    cvar.experiment = copy.deepcopy(shared_experiment)
    cvar.experiment.label = 'D_cvar_robust'
    cvar.experiment.control_method = 'cvar_robust'
    cvar.control = ControlConfig(method='cvar_robust')
    cvar.enable_e2e_baseline = False
    configs.append(cvar)

    # D+: CVaR plus convexity-aware payoff shaping
    cvar_convex = copy.deepcopy(base_config)
    cvar_convex.experiment = copy.deepcopy(shared_experiment)
    cvar_convex.experiment.label = 'D_plus_convexity'
    cvar_convex.experiment.control_method = 'cvar_robust'
    cvar_convex.control = ControlConfig(
        method='cvar_robust',
        convexity_enabled=True,
    )
    cvar_convex.enable_e2e_baseline = False
    configs.append(cvar_convex)

    # E: Expert-gated council
    council = copy.deepcopy(base_config)
    council.experiment = copy.deepcopy(shared_experiment)
    council.experiment.label = 'E_council'
    council.experiment.control_method = 'council'
    council.control = ControlConfig(method='council')
    council.enable_e2e_baseline = False
    configs.append(council)

    # E+: Expert-gated council plus convexity
    council_convex = copy.deepcopy(base_config)
    council_convex.experiment = copy.deepcopy(shared_experiment)
    council_convex.experiment.label = 'E_plus_convexity'
    council_convex.experiment.control_method = 'council'
    council_convex.control = ControlConfig(
        method='council',
        convexity_enabled=True,
    )
    council_convex.enable_e2e_baseline = False
    configs.append(council_convex)

    # G: MLP-gated meta-controller
    mlp_meta = copy.deepcopy(base_config)
    mlp_meta.experiment = copy.deepcopy(shared_experiment)
    mlp_meta.experiment.label = 'G_mlp_meta'
    mlp_meta.experiment.control_method = 'mlp_meta'
    mlp_meta.control = ControlConfig(method='mlp_meta')
    mlp_meta.enable_e2e_baseline = False
    configs.append(mlp_meta)

    # G+: MLP meta-controller plus convexity
    mlp_meta_convex = copy.deepcopy(base_config)
    mlp_meta_convex.experiment = copy.deepcopy(shared_experiment)
    mlp_meta_convex.experiment.label = 'G_plus_convexity'
    mlp_meta_convex.experiment.control_method = 'mlp_meta'
    mlp_meta_convex.control = ControlConfig(
        method='mlp_meta',
        convexity_enabled=True,
    )
    mlp_meta_convex.enable_e2e_baseline = False
    configs.append(mlp_meta_convex)

    # F: CMDP-style constrained controller
    cmdp = copy.deepcopy(base_config)
    cmdp.experiment = copy.deepcopy(shared_experiment)
    cmdp.experiment.label = 'F_cmdp_lagrangian'
    cmdp.experiment.control_method = 'cmdp_lagrangian'
    cmdp.control = ControlConfig(method='cmdp_lagrangian')
    cmdp.enable_e2e_baseline = False
    configs.append(cmdp)

    # RL: Tabular Q-learning (portfolio only, minimal state)
    q_learning = copy.deepcopy(base_config)
    q_learning.experiment = copy.deepcopy(shared_experiment)
    q_learning.experiment.label = 'RL_q_learning'
    q_learning.experiment.control_method = 'q_learning'
    q_learning.control = ControlConfig(method='q_learning')
    q_learning.enable_e2e_baseline = False
    configs.append(q_learning)

    return configs


def _control_train_fracs(
    ctrl_config: PipelineConfig,
    evaluation_config: EvaluationConfig,
) -> tuple[float, ...]:
    """Allow controller-specific split overrides while keeping the grid size stable."""
    adjusted: list[float] = []
    for train_frac in evaluation_config.train_fracs:
        candidate = float(train_frac)
        if ctrl_config.control.method == 'q_learning' and abs(candidate - 0.50) < 1e-12:
            candidate = 0.75
        if not any(abs(candidate - existing) < 1e-12 for existing in adjusted):
            adjusted.append(candidate)
    return tuple(adjusted)


def _control_reference_train_frac(
    ctrl_config: PipelineConfig,
    base_config: PipelineConfig,
) -> float:
    """Return the representative train split used for controller-to-controller comparisons."""
    if ctrl_config.control.method == 'q_learning':
        return 0.75
    return float(base_config.train_frac)


def build_ablation_suite(base_config: PipelineConfig) -> list[PipelineConfig]:
    """Legacy ablation suite — retained for backwards compatibility.

    For the revised architecture, use ``build_control_comparison_suite`` instead.
    """
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

    alpha_stack_fixed = copy.deepcopy(base_config)
    alpha_stack_fixed.experiment = ExperimentConfig(
        label='alpha_stack_fixed_weights',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    configs.append(alpha_stack_fixed)

    alpha_stack = copy.deepcopy(base_config)
    alpha_stack.experiment = ExperimentConfig(
        label='alpha_stack_no_rl',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=True,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    configs.append(alpha_stack)

    portfolio_fixed = copy.deepcopy(base_config)
    portfolio_fixed.experiment = ExperimentConfig(
        label='portfolio_rl_fixed_weights',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=True,
        use_hedge_rl=False,
    )
    portfolio_fixed.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(portfolio_fixed)

    full_pipeline = copy.deepcopy(base_config)
    full_pipeline.experiment = ExperimentConfig(label='full_pipeline')
    full_pipeline.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(full_pipeline)

    full_fixed = copy.deepcopy(base_config)
    full_fixed.experiment = ExperimentConfig(
        label='full_pipeline_fixed_weights',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=True,
        use_hedge_rl=False,
    )
    full_fixed.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
    configs.append(full_fixed)

    return configs


def _component_label(label: str) -> str:
    return label.rsplit('_tf', 1)[0] if '_tf' in label else label


def _control_component_label(label: str) -> str:
    base = _component_label(label)
    if base == 'factor_only':
        return 'alpha_engine_no_control'
    return base


def _display_label(label: str) -> str:
    mapping = {
        'factor_only': 'Factor Only',
        'alpha_engine_no_control': 'No Control\n(Alpha Engine)',
        'alpha_stack_fixed_weights': 'Allocator\nFixed Weights',
        'alpha_stack_no_rl': 'Alpha Stack\nNo RL',
        'portfolio_rl_fixed_weights': 'Portfolio RL\nFixed Weights',
        'full_pipeline': 'Full\nPipeline',
        'full_pipeline_fixed_weights': 'Full Pipeline\nFixed Weights',
        'SPY': 'SPY',
        'factor_benchmark': 'Factor Benchmark',
        'vol_target': 'Vol-Target',
        'dd_delever': 'DD-Delever',
        'e2e_rl': 'E2E RL\n(PPO)',
        'risk_parity': 'Risk Parity',
        # Control-method comparison labels (architecture v2)
        'A1_fixed': 'A1: Fixed',
        'A2_vol_target': 'A2: Vol-Target',
        'A3_dd_delever': 'A3: DD-Delever',
        'A4_regime_rules': 'A4: Regime Rules',
        'A5_ensemble_mean': 'A5: Ensemble\n(mean)',
        'A5_ensemble_min': 'A5: Ensemble\n(min)',
        'B1_linucb': 'B1: LinUCB',
        'B2_thompson': 'B2: Thompson',
        'B3_epsilon_greedy': 'B3: Eps-Greedy',
        'C_supervised': 'C: Supervised',
        'D_cvar_robust': 'D: CVaR-Robust',
        'D_plus_convexity': 'D+: CVaR + Convexity',
        'E_council': 'E: Council',
        'E_plus_convexity': 'E+: Council + Convexity',
        'G_mlp_meta': 'G: MLP Meta',
        'G_plus_convexity': 'G+: MLP Meta + Convexity',
        'F_cmdp_lagrangian': 'F: CMDP-Lagrangian',
        'RL_q_learning': 'RL: Q-Learning',
        'RL_ppo': 'RL: PPO',
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
    ordering = [
        'factor_only',
        'alpha_stack_fixed_weights',
        'alpha_stack_no_rl',
        'portfolio_rl_fixed_weights',
        'full_pipeline_fixed_weights',
        'full_pipeline',
    ]
    summary['order'] = summary['component_label'].apply(lambda x: ordering.index(x) if x in ordering else len(ordering))
    summary = summary.sort_values('order').drop(columns='order')
    return summary


def _build_control_comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the primary comparison table for architecture revision v2."""
    control = metrics[metrics['suite'] == 'control_comparison'].copy()
    if control.empty:
        return pd.DataFrame()

    control['component_label'] = control['label'].map(_control_component_label)
    summary = (
        control.groupby('component_label')
        .agg(
            mean_return=('ann_return', 'mean'),
            mean_vol=('ann_vol', 'mean'),
            mean_sharpe=('sharpe', 'mean'),
            mean_max_drawdown=('max_drawdown', 'mean'),
            mean_calmar=('calmar', 'mean'),
        )
        .reset_index()
    )
    ordering = [
        'alpha_engine_no_control',
        'A1_fixed',
        'A2_vol_target',
        'A3_dd_delever',
        'A4_regime_rules',
        'A5_ensemble_mean',
        'A5_ensemble_min',
        'B1_linucb',
        'B2_thompson',
        'B3_epsilon_greedy',
        'C_supervised',
        'D_cvar_robust',
        'D_plus_convexity',
        'E_council',
        'E_plus_convexity',
        'G_mlp_meta',
        'G_plus_convexity',
        'F_cmdp_lagrangian',
        'RL_q_learning',
        'RL_ppo',
    ]
    summary['order'] = summary['component_label'].apply(
        lambda x: ordering.index(x) if x in ordering else len(ordering)
    )
    summary = summary.sort_values('order').drop(columns='order')
    return summary


def _decorate_control_significance(significance: pd.DataFrame) -> pd.DataFrame:
    """Add component/display labels for controller-to-controller comparisons."""
    if significance.empty:
        return significance

    decorated = significance.copy()
    decorated['base_component_label'] = decorated['base_label'].map(_control_component_label)
    decorated['compare_component_label'] = decorated['compare_label'].map(_control_component_label)
    decorated['base_display_label'] = decorated['base_component_label'].map(_display_label)
    decorated['compare_display_label'] = decorated['compare_component_label'].map(_display_label)
    return decorated


def _control_family(label: str) -> str:
    if label in {'factor_only', 'alpha_engine_no_control'}:
        return 'alpha baseline'
    if label.startswith('A'):
        return 'rules'
    if label.startswith('B'):
        return 'bandits'
    if label.startswith('C'):
        return 'supervised'
    if label.startswith('D'):
        return 'robust opt'
    if label.startswith('E'):
        return 'meta control'
    if label.startswith('F'):
        return 'safe rl'
    if label.startswith('RL'):
        return 'rl'
    return 'other'


def _control_color(label: str) -> str:
    family_colors = {
        'alpha baseline': '#7f7f7f',
        'rules': '#4c78a8',
        'bandits': '#f58518',
        'supervised': '#54a24b',
        'robust opt': '#e45756',
        'meta control': '#72b7b2',
        'safe rl': '#8c6d31',
        'rl': '#b279a2',
        'other': '#9d9da1',
    }
    return family_colors[_control_family(label)]


def _pareto_frontier_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated points in return-vs-drawdown space.

    Lower absolute drawdown is better; higher return is better.
    """
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked['drawdown_abs'] = ranked['mean_max_drawdown'].abs()
    ranked = ranked.sort_values(['drawdown_abs', 'mean_return'], ascending=[True, False])
    frontier_rows: list[dict[str, object]] = []
    best_return = -np.inf
    for _, row in ranked.iterrows():
        ret = float(row['mean_return'])
        if ret > best_return + 1e-12:
            frontier_rows.append(row.to_dict())
            best_return = ret
    return pd.DataFrame(frontier_rows)


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
    output_path: Path,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def plot_research_evaluation(
    metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    baseline_results: dict[str, object] | None = None,
    rolling_references: pd.DataFrame | None = None,
    output_path: Path = Path('pipeline_research_eval.png'),
    frontier_output_path: Path = Path('control_pareto_frontier.png'),
) -> None:
    """Create a paper-facing summary figure for the revised control-method study."""
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
                x,
                y,
                s=95,
                color=_control_color(label),
                edgecolors='black',
                linewidth=0.6,
                alpha=0.9,
                zorder=4,
            )
            ax.annotate(
                _display_label(label).replace('\n', ' '),
                (x, y),
                fontsize=7,
                xytext=(5, 4),
                textcoords='offset points',
            )
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
                x,
                y,
                s=95,
                color=_control_color(label),
                edgecolors='black',
                linewidth=0.6,
                alpha=0.9,
                zorder=4,
            )
            ax.annotate(
                _display_label(label).replace('\n', ' '),
                (x, y),
                fontsize=7,
                xytext=(5, 4),
                textcoords='offset points',
            )
        frontier = _pareto_frontier_points(ranked)
        if not frontier.empty and len(frontier) >= 2:
            ax.plot(
                frontier['drawdown_abs'],
                frontier['mean_return'],
                color='black',
                linewidth=1.4,
                linestyle='--',
                alpha=0.8,
                label='Pareto frontier',
            )
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
            frontier_ax.scatter(
                x,
                y,
                s=110,
                color=_control_color(label),
                edgecolors='black',
                linewidth=0.6,
                alpha=0.9,
            )
            frontier_ax.annotate(
                _display_label(label).replace('\n', ' '),
                (x, y),
                fontsize=8,
                xytext=(5, 4),
                textcoords='offset points',
            )
        frontier = _pareto_frontier_points(ranked)
        if not frontier.empty:
            frontier_ax.plot(
                frontier['drawdown_abs'],
                frontier['mean_return'],
                color='black',
                linewidth=1.5,
                linestyle='--',
            )
        frontier_ax.set_title('Control-Method Pareto Frontier')
        frontier_ax.set_xlabel('Absolute mean max drawdown')
        frontier_ax.set_ylabel('Mean annualized return')
        frontier_ax.grid(True, alpha=0.3)
        frontier_fig.tight_layout()
        frontier_fig.savefig(frontier_output_path, dpi=150, bbox_inches='tight')
        plt.close(frontier_fig)


def _normalize_checkpoint_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _normalize_checkpoint_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_checkpoint_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_control_config(control_cfg: dict[str, object]) -> dict[str, object]:
    control_cfg = control_cfg or {}
    method = str(control_cfg.get('method', 'none'))
    canonical: dict[str, object] = {'method': method}

    method_fields = {
        'fixed': ('fixed_invested_fraction',),
        'vol_target': ('vol_target_annual', 'vol_lookback'),
        'dd_delever': ('dd_thresholds', 'dd_min_invested'),
        'regime_rules': (
            'regime_bull_threshold',
            'regime_bear_threshold',
            'regime_bull_fraction',
            'regime_neutral_fraction',
            'regime_bear_fraction',
        ),
        'ensemble_rules': (
            'vol_target_annual',
            'vol_lookback',
            'dd_thresholds',
            'dd_min_invested',
            'regime_bull_threshold',
            'regime_bear_threshold',
            'regime_bull_fraction',
            'regime_neutral_fraction',
            'regime_bear_fraction',
            'ensemble_mode',
        ),
        'linucb': (
            'bandit_n_actions',
            'bandit_reward_window',
            'bandit_alpha_ucb',
            'bandit_feature_lookback',
        ),
        'thompson': (
            'bandit_n_actions',
            'bandit_reward_window',
            'bandit_feature_lookback',
        ),
        'epsilon_greedy': (
            'bandit_n_actions',
            'bandit_reward_window',
            'bandit_epsilon',
            'bandit_feature_lookback',
        ),
        'supervised': (
            'bandit_n_actions',
            'supervised_model',
            'supervised_retrain_every',
            'supervised_label_window',
        ),
        'cvar_robust': (
            'cvar_confidence',
            'cvar_n_scenarios',
            'cvar_lambda_base',
            'cvar_regime_scaling',
            'cvar_dd_budget',
        ),
        'cmdp_lagrangian': (
            'ql_alpha',
            'ql_gamma',
            'ql_epsilon',
            'cmdp_constraint_type',
            'cmdp_constraint_kappa',
            'cmdp_lambda_init',
            'cmdp_lambda_lr',
            'cmdp_tail_loss_threshold',
        ),
        'council': (
            'council_experts',
            'council_gate_model',
            'council_retrain_every',
            'council_min_samples',
            'council_temperature',
            'council_min_weight',
            'council_default_bias',
            'bandit_n_actions',
            'bandit_reward_window',
            'bandit_alpha_ucb',
            'bandit_feature_lookback',
            'cvar_confidence',
            'cvar_n_scenarios',
            'cvar_lambda_base',
            'cvar_regime_scaling',
            'cvar_dd_budget',
            'regime_bull_threshold',
            'regime_bear_threshold',
            'regime_bull_fraction',
            'regime_neutral_fraction',
            'regime_bear_fraction',
        ),
        'q_learning': ('ql_alpha', 'ql_gamma', 'ql_epsilon'),
    }

    for field in method_fields.get(method, ()):
        if field in control_cfg:
            canonical[field] = _normalize_checkpoint_value(control_cfg[field])

    if bool(control_cfg.get('convexity_enabled', False)):
        canonical['convexity_enabled'] = True
        for field in (
            'convexity_threshold',
            'convexity_mode_carries',
            'convexity_mode_lambdas',
            'convexity_mild_drawdown',
            'convexity_strong_drawdown',
            'convexity_mild_vol',
            'convexity_strong_vol',
            'convexity_mild_regime',
            'convexity_strong_regime',
        ):
            if field in control_cfg:
                canonical[field] = _normalize_checkpoint_value(control_cfg[field])
    return canonical


def _canonical_pipeline_config(config_payload: dict[str, object]) -> dict[str, object]:
    config_payload = config_payload or {}
    experiment_cfg = dict(config_payload.get('experiment', {}) or {})
    feature_cfg = dict(config_payload.get('feature_availability', {}) or {})
    cost_cfg = dict(config_payload.get('cost_model', {}) or {})
    optimizer_cfg = dict(config_payload.get('optimizer', {}) or {})
    option_cfg = dict(config_payload.get('option_overlay', {}) or {})
    control_cfg = dict(config_payload.get('control', {}) or {})

    canonical = {
        'train_frac': _normalize_checkpoint_value(config_payload.get('train_frac')),
        'rebalance_band': _normalize_checkpoint_value(config_payload.get('rebalance_band')),
        'min_turnover': _normalize_checkpoint_value(config_payload.get('min_turnover')),
        'portfolio_reward_mode': _normalize_checkpoint_value(config_payload.get('portfolio_reward_mode')),
        'hedge_reward_mode': _normalize_checkpoint_value(config_payload.get('hedge_reward_mode')),
        'e2e_reward_mode': _normalize_checkpoint_value(config_payload.get('e2e_reward_mode')),
        'enable_e2e_baseline': _normalize_checkpoint_value(config_payload.get('enable_e2e_baseline')),
        'feature_availability': {
            'macro_lag_days': _normalize_checkpoint_value(feature_cfg.get('macro_lag_days')),
            'allow_static_sec_quality': _normalize_checkpoint_value(feature_cfg.get('allow_static_sec_quality')),
        },
        'cost_model': {
            'base_cost_bps': _normalize_checkpoint_value(cost_cfg.get('base_cost_bps')),
            'turnover_vol_multiplier': _normalize_checkpoint_value(cost_cfg.get('turnover_vol_multiplier')),
            'size_penalty_bps': _normalize_checkpoint_value(cost_cfg.get('size_penalty_bps')),
            'use_almgren_chriss': _normalize_checkpoint_value(cost_cfg.get('use_almgren_chriss')),
            'ac_permanent_beta': _normalize_checkpoint_value(cost_cfg.get('ac_permanent_beta')),
            'ac_temporary_eta': _normalize_checkpoint_value(cost_cfg.get('ac_temporary_eta')),
        },
        'optimizer': {
            'use_optimizer': _normalize_checkpoint_value(optimizer_cfg.get('use_optimizer')),
            'max_weight': _normalize_checkpoint_value(optimizer_cfg.get('max_weight')),
            'risk_aversion': _normalize_checkpoint_value(optimizer_cfg.get('risk_aversion')),
            'alpha_strength': _normalize_checkpoint_value(optimizer_cfg.get('alpha_strength')),
            'anchor_strength': _normalize_checkpoint_value(optimizer_cfg.get('anchor_strength')),
            'turnover_penalty': _normalize_checkpoint_value(optimizer_cfg.get('turnover_penalty')),
            'group_caps': _normalize_checkpoint_value(optimizer_cfg.get('group_caps')),
        },
        'option_overlay': {
            'use_option_overlay': _normalize_checkpoint_value(option_cfg.get('use_option_overlay')),
        },
        'experiment': {
            'label': _normalize_checkpoint_value(experiment_cfg.get('label')),
            'use_factor': _normalize_checkpoint_value(experiment_cfg.get('use_factor')),
            'use_pairs': _normalize_checkpoint_value(experiment_cfg.get('use_pairs')),
            'use_lstm': _normalize_checkpoint_value(experiment_cfg.get('use_lstm')),
            'adaptive_combiner': _normalize_checkpoint_value(experiment_cfg.get('adaptive_combiner')),
            'use_portfolio_rl': _normalize_checkpoint_value(experiment_cfg.get('use_portfolio_rl')),
            'use_hedge_rl': _normalize_checkpoint_value(experiment_cfg.get('use_hedge_rl')),
            'use_uncertainty_state': _normalize_checkpoint_value(experiment_cfg.get('use_uncertainty_state')),
            'use_regime_state': _normalize_checkpoint_value(experiment_cfg.get('use_regime_state')),
            'use_vol_state': _normalize_checkpoint_value(experiment_cfg.get('use_vol_state')),
            'control_method': _normalize_checkpoint_value(experiment_cfg.get('control_method')),
        },
        'control': _canonical_control_config(control_cfg),
    }
    return canonical


def _canonical_checkpoint_metadata(metadata: dict[str, object]) -> dict[str, object]:
    metadata = metadata or {}
    canonical = dict(metadata)
    canonical['config'] = _canonical_pipeline_config(dict(metadata.get('config', {}) or {}))
    for key in ('prices', 'volumes', 'returns', 'macro', 'sec_quality'):
        if key in canonical:
            canonical[key] = _normalize_checkpoint_value(canonical[key])
    return canonical


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
    return _canonical_checkpoint_metadata({
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
    })


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

    payload_metadata = _canonical_checkpoint_metadata(dict(payload.get('metadata', {}) or {}))
    if payload_metadata != expected_metadata:
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
    - research_metrics.csv
    - research_regime_summary.csv
    - research_summary.json
    - pipeline_research_eval.png
    """
    base_config = copy.deepcopy(base_config or PipelineConfig())
    evaluation_config = evaluation_config or EvaluationConfig()
    macro_data = macro_data if macro_data is not None else pd.DataFrame(index=returns.index)
    sec_quality_scores = sec_quality_scores if sec_quality_scores is not None else pd.Series(dtype=float)
    checkpoint_dir = Path(evaluation_config.checkpoint_dir)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path(evaluation_config.output_dir)
    output_dir = output_root / run_timestamp if evaluation_config.timestamp_outputs else output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Research outputs will be written to: {output_dir}")
    if evaluation_config.enable_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Research checkpoints enabled: {checkpoint_dir}")

    metric_rows: list[dict[str, float | str]] = []
    regime_rows: list[dict[str, float | str]] = []
    rolling_reference_rows: list[dict[str, float | str]] = []
    baseline_results: dict[str, object] | None = None
    deferred_baseline_config: PipelineConfig | None = None
    first_ppo_logs_shown = False
    ablation_suite = build_ablation_suite(base_config)
    control_suite = build_control_comparison_suite(base_config)
    rolling_starts = _rolling_starts(
        n_obs=len(returns),
        window_days=evaluation_config.rolling_window_days,
        step_days=evaluation_config.rolling_step_days,
        min_windows=evaluation_config.min_rolling_windows,
        max_windows=evaluation_config.max_rolling_windows,
    )
    total_runs = (
        len(ablation_suite) * len(evaluation_config.train_fracs)
        + sum(len(_control_train_fracs(cfg, evaluation_config)) for cfg in control_suite)
        + (
            0
            if any(abs(train_frac - base_config.train_frac) < 1e-12 for train_frac in evaluation_config.train_fracs)
            else 1
        )
        + 1  # strict timing
        + len(rolling_starts)
        + len(evaluation_config.cost_bps_grid)
        + len(evaluation_config.rebalance_band_grid)
        + (len(evaluation_config.hedge_scale_grid) if base_config.experiment.use_hedge_rl else 0)
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

    for config in ablation_suite:
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

    # ================================================================
    # CONTROL-METHOD COMPARISON (Architecture Revision v2)
    # ================================================================
    print("\n" + "=" * 60)
    print("CONTROL-METHOD COMPARISON SUITE")
    print("=" * 60)
    control_baseline_results: dict[str, object] | None = None
    control_results_by_label: dict[str, dict[str, object]] = {}
    for ctrl_config in control_suite:
        ctrl_label = ctrl_config.experiment.label
        reference_train_frac = _control_reference_train_frac(ctrl_config, base_config)
        for train_frac in _control_train_fracs(ctrl_config, evaluation_config):
            run_config = copy.deepcopy(ctrl_config)
            run_config.train_frac = train_frac
            run_config.experiment.label = f'{ctrl_label}_tf{train_frac:.2f}'
            results = _run_with_research_logging(
                prices, volumes, returns, macro_data, run_config,
                suite='control_comparison',
                include_e2e=False,
                run_key=f"control_{run_config.experiment.label}",
            )
            row = _metric_summary(results)
            row.update({
                'suite': 'control_comparison',
                'window_id': 'full_sample',
                'param_name': 'train_frac',
                'param_value': train_frac,
            })
            metric_rows.append(row)
            regime_rows.extend(_regime_summary(results))
            # Use factor_only as the control baseline for bootstrap comparisons
            if abs(train_frac - reference_train_frac) < 1e-12:
                control_results_by_label[ctrl_label] = results
                if ctrl_label == 'factor_only':
                    control_baseline_results = results

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

    for window_id, start in enumerate(rolling_starts):
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

    if base_config.experiment.use_hedge_rl:
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

        e2e_path = results.get('e2e_rl', [1.0])
        e2e_row = _path_metric_summary(e2e_path, f'e2e_reward_{reward_mode}')
        has_real_e2e = (
            reward_config.enable_e2e_baseline
            and len(e2e_path) > 1
            and np.any(np.abs(np.asarray(e2e_path[1:], dtype=float) - 1.0) > 1e-12)
        )
        if has_real_e2e and 'ann_return' in e2e_row:
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
    control_comparison_summary = _build_control_comparison_summary(metrics)
    robustness_summary = _build_robustness_summary(metrics, rolling_references)
    bootstrap_cis = pd.DataFrame()
    bootstrap_significance = pd.DataFrame()
    control_significance = pd.DataFrame()

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

    if control_results_by_label:
        control_path_map = {
            label: result.get('wealth', [1.0])
            for label, result in control_results_by_label.items()
        }
        spy_reference = next(
            (result.get('spy') for result in control_results_by_label.values() if result.get('spy') is not None),
            None,
        )
        if spy_reference is not None:
            control_path_map['SPY'] = spy_reference

        significance_frames: list[pd.DataFrame] = []
        significance_bases = [
            ('D_cvar_robust', evaluation_config.bootstrap_seed + 101),
            ('D_plus_convexity', evaluation_config.bootstrap_seed + 151),
        ]
        for base_label, seed in significance_bases:
            if base_label not in control_path_map:
                continue
            compare_labels = [
                label for label in [
                    'factor_only',
                    'A1_fixed',
                    'A2_vol_target',
                    'A3_dd_delever',
                    'A4_regime_rules',
                    'A5_ensemble_mean',
                    'A5_ensemble_min',
                    'B1_linucb',
                    'B2_thompson',
                    'B3_epsilon_greedy',
                    'C_supervised',
                    'D_cvar_robust',
                    'D_plus_convexity',
                    'E_council',
                    'E_plus_convexity',
                    'G_mlp_meta',
                    'G_plus_convexity',
                    'F_cmdp_lagrangian',
                    'RL_q_learning',
                    'SPY',
                ]
                if label in control_path_map and label != base_label
            ]
            significance_frames.append(
                _compute_bootstrap_pairwise_significance(
                    path_map=control_path_map,
                    base_label=base_label,
                    compare_labels=compare_labels,
                    n_samples=evaluation_config.bootstrap_samples,
                    block_size=evaluation_config.bootstrap_block_size,
                    seed=seed,
                )
            )
        if significance_frames:
            control_significance = _decorate_control_significance(
                pd.concat(significance_frames, ignore_index=True)
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
            jk_table.to_csv(output_dir / 'research_jobson_korkie.csv', index=False)
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
            ts_cv_results.to_csv(output_dir / 'research_ts_cv.csv', index=False)
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

    metrics.to_csv(output_dir / 'research_metrics.csv', index=False)
    regime_summary.to_csv(output_dir / 'research_regime_summary.csv', index=False)
    rolling_references.to_csv(output_dir / 'research_rolling_references.csv', index=False)
    ablation_summary.to_csv(output_dir / 'research_ablation_summary.csv', index=False)
    if not control_comparison_summary.empty:
        control_comparison_summary.to_csv(output_dir / 'research_control_comparison.csv', index=False)
        print("\n  Control Comparison Summary:")
        for _, row in control_comparison_summary.iterrows():
            print(f"    {row['component_label']:25s}  Sharpe={row['mean_sharpe']:.2f}  "
                  f"Calmar={row['mean_calmar']:.2f}  MaxDD={row['mean_max_drawdown']:.1%}")
    robustness_summary.to_csv(output_dir / 'research_robustness_summary.csv', index=False)
    if not bootstrap_cis.empty:
        bootstrap_cis.to_csv(output_dir / 'research_bootstrap_cis.csv', index=False)
    if not bootstrap_significance.empty:
        bootstrap_significance.to_csv(output_dir / 'research_bootstrap_significance.csv', index=False)
    if not control_significance.empty:
        control_significance.to_csv(output_dir / 'research_control_significance.csv', index=False)
    _write_research_tables(
        ablation_summary,
        robustness_summary,
        output_path=Path('paper/2col/research_paper_tables.tex'),
        bootstrap_significance=bootstrap_significance,
    )
    _write_research_tables(
        ablation_summary,
        robustness_summary,
        output_path=output_dir / 'research_paper_tables.tex',
        bootstrap_significance=bootstrap_significance,
    )
    plot_research_evaluation(
        metrics,
        regime_summary,
        baseline_results=baseline_results,
        rolling_references=rolling_references,
        output_path=output_dir / 'pipeline_research_eval.png',
        frontier_output_path=output_dir / 'control_pareto_frontier.png',
    )
    plot_rolling_windows(
        metrics,
        rolling_references_df=rolling_references if not rolling_references.empty else None,
        output_path=output_dir / 'pipeline_rolling_windows.png',
    )
    plot_reward_ablation(metrics, output_path=output_dir / 'pipeline_reward_ablation.png')

    summary = {
        'base_config': asdict(base_config),
        'evaluation_config': asdict(evaluation_config),
        'run_timestamp': run_timestamp,
        'output_dir': str(output_dir),
        'n_metric_rows': len(metrics),
        'n_regime_rows': len(regime_summary),
        'ablation_summary': ablation_summary.to_dict(orient='records'),
        'control_comparison_summary': control_comparison_summary.to_dict(orient='records') if not control_comparison_summary.empty else [],
        'robustness_summary': robustness_summary.to_dict(orient='records'),
        'bootstrap_cis': bootstrap_cis.to_dict(orient='records'),
        'bootstrap_significance': bootstrap_significance.to_dict(orient='records'),
        'control_significance': control_significance.to_dict(orient='records'),
        'best_sharpe_by_suite': {
            suite: (
                group.sort_values('sharpe', ascending=False)
                .head(1)[['label', 'sharpe', 'ann_return', 'max_drawdown']]
                .to_dict(orient='records')[0]
            )
            for suite, group in metrics.groupby('suite')
        },
    }
    with (output_dir / 'research_summary.json').open('w', encoding='utf-8') as handle:
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
        'control_comparison_summary': control_comparison_summary,
        'robustness_summary': robustness_summary,
        'bootstrap_cis': bootstrap_cis,
        'bootstrap_significance': bootstrap_significance,
        'control_significance': control_significance,
        'summary': summary,
        'output_dir': output_dir,
    }
