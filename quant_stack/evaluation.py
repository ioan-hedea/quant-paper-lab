"""Research evaluation engine for ablations, robustness, and regime analysis."""

from __future__ import annotations

import copy
from datetime import datetime
import json
from dataclasses import asdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_candidates,
    _checkpoint_metadata,
    _checkpoint_path,
    _load_checkpoint_results,
    _sanitize_for_checkpoint,
    _scope_universe_run_key,
    _universe_checkpoint_dir,
    _write_progress_manifest,
    _write_run_manifest,
)
from .config import (
    EvaluationConfig,
    PipelineConfig,
    UniverseProfile,
    get_universe_profile,
    use_universe,
)
from .evaluation_helpers import (
    _build_ablation_summary,
    _build_control_comparison_summary,
    _build_execution_summary,
    _build_robustness_summary,
    _control_color,
    _control_component_label,
    _control_reference_train_frac,
    _control_train_fracs,
    _daily_returns_from_path,
    _decorate_control_significance,
    _display_label,
    _latex_pct,
    _metric_summary,
    _pareto_frontier_points,
    _path_metric_summary,
    _regime_summary,
    _returns_metric_summary,
    _table_label,
    build_ablation_suite,
    build_control_comparison_suite,
)
from .pipeline import run_full_pipeline
from .plots import plot_research_evaluation
from .result_publisher import _write_research_tables, publish_research_outputs

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


def run_research_evaluation(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    sec_quality_scores: pd.Series | None = None,
    base_config: PipelineConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    universe_id: str | None = None,
    export_paper_tables: bool | None = None,
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
    universe_id = universe_id or 'A'
    if export_paper_tables is None:
        export_paper_tables = universe_id == 'A'
    macro_data = macro_data if macro_data is not None else pd.DataFrame(index=returns.index)
    sec_quality_scores = sec_quality_scores if sec_quality_scores is not None else pd.Series(dtype=float)
    checkpoint_root = Path(evaluation_config.checkpoint_dir)
    checkpoint_dir = _universe_checkpoint_dir(checkpoint_root, universe_id)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path(evaluation_config.output_dir)
    if evaluation_config.timestamp_outputs:
        output_dir = output_root / f"{run_timestamp}_universe_{universe_id}"
    else:
        output_dir = output_root / f"universe_{universe_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_dir,
        run_type='research_evaluation',
        base_config=base_config,
        evaluation_config=evaluation_config,
        universe_id=universe_id,
        run_timestamp=run_timestamp,
        status='running',
    )
    print(f"Research outputs for Universe {universe_id} will be written to: {output_dir}")
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
        + len(evaluation_config.cost_stress_multiplier_grid)
        + len(evaluation_config.rebalance_band_grid)
        + len(evaluation_config.adv_participation_cap_grid)
        + len(evaluation_config.execution_delay_grid)
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
        scoped_run_key = _scope_universe_run_key(run_key, universe_id)
        current_run = {
            'ordinal': run_counter,
            'total_runs': total_runs,
            'suite': suite,
            'label': run_config.experiment.label,
            'run_key': scoped_run_key,
            'include_e2e': include_e2e,
            'universe_id': universe_id,
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
        checkpoint_candidates = _checkpoint_candidates(checkpoint_dir, run_key, universe_id)
        active_checkpoint_key = scoped_run_key
        active_checkpoint_path = _checkpoint_path(checkpoint_dir, active_checkpoint_key)
        if evaluation_config.enable_checkpoints:
            for candidate_key, candidate_path in checkpoint_candidates:
                if not candidate_path.exists():
                    continue
                checkpoint_metadata = _checkpoint_metadata(
                    run_prices,
                    run_volumes,
                    run_returns,
                    run_macro,
                    sec_quality_scores,
                    run_config,
                    suite=suite,
                    include_e2e=include_e2e,
                    run_key=candidate_key,
                )
                cached_results = _load_checkpoint_results(
                    candidate_path,
                    checkpoint_metadata,
                    match_mode=evaluation_config.checkpoint_match_mode,
                )
                if cached_results is None:
                    continue
                if include_e2e:
                    first_ppo_logs_shown = True
                completed_run_keys.append(candidate_key)
                last_completed_run = {
                    **current_run,
                    'source': 'cache',
                    'completed_at': datetime.now().isoformat(),
                    'checkpoint_path': str(candidate_path),
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
            checkpoint_metadata = _checkpoint_metadata(
                run_prices,
                run_volumes,
                run_returns,
                run_macro,
                sec_quality_scores,
                run_config,
                suite=suite,
                include_e2e=include_e2e,
                run_key=active_checkpoint_key,
            )
            checkpoint_payload = {
                'schema_version': CHECKPOINT_SCHEMA_VERSION,
                'saved_at': datetime.now().isoformat(),
                'metadata': checkpoint_metadata,
                'results': _sanitize_for_checkpoint(results),
            }
            temp_checkpoint_path = active_checkpoint_path.with_suffix(f"{active_checkpoint_path.suffix}.tmp")
            with temp_checkpoint_path.open('wb') as handle:
                pickle.dump(checkpoint_payload, handle)
            temp_checkpoint_path.replace(active_checkpoint_path)
            print(f"  Saved checkpoint to {active_checkpoint_path}")
            completed_run_keys.append(active_checkpoint_key)
            last_completed_run = {
                **current_run,
                'source': 'fresh',
                'completed_at': datetime.now().isoformat(),
                'checkpoint_path': str(active_checkpoint_path),
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

    for cost_multiplier in evaluation_config.cost_stress_multiplier_grid:
        stress_config = copy.deepcopy(base_config)
        stress_config.cost_model.cost_stress_multiplier = cost_multiplier
        stress_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            stress_config,
            suite='cost_stress_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"cost_stress_sensitivity_{cost_multiplier}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'cost_stress_sensitivity', 'window_id': 'full_sample', 'param_name': 'cost_stress_multiplier', 'param_value': cost_multiplier})
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

    for adv_cap in evaluation_config.adv_participation_cap_grid:
        liquidity_config = copy.deepcopy(base_config)
        liquidity_config.cost_model.adv_participation_cap = adv_cap
        liquidity_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            liquidity_config,
            suite='liquidity_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"liquidity_sensitivity_{adv_cap}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'liquidity_sensitivity', 'window_id': 'full_sample', 'param_name': 'adv_participation_cap', 'param_value': adv_cap})
        metric_rows.append(row)

    for delay_days in evaluation_config.execution_delay_grid:
        delay_config = copy.deepcopy(base_config)
        delay_config.cost_model.execution_delay_days = int(delay_days)
        delay_config.experiment.label = 'full_pipeline'
        results = _run_with_research_logging(
            prices,
            volumes,
            returns,
            macro_data,
            delay_config,
            suite='delay_sensitivity',
            include_e2e=evaluation_config.research_e2e_scope == 'all',
            run_key=f"delay_sensitivity_{delay_days}",
        )
        row = _metric_summary(results)
        row.update({'suite': 'delay_sensitivity', 'window_id': 'full_sample', 'param_name': 'execution_delay_days', 'param_value': delay_days})
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
    execution_summary = _build_execution_summary(metrics)
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
                    'H_mpc',
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

    summary = {
        'universe_id': universe_id,
        'base_config': asdict(base_config),
        'evaluation_config': asdict(evaluation_config),
        'run_timestamp': run_timestamp,
        'output_dir': str(output_dir),
        'n_metric_rows': len(metrics),
        'n_regime_rows': len(regime_summary),
        'ablation_summary': ablation_summary.to_dict(orient='records'),
        'control_comparison_summary': control_comparison_summary.to_dict(orient='records') if not control_comparison_summary.empty else [],
        'execution_summary': execution_summary.to_dict(orient='records') if not execution_summary.empty else [],
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
    publish_research_outputs(
        output_dir=output_dir,
        metrics=metrics,
        regime_summary=regime_summary,
        rolling_references=rolling_references,
        ablation_summary=ablation_summary,
        control_comparison_summary=control_comparison_summary,
        execution_summary=execution_summary,
        robustness_summary=robustness_summary,
        bootstrap_cis=bootstrap_cis,
        bootstrap_significance=bootstrap_significance,
        control_significance=control_significance,
        baseline_results=baseline_results,
        summary=summary,
        export_paper_tables=export_paper_tables,
    )
    _write_run_manifest(
        output_dir,
        run_type='research_evaluation',
        base_config=base_config,
        evaluation_config=evaluation_config,
        universe_id=universe_id,
        run_timestamp=run_timestamp,
        status='completed',
        summary=summary,
    )

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


# ============================================================
# Meta-Learning / Controller-Selection Evaluation (RQ4)
# ============================================================

def _compute_environment_features_from_results(
    results: dict,
    universe_id: str,
    train_frac: float,
) -> dict[str, float]:
    """Extract environment-level features from a completed pipeline run.

    These features characterize the environment (universe + split) and
    are used as inputs to the meta-learning controller selector.
    """
    rets = _daily_returns_from_path(results.get('wealth', [1.0]))
    if len(rets) == 0:
        return {}

    beliefs = np.asarray(results.get('regime_beliefs', [0.5]), dtype=float)
    vols = np.asarray(results.get('realized_vols', [0.15]), dtype=float)
    alphas = np.asarray(results.get('alpha_strengths', [0.5]), dtype=float)

    # Alpha quality
    alpha_mean = float(alphas.mean()) if len(alphas) > 0 else 0.0
    alpha_std = float(alphas.std()) if len(alphas) > 1 else 0.0

    # Risk / volatility
    vol_mean = float(vols.mean()) if len(vols) > 0 else 0.15
    vol_of_vol = float(vols.std()) if len(vols) > 1 else 0.0

    # Regime
    regime_mean = float(beliefs.mean()) if len(beliefs) > 0 else 0.5
    if len(beliefs) > 1:
        regime_switch_freq = float(np.abs(np.diff((beliefs > 0.5).astype(float))).mean())
    else:
        regime_switch_freq = 0.0
    regime_risk_off = float(np.mean(beliefs < 0.30)) if len(beliefs) > 0 else 0.0

    # Market stress
    wealth = np.asarray(results.get('wealth', [1.0]), dtype=float)
    if len(wealth) > 1:
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / (peak + 1e-8)
        max_dd = float(dd.min())
        dd_freq = float(np.mean(dd < -0.02))
    else:
        max_dd = 0.0
        dd_freq = 0.0

    ann_ret = float(np.mean(rets) * 252) if len(rets) > 0 else 0.0
    ann_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 0 else 0.15

    return {
        'universe_id': universe_id,
        'train_frac': train_frac,
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std,
        'vol_mean': vol_mean,
        'vol_of_vol': vol_of_vol,
        'regime_mean': regime_mean,
        'regime_switch_freq': regime_switch_freq,
        'regime_risk_off_frac': regime_risk_off,
        'max_drawdown': max_dd,
        'dd_frequency': dd_freq,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
    }


def _build_meta_learning_dataset(
    all_results: list[dict],
) -> pd.DataFrame:
    """Assemble the environment × controller performance dataset.

    Each row is one (environment, controller) pair with:
      - environment features
      - controller label
      - performance metrics (sharpe, calmar, max_dd)
    """
    rows: list[dict] = []
    for entry in all_results:
        env_features = entry.get('env_features', {})
        controller_label = entry.get('controller_label', '')
        metrics = entry.get('metrics', {})

        row = dict(env_features)
        row['controller_label'] = controller_label
        row.update({
            'sharpe': metrics.get('sharpe', 0.0),
            'calmar': metrics.get('calmar', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'ann_return': metrics.get('ann_return', 0.0),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_controller_transfer(
    meta_dataset: pd.DataFrame,
    score_col: str = 'sharpe',
) -> dict[str, object]:
    """Evaluate whether controller rankings transfer across environments.

    For each environment, identifies the best controller by ``score_col``,
    then measures:
      - Rank correlation (Kendall's tau) of controller rankings across envs
      - Top-1 and top-2 accuracy of simple meta-predictors
      - Average regret vs. oracle selection

    Returns a summary dict suitable for inclusion in the paper results.
    """
    if meta_dataset.empty:
        return {'error': 'empty dataset'}

    feature_cols = [
        c for c in meta_dataset.columns
        if c not in ('controller_label', 'sharpe', 'calmar', 'max_drawdown',
                      'ann_return', 'universe_id', 'train_frac')
    ]

    # Build pivot: rows = environments, columns = controllers, values = score
    env_keys = ['universe_id', 'train_frac']
    available_keys = [k for k in env_keys if k in meta_dataset.columns]
    if not available_keys:
        return {'error': 'no environment keys found'}

    meta_dataset['env_id'] = meta_dataset[available_keys].astype(str).agg('_'.join, axis=1)
    pivot = meta_dataset.pivot_table(
        index='env_id', columns='controller_label', values=score_col, aggfunc='first',
    ).dropna(axis=1, how='all')

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return {'error': f'insufficient data: {pivot.shape[0]} envs, {pivot.shape[1]} controllers'}

    # Best controller per environment
    best_per_env = pivot.idxmax(axis=1)
    best_score_per_env = pivot.max(axis=1)

    # Baseline: always pick the best-on-average controller
    avg_score = pivot.mean(axis=0)
    best_avg_controller = avg_score.idxmax()

    # Rank correlation across environments
    rankings = pivot.rank(axis=1, ascending=False)
    tau_values: list[float] = []
    envs = list(pivot.index)
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            shared_cols = pivot.columns[pivot.loc[envs[i]].notna() & pivot.loc[envs[j]].notna()]
            if len(shared_cols) >= 3:
                tau, _ = sp_stats.kendalltau(
                    rankings.loc[envs[i], shared_cols],
                    rankings.loc[envs[j], shared_cols],
                )
                if not np.isnan(tau):
                    tau_values.append(float(tau))

    mean_tau = float(np.mean(tau_values)) if tau_values else float('nan')

    # Top-1 / top-2 accuracy of baselines
    n_envs = len(envs)
    # Always-best-average
    top1_avg = float(np.mean(best_per_env == best_avg_controller))
    top2_per_env = pivot.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
    top2_avg = float(np.mean([best_avg_controller in t2 for t2 in top2_per_env]))

    # Average regret: difference between oracle and best-average strategy
    regret_values: list[float] = []
    for env_id in envs:
        oracle_score = float(best_score_per_env[env_id])
        baseline_score = float(pivot.loc[env_id, best_avg_controller]) if best_avg_controller in pivot.columns else 0.0
        regret_values.append(oracle_score - baseline_score)
    avg_regret = float(np.mean(regret_values))

    # Random baseline
    n_controllers = pivot.shape[1]
    random_top1 = 1.0 / max(1, n_controllers)
    random_top2 = min(1.0, 2.0 / max(1, n_controllers))

    return {
        'n_environments': n_envs,
        'n_controllers': n_controllers,
        'best_avg_controller': best_avg_controller,
        'mean_rank_correlation_tau': mean_tau,
        'best_per_env': best_per_env.to_dict(),
        'top1_accuracy_best_avg': top1_avg,
        'top2_accuracy_best_avg': top2_avg,
        'top1_accuracy_random': random_top1,
        'top2_accuracy_random': random_top2,
        'avg_regret_best_avg': avg_regret,
        'avg_score_by_controller': avg_score.to_dict(),
        'controller_rankings_pivot': pivot.to_dict(),
    }


# ============================================================
# Cross-Universe Evaluation Runner
# ============================================================

def run_cross_universe_evaluation(
    universe_ids: tuple[str, ...] | None = None,
    base_config: PipelineConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Run the full controller suite on multiple universes and evaluate transfer.

    For each universe in ``universe_ids``:
      1. Swap the active universe via ``use_universe()``
      2. Load market data for that universe
      3. Run every controller across the configured train_fracs
      4. Collect per-environment features and per-controller performance

    Then assemble the meta-learning dataset and evaluate whether controller
    rankings transfer across universes (Kendall's tau, regret, top-k accuracy).

    Returns a dict with the meta-learning dataset, transfer results, and
    per-universe raw results.
    """
    from .data import load_market_data

    if base_config is None:
        base_config = PipelineConfig()
    if evaluation_config is None:
        evaluation_config = EvaluationConfig()
    if universe_ids is None:
        universe_ids = evaluation_config.meta_learning_universes

    control_suite = build_control_comparison_suite(base_config)
    checkpoint_root = Path(evaluation_config.checkpoint_dir)
    if evaluation_config.enable_checkpoints:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    all_meta_rows: list[dict] = []
    per_universe_results: dict[str, list[dict]] = {}

    for uid in universe_ids:
        print(f"\n{'='*60}")
        print(f"CROSS-UNIVERSE EVALUATION: Universe {uid}")
        print(f"{'='*60}")

        with use_universe(uid) as profile:
            checkpoint_dir = _universe_checkpoint_dir(checkpoint_root, uid)
            if evaluation_config.enable_checkpoints:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Tickers: {len(profile.tickers)}")
            print(f"  Pairs: {len(profile.pairs)}")

            # Load data for this universe
            prices, volumes, returns, macro_data, sec_quality = load_market_data(universe_id=uid)

            universe_runs: list[dict] = []

            for ctrl_config in control_suite:
                ctrl_label = ctrl_config.experiment.label

                for train_frac in _control_train_fracs(ctrl_config, evaluation_config):
                    run_config = copy.deepcopy(ctrl_config)
                    run_config.train_frac = train_frac

                    run_label = f"{uid}_{ctrl_label}_tf{train_frac:.2f}"
                    print(f"  Running {run_label} ...")

                    try:
                        base_run_key = f"control_{ctrl_label}_tf{train_frac:.2f}"
                        active_checkpoint_key = _scope_universe_run_key(base_run_key, uid)
                        active_checkpoint_path = _checkpoint_path(checkpoint_dir, active_checkpoint_key)
                        results = None
                        if evaluation_config.enable_checkpoints:
                            for candidate_key, candidate_path in _checkpoint_candidates(checkpoint_dir, base_run_key, uid):
                                if not candidate_path.exists():
                                    continue
                                checkpoint_metadata = _checkpoint_metadata(
                                    prices,
                                    volumes,
                                    returns,
                                    macro_data,
                                    sec_quality,
                                    run_config,
                                    suite='control_comparison',
                                    include_e2e=False,
                                    run_key=candidate_key,
                                )
                                results = _load_checkpoint_results(
                                    candidate_path,
                                    checkpoint_metadata,
                                    match_mode=evaluation_config.checkpoint_match_mode,
                                )
                                if results is not None:
                                    print(f"    Loaded cached result from {candidate_path}")
                                    break

                        if results is None:
                            results = run_full_pipeline(
                                prices, volumes, returns,
                                macro_data=macro_data,
                                sec_quality_scores=sec_quality,
                                config=run_config,
                            )
                            if evaluation_config.enable_checkpoints:
                                checkpoint_metadata = _checkpoint_metadata(
                                    prices,
                                    volumes,
                                    returns,
                                    macro_data,
                                    sec_quality,
                                    run_config,
                                    suite='control_comparison',
                                    include_e2e=False,
                                    run_key=active_checkpoint_key,
                                )
                                checkpoint_payload = {
                                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                                    'saved_at': datetime.now().isoformat(),
                                    'metadata': checkpoint_metadata,
                                    'results': _sanitize_for_checkpoint(results),
                                }
                                temp_checkpoint_path = active_checkpoint_path.with_suffix(f"{active_checkpoint_path.suffix}.tmp")
                                with temp_checkpoint_path.open('wb') as handle:
                                    pickle.dump(checkpoint_payload, handle)
                                temp_checkpoint_path.replace(active_checkpoint_path)
                                print(f"    Saved checkpoint to {active_checkpoint_path}")

                        # Extract metrics
                        metrics = _path_metric_summary(
                            results.get('wealth', [1.0]),
                            run_label,
                        )

                        # Extract environment features
                        env_features = _compute_environment_features_from_results(
                            results, uid, train_frac,
                        )

                        meta_row = {
                            'env_features': env_features,
                            'controller_label': ctrl_label,
                            'metrics': metrics,
                        }
                        all_meta_rows.append(meta_row)

                        universe_runs.append({
                            'label': run_label,
                            'controller': ctrl_label,
                            'train_frac': train_frac,
                            'metrics': metrics,
                            'env_features': env_features,
                        })

                    except Exception as e:
                        print(f"    FAILED: {e}")
                        continue

            per_universe_results[uid] = universe_runs

    # Assemble the meta-learning dataset
    meta_dataset = _build_meta_learning_dataset(all_meta_rows)

    # Evaluate transfer
    transfer_results = {}
    for score in ('sharpe', 'calmar'):
        transfer_results[score] = evaluate_controller_transfer(
            meta_dataset, score_col=score,
        )

    # Save outputs
    output_dir = Path(evaluation_config.output_dir)
    if evaluation_config.timestamp_outputs:
        transfer_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        universe_slug = "-".join(universe_ids)
        output_dir = output_dir / f"{transfer_timestamp}_transfer_{universe_slug}"
    else:
        transfer_timestamp = None
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_dir,
        run_type='cross_universe_transfer',
        base_config=None,
        evaluation_config=evaluation_config,
        universe_ids=tuple(universe_ids),
        run_timestamp=transfer_timestamp,
        status='running',
    )

    meta_dataset.to_csv(output_dir / 'meta_learning_dataset.csv', index=False)

    transfer_summary = {
        'universe_ids': list(universe_ids),
        'n_environments': int(
            meta_dataset[['universe_id', 'train_frac']].drop_duplicates().shape[0]
        ) if not meta_dataset.empty else 0,
        'transfer_sharpe': transfer_results.get('sharpe', {}),
        'transfer_calmar': transfer_results.get('calmar', {}),
    }
    with open(output_dir / 'meta_learning_transfer.json', 'w') as f:
        json.dump(transfer_summary, f, indent=2, default=str)
    _write_run_manifest(
        output_dir,
        run_type='cross_universe_transfer',
        base_config=None,
        evaluation_config=evaluation_config,
        universe_ids=tuple(universe_ids),
        run_timestamp=transfer_timestamp,
        status='completed',
        summary=transfer_summary,
    )

    print(f"\n{'='*60}")
    print("CROSS-UNIVERSE EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Universes: {list(universe_ids)}")
    print(f"  Meta-dataset rows: {len(meta_dataset)}")
    if 'sharpe' in transfer_results:
        tr = transfer_results['sharpe']
        if 'error' not in tr:
            print(f"  Mean rank correlation (tau): {tr['mean_rank_correlation_tau']:.3f}")
            print(f"  Best-avg controller: {tr['best_avg_controller']}")
            print(f"  Top-1 accuracy (best-avg): {tr['top1_accuracy_best_avg']:.2f}")
            print(f"  Avg regret vs oracle: {tr['avg_regret_best_avg']:.4f}")
    print(f"  Output: {output_dir}")

    return {
        'meta_dataset': meta_dataset,
        'transfer_results': transfer_results,
        'per_universe_results': per_universe_results,
        'output_dir': str(output_dir),
    }
