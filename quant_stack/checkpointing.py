"""Checkpointing, progress manifests, and run provenance helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import pickle
import platform
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

from .config import PipelineConfig

CHECKPOINT_SCHEMA_VERSION = 1


def _normalize_checkpoint_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _normalize_checkpoint_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_checkpoint_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_control_config(control_cfg: dict[str, object]) -> dict[str, object]:
    default_cfg = asdict(PipelineConfig().control)
    merged_cfg = dict(default_cfg)
    merged_cfg.update(control_cfg or {})
    method = str(merged_cfg.get('method', 'none'))
    canonical: dict[str, object] = {'method': method}

    method_fields = {
        'fixed': ('fixed_invested_fraction',),
        'vol_target': ('vol_target_annual', 'vol_lookback'),
        'dd_delever': ('dd_thresholds', 'dd_min_invested'),
        'regime_rules': (
            'regime_bull_threshold', 'regime_bear_threshold',
            'regime_bull_fraction', 'regime_neutral_fraction', 'regime_bear_fraction',
        ),
        'ensemble_rules': (
            'vol_target_annual', 'vol_lookback',
            'dd_thresholds', 'dd_min_invested',
            'regime_bull_threshold', 'regime_bear_threshold',
            'regime_bull_fraction', 'regime_neutral_fraction', 'regime_bear_fraction',
            'ensemble_mode',
        ),
        'linucb': ('bandit_n_actions', 'bandit_reward_window', 'bandit_alpha_ucb', 'bandit_feature_lookback'),
        'thompson': ('bandit_n_actions', 'bandit_reward_window', 'bandit_feature_lookback'),
        'epsilon_greedy': ('bandit_n_actions', 'bandit_reward_window', 'bandit_epsilon', 'bandit_feature_lookback'),
        'supervised': ('bandit_n_actions', 'supervised_model', 'supervised_retrain_every', 'supervised_label_window'),
        'cvar_robust': ('cvar_confidence', 'cvar_n_scenarios', 'cvar_lambda_base', 'cvar_regime_scaling', 'cvar_dd_budget'),
        'cmdp_lagrangian': (
            'ql_alpha', 'ql_gamma', 'ql_epsilon',
            'cmdp_constraint_type', 'cmdp_constraint_kappa',
            'cmdp_lambda_init', 'cmdp_lambda_lr', 'cmdp_tail_loss_threshold',
        ),
        'council': (
            'council_experts', 'council_gate_model', 'council_retrain_every',
            'council_min_samples', 'council_temperature', 'council_min_weight',
            'council_default_bias', 'bandit_n_actions', 'bandit_reward_window',
            'bandit_alpha_ucb', 'bandit_feature_lookback', 'cvar_confidence',
            'cvar_n_scenarios', 'cvar_lambda_base', 'cvar_regime_scaling',
            'cvar_dd_budget', 'regime_bull_threshold', 'regime_bear_threshold',
            'regime_bull_fraction', 'regime_neutral_fraction', 'regime_bear_fraction',
        ),
        'mlp_meta': (
            'mlp_meta_experts', 'mlp_meta_hidden_layers', 'mlp_meta_retrain_every',
            'mlp_meta_min_samples', 'mlp_meta_feature_lookback', 'mlp_meta_min_weight',
            'mlp_meta_default_bias', 'mlp_meta_learning_rate', 'mlp_meta_alpha_reg',
            'mlp_meta_temperature', 'bandit_n_actions', 'bandit_reward_window',
            'bandit_alpha_ucb', 'bandit_feature_lookback', 'cvar_confidence',
            'cvar_n_scenarios', 'cvar_lambda_base', 'cvar_regime_scaling',
            'cvar_dd_budget', 'regime_bull_threshold', 'regime_bear_threshold',
            'regime_bull_fraction', 'regime_neutral_fraction', 'regime_bear_fraction',
        ),
        'mpc': (
            'mpc_objective_version', 'mpc_horizon', 'mpc_replan_every', 'mpc_discount',
            'mpc_alpha_decay', 'mpc_stress_reversion', 'mpc_min_invested',
            'mpc_max_stabilizer', 'mpc_risk_penalty', 'mpc_turnover_penalty',
            'mpc_drawdown_penalty', 'mpc_stress_penalty', 'mpc_terminal_penalty',
            'mpc_max_daily_change', 'mpc_joint_convexity', 'mpc_convexity_tail_scale',
        ),
        'adaptive_allocator': (
            'adaptive_allocator_min_invested', 'adaptive_allocator_param_smoothing',
            'adaptive_allocator_risk_mult_range', 'adaptive_allocator_anchor_mult_range',
            'adaptive_allocator_turnover_mult_range', 'adaptive_allocator_alpha_mult_range',
            'adaptive_allocator_cap_scale_range', 'adaptive_allocator_group_cap_scale_range',
            'adaptive_allocator_policy_version',
        ),
        'q_learning': ('ql_alpha', 'ql_gamma', 'ql_epsilon'),
    }

    for field in method_fields.get(method, ()):
        if field in merged_cfg:
            canonical[field] = _normalize_checkpoint_value(merged_cfg[field])

    if bool(merged_cfg.get('convexity_enabled', False)):
        canonical['convexity_enabled'] = True
        for field in (
            'convexity_threshold', 'convexity_mode_carries', 'convexity_mode_lambdas',
            'convexity_mild_drawdown', 'convexity_strong_drawdown', 'convexity_mild_vol',
            'convexity_strong_vol', 'convexity_mild_regime', 'convexity_strong_regime',
        ):
            if field in merged_cfg:
                canonical[field] = _normalize_checkpoint_value(merged_cfg[field])
    return canonical


def _canonical_pipeline_config(config_payload: dict[str, object]) -> dict[str, object]:
    default_payload = asdict(PipelineConfig())
    config_payload = config_payload or {}
    merged_payload = dict(default_payload)
    for key, value in config_payload.items():
        if isinstance(value, dict) and isinstance(merged_payload.get(key), dict):
            nested = dict(merged_payload.get(key, {}))
            nested.update(value)
            merged_payload[key] = nested
        else:
            merged_payload[key] = value

    experiment_cfg = dict(merged_payload.get('experiment', {}) or {})
    feature_cfg = dict(merged_payload.get('feature_availability', {}) or {})
    cost_cfg = dict(merged_payload.get('cost_model', {}) or {})
    optimizer_cfg = dict(merged_payload.get('optimizer', {}) or {})
    option_cfg = dict(merged_payload.get('option_overlay', {}) or {})
    control_cfg = dict(merged_payload.get('control', {}) or {})

    return {
        'train_frac': _normalize_checkpoint_value(merged_payload.get('train_frac')),
        'rebalance_band': _normalize_checkpoint_value(merged_payload.get('rebalance_band')),
        'min_turnover': _normalize_checkpoint_value(merged_payload.get('min_turnover')),
        'portfolio_reward_mode': _normalize_checkpoint_value(merged_payload.get('portfolio_reward_mode')),
        'hedge_reward_mode': _normalize_checkpoint_value(merged_payload.get('hedge_reward_mode')),
        'e2e_reward_mode': _normalize_checkpoint_value(merged_payload.get('e2e_reward_mode')),
        'enable_e2e_baseline': _normalize_checkpoint_value(merged_payload.get('enable_e2e_baseline')),
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
            'adaptive_allocator': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator')),
            'adaptive_allocator_min_invested': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_min_invested')),
            'adaptive_allocator_param_smoothing': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_param_smoothing')),
            'adaptive_allocator_risk_mult_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_risk_mult_range')),
            'adaptive_allocator_anchor_mult_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_anchor_mult_range')),
            'adaptive_allocator_turnover_mult_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_turnover_mult_range')),
            'adaptive_allocator_alpha_mult_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_alpha_mult_range')),
            'adaptive_allocator_cap_scale_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_cap_scale_range')),
            'adaptive_allocator_group_cap_scale_range': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_group_cap_scale_range')),
            'adaptive_allocator_policy_version': _normalize_checkpoint_value(optimizer_cfg.get('adaptive_allocator_policy_version')),
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


def _universe_checkpoint_dir(base_checkpoint_dir: Path, universe_id: str | None) -> Path:
    if universe_id in (None, "", "A"):
        return base_checkpoint_dir / "universe_A"
    return base_checkpoint_dir / f"universe_{universe_id}"


def _scope_universe_run_key(run_key: str, universe_id: str | None) -> str:
    if universe_id in (None, "", "A"):
        return run_key
    return f"universe_{universe_id}_{run_key}"


def _checkpoint_key_candidates(run_key: str, universe_id: str | None) -> list[str]:
    scoped = _scope_universe_run_key(run_key, universe_id)
    candidates: list[str] = []
    for key in (scoped, run_key):
        if key not in candidates:
            candidates.append(key)
    return candidates


def _checkpoint_candidates(checkpoint_dir: Path, run_key: str, universe_id: str | None) -> list[tuple[str, Path]]:
    candidates = [
        (candidate_key, _checkpoint_path(checkpoint_dir, candidate_key))
        for candidate_key in _checkpoint_key_candidates(run_key, universe_id)
    ]
    legacy_dir = checkpoint_dir.parent if checkpoint_dir.name.startswith("universe_") else checkpoint_dir
    if legacy_dir != checkpoint_dir:
        for candidate_key in _checkpoint_key_candidates(run_key, universe_id):
            legacy_path = _checkpoint_path(legacy_dir, candidate_key)
            pair = (candidate_key, legacy_path)
            if pair not in candidates:
                candidates.append(pair)
    return candidates


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
    return {'rows': int(len(series)), 'sum': round(float(numeric.sum()), 10)}


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


def _compatible_checkpoint_view(metadata: dict[str, object]) -> dict[str, object]:
    prices = dict(metadata.get('prices', {}) or {})
    volumes = dict(metadata.get('volumes', {}) or {})
    returns = dict(metadata.get('returns', {}) or {})
    macro = dict(metadata.get('macro', {}) or {})
    sec_quality = dict(metadata.get('sec_quality', {}) or {})
    return {
        'suite': metadata.get('suite'),
        'include_e2e': metadata.get('include_e2e'),
        'config': metadata.get('config'),
        'prices_columns': tuple(prices.get('columns', ()) or ()),
        'volumes_columns': tuple(volumes.get('columns', ()) or ()),
        'returns_columns': tuple(returns.get('columns', ()) or ()),
        'macro_columns': tuple(macro.get('columns', ()) or ()),
        'sec_quality_present': int(sec_quality.get('rows', 0) or 0) > 0,
    }


def _config_only_checkpoint_view(metadata: dict[str, object]) -> dict[str, object]:
    return {
        'suite': metadata.get('suite'),
        'include_e2e': metadata.get('include_e2e'),
        'config': metadata.get('config'),
    }


def _load_checkpoint_results(
    checkpoint_path: Path,
    expected_metadata: dict[str, object],
    match_mode: str = 'strict',
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
        if match_mode == 'compatible':
            if _compatible_checkpoint_view(payload_metadata) == _compatible_checkpoint_view(expected_metadata):
                print(f"  Loading compatible checkpoint at {checkpoint_path}; data window or patch metadata changed.")
                return payload['results']
        elif match_mode == 'config_only':
            if _config_only_checkpoint_view(payload_metadata) == _config_only_checkpoint_view(expected_metadata):
                print(f"  Loading config-matched checkpoint at {checkpoint_path}; data signatures differ.")
                return payload['results']
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
    return {'__checkpoint_repr__': repr(value), '__checkpoint_type__': type(value).__name__}


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
        'updated_at': pd.Timestamp.now().isoformat(),
        'status': status,
        'total_runs': total_runs,
        'completed_runs': len(completed_run_keys),
        'completed_run_keys': completed_run_keys,
        'current_run': current_run,
        'last_completed_run': last_completed_run,
    }
    _progress_path(checkpoint_dir).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root, check=True, capture_output=True, text=True)
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _config_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(_normalize_checkpoint_value(payload), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': getattr(matplotlib, '__version__', 'unknown'),
    }
    try:
        import sklearn
        versions['scikit_learn'] = sklearn.__version__
    except Exception:
        pass
    return versions


def _write_run_manifest(
    output_dir: Path,
    *,
    run_type: str,
    base_config: PipelineConfig | None,
    evaluation_config,
    universe_id: str | None = None,
    universe_ids: tuple[str, ...] | None = None,
    run_timestamp: str | None = None,
    status: str,
    summary: dict[str, object] | None = None,
) -> None:
    repo_root = _repo_root()
    config_payload = {
        'base_config': asdict(base_config) if base_config is not None else None,
        'evaluation_config': asdict(evaluation_config),
        'universe_id': universe_id,
        'universe_ids': list(universe_ids) if universe_ids is not None else None,
    }
    payload = {
        'run_type': run_type,
        'status': status,
        'created_at': pd.Timestamp.now().isoformat(),
        'run_timestamp': run_timestamp,
        'repo_root': str(repo_root),
        'git_commit': _safe_git_commit(repo_root),
        'config_hash': _config_hash(config_payload),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'package_versions': _package_versions(),
        'universe_id': universe_id,
        'universe_ids': list(universe_ids) if universe_ids is not None else None,
        'config': config_payload,
        'summary': summary or {},
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
