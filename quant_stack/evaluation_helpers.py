"""Metrics, labels, and suite builders for research evaluation."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .config import (
    ControlConfig,
    ExperimentConfig,
    PipelineConfig,
    RISK_FREE_RATE,
    get_active_benchmark_label,
)


def _daily_returns_from_path(path: list[float] | np.ndarray) -> np.ndarray:
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
        'avg_desired_turnover': float(np.mean(results.get('desired_turnover', [0.0]))),
        'avg_buy_turnover': float(np.mean(results.get('buy_turnover', [0.0]))),
        'avg_sell_turnover': float(np.mean(results.get('sell_turnover', [0.0]))),
        'avg_participation_rate': float(np.mean(results.get('avg_participation_rates', [0.0]))),
        'avg_max_participation_rate': float(np.mean(results.get('max_participation_rates', [0.0]))),
        'liquidity_cap_hit_rate': float(np.mean(results.get('adv_cap_hits', [0.0]))),
        'avg_adv_excess_ratio': float(np.mean(results.get('adv_excess_ratios', [0.0]))),
        'avg_execution_weight_gap': float(np.mean(results.get('execution_weight_gaps', [0.0]))),
        'avg_execution_delay_gap': float(np.mean(results.get('execution_delay_gaps', [0.0]))),
        'avg_execution_shortfall': float(np.mean(results.get('execution_shortfalls', [0.0]))),
        'execution_delay_days': float(np.mean(results.get('execution_delay_days', [0.0]))),
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
    desired_turnover = np.asarray(results.get('desired_turnover', []), dtype=float)
    buy_turnover = np.asarray(results.get('buy_turnover', []), dtype=float)
    sell_turnover = np.asarray(results.get('sell_turnover', []), dtype=float)
    avg_participation = np.asarray(results.get('avg_participation_rates', []), dtype=float)
    max_participation = np.asarray(results.get('max_participation_rates', []), dtype=float)
    adv_cap_hits = np.asarray(results.get('adv_cap_hits', []), dtype=float)
    execution_weight_gaps = np.asarray(results.get('execution_weight_gaps', []), dtype=float)
    execution_delay_gaps = np.asarray(results.get('execution_delay_gaps', []), dtype=float)
    execution_shortfalls = np.asarray(results.get('execution_shortfalls', []), dtype=float)
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
        len(desired_turnover) if len(desired_turnover) > 0 else len(beliefs),
        len(buy_turnover) if len(buy_turnover) > 0 else len(beliefs),
        len(sell_turnover) if len(sell_turnover) > 0 else len(beliefs),
        len(avg_participation) if len(avg_participation) > 0 else len(beliefs),
        len(max_participation) if len(max_participation) > 0 else len(beliefs),
        len(adv_cap_hits) if len(adv_cap_hits) > 0 else len(beliefs),
        len(execution_weight_gaps) if len(execution_weight_gaps) > 0 else len(beliefs),
        len(execution_delay_gaps) if len(execution_delay_gaps) > 0 else len(beliefs),
        len(execution_shortfalls) if len(execution_shortfalls) > 0 else len(beliefs),
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
    desired_turnover = desired_turnover[:n_obs] if len(desired_turnover) > 0 else np.zeros(n_obs, dtype=float)
    buy_turnover = buy_turnover[:n_obs] if len(buy_turnover) > 0 else np.zeros(n_obs, dtype=float)
    sell_turnover = sell_turnover[:n_obs] if len(sell_turnover) > 0 else np.zeros(n_obs, dtype=float)
    avg_participation = avg_participation[:n_obs] if len(avg_participation) > 0 else np.zeros(n_obs, dtype=float)
    max_participation = max_participation[:n_obs] if len(max_participation) > 0 else np.zeros(n_obs, dtype=float)
    adv_cap_hits = adv_cap_hits[:n_obs] if len(adv_cap_hits) > 0 else np.zeros(n_obs, dtype=float)
    execution_weight_gaps = execution_weight_gaps[:n_obs] if len(execution_weight_gaps) > 0 else np.zeros(n_obs, dtype=float)
    execution_delay_gaps = execution_delay_gaps[:n_obs] if len(execution_delay_gaps) > 0 else np.zeros(n_obs, dtype=float)
    execution_shortfalls = execution_shortfalls[:n_obs] if len(execution_shortfalls) > 0 else np.zeros(n_obs, dtype=float)
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
            'avg_desired_turnover': float(desired_turnover[mask].mean()),
            'avg_buy_turnover': float(buy_turnover[mask].mean()),
            'avg_sell_turnover': float(sell_turnover[mask].mean()),
            'avg_participation_rate': float(avg_participation[mask].mean()),
            'avg_max_participation_rate': float(max_participation[mask].mean()),
            'liquidity_cap_hit_rate': float(adv_cap_hits[mask].mean()),
            'avg_execution_weight_gap': float(execution_weight_gaps[mask].mean()),
            'avg_execution_delay_gap': float(execution_delay_gaps[mask].mean()),
            'avg_execution_shortfall': float(execution_shortfalls[mask].mean()),
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


def build_control_comparison_suite(base_config: PipelineConfig) -> list[PipelineConfig]:
    configs: list[PipelineConfig] = []
    shared_experiment = ExperimentConfig(
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=True,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )

    def add(label: str, method: str, control: ControlConfig | None = None) -> None:
        cfg = copy.deepcopy(base_config)
        cfg.experiment = copy.deepcopy(shared_experiment)
        cfg.experiment.label = label
        cfg.experiment.control_method = method if method != 'none' else ''
        cfg.control = control or ControlConfig(method=method)
        cfg.enable_e2e_baseline = False
        configs.append(cfg)

    add('factor_only', 'none', ControlConfig(method='none'))
    add('A1_fixed', 'fixed', ControlConfig(method='fixed', fixed_invested_fraction=0.95))
    add('A2_vol_target', 'vol_target', ControlConfig(method='vol_target', vol_target_annual=0.12))
    add('A3_dd_delever', 'dd_delever', ControlConfig(method='dd_delever'))
    add('A4_regime_rules', 'regime_rules', ControlConfig(method='regime_rules'))
    add('A5_ensemble_mean', 'ensemble_rules', ControlConfig(method='ensemble_rules', ensemble_mode='mean'))
    add('B1_linucb', 'linucb', ControlConfig(method='linucb'))
    add('B2_thompson', 'thompson', ControlConfig(method='thompson'))
    add('B3_epsilon_greedy', 'epsilon_greedy', ControlConfig(method='epsilon_greedy'))
    add('C_supervised', 'supervised', ControlConfig(method='supervised', supervised_model='logistic'))
    add('D_cvar_robust', 'cvar_robust', ControlConfig(method='cvar_robust'))
    add('D_plus_convexity', 'cvar_robust', ControlConfig(method='cvar_robust', convexity_enabled=True))
    add('H_mpc', 'mpc', ControlConfig(method='mpc'))
    add('E_council', 'council', ControlConfig(method='council'))
    add('E_plus_convexity', 'council', ControlConfig(method='council', convexity_enabled=True))
    add('G_mlp_meta', 'mlp_meta', ControlConfig(method='mlp_meta'))
    add('G_plus_convexity', 'mlp_meta', ControlConfig(method='mlp_meta', convexity_enabled=True))
    add('F_cmdp_lagrangian', 'cmdp_lagrangian', ControlConfig(method='cmdp_lagrangian'))
    add('RL_q_learning', 'q_learning', ControlConfig(method='q_learning'))
    return configs


def _control_train_fracs(
    ctrl_config: PipelineConfig,
    evaluation_config,
) -> tuple[float, ...]:
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
    if ctrl_config.control.method == 'q_learning':
        return 0.75
    return float(base_config.train_frac)


def build_ablation_suite(base_config: PipelineConfig) -> list[PipelineConfig]:
    configs: list[PipelineConfig] = []

    def ablation(label: str, **kwargs) -> PipelineConfig:
        cfg = copy.deepcopy(base_config)
        cfg.experiment = ExperimentConfig(label=label, **kwargs)
        cfg.hedge_ratios = (0.0, 0.0, 0.0, 0.0)
        return cfg

    factor_only = ablation(
        'factor_only',
        use_factor=True,
        use_pairs=False,
        use_lstm=False,
        adaptive_combiner=False,
        use_portfolio_rl=False,
        use_hedge_rl=False,
    )
    factor_only.optimizer.use_optimizer = False
    configs.append(factor_only)
    configs.append(ablation('alpha_stack_fixed_weights', use_factor=True, use_pairs=False, use_lstm=False, adaptive_combiner=False, use_portfolio_rl=False, use_hedge_rl=False))
    configs.append(ablation('alpha_stack_no_rl', use_factor=True, use_pairs=False, use_lstm=False, adaptive_combiner=True, use_portfolio_rl=False, use_hedge_rl=False))
    configs.append(ablation('portfolio_rl_fixed_weights', use_factor=True, use_pairs=False, use_lstm=False, adaptive_combiner=False, use_portfolio_rl=True, use_hedge_rl=False))
    configs.append(ablation('full_pipeline'))
    configs.append(ablation('full_pipeline_fixed_weights', use_factor=True, use_pairs=False, use_lstm=False, adaptive_combiner=False, use_portfolio_rl=True, use_hedge_rl=False))
    return configs


def _component_label(label: str) -> str:
    return label.rsplit('_tf', 1)[0] if '_tf' in label else label


def _control_component_label(label: str) -> str:
    base = _component_label(label)
    if base == 'factor_only':
        return 'alpha_engine_no_control'
    return base


def _display_label(label: str) -> str:
    benchmark_label = get_active_benchmark_label()
    if label == benchmark_label:
        return benchmark_label
    mapping = {
        'factor_only': 'Factor Only',
        'alpha_engine_no_control': 'No Control\n(Alpha Engine)',
        'alpha_stack_fixed_weights': 'Allocator\nFixed Weights',
        'alpha_stack_no_rl': 'Alpha Stack\nNo RL',
        'portfolio_rl_fixed_weights': 'Portfolio RL\nFixed Weights',
        'full_pipeline': 'Full\nPipeline',
        'full_pipeline_fixed_weights': 'Full Pipeline\nFixed Weights',
        'SPY': benchmark_label,
        'factor_benchmark': 'Factor Benchmark',
        'vol_target': 'Vol-Target',
        'dd_delever': 'DD-Delever',
        'e2e_rl': 'E2E RL\n(PPO)',
        'risk_parity': 'Risk Parity',
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
        'H_mpc': 'H: MPC',
        'I_adaptive_allocator': 'I: Adaptive Allocator',
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
    return summary.sort_values('order').drop(columns='order')


def _build_control_comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
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
        'RL_ppo',
    ]
    summary['order'] = summary['component_label'].apply(lambda x: ordering.index(x) if x in ordering else len(ordering))
    return summary.sort_values('order').drop(columns='order')


def _decorate_control_significance(significance: pd.DataFrame) -> pd.DataFrame:
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
    if label.startswith('H') or label.startswith('I'):
        return 'predictive control'
    if label.startswith('E') or label.startswith('G'):
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
        'predictive control': '#6f4e7c',
        'meta control': '#72b7b2',
        'safe rl': '#8c6d31',
        'rl': '#b279a2',
        'other': '#9d9da1',
    }
    return family_colors[_control_family(label)]


def _pareto_frontier_points(summary: pd.DataFrame) -> pd.DataFrame:
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


def _build_execution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    execution = metrics[metrics['suite'] == 'control_comparison'].copy()
    if execution.empty:
        return pd.DataFrame()
    execution['component_label'] = execution['label'].map(_control_component_label)
    return (
        execution.groupby('component_label')
        .agg(
            mean_turnover=('avg_turnover', 'mean'),
            mean_desired_turnover=('avg_desired_turnover', 'mean'),
            mean_buy_turnover=('avg_buy_turnover', 'mean'),
            mean_sell_turnover=('avg_sell_turnover', 'mean'),
            mean_participation_rate=('avg_participation_rate', 'mean'),
            mean_max_participation_rate=('avg_max_participation_rate', 'mean'),
            liquidity_cap_hit_rate=('liquidity_cap_hit_rate', 'mean'),
            mean_execution_weight_gap=('avg_execution_weight_gap', 'mean'),
            mean_execution_delay_gap=('avg_execution_delay_gap', 'mean'),
            mean_execution_shortfall=('avg_execution_shortfall', 'mean'),
            mean_transaction_cost=('avg_transaction_cost', 'mean'),
        )
        .reset_index()
        .sort_values('mean_execution_shortfall', ascending=False)
    )


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
        benchmark_label = get_active_benchmark_label()
        spy = rolling_references[rolling_references['label'] == benchmark_label].copy()
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
