"""Backtest orchestration for the quant trading pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .alpha import (
    AlphaCombiner,
    FamaFrenchFactors,
    GARCHForecaster,
    HMMRegimeDetector,
)
from .config import (
    BENCHMARK,
    BENCHMARK_COMPONENTS,
    BENCHMARK_REBALANCE,
    ControlConfig,
    PipelineConfig,
    RISK_FREE_RATE,
    UNIVERSE,
)
from .controllers import (
    BaseController,
    CouncilController,
    CVaRRobustController,
    build_control_state,
    build_controller,
)
from .controllers_extended import CMDPLagrangianController, MLPMetaController
from .data import compute_macro_regime_signal
from .execution import (
    _apply_execution_constraints,
    _build_feature_contracts,
    _compute_transaction_cost,
    _net_portfolio_return,
)
from .rl import PortfolioConstructionRL


def _apply_macro_lag(macro_data: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    if macro_data.empty or lag_days <= 0:
        return macro_data
    return macro_data.shift(lag_days).ffill()


def _regime_entropy(regime_belief: float) -> float:
    p = float(np.clip(regime_belief, 1e-6, 1 - 1e-6))
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.clip(entropy / np.log(2.0), 0.0, 1.0))


def _compute_portfolio_reward(
    portfolio_ret: float,
    recent_portfolio_rets: np.ndarray,
    mode: str,
) -> float:
    reward_mode = str(mode or 'differential_sharpe').lower()
    if reward_mode == 'return':
        return float(portfolio_ret * 100.0)
    if reward_mode == 'sortino':
        if len(recent_portfolio_rets) <= 5:
            return float(portfolio_ret * 100.0)
        downside = recent_portfolio_rets[recent_portfolio_rets < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 1e-8
        return float((portfolio_ret - recent_portfolio_rets.mean()) / (downside_std + 1e-8))
    if reward_mode == 'mean_variance':
        if len(recent_portfolio_rets) <= 5:
            return float(portfolio_ret * 100.0)
        lam = 2.0
        return float(portfolio_ret * 100.0 - lam * float(np.var(recent_portfolio_rets)) * 100.0)
    if len(recent_portfolio_rets) <= 5:
        return float(portfolio_ret * 100.0)
    return float((portfolio_ret - recent_portfolio_rets.mean()) / (recent_portfolio_rets.std() + 1e-8))


def _initialize_benchmark_state(
    returns: pd.DataFrame,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    components = [
        (ticker, float(weight))
        for ticker, weight in BENCHMARK_COMPONENTS
        if ticker in returns.columns and float(weight) > 0.0
    ]
    if not components:
        raise RuntimeError(
            f"No benchmark components for {BENCHMARK!r} are available in the returns frame."
        )
    total_weight = sum(weight for _, weight in components)
    normalized = [(ticker, weight / total_weight) for ticker, weight in components]
    current_weights = {ticker: weight for ticker, weight in normalized}
    return normalized, current_weights


def _benchmark_step_return(
    returns: pd.DataFrame,
    t: int,
    dates: pd.Index,
    target_components: list[tuple[str, float]],
    current_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    if BENCHMARK_REBALANCE == 'monthly' and t > 0:
        current_date = pd.Timestamp(dates[t])
        prev_date = pd.Timestamp(dates[t - 1])
        if current_date.month != prev_date.month or current_date.year != prev_date.year:
            current_weights = {ticker: weight for ticker, weight in target_components}

    bench_ret = 0.0
    next_weights: dict[str, float] = {}
    for ticker, _ in target_components:
        ticker_ret = float(returns[ticker].iloc[t])
        weight = float(current_weights.get(ticker, 0.0))
        bench_ret += weight * ticker_ret
        next_weights[ticker] = weight * (1.0 + ticker_ret)

    denom = max(1.0 + float(bench_ret), 1e-8)
    next_weights = {ticker: value / denom for ticker, value in next_weights.items()}
    return float(bench_ret), next_weights


def _resolve_control_mode(config: PipelineConfig) -> tuple[str, bool, bool]:
    """Determine which control path to use. Returns (method, use_new, use_legacy_rl)."""
    control_method = config.experiment.control_method
    if control_method and control_method not in ('none', ''):
        return control_method, True, False
    if config.control.method and config.control.method not in ('none', ''):
        return config.control.method, True, False
    if config.experiment.use_portfolio_rl:
        return 'legacy_rl', False, True
    return 'none', False, False


def run_full_pipeline(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    sec_quality_scores: pd.Series | None = None,
    train_frac: float = 0.5,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """
    Run the complete quant pipeline:
    Data -> Alpha -> Constrained Allocator -> Control Layer -> Execution

    The control layer is pluggable via config.control.method or
    config.experiment.control_method.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Running Full Pipeline Backtest")
    print("=" * 60)

    config = config or PipelineConfig(train_frac=train_frac)
    train_frac = config.train_frac

    tickers = [t for t in UNIVERSE if t in returns.columns]
    benchmark_components, benchmark_weights = _initialize_benchmark_state(returns)
    dates = returns.index
    n = len(dates)
    train_end = int(n * train_frac)

    # Initialize alpha models
    macro_data = macro_data if macro_data is not None else pd.DataFrame(index=returns.index)
    macro_data = _apply_macro_lag(macro_data, config.feature_availability.macro_lag_days)
    sec_quality_scores = sec_quality_scores if sec_quality_scores is not None else pd.Series(dtype=float)
    if not config.feature_availability.allow_static_sec_quality:
        sec_quality_scores = pd.Series(dtype=float)

    ff = FamaFrenchFactors(prices, returns, sec_quality_scores=sec_quality_scores)
    garch = GARCHForecaster(returns, refit_every=21)
    hmm = HMMRegimeDetector(lookback=252)
    combiner = AlphaCombiner()

    # Shared allocator (factor-anchored target book + constrained optimizer)
    allocator = PortfolioConstructionRL(
        rebalance_band=config.rebalance_band,
        min_turnover=config.min_turnover,
        optimizer_config=config.optimizer,
    )

    # Resolve control mode
    control_method, use_new_controller, use_legacy_rl = _resolve_control_mode(config)

    controller: BaseController | None = None
    if use_new_controller:
        controller = build_controller(config.control)
        print(f"  Control method: {control_method} ({controller.label})")
    elif use_legacy_rl:
        print("  Control method: legacy portfolio RL (Q-learning)")
    else:
        print("  Control method: none (factor-only)")

    # Market return for regime detection
    market_ret = returns[tickers].mean(axis=1)
    _portfolio_vol_20d = returns[tickers].rolling(20).std().mean(axis=1)
    execution_queue: list[pd.Series] = []

    # Storage
    wealth = 1.0
    spy_wealth = 1.0
    equal_wealth = 1.0
    factor_wealth = 1.0
    voltarget_wealth = 1.0
    ddlever_wealth = 1.0
    riskparity_wealth = 1.0

    results = {
        'dates': [], 'wealth': [1.0], 'spy': [1.0], 'equal': [1.0], 'factor': [1.0],
        'voltarget': [1.0], 'ddlever': [1.0], 'risk_parity': [1.0],
        'actions': [], 'hedge_actions': [], 'regime_beliefs': [],
        'invested_fractions': [], 'overlay_sizes': [],
        'factor_scores_hist': [], 'portfolio_weights': [],
        'garch_vols': [], 'pairs_info_hist': [],
        'drawdowns': [], 'spy_drawdowns': [],
        'turnover': [], 'hedge_pnl': [], 'source_weights_hist': [],
        'transaction_costs': [],
        'desired_turnover': [], 'buy_turnover': [], 'sell_turnover': [],
        'avg_participation_rates': [], 'max_participation_rates': [],
        'liquidity_scales': [], 'adv_cap_hits': [], 'adv_excess_ratios': [],
        'execution_weight_gaps': [], 'execution_delay_gaps': [],
        'execution_shortfalls': [], 'execution_delay_days': [],
        'portfolio_returns': [], 'cash_weights': [], 'hedge_ratios': [],
        'alpha_dispersion': [], 'regime_entropy': [], 'ic_instability': [],
        'garch_vol_uncertainty': [], 'uncertainty_score': [],
        'hedge_type_actions': [], 'hedge_types': [],
        'effective_hedge_ratios': [], 'hedge_costs': [], 'hedge_benefits': [],
        'iv_annualized': [], 'iv_percentile': [], 'iv_realized_spread': [], 'iv_regime_score': [],
        'raw_portfolio_returns': [],
        'convexity_modes': [], 'convexity_mode_names': [],
        'convexity_carries': [], 'convexity_benefits': [], 'convexity_net_adjustments': [],
        'council_weight_regime_rules': [], 'council_weight_linucb': [], 'council_weight_cvar_robust': [],
        'council_dominant_expert': [], 'council_best_expert': [], 'council_gate_entropy': [],
        'mlp_meta_weight_regime_rules': [], 'mlp_meta_weight_linucb': [], 'mlp_meta_weight_cvar_robust': [],
        'mlp_meta_dominant_expert': [], 'mlp_meta_best_expert': [], 'mlp_meta_gate_entropy': [],
        'mlp_meta_gate_source': [], 'mlp_meta_n_training_samples': [],
        'mpc_invested_targets': [], 'mpc_stabilizer_mixes': [],
        'mpc_plan_steps': [], 'mpc_plan_sources': [], 'mpc_plan_objectives': [],
        'adaptive_allocator_stress_scores': [],
        'adaptive_allocator_invested_targets': [],
        'adaptive_allocator_risk_mults': [],
        'adaptive_allocator_anchor_mults': [],
        'adaptive_allocator_turnover_mults': [],
        'adaptive_allocator_alpha_mults': [],
        'adaptive_allocator_cap_scales': [],
        'adaptive_allocator_group_cap_scales': [],
        'adaptive_allocator_max_weights': [],
        'cmdp_lambdas': [], 'cmdp_constraint_costs': [], 'cmdp_violations': [],
        'control_method': control_method,
        'benchmark_label': BENCHMARK,
    }

    peak = 1.0
    spy_peak = 1.0
    ddlever_peak = 1.0

    # Benchmark parameters
    vol_target_annual = 0.15
    vol_lookback = 60
    dd_thresholds = [(-0.05, 1.0), (-0.08, 0.70), (-0.12, 0.40)]

    print(f"  Training period: {dates[0].date()} to {dates[train_end].date()}")
    print(f"  Test period: {dates[train_end].date()} to {dates[-1].date()}")
    print("  Preparing factor/GARCH/HMM alpha sleeves...")

    _rp_weights_cache: np.ndarray | None = None
    _rp_last_fit: int = -999
    _macro_belief_cache: float = 0.5
    _macro_last_compute: int = -999

    print("  Running walk-forward backtest...")
    for t in range(max(252, train_end), n):
        if t % 100 == 0:
            progress = (t - train_end) / (n - train_end) * 100
            print(f"    Progress: {progress:.0f}%  Wealth: {wealth:.3f}  "
                  f"SPY: {spy_wealth:.3f}  Date: {dates[t].date()}")

        # --- Alpha Generation ---
        factor_scores, factor_detail = ff.get_factor_scores(t - 1)
        garch_vols = garch.forecast_all(t - 1)
        garch.record_avg_vol(float(garch_vols.mean()))
        regime_belief = hmm.get_regime_belief(t - 1, market_ret)
        if not macro_data.empty and dates[t - 1] in macro_data.index:
            if t - _macro_last_compute >= 5:
                macro_window = macro_data.loc[:dates[t - 1]].tail(252)
                _macro_belief_cache = compute_macro_regime_signal(macro_window)
                _macro_last_compute = t
            regime_belief = 0.75 * regime_belief + 0.25 * _macro_belief_cache

        portfolio_vol = _portfolio_vol_20d.iloc[t - 1] if t > 20 else returns[tickers].iloc[:t].std().mean()
        prev_weights = None
        if results['portfolio_weights']:
            prev_weights = pd.Series(results['portfolio_weights'][-1], index=tickers)
        recent_returns = returns[tickers].iloc[max(0, t - 126):t]
        # --- Signal Combination ---
        expected, confidence, source_signals, source_weights = combiner.combine(
            factor_scores, garch_vols, regime_belief, tickers,
            use_factor=config.experiment.use_factor,
            adaptive=config.experiment.adaptive_combiner,
        )

        # --- Uncertainty diagnostics ---
        weighted_alpha = (expected * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        alpha_dispersion = float(weighted_alpha.std())
        regime_uncertainty = _regime_entropy(regime_belief)
        ic_instability = float(np.clip(combiner.get_ic_instability(), 0.0, 1.0))
        garch_vol_uncertainty = garch.forecast_vol_uncertainty(t - 1)
        alpha_dispersion_norm = float(np.clip(np.tanh(alpha_dispersion / 2.0), 0.0, 1.0))
        ic_instability_norm = float(np.clip(ic_instability / 0.20, 0.0, 1.0))
        uncertainty_score = float(np.clip(
            0.35 * alpha_dispersion_norm + 0.30 * regime_uncertainty
            + 0.20 * ic_instability_norm + 0.15 * garch_vol_uncertainty,
            0.0, 1.0,
        ))

        # ================================================================
        # CONTROL LAYER
        # ================================================================
        invested_fraction_selected = 1.0
        overlay_size_selected = 0.0
        action = 0
        allocator_state = build_control_state(
            alpha_scores=expected,
            portfolio_weights=prev_weights,
            recent_returns=recent_returns,
            regime_belief=regime_belief,
            wealth_path=results['wealth'],
            t=t,
        )
        ctrl_state = allocator_state if use_new_controller and controller is not None else None
        expert_feedback: dict[str, float] | None = None

        if use_new_controller and controller is not None:
            if controller.uses_direct_weights():
                built_weights = controller.build_target_weights(
                    allocator=allocator,
                    factor_scores=factor_scores,
                    alpha_scores=expected,
                    confidence=confidence,
                    recent_returns=recent_returns,
                    prev_weights=prev_weights,
                    optimizer_config=config.optimizer,
                    state=ctrl_state,
                )
                if built_weights is None:
                    raise ValueError(f"Controller {controller.label} signaled direct weights but returned None")
                weights = built_weights.reindex(tickers).fillna(0.0).clip(lower=0.0)
                if float(weights.sum()) > 1.0:
                    weights = weights / float(weights.sum())
                invested_fraction_selected = float(np.clip(weights.sum(), 0.0, 1.0))
                overlay_size_selected = max(0.0, invested_fraction_selected - float(ctrl_state.invested_fraction))
                cash_weight = max(0.0, 1.0 - float(weights.sum()))
            else:
                invested_fraction_selected = float(np.clip(
                    controller.compute_invested_fraction(ctrl_state), 0.0, 1.0
                ))
                weights = allocator.optimize_target_book(
                    factor_scores=factor_scores,
                    alpha_scores=expected,
                    confidence=confidence,
                    recent_returns=recent_returns,
                    prev_weights=prev_weights,
                    control_state=allocator_state,
                )
                weights = weights * invested_fraction_selected
                cash_weight = max(0.0, 1.0 - float(weights.sum()))

            if prev_weights is not None:
                weights = allocator.apply_rebalance_band(weights, prev_weights)
                cash_weight = max(0.0, 1.0 - float(weights.sum()))

        elif use_legacy_rl:
            state = allocator.get_state(expected, prev_weights, recent_returns)
            action = allocator.select_action(state)
            weights, cash_weight = allocator.construct_portfolio(
                factor_scores, expected, confidence, action,
                recent_returns=recent_returns, prev_weights=prev_weights,
                control_state=allocator_state,
            )
            if prev_weights is not None:
                weights = allocator.apply_rebalance_band(weights, prev_weights)
            cash_weight = max(0.0, 1.0 - float(weights.sum()))
            invested_fraction_selected, overlay_size_selected = allocator.decode_action(action)

        else:
            weights, cash_weight = allocator.construct_allocator_only(
                factor_scores, expected, confidence,
                recent_returns=recent_returns, prev_weights=prev_weights,
                control_state=allocator_state,
            )
            if prev_weights is not None:
                weights = allocator.apply_rebalance_band(weights, prev_weights)
            cash_weight = max(0.0, 1.0 - float(weights.sum()))
            invested_fraction_selected = float(np.clip(weights.sum(), 0.0, 1.0))

        # --- Apply Portfolio ---
        daily_ret = returns[tickers].iloc[t]
        spy_ret, benchmark_weights = _benchmark_step_return(
            returns, t, dates, benchmark_components, benchmark_weights,
        )
        prev_weight_series = None
        if results['portfolio_weights']:
            prev_weight_series = pd.Series(results['portfolio_weights'][-1], index=tickers)
        desired_weights = weights.copy()
        desired_turnover = 0.0
        desired_cash_weight = max(0.0, 1.0 - float(desired_weights.sum()))
        if prev_weight_series is not None and len(prev_weight_series) > 0:
            desired_turnover = float(
                np.abs(
                    desired_weights.reindex(tickers).fillna(0.0).values
                    - prev_weight_series.reindex(tickers).fillna(0.0).values
                ).sum()
            )

        executed_target_weights = desired_weights
        if config.cost_model.execution_delay_days > 0 and prev_weight_series is not None and len(prev_weight_series) > 0:
            execution_queue.append(desired_weights.copy())
            if len(execution_queue) > int(config.cost_model.execution_delay_days):
                executed_target_weights = execution_queue.pop(0)
            else:
                executed_target_weights = prev_weight_series.copy()

        prices_row = prices[tickers].iloc[t - 1] if t > 0 else prices[tickers].iloc[t]
        volume_window = volumes[tickers].iloc[max(0, t - config.cost_model.adv_lookback_days):t]
        executed_weights, execution_stats = _apply_execution_constraints(
            target_weights=executed_target_weights,
            prev_weights=prev_weight_series,
            prices_row=prices_row,
            volume_window=volume_window,
            wealth=results['wealth'][-1],
            config=config,
        )
        weights = executed_weights
        cash_weight = max(0.0, 1.0 - float(weights.sum()))
        desired_gross_ret = float((desired_weights * daily_ret).sum()) + desired_cash_weight * (RISK_FREE_RATE / 252.0)
        raw_portfolio_ret, turnover, transaction_cost = _net_portfolio_return(
            weights=weights,
            daily_ret=daily_ret,
            risk_free_rate=RISK_FREE_RATE,
            execution_stats=execution_stats,
            portfolio_vol=portfolio_vol,
            config=config,
        )
        portfolio_ret = raw_portfolio_ret
        execution_weight_gap = float(
            np.abs(
                desired_weights.reindex(tickers).fillna(0.0).values
                - weights.reindex(tickers).fillna(0.0).values
            ).sum()
        )
        execution_delay_gap = float(
            np.abs(
                desired_weights.reindex(tickers).fillna(0.0).values
                - executed_target_weights.reindex(tickers).fillna(0.0).values
            ).sum()
        )
        execution_shortfall = float(desired_gross_ret - raw_portfolio_ret)

        controller_diagnostics = allocator.current_allocator_diagnostics()
        if use_new_controller and controller is not None and ctrl_state is not None:
            portfolio_ret, convexity_diagnostics = controller.apply_return_overlay(portfolio_ret, ctrl_state)
            controller_diagnostics = {
                **controller_diagnostics,
                **controller.current_diagnostics(),
            }
            overlay_size_selected = float(controller_diagnostics.get('overlay_size', overlay_size_selected))
        else:
            convexity_diagnostics = {
                'convexity_mode': 0,
                'convexity_mode_name': 'none',
                'convexity_carry': 0.0,
                'convexity_benefit': 0.0,
                'convexity_net_adjustment': 0.0,
            }

        if isinstance(controller, (CouncilController, MLPMetaController)):
            expert_feedback = {}
            for expert_name, expert_weights in controller.get_pending_expert_books().items():
                eval_weights = expert_weights.reindex(tickers).fillna(0.0).clip(lower=0.0)
                if prev_weight_series is not None:
                    eval_weights = allocator.apply_rebalance_band(eval_weights, prev_weight_series)
                expert_net_ret, _, _ = _net_portfolio_return(
                    weights=eval_weights,
                    daily_ret=daily_ret,
                    risk_free_rate=RISK_FREE_RATE,
                    execution_stats=None,
                    portfolio_vol=portfolio_vol,
                    config=config,
                )
                expert_feedback[expert_name] = float(expert_net_ret)

        wealth *= (1 + portfolio_ret)
        peak = max(peak, wealth)

        # SPY benchmark
        spy_wealth *= (1 + spy_ret)
        spy_peak = max(spy_peak, spy_wealth)

        # Equal weight benchmark
        equal_wealth *= (1 + daily_ret.mean())

        # Factor-only benchmark
        shifted_f = factor_scores - factor_scores.max()
        factor_w = np.exp(shifted_f * 2.0)
        factor_w = factor_w / (factor_w.sum() + 1e-8)
        factor_ret = (factor_w * daily_ret).sum()
        factor_wealth *= (1 + factor_ret)

        # Vol-targeting baseline
        factor_rets_window = returns[tickers].iloc[max(0, t - vol_lookback):t]
        factor_port_vol = (factor_w * factor_rets_window).sum(axis=1).std() * np.sqrt(252)
        vt_invested = min(1.0, vol_target_annual / (factor_port_vol + 1e-8))
        vt_ret = vt_invested * factor_ret + (1 - vt_invested) * (RISK_FREE_RATE / 252)
        voltarget_wealth *= (1 + vt_ret)

        # Drawdown-based deleveraging baseline
        ddlever_dd = (ddlever_wealth - ddlever_peak) / ddlever_peak
        dd_invested = 1.0
        for dd_thresh, dd_frac in dd_thresholds:
            if ddlever_dd < dd_thresh:
                dd_invested = dd_frac
        if ddlever_dd > dd_thresholds[0][0]:
            dd_invested = 1.0
        dd_ret = dd_invested * factor_ret + (1 - dd_invested) * (RISK_FREE_RATE / 252)
        ddlever_wealth *= (1 + dd_ret)
        ddlever_peak = max(ddlever_peak, ddlever_wealth)

        # Risk parity baseline
        if t - _rp_last_fit >= 5 or _rp_weights_cache is None:
            rp_window = returns[tickers].iloc[max(0, t - 60):t]
            if len(rp_window) >= 20:
                try:
                    lw = LedoitWolf().fit(rp_window.values)
                    inv_var = 1.0 / (np.diag(lw.covariance_) + 1e-8)
                    _rp_weights_cache = inv_var / inv_var.sum()
                except Exception:
                    _rp_weights_cache = np.ones(len(tickers)) / len(tickers)
            else:
                _rp_weights_cache = np.ones(len(tickers)) / len(tickers)
            _rp_last_fit = t
        rp_ret = float(np.dot(_rp_weights_cache, daily_ret.values))
        riskparity_wealth *= (1 + rp_ret)

        # --- Controller / RL Updates ---
        results['wealth'].append(wealth)
        port_rets = np.diff(results['wealth'][-min(60, len(results['wealth'])):]) / \
                    np.array(results['wealth'][-min(60, len(results['wealth'])):-1])
        sharpe_reward = _compute_portfolio_reward(
            portfolio_ret=portfolio_ret,
            recent_portfolio_rets=port_rets,
            mode=config.portfolio_reward_mode,
        )

        if config.experiment.adaptive_combiner:
            combiner.update_signal_quality(source_signals, daily_ret)

        if use_new_controller and controller is not None and ctrl_state is not None:
            next_ctrl_state = build_control_state(
                alpha_scores=expected,
                portfolio_weights=weights,
                recent_returns=recent_returns,
                regime_belief=regime_belief,
                wealth_path=results['wealth'],
                t=t + 1,
            )
            if isinstance(controller, (CouncilController, MLPMetaController)):
                controller.update(
                    ctrl_state,
                    sharpe_reward,
                    next_ctrl_state,
                    expert_feedback=expert_feedback,
                )
            elif isinstance(controller, CMDPLagrangianController):
                constraint_cost = max(0.0, -float(next_ctrl_state.recent_drawdown))
                controller.update(
                    ctrl_state,
                    sharpe_reward,
                    next_ctrl_state,
                    constraint_cost=constraint_cost,
                    realized_return=portfolio_ret,
                )
            else:
                controller.update(ctrl_state, sharpe_reward, next_ctrl_state)

        elif use_legacy_rl:
            next_factor_scores, _ = ff.get_factor_scores(t)
            next_garch_vols = garch.forecast_all(t)
            next_regime_belief = hmm.get_regime_belief(t, market_ret)
            if not macro_data.empty and dates[t] in macro_data.index:
                if t + 1 - _macro_last_compute >= 5:
                    macro_window = macro_data.loc[:dates[t]].tail(252)
                    _macro_belief_cache = compute_macro_regime_signal(macro_window)
                    _macro_last_compute = t + 1
                next_regime_belief = 0.75 * next_regime_belief + 0.25 * _macro_belief_cache
            next_expected, _, _, _ = combiner.combine(
                next_factor_scores, next_garch_vols, next_regime_belief,
                tickers, use_factor=config.experiment.use_factor,
                adaptive=config.experiment.adaptive_combiner,
            )
            next_recent_returns = returns[tickers].iloc[max(0, t - 125):t + 1]
            next_weights = pd.Series(weights, copy=False)
            next_state = allocator.get_state(next_expected, next_weights, next_recent_returns)
            allocator.update(state, action, sharpe_reward, next_state)
            if t > train_end:
                allocator.epsilon = max(0.01, allocator.epsilon * 0.9998)

        if use_new_controller and controller is not None:
            controller_diagnostics = {
                **allocator.current_allocator_diagnostics(),
                **controller.current_diagnostics(),
            }
        else:
            controller_diagnostics = allocator.current_allocator_diagnostics()

        # Store results
        results['dates'].append(dates[t])
        results['spy'].append(spy_wealth)
        results['equal'].append(equal_wealth)
        results['factor'].append(factor_wealth)
        results['voltarget'].append(voltarget_wealth)
        results['ddlever'].append(ddlever_wealth)
        results['risk_parity'].append(riskparity_wealth)
        results['actions'].append(action)
        results['invested_fractions'].append(invested_fraction_selected)
        results['overlay_sizes'].append(overlay_size_selected)
        results['hedge_actions'].append(0)
        results['hedge_type_actions'].append(0)
        results['hedge_types'].append('none')
        results['regime_beliefs'].append(regime_belief)
        results['factor_scores_hist'].append(factor_scores.values.copy())
        results['portfolio_weights'].append(weights.values.copy())
        results['garch_vols'].append(garch_vols.values.copy() if len(garch_vols) > 0 else np.zeros(len(tickers)))
        results['pairs_info_hist'].append(0)
        results['drawdowns'].append((wealth - peak) / peak)
        results['spy_drawdowns'].append((spy_wealth - spy_peak) / spy_peak)
        results['turnover'].append(turnover)
        results['transaction_costs'].append(transaction_cost)
        results['desired_turnover'].append(desired_turnover)
        results['buy_turnover'].append(float(execution_stats.get('buy_turnover', 0.0)))
        results['sell_turnover'].append(float(execution_stats.get('sell_turnover', 0.0)))
        results['avg_participation_rates'].append(float(execution_stats.get('avg_participation_rate', 0.0)))
        results['max_participation_rates'].append(float(execution_stats.get('max_participation_rate', 0.0)))
        results['liquidity_scales'].append(float(execution_stats.get('liquidity_scale', 1.0)))
        results['adv_cap_hits'].append(float(execution_stats.get('adv_cap_hit', 0.0)))
        results['adv_excess_ratios'].append(float(execution_stats.get('adv_excess_ratio', 0.0)))
        results['execution_weight_gaps'].append(execution_weight_gap)
        results['execution_delay_gaps'].append(execution_delay_gap)
        results['execution_shortfalls'].append(execution_shortfall)
        results['execution_delay_days'].append(int(config.cost_model.execution_delay_days))
        results['hedge_pnl'].append(0.0)
        results['hedge_costs'].append(0.0)
        results['hedge_benefits'].append(0.0)
        results['raw_portfolio_returns'].append(raw_portfolio_ret)
        results['portfolio_returns'].append(portfolio_ret)
        results['cash_weights'].append(cash_weight)
        results['hedge_ratios'].append(0.0)
        results['effective_hedge_ratios'].append(0.0)
        results['source_weights_hist'].append([source_weights['factor'], source_weights['garch'], source_weights['hmm']])
        results['alpha_dispersion'].append(alpha_dispersion)
        results['regime_entropy'].append(regime_uncertainty)
        results['ic_instability'].append(ic_instability)
        results['garch_vol_uncertainty'].append(garch_vol_uncertainty)
        results['uncertainty_score'].append(uncertainty_score)
        results['iv_annualized'].append(0.0)
        results['iv_percentile'].append(0.0)
        results['iv_realized_spread'].append(0.0)
        results['iv_regime_score'].append(0.0)
        results['convexity_modes'].append(int(convexity_diagnostics.get('convexity_mode', 0)))
        results['convexity_mode_names'].append(str(convexity_diagnostics.get('convexity_mode_name', 'none')))
        results['convexity_carries'].append(float(convexity_diagnostics.get('convexity_carry', 0.0)))
        results['convexity_benefits'].append(float(convexity_diagnostics.get('convexity_benefit', 0.0)))
        results['convexity_net_adjustments'].append(float(convexity_diagnostics.get('convexity_net_adjustment', 0.0)))
        council_weights = controller_diagnostics.get('council_weights', {}) if isinstance(controller_diagnostics, dict) else {}
        results['council_weight_regime_rules'].append(float(council_weights.get('regime_rules', 0.0)))
        results['council_weight_linucb'].append(float(council_weights.get('linucb', 0.0)))
        results['council_weight_cvar_robust'].append(float(council_weights.get('cvar_robust', 0.0)))
        results['council_dominant_expert'].append(str(controller_diagnostics.get('council_dominant_expert', 'none')))
        results['council_best_expert'].append(str(controller_diagnostics.get('council_best_expert', 'none')))
        results['council_gate_entropy'].append(float(controller_diagnostics.get('council_gate_entropy', 0.0)))
        mlp_meta_weights = controller_diagnostics.get('mlp_meta_gate_weights', {}) if isinstance(controller_diagnostics, dict) else {}
        results['mlp_meta_weight_regime_rules'].append(float(mlp_meta_weights.get('regime_rules', 0.0)))
        results['mlp_meta_weight_linucb'].append(float(mlp_meta_weights.get('linucb', 0.0)))
        results['mlp_meta_weight_cvar_robust'].append(float(mlp_meta_weights.get('cvar_robust', 0.0)))
        results['mlp_meta_dominant_expert'].append(str(controller_diagnostics.get('mlp_meta_dominant_expert', 'none')))
        results['mlp_meta_best_expert'].append(str(controller_diagnostics.get('mlp_meta_best_expert', 'none')))
        results['mlp_meta_gate_entropy'].append(float(controller_diagnostics.get('mlp_meta_gate_entropy', 0.0)))
        results['mlp_meta_gate_source'].append(str(controller_diagnostics.get('mlp_meta_gate_source', 'none')))
        results['mlp_meta_n_training_samples'].append(int(controller_diagnostics.get('mlp_meta_n_training_samples', 0)))
        results['mpc_invested_targets'].append(float(controller_diagnostics.get('mpc_invested_target', 0.0)))
        results['mpc_stabilizer_mixes'].append(float(controller_diagnostics.get('mpc_stabilizer_mix', 0.0)))
        results['mpc_plan_steps'].append(int(controller_diagnostics.get('mpc_plan_step', 0)))
        results['mpc_plan_sources'].append(str(controller_diagnostics.get('mpc_plan_source', 'none')))
        results['mpc_plan_objectives'].append(float(controller_diagnostics.get('mpc_plan_objective', 0.0)))
        results['adaptive_allocator_stress_scores'].append(float(controller_diagnostics.get('adaptive_allocator_stress_score', 0.0)))
        results['adaptive_allocator_invested_targets'].append(float(controller_diagnostics.get('adaptive_allocator_invested_target', invested_fraction_selected)))
        results['adaptive_allocator_risk_mults'].append(float(controller_diagnostics.get('adaptive_allocator_risk_mult', 1.0)))
        results['adaptive_allocator_anchor_mults'].append(float(controller_diagnostics.get('adaptive_allocator_anchor_mult', 1.0)))
        results['adaptive_allocator_turnover_mults'].append(float(controller_diagnostics.get('adaptive_allocator_turnover_mult', 1.0)))
        results['adaptive_allocator_alpha_mults'].append(float(controller_diagnostics.get('adaptive_allocator_alpha_mult', 1.0)))
        results['adaptive_allocator_cap_scales'].append(float(controller_diagnostics.get('adaptive_allocator_cap_scale', 1.0)))
        results['adaptive_allocator_group_cap_scales'].append(float(controller_diagnostics.get('adaptive_allocator_group_cap_scale', 1.0)))
        results['adaptive_allocator_max_weights'].append(float(controller_diagnostics.get('adaptive_allocator_max_weight', config.optimizer.max_weight)))
        results['cmdp_lambdas'].append(float(controller_diagnostics.get('cmdp_lambda', 0.0)))
        results['cmdp_constraint_costs'].append(float(controller_diagnostics.get('cmdp_constraint_cost', 0.0)))
        results['cmdp_violations'].append(float(controller_diagnostics.get('cmdp_violation', 0.0)))

    results['tickers'] = tickers
    results['train_end_idx'] = train_end - max(252, train_end)
    results['portfolio_rl'] = allocator
    results['hedging_rl'] = None
    results['controller'] = controller
    results['config'] = config
    results['experiment_label'] = config.experiment.label
    results['feature_contracts'] = _build_feature_contracts(config)

    # --- End-to-End RL Baseline (PPO) ---
    if config.enable_e2e_baseline:
        try:
            from .rl_e2e import build_e2e_features, run_e2e_baseline

            print("  Running end-to-end RL (PPO) baseline...")
            backtest_start = max(252, train_end)

            def _factor_scores_fn(t):
                scores, _ = ff.get_factor_scores(max(0, t - 1))
                return scores.values

            def _garch_vols_fn(t):
                vols = garch.forecast_all(max(0, t - 1))
                return vols.values if len(vols) > 0 else np.zeros(len(tickers))

            def _regime_fn(t):
                return hmm.get_regime_belief(max(0, t - 1), market_ret)

            def _macro_fn(t):
                if not macro_data.empty and dates[max(0, t - 1)] in macro_data.index:
                    mw = macro_data.loc[:dates[max(0, t - 1)]].tail(252)
                    return compute_macro_regime_signal(mw)
                return 0.5

            def _option_features_fn(t):
                return {
                    'iv_annualized': 0.20,
                    'iv_percentile': 0.50,
                    'iv_realized_spread': 0.0,
                    'iv_regime_score': 0.50,
                }

            e2e_feature_fn = build_e2e_features(
                returns, tickers,
                _factor_scores_fn, _garch_vols_fn,
                _regime_fn, _macro_fn, _option_features_fn,
            )

            e2e_results = run_e2e_baseline(
                returns=returns,
                tickers=tickers,
                feature_fn=e2e_feature_fn,
                train_start=backtest_start,
                train_end=train_end + (n - train_end) // 2 if train_end < n else train_end,
                test_end=n,
                cost_bps=config.cost_model.base_cost_bps,
                risk_free_rate=RISK_FREE_RATE,
                reward_mode=config.e2e_reward_mode,
                total_timesteps=30_000,
                verbose=config.e2e_ppo_verbose,
                log_interval=config.e2e_ppo_log_interval,
            )

            e2e_wealth = e2e_results['wealth']
            n_test_dates = len(results['dates'])
            n_e2e = len(e2e_wealth) - 1
            pad_len = n_test_dates - n_e2e
            results['e2e_rl'] = [1.0] * (pad_len + 1) + e2e_wealth[1:]
            print(f"  E2E RL final wealth: {e2e_wealth[-1]:.3f}")

        except Exception as e:
            print(f"  End-to-end RL baseline skipped: {e}")
            results['e2e_rl'] = [1.0] * (len(results['dates']) + 1)
    else:
        print("  End-to-end RL baseline disabled for this run.")
        results['e2e_rl'] = [1.0] * (len(results['dates']) + 1)

    print(f"\n  Final wealth: {wealth:.3f} (Pipeline) vs {spy_wealth:.3f} (SPY)")
    return results
