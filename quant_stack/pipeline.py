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
    LSTMAlpha,
    PairsTrading,
    compute_alpha_opportunity,
)
from .config import BENCHMARK, LSTM_TICKERS, PipelineConfig, RISK_FREE_RATE, UNIVERSE
from .data import build_option_overlay_features, compute_macro_regime_signal
from .rl import DynamicHedgingRL, PortfolioConstructionRL


def _apply_macro_lag(macro_data: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    if macro_data.empty or lag_days <= 0:
        return macro_data
    return macro_data.shift(lag_days).ffill()


def _compute_transaction_cost(
    turnover: float,
    max_trade_change: float,
    portfolio_vol: float,
    config: PipelineConfig,
    avg_daily_volume_frac: float = 0.01,
) -> float:
    if config.cost_model.use_almgren_chriss:
        # Almgren-Chriss permanent + temporary market-impact model.
        # participation_rate approximated from turnover as fraction of ADV.
        participation_rate = turnover * avg_daily_volume_frac
        permanent = config.cost_model.ac_permanent_beta * participation_rate
        temporary = (config.cost_model.ac_temporary_eta
                     * participation_rate * portfolio_vol)
        return float(permanent + temporary)
    base_cost = turnover * config.cost_model.base_cost_bps / 10000
    vol_cost = turnover * portfolio_vol * config.cost_model.turnover_vol_multiplier
    size_cost = max_trade_change * config.cost_model.size_penalty_bps / 10000
    return float(base_cost + vol_cost + size_cost)


def _build_feature_contracts(config: PipelineConfig) -> dict[str, str]:
    macro_contract = (
        f"Macro inputs shifted by {config.feature_availability.macro_lag_days} trading days"
        if config.feature_availability.macro_lag_days > 0
        else "Macro inputs used without lag"
    )
    sec_contract = (
        config.feature_availability.sec_quality_note
        if config.feature_availability.allow_static_sec_quality
        else "SEC quality disabled for stricter timing discipline"
    )
    return {
        'macro': macro_contract,
        'sec_quality': sec_contract,
        'transaction_costs': (
            f"{config.cost_model.base_cost_bps:.1f} bps base + "
            f"{config.cost_model.turnover_vol_multiplier:.2f} * turnover * vol + "
            f"{config.cost_model.size_penalty_bps:.1f} bps max-trade penalty"
        ),
        'rebalancing': (
            f"Band {config.rebalance_band:.2%}, minimum turnover {config.min_turnover:.2%}"
        ),
        'optimizer': (
            "Constrained long-only allocator with anchor, risk, turnover, and group caps"
            if config.optimizer.use_optimizer
            else "Factor target used directly without constrained optimizer"
        ),
        'uncertainty_features': (
            "State includes alpha dispersion, regime entropy, and IC instability"
        ),
        'reward_modes': (
            f"portfolio={config.portfolio_reward_mode}, "
            f"hedge={config.hedge_reward_mode}, "
            f"e2e={config.e2e_reward_mode}"
        ),
        'hedge_overlay': (
            "Option overlay RL selects hedge type and intensity using IV-aware state"
            if config.option_overlay.use_option_overlay
            else "Stylized non-option hedge overlay"
        ),
    }


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
        lam = 2.0  # risk-aversion coefficient
        return float(portfolio_ret * 100.0 - lam * float(np.var(recent_portfolio_rets)) * 100.0)
    if len(recent_portfolio_rets) <= 5:
        return float(portfolio_ret * 100.0)
    return float((portfolio_ret - recent_portfolio_rets.mean()) / (recent_portfolio_rets.std() + 1e-8))

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
    Data → Alpha → Signal Combination → RL Portfolio → Execution → Hedging
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Running Full Pipeline Backtest")
    print("=" * 60)

    config = config or PipelineConfig(train_frac=train_frac)
    train_frac = config.train_frac

    tickers = [t for t in UNIVERSE if t in returns.columns]
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
    pairs = PairsTrading(prices)
    garch = GARCHForecaster(returns, refit_every=21)
    hmm = HMMRegimeDetector(lookback=252)
    lstm_models = {
        t: LSTMAlpha(seq_len=20, hidden_size=8, n_features=5)
        for t in LSTM_TICKERS if t in tickers
    }
    combiner = AlphaCombiner()

    # Initialize RL agents
    portfolio_rl = PortfolioConstructionRL(
        rebalance_band=config.rebalance_band,
        min_turnover=config.min_turnover,
        optimizer_config=config.optimizer,
    )
    hedging_rl = DynamicHedgingRL(
        hedge_ratios=config.hedge_ratios,
        hedge_types=config.option_overlay.hedge_types,
        option_overlay_config=config.option_overlay,
    )

    # Market return for regime detection
    market_ret = returns[tickers].mean(axis=1)
    option_features = build_option_overlay_features(returns.index, macro_data, market_ret)

    # Precompute rolling stats for the full series to avoid per-step recomputation
    _market_rolling_vol = market_ret.rolling(60).std()
    _portfolio_vol_20d = returns[tickers].rolling(20).std().mean(axis=1)

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
        'factor_scores_hist': [], 'portfolio_weights': [],
        'garch_vols': [], 'pairs_info_hist': [],
        'drawdowns': [], 'spy_drawdowns': [],
        'turnover': [], 'hedge_pnl': [], 'source_weights_hist': [],
        'transaction_costs': [],
        'portfolio_returns': [], 'cash_weights': [], 'hedge_ratios': [],
        'alpha_dispersion': [], 'regime_entropy': [], 'ic_instability': [],
        'garch_vol_uncertainty': [], 'uncertainty_score': [],
        'hedge_type_actions': [], 'hedge_types': [],
        'effective_hedge_ratios': [], 'hedge_costs': [], 'hedge_benefits': [],
        'iv_annualized': [], 'iv_percentile': [], 'iv_realized_spread': [], 'iv_regime_score': [],
    }

    peak = 1.0
    spy_peak = 1.0
    voltarget_peak = 1.0
    ddlever_peak = 1.0

    # Vol-targeting baseline parameters
    vol_target_annual = 0.15  # target 15% annual vol (matches pipeline avg)
    vol_lookback = 60  # 60-day realized vol estimate

    # Drawdown-based deleveraging baseline parameters
    dd_thresholds = [(-0.05, 1.0), (-0.08, 0.70), (-0.12, 0.40)]
    dd_reentry_rate = 0.10  # increase invested fraction by 10% per week (5 days)

    print(f"  Training period: {dates[0].date()} to {dates[train_end].date()}")
    print(f"  Test period: {dates[train_end].date()} to {dates[-1].date()}")

    # Train LSTMs on initial window
    print("  Training LSTM models...")
    for ticker, lstm in lstm_models.items():
        if ticker in returns.columns:
            lstm.train_on_window(returns, train_end, ticker, window=504)

    # Caches for expensive operations that don't need daily recomputation
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
        pairs_signals, pairs_info = pairs.get_pairs_signals(t - 1)
        garch_vols = garch.forecast_all(t - 1)
        garch.record_avg_vol(float(garch_vols.mean()))
        regime_belief = hmm.get_regime_belief(t - 1, market_ret)
        if not macro_data.empty and dates[t - 1] in macro_data.index:
            if t - _macro_last_compute >= 5:
                macro_window = macro_data.loc[:dates[t - 1]].tail(252)
                _macro_belief_cache = compute_macro_regime_signal(macro_window)
                _macro_last_compute = t
            regime_belief = 0.75 * regime_belief + 0.25 * _macro_belief_cache

        lstm_preds = {}
        for ticker, lstm in lstm_models.items():
            if ticker in returns.columns:
                lstm_preds[ticker] = lstm.predict_return(returns, t - 1, ticker)

        # --- Signal Combination ---
        expected, confidence, source_signals, source_weights = combiner.combine(
            factor_scores, pairs_signals, garch_vols,
            regime_belief, lstm_preds, tickers,
            use_factor=config.experiment.use_factor,
            use_pairs=config.experiment.use_pairs,
            use_lstm=config.experiment.use_lstm,
            adaptive=config.experiment.adaptive_combiner,
        )

        # --- RL Portfolio Construction ---
        weighted_alpha = (expected * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        alpha_dispersion = float(weighted_alpha.std())
        regime_uncertainty = _regime_entropy(regime_belief)
        ic_instability = float(np.clip(combiner.get_ic_instability(), 0.0, 1.0))
        garch_vol_uncertainty = garch.forecast_vol_uncertainty(t - 1)
        alpha_dispersion_norm = float(np.clip(np.tanh(alpha_dispersion / 2.0), 0.0, 1.0))
        ic_instability_norm = float(np.clip(ic_instability / 0.20, 0.0, 1.0))
        # Composite uncertainty: alpha dispersion (35%), regime entropy (30%),
        # IC instability (20%), GARCH vol forecast CV (15%)
        uncertainty_score = float(np.clip(
            0.35 * alpha_dispersion_norm
            + 0.30 * regime_uncertainty
            + 0.20 * ic_instability_norm
            + 0.15 * garch_vol_uncertainty,
            0.0,
            1.0,
        ))

        alpha_opportunity = compute_alpha_opportunity(expected, confidence)
        portfolio_vol = _portfolio_vol_20d.iloc[t - 1] if t > 20 else returns[tickers].iloc[:t].std().mean()
        prev_weights = None
        if results['portfolio_weights']:
            prev_weights = pd.Series(results['portfolio_weights'][-1], index=tickers)
        _vol_for_state = portfolio_vol if config.experiment.use_vol_state else 0.012
        _regime_for_state = regime_belief if config.experiment.use_regime_state else 0.5
        _uncert_for_state = uncertainty_score if config.experiment.use_uncertainty_state else 0.0
        state = portfolio_rl.get_state(alpha_opportunity, _vol_for_state, _regime_for_state, _uncert_for_state)
        action = portfolio_rl.select_action(state) if config.experiment.use_portfolio_rl else (portfolio_rl.n_risk_levels - 1)
        recent_returns = returns[tickers].iloc[max(0, t - 126):t]
        weights, cash_weight = portfolio_rl.construct_portfolio(
            factor_scores,
            expected,
            confidence,
            action,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
        )
        if results['portfolio_weights']:
            weights = portfolio_rl.apply_rebalance_band(weights, prev_weights)
        cash_weight = max(0.0, 1.0 - float(weights.sum()))

        # --- RL Dynamic Hedging ---
        drawdown = (wealth - peak) / peak
        current_vol = _market_rolling_vol.iloc[t] if t < len(_market_rolling_vol) else market_ret.iloc[max(0, t - 60):t].std()
        if t > 312:
            hist_vols = _market_rolling_vol.iloc[max(0, t - 252):t].dropna().values
            vol_percentile = float(np.searchsorted(np.sort(hist_vols), current_vol) / (len(hist_vols) + 1e-8))
        else:
            vol_percentile = 0.5
        momentum = market_ret.iloc[max(0, t - 20):t].sum()
        option_row = option_features.iloc[t] if not option_features.empty else pd.Series(dtype=float)
        iv_annualized = float(option_row.get('iv_annualized', 0.20))
        iv_percentile = float(option_row.get('iv_percentile', 0.50))
        iv_realized_spread = float(option_row.get('iv_realized_spread', 0.0))
        iv_regime_score = float(option_row.get('iv_regime_score', 0.50))

        hedge_state = hedging_rl.get_state(
            drawdown,
            vol_percentile,
            momentum,
            iv_percentile=iv_percentile,
            iv_realized_spread=iv_realized_spread,
        )
        hedge_joint_action = hedging_rl.select_action(hedge_state) if config.experiment.use_hedge_rl else 0
        hedge_type_idx, hedge_ratio_idx = hedging_rl.decode_action(hedge_joint_action)
        hedge_ratio = hedging_rl.hedge_ratios[hedge_ratio_idx] if config.experiment.use_hedge_rl else 0.0
        hedge_type = (
            hedging_rl.hedge_types[hedge_type_idx]
            if config.experiment.use_hedge_rl and hedge_ratio > 1e-8
            else 'none'
        )

        # --- Apply Portfolio ---
        daily_ret = returns[tickers].iloc[t]
        spy_ret = returns[BENCHMARK].iloc[t]
        invested_ret = (weights * daily_ret).sum()
        cash_ret = cash_weight * (RISK_FREE_RATE / 252)
        portfolio_ret, hedge_overlay = hedging_rl.apply_hedge(
            invested_ret,
            cash_ret,
            spy_ret,
            hedge_type,
            hedge_ratio,
            vol_percentile,
            iv_annualized,
            iv_percentile,
            iv_realized_spread,
            drawdown=drawdown,
            momentum=momentum,
        )
        hedge_pnl = float(hedge_overlay['hedge_pnl'])
        hedge_cost = float(hedge_overlay['hedge_cost'])
        hedge_benefit = float(hedge_overlay['hedge_benefit'])
        effective_hedge_ratio = float(hedge_overlay['effective_hedge_ratio'])

        # Transaction cost (on weight changes)
        turnover = 0.0
        transaction_cost = 0.0
        if results['portfolio_weights']:
            prev_w = results['portfolio_weights'][-1]
            turnover = np.abs(weights.values - prev_w).sum()
            max_trade_change = np.abs(weights.values - prev_w).max()
            transaction_cost = _compute_transaction_cost(turnover, max_trade_change, portfolio_vol, config)
            portfolio_ret -= transaction_cost

        wealth *= (1 + portfolio_ret)
        peak = max(peak, wealth)

        # SPY benchmark
        spy_wealth *= (1 + spy_ret)
        spy_peak = max(spy_peak, spy_wealth)

        # Equal weight benchmark
        eq_ret = daily_ret.mean()
        equal_wealth *= (1 + eq_ret)

        # Factor-only benchmark: softmax weights from alpha, fully invested
        shifted_f = factor_scores - factor_scores.max()
        factor_w = np.exp(shifted_f * 2.0)
        factor_w = factor_w / (factor_w.sum() + 1e-8)
        factor_ret = (factor_w * daily_ret).sum()
        factor_wealth *= (1 + factor_ret)

        # Vol-targeting baseline: scale factor portfolio by sigma_target / sigma_forecast
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
        # Gradual re-entry: if recovering, increase toward 1.0
        if ddlever_dd > dd_thresholds[0][0]:
            dd_invested = 1.0
        dd_ret = dd_invested * factor_ret + (1 - dd_invested) * (RISK_FREE_RATE / 252)
        ddlever_wealth *= (1 + dd_ret)
        ddlever_peak = max(ddlever_peak, ddlever_wealth)

        # Risk parity baseline: inverse-variance weights via Ledoit-Wolf covariance
        # Refit only every 5 days — covariance moves slowly
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

        # --- RL Updates ---
        # Portfolio RL reward: differential Sharpe
        results['wealth'].append(wealth)
        port_rets = np.diff(results['wealth'][-min(60, len(results['wealth'])):]) / \
                    np.array(results['wealth'][-min(60, len(results['wealth'])):-1])
        sharpe_reward = _compute_portfolio_reward(
            portfolio_ret=portfolio_ret,
            recent_portfolio_rets=port_rets,
            mode=config.portfolio_reward_mode,
        )

        next_alpha = alpha_opportunity
        next_vol = _portfolio_vol_20d.iloc[t] if t < len(_portfolio_vol_20d) else portfolio_vol
        _next_vol_for_state = next_vol if config.experiment.use_vol_state else 0.012
        next_state = portfolio_rl.get_state(next_alpha, _next_vol_for_state, _regime_for_state, _uncert_for_state)
        if config.experiment.use_portfolio_rl:
            portfolio_rl.update(state, action, sharpe_reward, next_state)
        if config.experiment.adaptive_combiner:
            combiner.update_signal_quality(source_signals, daily_ret)

        # Hedging RL reward — pass the realized return so it learns the tradeoff
        hedge_reward = hedging_rl.compute_reward(
            portfolio_ret,
            hedge_ratio,
            mode=config.hedge_reward_mode,
        )
        next_dd = (wealth - peak) / peak
        next_hedge_state = hedging_rl.get_state(
            next_dd,
            vol_percentile,
            momentum,
            iv_percentile=iv_percentile,
            iv_realized_spread=iv_realized_spread,
        )
        if config.experiment.use_hedge_rl:
            hedging_rl.update(hedge_state, hedge_joint_action, hedge_reward, next_hedge_state)

        # Decay exploration
        if t > train_end:
            portfolio_rl.epsilon = max(0.01, portfolio_rl.epsilon * 0.9998)
            hedging_rl.epsilon = max(0.01, hedging_rl.epsilon * 0.9998)

        # Store
        results['dates'].append(dates[t])
        results['spy'].append(spy_wealth)
        results['equal'].append(equal_wealth)
        results['factor'].append(factor_wealth)
        results['voltarget'].append(voltarget_wealth)
        results['ddlever'].append(ddlever_wealth)
        results['risk_parity'].append(riskparity_wealth)
        results['actions'].append(action)
        results['hedge_actions'].append(hedge_ratio_idx)
        results['hedge_type_actions'].append(hedge_type_idx)
        results['hedge_types'].append(hedge_type)
        results['regime_beliefs'].append(regime_belief)
        results['factor_scores_hist'].append(factor_scores.values.copy())
        results['portfolio_weights'].append(weights.values.copy())
        results['garch_vols'].append(garch_vols.values.copy() if len(garch_vols) > 0 else np.zeros(len(tickers)))
        results['pairs_info_hist'].append(len(pairs_info))
        results['drawdowns'].append((wealth - peak) / peak)
        results['spy_drawdowns'].append((spy_wealth - spy_peak) / spy_peak)
        results['turnover'].append(turnover)
        results['transaction_costs'].append(transaction_cost)
        results['hedge_pnl'].append(hedge_pnl)
        results['hedge_costs'].append(hedge_cost)
        results['hedge_benefits'].append(hedge_benefit)
        results['portfolio_returns'].append(portfolio_ret)
        results['cash_weights'].append(cash_weight)
        results['hedge_ratios'].append(hedge_ratio)
        results['effective_hedge_ratios'].append(effective_hedge_ratio)
        results['source_weights_hist'].append([source_weights['factor'], source_weights['pairs'], source_weights['lstm']])
        results['alpha_dispersion'].append(alpha_dispersion)
        results['regime_entropy'].append(regime_uncertainty)
        results['ic_instability'].append(ic_instability)
        results['garch_vol_uncertainty'].append(garch_vol_uncertainty)
        results['uncertainty_score'].append(uncertainty_score)
        results['iv_annualized'].append(iv_annualized)
        results['iv_percentile'].append(iv_percentile)
        results['iv_realized_spread'].append(iv_realized_spread)
        results['iv_regime_score'].append(iv_regime_score)

    results['tickers'] = tickers
    results['train_end_idx'] = train_end - max(252, train_end)
    results['portfolio_rl'] = portfolio_rl
    results['hedging_rl'] = hedging_rl
    results['config'] = config
    results['experiment_label'] = config.experiment.label
    results['feature_contracts'] = _build_feature_contracts(config)

    # --- End-to-End RL Baseline (PPO) ---
    if config.enable_e2e_baseline:
        try:
            from .rl import build_e2e_features, run_e2e_baseline

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
                if option_features.empty:
                    return {
                        'iv_annualized': 0.20,
                        'iv_percentile': 0.50,
                        'iv_realized_spread': 0.0,
                        'iv_regime_score': 0.50,
                    }
                row = option_features.iloc[max(0, min(len(option_features) - 1, t - 1))]
                return {
                    'iv_annualized': float(row.get('iv_annualized', 0.20)),
                    'iv_percentile': float(row.get('iv_percentile', 0.50)),
                    'iv_realized_spread': float(row.get('iv_realized_spread', 0.0)),
                    'iv_regime_score': float(row.get('iv_regime_score', 0.50)),
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

            # Align e2e wealth path to test period dates
            e2e_wealth = e2e_results['wealth']
            # Pad with 1.0 for dates before e2e test starts
            n_test_dates = len(results['dates'])
            n_e2e = len(e2e_wealth) - 1  # exclude initial 1.0
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
