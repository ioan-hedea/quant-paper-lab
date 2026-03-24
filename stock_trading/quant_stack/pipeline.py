"""Backtest orchestration for the quant trading pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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
from .data import compute_macro_regime_signal
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
) -> float:
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
    }

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
    hedging_rl = DynamicHedgingRL(hedge_ratios=config.hedge_ratios)

    # Market return for regime detection
    market_ret = returns[tickers].mean(axis=1)

    # Storage
    wealth = 1.0
    spy_wealth = 1.0
    equal_wealth = 1.0
    factor_wealth = 1.0

    results = {
        'dates': [], 'wealth': [1.0], 'spy': [1.0], 'equal': [1.0], 'factor': [1.0],
        'actions': [], 'hedge_actions': [], 'regime_beliefs': [],
        'factor_scores_hist': [], 'portfolio_weights': [],
        'garch_vols': [], 'pairs_info_hist': [],
        'drawdowns': [], 'spy_drawdowns': [],
        'turnover': [], 'hedge_pnl': [], 'source_weights_hist': [],
        'transaction_costs': [],
        'portfolio_returns': [], 'cash_weights': [], 'hedge_ratios': [],
    }

    peak = 1.0
    spy_peak = 1.0

    print(f"  Training period: {dates[0].date()} to {dates[train_end].date()}")
    print(f"  Test period: {dates[train_end].date()} to {dates[-1].date()}")

    # Train LSTMs on initial window
    print("  Training LSTM models...")
    for ticker, lstm in lstm_models.items():
        if ticker in returns.columns:
            lstm.train_on_window(returns, train_end, ticker, window=504)

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
        regime_belief = hmm.get_regime_belief(t - 1, market_ret)
        if not macro_data.empty and dates[t - 1] in macro_data.index:
            macro_window = macro_data.loc[:dates[t - 1]].tail(252)
            macro_belief = compute_macro_regime_signal(macro_window)
            regime_belief = 0.75 * regime_belief + 0.25 * macro_belief

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
        alpha_opportunity = compute_alpha_opportunity(expected, confidence)
        portfolio_vol = returns[tickers].iloc[max(0, t - 20):t].std().mean()
        prev_weights = None
        if results['portfolio_weights']:
            prev_weights = pd.Series(results['portfolio_weights'][-1], index=tickers)
        state = portfolio_rl.get_state(alpha_opportunity, portfolio_vol, regime_belief)
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
        vol_window = market_ret.iloc[max(0, t - 60):t]
        current_vol = vol_window.std()
        vol_percentile = sp_stats.percentileofscore(
            market_ret.iloc[max(0, t - 252):t].rolling(60).std().dropna().values,
            current_vol) / 100 if t > 312 else 0.5
        momentum = market_ret.iloc[max(0, t - 20):t].sum()

        hedge_state = hedging_rl.get_state(drawdown, vol_percentile, momentum)
        hedge_action = hedging_rl.select_action(hedge_state) if config.experiment.use_hedge_rl else 0
        hedge_ratio = hedging_rl.hedge_ratios[hedge_action] if config.experiment.use_hedge_rl else 0.0

        # --- Apply Portfolio ---
        daily_ret = returns[tickers].iloc[t]
        spy_ret = returns[BENCHMARK].iloc[t]
        invested_ret = (weights * daily_ret).sum()
        cash_ret = cash_weight * (RISK_FREE_RATE / 252)
        portfolio_ret, hedge_pnl = hedging_rl.apply_hedge(
            invested_ret,
            cash_ret,
            spy_ret,
            hedge_ratio,
            vol_percentile,
            drawdown=drawdown,
            momentum=momentum,
        )

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

        # --- RL Updates ---
        # Portfolio RL reward: differential Sharpe
        results['wealth'].append(wealth)
        port_rets = np.diff(results['wealth'][-min(60, len(results['wealth'])):]) / \
                    np.array(results['wealth'][-min(60, len(results['wealth'])):-1])
        if len(port_rets) > 5:
            sharpe_reward = (portfolio_ret - np.mean(port_rets)) / (np.std(port_rets) + 1e-8)
        else:
            sharpe_reward = portfolio_ret * 100

        next_alpha = alpha_opportunity
        next_vol = returns[tickers].iloc[max(0, t - 19):t + 1].std().mean()
        next_state = portfolio_rl.get_state(next_alpha, next_vol, regime_belief)
        if config.experiment.use_portfolio_rl:
            portfolio_rl.update(state, action, sharpe_reward, next_state)
        if config.experiment.adaptive_combiner:
            combiner.update_signal_quality(source_signals, daily_ret)

        # Hedging RL reward — pass the realized return so it learns the tradeoff
        hedge_reward = hedging_rl.compute_reward(portfolio_ret, hedge_ratio)
        next_dd = (wealth - peak) / peak
        next_hedge_state = hedging_rl.get_state(next_dd, vol_percentile, momentum)
        if config.experiment.use_hedge_rl:
            hedging_rl.update(hedge_state, hedge_action, hedge_reward, next_hedge_state)

        # Decay exploration
        if t > train_end:
            portfolio_rl.epsilon = max(0.01, portfolio_rl.epsilon * 0.9998)
            hedging_rl.epsilon = max(0.01, hedging_rl.epsilon * 0.9998)

        # Store
        results['dates'].append(dates[t])
        results['spy'].append(spy_wealth)
        results['equal'].append(equal_wealth)
        results['factor'].append(factor_wealth)
        results['actions'].append(action)
        results['hedge_actions'].append(hedge_action)
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
        results['portfolio_returns'].append(portfolio_ret)
        results['cash_weights'].append(cash_weight)
        results['hedge_ratios'].append(hedge_ratio)
        results['source_weights_hist'].append([source_weights['factor'], source_weights['pairs'], source_weights['lstm']])

    results['tickers'] = tickers
    results['train_end_idx'] = train_end - max(252, train_end)
    results['portfolio_rl'] = portfolio_rl
    results['hedging_rl'] = hedging_rl
    results['config'] = config
    results['experiment_label'] = config.experiment.label
    results['feature_contracts'] = _build_feature_contracts(config)

    print(f"\n  Final wealth: {wealth:.3f} (Pipeline) vs {spy_wealth:.3f} (SPY)")
    return results
