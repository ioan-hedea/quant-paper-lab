"""Execution and transaction-cost helpers for the backtest pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _compute_transaction_cost(
    turnover: float,
    max_trade_change: float,
    portfolio_vol: float,
    config: PipelineConfig,
    avg_participation_rate: float = 0.0,
    max_participation_rate: float = 0.0,
    adv_excess_ratio: float = 0.0,
) -> float:
    stress = float(max(config.cost_model.cost_stress_multiplier, 0.0))
    base_cost = stress * turnover * config.cost_model.base_cost_bps / 10000
    vol_cost = stress * turnover * portfolio_vol * config.cost_model.turnover_vol_multiplier
    size_cost = stress * max_trade_change * config.cost_model.size_penalty_bps / 10000
    liquidity_penalty = stress * adv_excess_ratio * config.cost_model.adv_penalty_bps / 10000
    impact_cost = 0.0
    if config.cost_model.use_almgren_chriss:
        permanent = stress * config.cost_model.ac_permanent_beta * avg_participation_rate
        temporary = stress * config.cost_model.ac_temporary_eta * max_participation_rate * portfolio_vol
        impact_cost = float(permanent + temporary)
    return float(base_cost + vol_cost + size_cost + liquidity_penalty + impact_cost)


def _estimate_dollar_adv(
    prices_row: pd.Series,
    volume_window: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    if volume_window.empty:
        avg_volume = pd.Series(np.nan, index=tickers, dtype=float)
    else:
        avg_volume = volume_window.reindex(columns=tickers).astype(float).mean(axis=0)
    px = prices_row.reindex(tickers).astype(float)
    dollar_adv = (px * avg_volume).replace([np.inf, -np.inf], np.nan)
    return dollar_adv.clip(lower=1.0).fillna(np.inf)


def _apply_execution_constraints(
    target_weights: pd.Series,
    prev_weights: pd.Series | None,
    prices_row: pd.Series,
    volume_window: pd.DataFrame,
    wealth: float,
    config: PipelineConfig,
) -> tuple[pd.Series, dict[str, float]]:
    target_weights = target_weights.astype(float).reindex(target_weights.index).fillna(0.0).clip(lower=0.0)
    diagnostics = {
        'turnover': 0.0,
        'buy_turnover': 0.0,
        'sell_turnover': 0.0,
        'max_trade_change': 0.0,
        'avg_participation_rate': 0.0,
        'max_participation_rate': 0.0,
        'adv_excess_ratio': 0.0,
        'liquidity_scale': 1.0,
        'adv_cap_hit': 0.0,
    }
    if prev_weights is None or len(prev_weights) == 0:
        return target_weights, diagnostics

    aligned_prev = prev_weights.reindex(target_weights.index).fillna(0.0).astype(float)
    raw_trade = target_weights - aligned_prev
    abs_trade = raw_trade.abs()
    diagnostics['buy_turnover'] = float(raw_trade.clip(lower=0.0).sum())
    diagnostics['sell_turnover'] = float((-raw_trade.clip(upper=0.0)).sum())

    dollar_adv = _estimate_dollar_adv(prices_row, volume_window, list(target_weights.index))
    trade_notional = abs_trade * max(float(wealth), 1e-8)
    participation = (trade_notional / dollar_adv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_participation = float(participation.max()) if len(participation) > 0 else 0.0
    traded_mask = abs_trade > 1e-12
    avg_participation = float(participation[traded_mask].mean()) if traded_mask.any() else 0.0

    diagnostics['adv_excess_ratio'] = float(max(0.0, max_participation - config.cost_model.adv_participation_cap))
    diagnostics['avg_participation_rate'] = avg_participation
    diagnostics['max_participation_rate'] = max_participation

    cap = float(config.cost_model.adv_participation_cap)
    liquidity_scale = 1.0
    if cap > 0.0 and np.isfinite(max_participation) and max_participation > cap:
        liquidity_scale = float(np.clip(cap / (max_participation + 1e-12), 0.0, 1.0))
        diagnostics['adv_cap_hit'] = 1.0

    executed_weights = aligned_prev + raw_trade * liquidity_scale
    executed_weights = executed_weights.clip(lower=0.0)
    total_weight = float(executed_weights.sum())
    if total_weight > 1.0:
        executed_weights = executed_weights / total_weight

    exec_trade = (executed_weights - aligned_prev).astype(float)
    exec_abs_trade = exec_trade.abs()
    exec_trade_notional = exec_abs_trade * max(float(wealth), 1e-8)
    exec_participation = (exec_trade_notional / dollar_adv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exec_traded_mask = exec_abs_trade > 1e-12

    diagnostics['turnover'] = float(exec_abs_trade.sum())
    diagnostics['buy_turnover'] = float(exec_trade.clip(lower=0.0).sum())
    diagnostics['sell_turnover'] = float((-exec_trade.clip(upper=0.0)).sum())
    diagnostics['max_trade_change'] = float(exec_abs_trade.max()) if len(exec_abs_trade) > 0 else 0.0
    diagnostics['avg_participation_rate'] = float(exec_participation[exec_traded_mask].mean()) if exec_traded_mask.any() else 0.0
    diagnostics['max_participation_rate'] = float(exec_participation.max()) if len(exec_participation) > 0 else 0.0
    diagnostics['liquidity_scale'] = liquidity_scale
    return executed_weights, diagnostics


def _net_portfolio_return(
    weights: pd.Series,
    daily_ret: pd.Series,
    risk_free_rate: float,
    execution_stats: dict[str, float] | None,
    portfolio_vol: float,
    config: PipelineConfig,
) -> tuple[float, float, float]:
    """Compute daily net return, turnover, and transaction cost for a candidate book."""
    invested_ret = float((weights * daily_ret).sum())
    cash_weight = max(0.0, 1.0 - float(weights.sum()))
    net_ret = invested_ret + cash_weight * (risk_free_rate / 252.0)
    stats = execution_stats or {}
    turnover = float(stats.get('turnover', 0.0))
    max_trade_change = float(stats.get('max_trade_change', 0.0))
    avg_participation_rate = float(stats.get('avg_participation_rate', 0.0))
    max_participation_rate = float(stats.get('max_participation_rate', 0.0))
    adv_excess_ratio = float(stats.get('adv_excess_ratio', 0.0))
    transaction_cost = _compute_transaction_cost(
        turnover,
        max_trade_change,
        portfolio_vol,
        config,
        avg_participation_rate=avg_participation_rate,
        max_participation_rate=max_participation_rate,
        adv_excess_ratio=adv_excess_ratio,
    )
    net_ret -= transaction_cost
    return float(net_ret), float(turnover), float(transaction_cost)


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
            f"{config.cost_model.base_cost_bps:.1f} bps base, stress x{config.cost_model.cost_stress_multiplier:.2f}, "
            f"ADV cap {config.cost_model.adv_participation_cap:.1%}, delay {config.cost_model.execution_delay_days}d"
        ),
        'rebalancing': (
            f"Band {config.rebalance_band:.2%}, minimum turnover {config.min_turnover:.2%}"
        ),
        'optimizer': (
            (
                "State-adaptive constrained long-only allocator with anchor, risk, turnover, and group caps"
                if config.optimizer.adaptive_allocator
                else "Constrained long-only allocator with anchor, risk, turnover, and group caps"
            )
            if config.optimizer.use_optimizer
            else "Factor target used directly without constrained optimizer"
        ),
        'control_method': config.control.method,
        'reward_modes': (
            f"portfolio={config.portfolio_reward_mode}, e2e={config.e2e_reward_mode}"
        ),
    }
