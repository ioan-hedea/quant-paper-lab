"""Reinforcement-learning agents for portfolio construction, execution, and hedging."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class PortfolioConstructionRL:
    """
    RL agent that decides portfolio weights given alpha signals.

    State: (cross-sectional alpha opportunity, current volatility, regime)
    Action: total risk budget allocation (how aggressively to follow signals)
    Reward: Sharpe ratio of portfolio

    This is the key insight: RL doesn't generate alpha, it decides
    HOW MUCH to bet on each alpha signal given the current conditions.
    """
    def __init__(
        self,
        n_risk_levels: int = 5,
        alpha: float = 0.03,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        rebalance_band: float = 0.015,
        min_turnover: float = 0.08,
    ) -> None:
        self.n_risk_levels = n_risk_levels
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rebalance_band = rebalance_band
        self.min_turnover = min_turnover
        self.Q = defaultdict(lambda: np.zeros(n_risk_levels))
        self.reward_buffer = deque(maxlen=60)

        # RL now mostly adjusts total exposure and a small active overlay
        # around a factor-anchored target book.
        self.invest_fractions = [0.82, 0.90, 0.95, 0.98, 1.00]
        self.tilt_fractions = [0.05, 0.08, 0.12, 0.16, 0.20]

    def get_state(
        self,
        alpha_opportunity: float,
        portfolio_vol: float,
        regime_belief: float,
    ) -> tuple[int, int, int]:
        """Discretize continuous state."""
        alpha_bin = int(np.clip(np.digitize(alpha_opportunity,
                        [0.25, 0.60, 1.00, 1.50]), 0, 4))
        vol_bin = int(np.clip(np.digitize(portfolio_vol,
                      [0.008, 0.012, 0.016, 0.022]), 0, 4))
        regime_bin = int(regime_belief > 0.5)
        return (alpha_bin, vol_bin, regime_bin)

    def select_action(self, state: tuple[int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_risk_levels)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: tuple[int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int],
    ) -> None:
        best_next = np.max(self.Q[next_state])
        td = reward + self.gamma * best_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td

    def get_action_labels(self) -> list[str]:
        return [f'{frac:.0%}' for frac in self.invest_fractions]

    def apply_rebalance_band(
        self,
        target_weights: pd.Series,
        prev_weights: pd.Series | None,
    ) -> pd.Series:
        """Skip tiny trades and only rebalance when the change is meaningful."""
        if prev_weights is None or len(prev_weights) == 0:
            return target_weights

        prev_weights = prev_weights.reindex(target_weights.index).fillna(0.0)
        raw_delta = target_weights - prev_weights
        raw_turnover = raw_delta.abs().sum()
        if raw_turnover < self.min_turnover:
            return prev_weights

        adjusted = target_weights.copy()
        small_moves = raw_delta.abs() < self.rebalance_band
        adjusted[small_moves] = prev_weights[small_moves]

        target_budget = target_weights.sum()
        adjusted_budget = adjusted.sum()
        if adjusted_budget > 1e-8:
            adjusted *= target_budget / adjusted_budget
        return adjusted

    def estimate_min_var_core(
        self,
        tickers: pd.Index,
        recent_returns: pd.DataFrame | None,
        confidence: pd.Series,
    ) -> pd.Series:
        """
        Estimate a long-only minimum-variance core using Ledoit-Wolf shrinkage.
        Falls back to confidence-based weights if data is thin.
        """
        fallback = confidence.reindex(tickers).fillna(1.0).clip(lower=0.5, upper=2.0)
        fallback = fallback / (fallback.sum() + 1e-8)

        if recent_returns is None or len(recent_returns) < 40:
            return fallback

        returns_window = recent_returns.reindex(columns=tickers).dropna(axis=1, how='all').fillna(0.0)
        if returns_window.shape[1] < 2:
            return fallback

        try:
            cov = LedoitWolf().fit(returns_window.values).covariance_
            inv_cov = np.linalg.pinv(cov + 1e-8 * np.eye(cov.shape[0]))
            ones = np.ones(inv_cov.shape[0])
            raw = inv_cov @ ones
            raw = np.clip(raw, 0.0, None)
            if raw.sum() < 1e-8:
                return fallback
            min_var = raw / raw.sum()
            min_var = pd.Series(min_var, index=returns_window.columns)

            aligned = pd.Series(0.0, index=tickers)
            aligned.loc[min_var.index] = min_var.values

            # Blend shrinkage min-var with inverse-vol confidence so the core
            # stays stable even when the sample covariance is noisy.
            blended = 0.65 * aligned + 0.35 * fallback
            return blended / (blended.sum() + 1e-8)
        except Exception:
            return fallback

    def build_factor_target(
        self,
        factor_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
    ) -> pd.Series:
        """Build the target book from factor signals, stabilized by min-var weights."""
        tickers = factor_scores.index
        shifted_factor = factor_scores - factor_scores.max()
        factor_book = np.exp(shifted_factor * 2.0)
        factor_book = factor_book / (factor_book.sum() + 1e-8)

        stabilizer = self.estimate_min_var_core(tickers, recent_returns, confidence)
        anchored = 0.90 * factor_book + 0.10 * stabilizer
        return anchored / (anchored.sum() + 1e-8)

    def construct_portfolio(
        self,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        action: int,
        recent_returns: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, float]:
        """
        Build a factor-anchored portfolio.
        The factor book is the main target; RL controls total exposure and the
        size of the active overlay around that target.
        """
        invest_frac = self.invest_fractions[action]
        tilt_frac = self.tilt_fractions[action]

        weighted_alpha = (alpha_scores * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        factor_target = self.build_factor_target(
            factor_scores.reindex(weighted_alpha.index).fillna(0.0),
            confidence,
            recent_returns,
        )
        n_assets = len(weighted_alpha)

        if n_assets == 0:
            return pd.Series(dtype=float), 1.0

        centered_alpha = weighted_alpha - weighted_alpha.median()
        positive_alpha = centered_alpha.clip(lower=0.0)

        top_k = max(3, int(np.ceil(n_assets * 0.25)))
        if positive_alpha.gt(0).sum() > top_k:
            keep = positive_alpha.nlargest(top_k).index
            positive_alpha = positive_alpha.where(positive_alpha.index.isin(keep), 0.0)

        if positive_alpha.sum() < 1e-8:
            satellite_weights = factor_target.copy()
        else:
            alpha_scale = positive_alpha / (positive_alpha.max() + 1e-8)
            tilted_target = factor_target * (1.0 + alpha_scale)
            satellite_weights = tilted_target / (tilted_target.sum() + 1e-8)
        core_budget = invest_frac * (1.0 - tilt_frac)
        satellite_budget = invest_frac * tilt_frac
        core_weights = factor_target * core_budget

        weights = core_weights + satellite_weights * satellite_budget
        cash_weight = max(0.0, 1.0 - invest_frac)

        return weights, cash_weight


# --- 4B. Execution RL ---

class ExecutionRL:
    """
    RL agent for optimal execution: splits large orders to minimize
    market impact. This is where RL has the strongest track record
    in real finance (JP Morgan LOXM, Goldman execution).

    State: (remaining shares, time left, recent volume, spread)
    Action: fraction of remaining order to execute now
    Reward: negative implementation shortfall
    """
    def __init__(
        self,
        n_time_slices: int = 5,
        alpha: float = 0.05,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        self.n_slices = n_time_slices
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_time_slices))

        # Action: execute 0%, 20%, 40%, 60%, 100% of remaining
        self.exec_fractions = [0.0, 0.2, 0.4, 0.6, 1.0]

    def market_impact(self, exec_fraction: float, volume_ratio: float) -> float:
        """
        Square-root market impact model (Almgren-Chriss).
        Impact ~ sigma * sqrt(exec_rate / daily_volume)
        """
        return 0.001 * np.sqrt(exec_fraction * volume_ratio + 1e-8)

    def get_state(self, remaining_frac: float, time_frac: float, vol_regime: int) -> tuple[int, int, int]:
        """Discretize execution state."""
        rem_bin = int(np.clip(remaining_frac * 4, 0, 4))
        time_bin = int(np.clip(time_frac * 4, 0, 4))
        return (rem_bin, time_bin, vol_regime)

    def select_action(self, state: tuple[int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_slices)
        return int(np.argmax(self.Q[state]))

    def execute_order(
        self,
        total_shares: float,
        n_periods: int,
        volume_profile: Sequence[float],
    ) -> tuple[float, list[dict[str, float]]]:
        """Simulate order execution over n_periods."""
        remaining = total_shares
        total_cost = 0
        exec_log = []

        for t in range(n_periods):
            frac_remaining = remaining / (total_shares + 1e-8)
            time_frac = t / n_periods
            vol_regime = 0 if volume_profile[t] > np.mean(volume_profile) else 1

            state = self.get_state(frac_remaining, time_frac, vol_regime)
            action = self.select_action(state)
            exec_frac = self.exec_fractions[action]

            shares_to_exec = remaining * exec_frac
            vol_ratio = shares_to_exec / (volume_profile[t] + 1e-8)
            impact = self.market_impact(exec_frac, vol_ratio)

            total_cost += shares_to_exec * impact
            remaining -= shares_to_exec

            exec_log.append({
                'time': t,
                'executed': shares_to_exec,
                'remaining': remaining,
                'impact': impact,
                'cost': shares_to_exec * impact,
            })

            # Update Q
            if t < n_periods - 1:
                next_frac = remaining / (total_shares + 1e-8)
                next_time = (t + 1) / n_periods
                next_vol = 0 if t + 1 < len(volume_profile) and volume_profile[t + 1] > np.mean(volume_profile) else 1
                next_state = self.get_state(next_frac, next_time, next_vol)
                reward = -impact * 1000  # penalize impact
                best_next = np.max(self.Q[next_state])
                td = reward + self.gamma * best_next - self.Q[state][action]
                self.Q[state][action] += self.alpha * td

        return total_cost, exec_log


# --- 4C. Dynamic Hedging RL ---

class DynamicHedgingRL:
    """
    RL agent for dynamic tail-risk hedging.
    Decides how much of the portfolio to hedge via a protective overlay
    based on current risk regime.

    This approximates an options-like payoff without requiring an options
    chain in the base demo:
    - In calm markets: stay fully invested
    - In stressed markets: dynamically increase hedge
    - After crash: remove hedge to capture recovery

    State: (portfolio drawdown, vol regime, momentum regime)
    Action: hedge ratio (0%, 3%, 8%, 15%)
    """
    def __init__(self, alpha: float = 0.03, gamma: float = 0.95, epsilon: float = 0.15) -> None:
        self.n_actions = 4
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        # Keep the hedge sleeve light so it protects in stress without
        # dominating performance in normal markets.
        self.hedge_ratios = [0.0, 0.03, 0.08, 0.15]
        self.portfolio_peak = 1.0
        self.reward_buffer = deque(maxlen=60)

    def get_state(self, drawdown: float, vol_percentile: float, momentum: float) -> tuple[int, int, int]:
        dd_bin = int(np.clip(np.digitize(-drawdown, [0.05, 0.10, 0.16, 0.24]), 0, 4))
        vol_bin = int(np.clip(np.digitize(vol_percentile, [0.50, 0.70, 0.85, 0.95]), 0, 4))
        mom_bin = int(np.clip(np.digitize(momentum, [-0.08, -0.03, 0.0, 0.04]), 0, 4))
        return (dd_bin, vol_bin, mom_bin)

    def select_action(self, state: tuple[int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def compute_reward(self, portfolio_return: float, hedge_ratio: float) -> float:
        """
        Reward = realized return. The agent learns that hedging
        costs returns in bull markets but saves money in crashes.
        Simple and clean — let RL figure out the tradeoff.
        """
        self.reward_buffer.append(portfolio_return)

        # Asymmetric reward: losses hurt 2x more than gains help
        if portfolio_return >= 0:
            return portfolio_return * 100
        else:
            return portfolio_return * 200  # double penalty for losses

    def apply_hedge(
        self,
        risky_ret: float,
        cash_ret: float,
        market_ret: float,
        hedge_ratio: float,
        vol_percentile: float,
    ) -> tuple[float, float]:
        """
        Protective sleeve:
        - hedged notional offsets some market beta
        - a convex bonus helps in sharper selloffs
        - carry cost bleeds slowly in calm periods
        """
        stress_gate = float((vol_percentile > 0.75) or (market_ret < -0.012))
        effective_hedge = hedge_ratio * (0.35 + 0.65 * stress_gate)
        protected_risky_ret = risky_ret * (1 - 0.5 * effective_hedge)
        carry_cost = effective_hedge * 0.00012
        convexity_bonus = max(-market_ret - 0.015, 0.0) * effective_hedge * (1.0 + 0.35 * vol_percentile)
        hedge_pnl = (-effective_hedge * market_ret) + convexity_bonus - carry_cost
        total_ret = protected_risky_ret + cash_ret + hedge_pnl
        return total_ret, hedge_pnl

    def update(
        self,
        state: tuple[int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int],
    ) -> None:
        best_next = np.max(self.Q[next_state])
        td = reward + self.gamma * best_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td

