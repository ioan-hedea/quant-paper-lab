"""Reinforcement-learning agents for portfolio construction, execution, and hedging."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import OptimizerConfig, TICKER_TO_GROUP

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
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        self.n_risk_levels = n_risk_levels
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rebalance_band = rebalance_band
        self.min_turnover = min_turnover
        self.optimizer_config = optimizer_config or OptimizerConfig()
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
        uncertainty_score: float = 0.0,
    ) -> tuple[int, int, int, int]:
        """Discretize continuous state."""
        alpha_bin = int(np.clip(np.digitize(alpha_opportunity,
                        [0.25, 0.60, 1.00, 1.50]), 0, 4))
        vol_bin = int(np.clip(np.digitize(portfolio_vol,
                      [0.008, 0.012, 0.016, 0.022]), 0, 4))
        regime_bin = int(regime_belief > 0.5)
        uncertainty_bin = int(np.clip(np.digitize(uncertainty_score, [0.20, 0.35, 0.55, 0.75]), 0, 4))
        return (alpha_bin, vol_bin, regime_bin, uncertainty_bin)

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

    def optimize_target_book(
        self,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Constrained allocator between the alpha layer and RL.
        It converts expected-return information into a capped, long-only target
        book with anchor, risk, turnover, and group-concentration controls.
        """
        base_target = self.build_factor_target(factor_scores, confidence, recent_returns)
        config = self.optimizer_config
        tickers = factor_scores.index
        n_assets = len(tickers)
        effective_max_weight = max(config.max_weight, 1.0 / max(1, n_assets))

        if not config.use_optimizer or n_assets == 0:
            return base_target

        if recent_returns is None or len(recent_returns) < 40:
            return base_target

        aligned_returns = recent_returns.reindex(columns=tickers).fillna(0.0)
        if aligned_returns.shape[1] < 2:
            return base_target

        alpha_view = (alpha_scores * confidence).reindex(tickers).fillna(0.0)
        if alpha_view.std() > 1e-8:
            alpha_view = (alpha_view - alpha_view.mean()) / alpha_view.std()
        else:
            alpha_view = alpha_view * 0.0

        prev_book = None
        if prev_weights is not None and len(prev_weights) > 0 and float(prev_weights.sum()) > 1e-8:
            prev_book = prev_weights.reindex(tickers).fillna(0.0)
            prev_book = prev_book / (prev_book.sum() + 1e-8)

        try:
            cov = LedoitWolf().fit(aligned_returns.values).covariance_
        except Exception:
            return base_target

        bounds = [(0.0, effective_max_weight) for _ in tickers]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        for group, cap in config.group_caps.items():
            indices = [i for i, ticker in enumerate(tickers) if TICKER_TO_GROUP.get(ticker) == group]
            if indices:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=indices, group_cap=cap: group_cap - np.sum(w[idx]),
                })

        x0 = np.clip(base_target.values, 0.0, effective_max_weight)
        x0 = x0 / (x0.sum() + 1e-8)

        def objective(weights: np.ndarray) -> float:
            risk_term = config.risk_aversion * float(weights @ cov @ weights)
            alpha_term = -config.alpha_strength * float(alpha_view.values @ weights)
            anchor_term = config.anchor_strength * float(np.sum((weights - base_target.values) ** 2))
            turnover_term = 0.0
            if prev_book is not None:
                turnover_term = config.turnover_penalty * float(np.sum((weights - prev_book.values) ** 2))
            return risk_term + alpha_term + anchor_term + turnover_term

        try:
            result = minimize(objective, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                return base_target
            optimized = np.clip(result.x, 0.0, effective_max_weight)
            optimized = optimized / (optimized.sum() + 1e-8)
            return pd.Series(optimized, index=tickers)
        except Exception:
            return base_target

    @staticmethod
    def _enforce_absolute_weight_cap(weights: pd.Series, max_weight: float) -> pd.Series:
        adjusted = weights.copy()
        for _ in range(10):
            excess_mask = adjusted > max_weight + 1e-8
            if not excess_mask.any():
                break
            excess = float((adjusted[excess_mask] - max_weight).sum())
            adjusted[excess_mask] = max_weight
            room_mask = adjusted < max_weight - 1e-8
            if not room_mask.any() or excess <= 1e-10:
                break
            room = max_weight - adjusted[room_mask]
            adjusted.loc[room_mask] += excess * room / (room.sum() + 1e-8)
        return adjusted

    def construct_portfolio(
        self,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        action: int,
        recent_returns: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        """
        Build a factor-anchored portfolio.
        The factor book is the main target; RL controls total exposure and the
        size of the active overlay around that target.
        """
        invest_frac = self.invest_fractions[action]
        tilt_frac = self.tilt_fractions[action]

        weighted_alpha = (alpha_scores * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        optimized_target = self.optimize_target_book(
            factor_scores.reindex(weighted_alpha.index).fillna(0.0),
            alpha_scores.reindex(weighted_alpha.index).fillna(0.0),
            confidence,
            recent_returns,
            prev_weights=prev_weights,
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
            satellite_weights = optimized_target.copy()
        else:
            alpha_scale = positive_alpha / (positive_alpha.max() + 1e-8)
            tilted_target = optimized_target * (1.0 + alpha_scale)
            satellite_weights = tilted_target / (tilted_target.sum() + 1e-8)
        core_budget = invest_frac * (1.0 - tilt_frac)
        satellite_budget = invest_frac * tilt_frac
        core_weights = optimized_target * core_budget

        weights = core_weights + satellite_weights * satellite_budget
        effective_cap = max(self.optimizer_config.max_weight, invest_frac / max(1, n_assets))
        weights = self._enforce_absolute_weight_cap(weights, effective_cap)
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
    def __init__(
        self,
        alpha: float = 0.03,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        hedge_ratios: Sequence[float] | None = None,
    ) -> None:
        self.n_actions = 4
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        # Keep the hedge sleeve light so it protects in stress without
        # dominating performance in normal markets.
        self.hedge_ratios = list(hedge_ratios or (0.0, 0.03, 0.08, 0.15))
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

    def compute_reward(
        self,
        portfolio_return: float,
        hedge_ratio: float,
        mode: str = 'asymmetric_return',
    ) -> float:
        """
        Reward = realized return. The agent learns that hedging
        costs returns in bull markets but saves money in crashes.
        Simple and clean — let RL figure out the tradeoff.
        """
        self.reward_buffer.append(portfolio_return)
        reward_mode = str(mode or 'asymmetric_return').lower()

        if reward_mode == 'return':
            return portfolio_return * 100

        if reward_mode == 'sortino':
            rets = np.array(self.reward_buffer, dtype=float)
            if len(rets) < 6:
                return portfolio_return * 100
            downside = rets[rets < 0]
            downside_std = float(downside.std()) if len(downside) > 0 else 1e-8
            return float((portfolio_return - rets.mean()) / (downside_std + 1e-8))

        # Default: asymmetric return reward (losses penalized harder).
        if portfolio_return >= 0:
            return portfolio_return * 100
        return portfolio_return * 200

    def apply_hedge(
        self,
        risky_ret: float,
        cash_ret: float,
        market_ret: float,
        hedge_ratio: float,
        vol_percentile: float,
        drawdown: float = 0.0,
        momentum: float = 0.0,
    ) -> tuple[float, float]:
        """
        Protective sleeve:
        - hedged notional offsets some market beta
        - a convex bonus helps in sharper selloffs
        - carry cost bleeds slowly in calm periods
        """
        stress_gate = float((vol_percentile > 0.75) or (market_ret < -0.012) or (drawdown < -0.06))
        crash_overlay = np.clip(max(-drawdown - 0.06, 0.0) * 0.5 + max(-momentum - 0.02, 0.0) * 1.2, 0.0, 0.10)
        effective_hedge = min(0.35, hedge_ratio * (0.40 + 0.60 * stress_gate) + crash_overlay)
        vol_target_scale = np.clip(1.05 - 0.45 * max(vol_percentile - 0.55, 0.0), 0.65, 1.0)
        protected_risky_ret = risky_ret * (1 - 0.45 * effective_hedge) * vol_target_scale
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


# ============================================================
# End-to-End RL Baseline (PPO via Stable-Baselines3)
# ============================================================

import gymnasium as gym
from gymnasium import spaces


class EndToEndTradingEnv(gym.Env):
    """
    Gymnasium environment for end-to-end RL portfolio management.

    The agent directly outputs portfolio weights from raw features,
    without the modular alpha/RL separation of the main pipeline.

    State: factor scores, vol forecasts, regime belief, macro signal,
           recent returns, current drawdown, portfolio vol
    Action: continuous portfolio weights (softmax-normalized)
    Reward: differential Sharpe ratio (same as pipeline RL)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        returns: pd.DataFrame,
        tickers: list[str],
        feature_fn: callable,
        start_idx: int,
        end_idx: int,
        cost_bps: float = 5.0,
        risk_free_rate: float = 0.035,
        reward_mode: str = 'differential_sharpe',
    ):
        super().__init__()
        self.returns = returns
        self.tickers = tickers
        self.feature_fn = feature_fn
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.cost_bps = cost_bps
        self.risk_free_rate = risk_free_rate
        self.reward_mode = reward_mode

        self.n_assets = len(tickers)
        # Observation: per-asset features + global features
        n_features = feature_fn(start_idx).shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
        )
        # Action: raw logits for each asset (softmax applied internally)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32,
        )

        self._t = start_idx
        self._wealth = 1.0
        self._peak = 1.0
        self._prev_weights = np.ones(self.n_assets) / self.n_assets
        self._recent_returns = deque(maxlen=60)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.start_idx
        self._wealth = 1.0
        self._peak = 1.0
        self._prev_weights = np.ones(self.n_assets) / self.n_assets
        self._recent_returns.clear()
        obs = self.feature_fn(self._t)
        return obs.astype(np.float32), {}

    def step(self, action):
        # Convert action to portfolio weights via softmax
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - action.max())
        weights = exp_a / (exp_a.sum() + 1e-8)

        # Daily return
        daily_ret = self.returns[self.tickers].iloc[self._t].values
        portfolio_ret = float(np.dot(weights, daily_ret))

        # Transaction costs
        turnover = float(np.abs(weights - self._prev_weights).sum())
        tx_cost = turnover * self.cost_bps / 10000
        portfolio_ret -= tx_cost

        self._wealth *= (1 + portfolio_ret)
        self._peak = max(self._peak, self._wealth)
        self._recent_returns.append(portfolio_ret)
        self._prev_weights = weights.copy()

        # Reward-function ablation support
        reward_mode = str(self.reward_mode or 'differential_sharpe').lower()
        if reward_mode == 'return':
            reward = float(portfolio_ret * 100)
        elif reward_mode == 'sortino':
            if len(self._recent_returns) > 5:
                rets_arr = np.array(self._recent_returns)
                downside = rets_arr[rets_arr < 0]
                downside_std = float(downside.std()) if len(downside) > 0 else 1e-8
                reward = float((portfolio_ret - rets_arr.mean()) / (downside_std + 1e-8))
            else:
                reward = float(portfolio_ret * 100)
        else:
            if len(self._recent_returns) > 5:
                rets_arr = np.array(self._recent_returns)
                reward = float((portfolio_ret - rets_arr.mean()) / (rets_arr.std() + 1e-8))
            else:
                reward = float(portfolio_ret * 100)

        self._t += 1
        terminated = self._t >= self.end_idx
        truncated = False

        obs = self.feature_fn(min(self._t, self.end_idx - 1)).astype(np.float32)
        info = {'wealth': self._wealth, 'portfolio_ret': portfolio_ret}

        return obs, reward, terminated, truncated, info


def build_e2e_features(
    returns: pd.DataFrame,
    tickers: list[str],
    factor_scores_fn: callable,
    garch_vols_fn: callable,
    regime_belief_fn: callable,
    macro_belief_fn: callable,
) -> callable:
    """
    Build a feature function for the end-to-end environment.

    Returns the same features the modular pipeline uses, so the
    comparison is fair (same information set).
    """
    n_assets = len(tickers)
    n_obs = len(returns)

    factor_cache: dict[int, np.ndarray] = {}
    garch_cache: dict[int, np.ndarray] = {}

    def _factor_scores(t: int) -> np.ndarray:
        t_idx = int(np.clip(t, 0, n_obs - 1))
        if t_idx not in factor_cache:
            factor_cache[t_idx] = np.asarray(factor_scores_fn(t_idx), dtype=float)
        return factor_cache[t_idx]

    def _garch_vols(t: int) -> np.ndarray:
        t_idx = int(np.clip(t, 0, n_obs - 1))
        if t_idx not in garch_cache:
            garch_cache[t_idx] = np.asarray(garch_vols_fn(t_idx), dtype=float)
        return garch_cache[t_idx]

    # Precompute a simple cross-sectional IC-instability proxy.
    ic_proxy = np.zeros(n_obs, dtype=float)
    for t in range(1, n_obs):
        fs_prev = _factor_scores(t - 1)
        realized = returns[tickers].iloc[t].values
        if fs_prev.std() < 1e-8 or realized.std() < 1e-8:
            continue
        corr = np.corrcoef(fs_prev, realized)[0, 1]
        ic_proxy[t] = float(corr) if np.isfinite(corr) else 0.0
    ic_instability = (
        pd.Series(ic_proxy)
        .rolling(20, min_periods=5)
        .std()
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .values
    )

    def feature_fn(t: int) -> np.ndarray:
        # Per-asset features
        factor_scores = _factor_scores(t)  # (n_assets,)
        garch_vols = _garch_vols(t)  # (n_assets,)

        # Recent returns (5d, 20d)
        ret_5d = returns[tickers].iloc[max(0, t - 5):t].mean().values
        ret_20d = returns[tickers].iloc[max(0, t - 20):t].mean().values
        vol_20d = returns[tickers].iloc[max(0, t - 20):t].std().values

        # Global features
        regime = regime_belief_fn(t)
        macro = macro_belief_fn(t)
        p = float(np.clip(regime, 1e-6, 1.0 - 1e-6))
        regime_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))
        alpha_dispersion = float(np.tanh(np.std(factor_scores) / 2.0))
        market_ret_20d = returns[tickers].iloc[max(0, t - 20):t].mean(axis=1).sum()
        market_vol = returns[tickers].iloc[max(0, t - 60):t].mean(axis=1).std() * np.sqrt(252)

        features = np.concatenate([
            factor_scores,       # n_assets
            garch_vols,          # n_assets
            ret_5d,              # n_assets
            ret_20d,             # n_assets
            vol_20d,             # n_assets
            np.array([
                regime,
                macro,
                market_ret_20d,
                market_vol,
                alpha_dispersion,
                regime_entropy,
                float(ic_instability[int(np.clip(t, 0, n_obs - 1))]),
            ]),  # 7 global
        ])
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    return feature_fn


def run_e2e_baseline(
    returns: pd.DataFrame,
    tickers: list[str],
    feature_fn: callable,
    train_start: int,
    train_end: int,
    test_end: int,
    cost_bps: float = 5.0,
    risk_free_rate: float = 0.035,
    reward_mode: str = 'differential_sharpe',
    total_timesteps: int = 50_000,
) -> dict[str, object]:
    """
    Train PPO on the training period and evaluate on the test period.

    Returns wealth path and daily returns for the test period.
    """
    from stable_baselines3 import PPO

    # Training environment
    train_env = EndToEndTradingEnv(
        returns=returns,
        tickers=tickers,
        feature_fn=feature_fn,
        start_idx=train_start,
        end_idx=train_end,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
        reward_mode=reward_mode,
    )

    # Train PPO
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=min(256, train_end - train_start - 1),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

    # Evaluate on test period
    test_env = EndToEndTradingEnv(
        returns=returns,
        tickers=tickers,
        feature_fn=feature_fn,
        start_idx=train_end,
        end_idx=test_end,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
        reward_mode=reward_mode,
    )

    obs, _ = test_env.reset()
    wealth_path = [1.0]
    daily_returns = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        wealth_path.append(info['wealth'])
        daily_returns.append(info['portfolio_ret'])
        done = terminated or truncated

    return {
        'wealth': wealth_path,
        'daily_returns': daily_returns,
        'model': model,
    }
