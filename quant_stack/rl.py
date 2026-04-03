"""Reinforcement-learning agents for portfolio construction, execution, and hedging."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import OptimizerConfig, OptionOverlayConfig, TICKER_TO_GROUP

class PortfolioConstructionRL:
    """
    RL agent that decides how aggressively to express an allocator book.

    State summaries are derived from:
    - alpha scores,
    - current portfolio weights,
    - recent return history.

    Actions control the invested fraction and active-overlay size around the
    allocator's target book. The agent is therefore a controller, not an alpha
    generator.
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
        self._cov_cache: dict[int, np.ndarray] = {}  # id(returns_df) -> covariance
        self._latest_allocator_diagnostics: dict[str, float | int | str] = {}
        self._smoothed_allocator_theta: dict[str, float] | None = None

        # RL now mostly adjusts total exposure and a small active overlay
        # around a factor-anchored target book.
        self.invest_fractions = [0.82, 0.90, 0.95, 0.98, 1.00]
        self.overlay_sizes = [0.05, 0.08, 0.12, 0.16, 0.20]

    @staticmethod
    def _interp(bounds: tuple[float, float], signal: float) -> float:
        low, high = map(float, bounds)
        clipped = float(np.clip(signal, 0.0, 1.0))
        return low + (high - low) * clipped

    def _adaptive_stress_score(self, control_state) -> float:
        if control_state is None:
            return 0.0
        dd_component = float(np.clip((-float(getattr(control_state, 'recent_drawdown', 0.0)) - 0.02) / 0.16, 0.0, 1.0))
        vol_component = float(np.clip((float(getattr(control_state, 'recent_vol', 0.15)) - 0.14) / 0.20, 0.0, 1.0))
        regime_component = float(np.clip((0.52 - float(getattr(control_state, 'regime_belief', 0.5))) / 0.42, 0.0, 1.0))
        concentration_component = float(np.clip((float(getattr(control_state, 'concentration', 0.0)) - 0.16) / 0.20, 0.0, 1.0))
        return float(np.clip(
            0.40 * dd_component
            + 0.28 * vol_component
            + 0.22 * regime_component
            + 0.10 * concentration_component,
            0.0,
            1.0,
        ))

    def _adapt_optimizer_config(
        self,
        optimizer_config: OptimizerConfig,
        control_state=None,
    ) -> OptimizerConfig:
        if not bool(getattr(optimizer_config, 'adaptive_allocator', False)):
            self._latest_allocator_diagnostics = {
                'adaptive_allocator_enabled': 0,
                'adaptive_allocator_stress_score': 0.0,
                'adaptive_allocator_risk_mult': 1.0,
                'adaptive_allocator_anchor_mult': 1.0,
                'adaptive_allocator_turnover_mult': 1.0,
                'adaptive_allocator_alpha_mult': 1.0,
                'adaptive_allocator_cap_scale': 1.0,
                'adaptive_allocator_group_cap_scale': 1.0,
                'adaptive_allocator_max_weight': float(optimizer_config.max_weight),
            }
            return optimizer_config

        stress = self._adaptive_stress_score(control_state)
        alpha_strength = float(getattr(control_state, 'alpha_strength', 0.0))
        trend = float(getattr(control_state, 'trend', 0.0))
        concentration = float(getattr(control_state, 'concentration', 0.0))

        alpha_boost = float(np.tanh(6.0 * alpha_strength))
        trend_signal = float(np.tanh(2.0 * trend))
        concentration_penalty = float(np.clip((concentration - 0.16) / 0.18, 0.0, 1.0))

        risk_signal = float(np.clip(0.35 + 0.60 * stress + 0.15 * concentration_penalty - 0.18 * alpha_boost, 0.0, 1.0))
        anchor_signal = float(np.clip(0.25 + 0.55 * stress + 0.20 * concentration_penalty - 0.10 * trend_signal, 0.0, 1.0))
        turnover_signal = float(np.clip(0.20 + 0.65 * stress + 0.15 * concentration_penalty, 0.0, 1.0))
        alpha_signal = float(np.clip(0.35 + 0.28 * alpha_boost + 0.12 * trend_signal - 0.20 * stress, 0.0, 1.0))
        cap_signal = float(np.clip(0.62 - 0.35 * stress + 0.28 * alpha_boost - 0.18 * concentration_penalty, 0.0, 1.0))
        group_signal = float(np.clip(0.70 - 0.32 * stress + 0.10 * alpha_boost, 0.0, 1.0))

        raw_theta = {
            'risk_mult': self._interp(optimizer_config.adaptive_allocator_risk_mult_range, risk_signal),
            'anchor_mult': self._interp(optimizer_config.adaptive_allocator_anchor_mult_range, anchor_signal),
            'turnover_mult': self._interp(optimizer_config.adaptive_allocator_turnover_mult_range, turnover_signal),
            'alpha_mult': self._interp(optimizer_config.adaptive_allocator_alpha_mult_range, alpha_signal),
            'cap_scale': self._interp(optimizer_config.adaptive_allocator_cap_scale_range, cap_signal),
            'group_cap_scale': self._interp(optimizer_config.adaptive_allocator_group_cap_scale_range, group_signal),
        }

        smoothing = float(np.clip(optimizer_config.adaptive_allocator_param_smoothing, 0.0, 0.95))
        if self._smoothed_allocator_theta is None:
            theta = raw_theta
        else:
            theta = {
                key: float(smoothing * self._smoothed_allocator_theta[key] + (1.0 - smoothing) * raw_theta[key])
                for key in raw_theta
            }
        self._smoothed_allocator_theta = dict(theta)

        max_weight = float(np.clip(
            optimizer_config.max_weight * theta['cap_scale'],
            0.05,
            0.30,
        ))
        group_caps = {
            group: float(np.clip(cap * theta['group_cap_scale'], 0.10, 1.0))
            for group, cap in optimizer_config.group_caps.items()
        }
        adapted = replace(
            optimizer_config,
            risk_aversion=float(optimizer_config.risk_aversion * theta['risk_mult']),
            alpha_strength=float(optimizer_config.alpha_strength * theta['alpha_mult']),
            anchor_strength=float(optimizer_config.anchor_strength * theta['anchor_mult']),
            turnover_penalty=float(optimizer_config.turnover_penalty * theta['turnover_mult']),
            max_weight=max_weight,
            group_caps=group_caps,
        )
        self._latest_allocator_diagnostics = {
            'adaptive_allocator_enabled': 1,
            'adaptive_allocator_policy_version': int(getattr(optimizer_config, 'adaptive_allocator_policy_version', 1)),
            'adaptive_allocator_stress_score': float(stress),
            'adaptive_allocator_risk_mult': float(theta['risk_mult']),
            'adaptive_allocator_anchor_mult': float(theta['anchor_mult']),
            'adaptive_allocator_turnover_mult': float(theta['turnover_mult']),
            'adaptive_allocator_alpha_mult': float(theta['alpha_mult']),
            'adaptive_allocator_cap_scale': float(theta['cap_scale']),
            'adaptive_allocator_group_cap_scale': float(theta['group_cap_scale']),
            'adaptive_allocator_max_weight': float(max_weight),
        }
        return adapted

    def current_allocator_diagnostics(self) -> dict[str, float | int | str]:
        return dict(self._latest_allocator_diagnostics)

    def adapt_optimizer_config(
        self,
        optimizer_config: OptimizerConfig | None = None,
        control_state=None,
    ) -> OptimizerConfig:
        """Expose the allocator's state-conditional parameter adaptation."""
        base_config = optimizer_config or self.optimizer_config
        return self._adapt_optimizer_config(base_config, control_state=control_state)

    def get_state(
        self,
        alpha_scores: pd.Series,
        portfolio_weights: pd.Series | None,
        recent_returns: pd.DataFrame | pd.Series | None,
    ) -> tuple[int, int, int, int]:
        """Discretize the allocator state from alpha, book shape, and realized path."""
        alpha_scores = pd.Series(alpha_scores, copy=False).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(alpha_scores) == 0:
            alpha_spread = 0.0
        else:
            k = max(1, min(3, len(alpha_scores)))
            alpha_spread = float(alpha_scores.nlargest(k).mean() - alpha_scores.nsmallest(k).mean())

        if portfolio_weights is None or len(portfolio_weights) == 0:
            invested_fraction = 0.0
            concentration = 0.0
        else:
            weights = pd.Series(portfolio_weights, copy=False).fillna(0.0).clip(lower=0.0)
            invested_fraction = float(weights.sum())
            normalized = weights / (invested_fraction + 1e-8)
            concentration = float((normalized ** 2).sum())

        if recent_returns is None or len(recent_returns) == 0:
            trend = 0.0
            realized_vol = 0.0
        else:
            if isinstance(recent_returns, pd.DataFrame):
                path_proxy = recent_returns.mean(axis=1)
            else:
                path_proxy = pd.Series(recent_returns, copy=False)
            path_proxy = path_proxy.dropna()
            if len(path_proxy) == 0:
                trend = 0.0
                realized_vol = 0.0
            else:
                trend = float(path_proxy.tail(20).mean() * 252)
                realized_vol = float(path_proxy.tail(20).std() * np.sqrt(252))

        alpha_bin = int(np.clip(np.digitize(alpha_spread, [0.40, 0.80, 1.20, 1.80]), 0, 4))
        invest_bin = int(np.clip(np.digitize(invested_fraction, [0.82, 0.90, 0.96, 0.995]), 0, 4))
        trend_bin = int(np.clip(np.digitize(trend, [-0.10, 0.0, 0.08, 0.18]), 0, 4))
        concentration_bin = int(np.clip(np.digitize(concentration, [0.10, 0.16, 0.24, 0.35]), 0, 4))
        return (alpha_bin, invest_bin, trend_bin, concentration_bin)

    def select_action(self, state: tuple[int, int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_risk_levels)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
    ) -> None:
        best_next = np.max(self.Q[next_state])
        td = reward + self.gamma * best_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td

    def get_action_labels(self) -> list[str]:
        return [f'{frac:.0%}' for frac in self.invest_fractions]

    def get_overlay_labels(self) -> list[str]:
        return [f'{frac:.0%}' for frac in self.overlay_sizes]

    def decode_action(self, action: int) -> tuple[float, float]:
        idx = int(np.clip(action, 0, len(self.invest_fractions) - 1))
        return self.invest_fractions[idx], self.overlay_sizes[idx]

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

    def _get_cached_cov(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Compute LedoitWolf covariance once per returns window, then cache."""
        key = id(returns_window)
        if key not in self._cov_cache:
            self._cov_cache.clear()  # only keep one entry
            self._cov_cache[key] = LedoitWolf().fit(returns_window.values).covariance_
        return self._cov_cache[key]

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
            cov = self._get_cached_cov(returns_window)
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
        optimizer_config: OptimizerConfig | None = None,
        control_state=None,
        adapt_config: bool = True,
    ) -> pd.Series:
        """
        Constrained allocator between the alpha layer and RL.
        It converts expected-return information into a capped, long-only target
        book with anchor, risk, turnover, and group-concentration controls.
        """
        base_target = self.build_factor_target(factor_scores, confidence, recent_returns)
        base_config = optimizer_config or self.optimizer_config
        config = (
            self._adapt_optimizer_config(base_config, control_state=control_state)
            if adapt_config else base_config
        )
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
            cov = self._get_cached_cov(aligned_returns)
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
        control_state=None,
    ) -> tuple[pd.Series, float]:
        """
        Build a factor-anchored portfolio.
        The factor book is the main target; RL controls total exposure and the
        size of the active overlay around that target.
        """
        invest_frac, overlay_size = self.decode_action(action)

        weighted_alpha = (alpha_scores * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        optimized_target = self.optimize_target_book(
            factor_scores.reindex(weighted_alpha.index).fillna(0.0),
            alpha_scores.reindex(weighted_alpha.index).fillna(0.0),
            confidence,
            recent_returns,
            prev_weights=prev_weights,
            control_state=control_state,
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
        core_budget = invest_frac * (1.0 - overlay_size)
        satellite_budget = invest_frac * overlay_size
        core_weights = optimized_target * core_budget

        weights = core_weights + satellite_weights * satellite_budget
        allocator_cap = float(self._latest_allocator_diagnostics.get('adaptive_allocator_max_weight', self.optimizer_config.max_weight))
        effective_cap = max(allocator_cap, invest_frac / max(1, n_assets))
        weights = self._enforce_absolute_weight_cap(weights, effective_cap)
        cash_weight = max(0.0, 1.0 - invest_frac)

        return weights, cash_weight

    def construct_allocator_only(
        self,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
        control_state=None,
    ) -> tuple[pd.Series, float]:
        """
        Build the constrained allocator book with no RL control layer.
        This is the non-RL baseline corresponding to the simplified architecture.
        """
        weights = self.optimize_target_book(
            factor_scores=factor_scores,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            control_state=control_state,
        )
        return weights, 0.0


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
    RL agent for dynamic option-based tail-risk hedging.

    The policy chooses both:
    - hedge type: protective put, collar, or put spread
    - hedge intensity: 0%, 3%, 8%, 15%

    State: (portfolio drawdown, vol regime, momentum regime,
            implied-vol percentile, IV minus realized-vol spread)
    """
    def __init__(
        self,
        alpha: float = 0.03,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        hedge_ratios: Sequence[float] | None = None,
        hedge_types: Sequence[str] | None = None,
        option_overlay_config: OptionOverlayConfig | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.hedge_ratios = list(hedge_ratios or (0.0, 0.03, 0.08, 0.15))
        self.hedge_types = list(hedge_types or ('protective_put', 'collar', 'put_spread'))
        self.option_overlay_config = option_overlay_config or OptionOverlayConfig()
        self.n_ratio_actions = len(self.hedge_ratios)
        self.n_type_actions = len(self.hedge_types)
        self.n_actions = self.n_ratio_actions * self.n_type_actions
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.portfolio_peak = 1.0
        self.reward_buffer = deque(maxlen=60)

    def get_state(
        self,
        drawdown: float,
        vol_percentile: float,
        momentum: float,
        iv_percentile: float = 0.5,
        iv_realized_spread: float = 0.0,
    ) -> tuple[int, int, int, int, int]:
        dd_bin = int(np.clip(np.digitize(-drawdown, [0.05, 0.10, 0.16, 0.24]), 0, 4))
        vol_bin = int(np.clip(np.digitize(vol_percentile, [0.50, 0.70, 0.85, 0.95]), 0, 4))
        mom_bin = int(np.clip(np.digitize(momentum, [-0.08, -0.03, 0.0, 0.04]), 0, 4))
        iv_bin = int(np.clip(np.digitize(iv_percentile, [0.40, 0.60, 0.80, 0.92]), 0, 4))
        spread_bin = int(np.clip(np.digitize(iv_realized_spread, [-0.04, 0.0, 0.05, 0.12]), 0, 4))
        return (dd_bin, vol_bin, mom_bin, iv_bin, spread_bin)

    def select_action(self, state: tuple[int, int, int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def decode_action(self, action: int) -> tuple[int, int]:
        type_idx = int(action // self.n_ratio_actions)
        ratio_idx = int(action % self.n_ratio_actions)
        type_idx = int(np.clip(type_idx, 0, self.n_type_actions - 1))
        ratio_idx = int(np.clip(ratio_idx, 0, self.n_ratio_actions - 1))
        return type_idx, ratio_idx

    def get_type_labels(self) -> list[str]:
        return [hedge_type.replace('_', ' ').title() for hedge_type in self.hedge_types]

    def get_joint_action_labels(self) -> list[str]:
        labels: list[str] = []
        for hedge_type in self.hedge_types:
            prefix = hedge_type.replace('_', ' ').title()
            for hedge_ratio in self.hedge_ratios:
                labels.append(f'{prefix} {hedge_ratio:.0%}')
        return labels

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

        if reward_mode == 'mean_variance':
            rets = np.array(self.reward_buffer, dtype=float)
            lam = 2.0
            return float(portfolio_return * 100.0 - lam * float(np.var(rets)) * 100.0 if len(rets) >= 6 else portfolio_return * 100.0)

        # Default: asymmetric return reward (losses penalized harder).
        if portfolio_return >= 0:
            return portfolio_return * 100
        return portfolio_return * 200

    def _effective_hedge_ratio(
        self,
        hedge_ratio: float,
        vol_percentile: float,
        drawdown: float,
        momentum: float,
        iv_percentile: float,
    ) -> float:
        stress_multiplier = (
            0.70
            + 0.25 * float(vol_percentile > 0.70)
            + 0.20 * float(drawdown < -0.06)
            + 0.15 * float(momentum < -0.02)
            + 0.15 * float(iv_percentile > 0.75)
        )
        return float(np.clip(
            hedge_ratio * stress_multiplier,
            0.0,
            self.option_overlay_config.max_effective_hedge,
        ))

    def apply_hedge(
        self,
        risky_ret: float,
        cash_ret: float,
        market_ret: float,
        hedge_type: str,
        hedge_ratio: float,
        vol_percentile: float,
        iv_annualized: float,
        iv_percentile: float,
        iv_realized_spread: float,
        drawdown: float = 0.0,
        momentum: float = 0.0,
    ) -> tuple[float, dict[str, float | str]]:
        """
        Approximate daily PnL for simple option overlays.

        The sleeve treats VIX-derived implied vol as the option-pricing driver
        and applies daily theta plus downside payoff approximations. This is
        still stylized, but it is much closer to an explicit options sleeve than
        the prior pure convex-bonus hedge.
        """
        overlay_type = hedge_type if hedge_ratio > 1e-8 else 'none'
        effective_hedge = self._effective_hedge_ratio(
            hedge_ratio, vol_percentile, drawdown, momentum, iv_percentile,
        )
        if overlay_type == 'none' or effective_hedge <= 1e-8:
            total_ret = risky_ret + cash_ret
            return total_ret, {
                'hedge_pnl': 0.0,
                'hedge_cost': 0.0,
                'hedge_benefit': 0.0,
                'effective_hedge_ratio': 0.0,
                'hedge_type': 'none',
            }

        cfg = self.option_overlay_config
        iv_annualized = float(np.clip(iv_annualized, 0.08, 0.90))
        iv_day = iv_annualized / np.sqrt(252.0)
        theta_daily = effective_hedge * iv_day * cfg.theta_premium_scale * (0.85 + 0.65 * iv_percentile)
        downside_move = max(-market_ret, 0.0)
        upside_move = max(market_ret, 0.0)
        downside_boost = 1.0 + 0.35 * vol_percentile + 0.25 * max(iv_realized_spread, 0.0)
        put_strike = cfg.put_strike_otm
        call_strike = cfg.call_strike_otm
        spread_width = cfg.spread_width

        payoff = 0.0
        carry_credit = 0.0
        upside_giveup = 0.0

        if overlay_type == 'protective_put':
            payoff = (
                effective_hedge
                * cfg.convexity_scale
                * max(downside_move - put_strike, 0.0)
                * downside_boost
            )
        elif overlay_type == 'collar':
            payoff = (
                effective_hedge
                * 0.95
                * max(downside_move - put_strike, 0.0)
                * downside_boost
            )
            carry_credit = theta_daily * cfg.collar_financing_ratio
            upside_giveup = (
                effective_hedge
                * 0.80
                * max(upside_move - call_strike, 0.0)
                * (0.85 + 0.30 * iv_percentile)
            )
        elif overlay_type == 'put_spread':
            raw_spread_payoff = max(downside_move - put_strike, 0.0)
            payoff = (
                effective_hedge
                * 1.10
                * min(raw_spread_payoff, spread_width)
                * downside_boost
            )
            theta_daily *= 0.70

        hedge_cost = theta_daily + upside_giveup
        hedge_benefit = payoff + carry_credit
        hedge_pnl = hedge_benefit - hedge_cost
        total_ret = risky_ret + cash_ret + hedge_pnl
        return total_ret, {
            'hedge_pnl': float(hedge_pnl),
            'hedge_cost': float(hedge_cost),
            'hedge_benefit': float(hedge_benefit),
            'effective_hedge_ratio': float(effective_hedge),
            'hedge_type': overlay_type,
        }

    def update(
        self,
        state: tuple[int, int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int, int],
    ) -> None:
        best_next = np.max(self.Q[next_state])
        td = reward + self.gamma * best_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td


# ============================================================
# End-to-End RL Baseline
# ============================================================

from .rl_e2e import EndToEndTradingEnv, build_e2e_features, run_e2e_baseline
