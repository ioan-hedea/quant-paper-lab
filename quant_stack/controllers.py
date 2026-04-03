"""Pluggable control layer for the factor-anchored portfolio pipeline.

Architecture Revision v2: all control candidates share a common interface
so they can be compared on equal footing. Each controller receives a
``ControlState`` and returns an invested fraction in [0, 1].
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ControlConfig


# ============================================================
# Shared State Container
# ============================================================

@dataclass
class ControlState:
    """State information passed to all controllers each day."""

    alpha_strength: float = 0.0       # mean |alpha| across universe
    recent_drawdown: float = 0.0      # trailing 63-day max drawdown
    recent_vol: float = 0.15          # trailing 21-day realized portfolio vol (annualized)
    regime_belief: float = 0.5        # P(bull) from HMM + macro blend
    trend: float = 0.0                # trailing 20-day annualized return
    concentration: float = 0.0        # Herfindahl index of portfolio weights
    invested_fraction: float = 1.0    # current sum of weights
    t: int = 0                        # current time step index


def build_control_state(
    alpha_scores: pd.Series,
    portfolio_weights: pd.Series | None,
    recent_returns: pd.DataFrame | None,
    regime_belief: float,
    wealth_path: list[float],
    t: int,
) -> ControlState:
    """Construct a ControlState from pipeline data."""
    alpha_scores = pd.Series(alpha_scores, copy=False).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    alpha_strength = float(alpha_scores.abs().mean()) if len(alpha_scores) > 0 else 0.0

    if portfolio_weights is None or len(portfolio_weights) == 0:
        invested_fraction = 0.0
        concentration = 0.0
    else:
        w = pd.Series(portfolio_weights, copy=False).fillna(0.0).clip(lower=0.0)
        invested_fraction = float(w.sum())
        norm = w / (invested_fraction + 1e-8)
        concentration = float((norm ** 2).sum())

    if recent_returns is None or len(recent_returns) == 0:
        trend = 0.0
        recent_vol = 0.15
    else:
        if isinstance(recent_returns, pd.DataFrame):
            path_proxy = recent_returns.mean(axis=1)
        else:
            path_proxy = pd.Series(recent_returns, copy=False)
        path_proxy = path_proxy.dropna()
        trend = float(path_proxy.tail(20).mean() * 252) if len(path_proxy) > 0 else 0.0
        recent_vol = float(path_proxy.tail(21).std() * np.sqrt(252)) if len(path_proxy) > 5 else 0.15

    # Trailing drawdown from wealth path
    if len(wealth_path) > 1:
        w_arr = np.asarray(wealth_path[-63:], dtype=float)
        peak = np.maximum.accumulate(w_arr)
        dd = (w_arr - peak) / (peak + 1e-8)
        recent_drawdown = float(dd.min())
    else:
        recent_drawdown = 0.0

    return ControlState(
        alpha_strength=alpha_strength,
        recent_drawdown=recent_drawdown,
        recent_vol=recent_vol,
        regime_belief=regime_belief,
        trend=trend,
        concentration=concentration,
        invested_fraction=invested_fraction,
        t=t,
    )


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax helper used by the council gate."""
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    z = z - np.max(z)
    weights = np.exp(z)
    return weights / (weights.sum() + 1e-8)


# ============================================================
# Base Controller
# ============================================================

class BaseController:
    """Abstract base for all control candidates."""

    label: str = 'base'

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.convexity_enabled = bool(getattr(config, 'convexity_enabled', False)) if config is not None else False
        self.convexity_threshold = float(getattr(config, 'convexity_threshold', 0.0)) if config is not None else 0.0
        self.convexity_mode_carries = tuple(getattr(config, 'convexity_mode_carries', (0.0, 0.0, 0.0)))
        self.convexity_mode_lambdas = tuple(getattr(config, 'convexity_mode_lambdas', (0.0, 0.0, 0.0)))
        self.convexity_mild_drawdown = float(getattr(config, 'convexity_mild_drawdown', -0.05)) if config is not None else -0.05
        self.convexity_strong_drawdown = float(getattr(config, 'convexity_strong_drawdown', -0.10)) if config is not None else -0.10
        self.convexity_mild_vol = float(getattr(config, 'convexity_mild_vol', 0.18)) if config is not None else 0.18
        self.convexity_strong_vol = float(getattr(config, 'convexity_strong_vol', 0.26)) if config is not None else 0.26
        self.convexity_mild_regime = float(getattr(config, 'convexity_mild_regime', 0.45)) if config is not None else 0.45
        self.convexity_strong_regime = float(getattr(config, 'convexity_strong_regime', 0.30)) if config is not None else 0.30
        self._latest_diagnostics: dict[str, object] = {}

    def compute_invested_fraction(self, state: ControlState) -> float:
        raise NotImplementedError

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        pass

    def uses_direct_weights(self) -> bool:
        return False

    def build_target_weights(
        self,
        allocator,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
        prev_weights: pd.Series | None,
        optimizer_config,
        state: ControlState,
    ) -> pd.Series | None:
        return None

    def apply_return_overlay(
        self,
        portfolio_ret: float,
        state: ControlState,
    ) -> tuple[float, dict[str, object]]:
        mode = 0
        if self.convexity_enabled:
            if (
                state.recent_drawdown <= self.convexity_strong_drawdown
                or state.recent_vol >= self.convexity_strong_vol
                or state.regime_belief <= self.convexity_strong_regime
            ):
                mode = 2
            elif (
                state.recent_drawdown <= self.convexity_mild_drawdown
                or state.recent_vol >= self.convexity_mild_vol
                or state.regime_belief <= self.convexity_mild_regime
            ):
                mode = 1

        carry = float(self.convexity_mode_carries[min(mode, len(self.convexity_mode_carries) - 1)])
        lam = float(self.convexity_mode_lambdas[min(mode, len(self.convexity_mode_lambdas) - 1)])
        benefit = float(lam * max(0.0, self.convexity_threshold - float(portfolio_ret)) ** 2)
        adjusted_ret = float(portfolio_ret - carry + benefit)
        diagnostics = {
            'convexity_mode': mode,
            'convexity_mode_name': ('none', 'mild', 'strong')[mode],
            'convexity_carry': carry,
            'convexity_benefit': benefit,
            'convexity_net_adjustment': benefit - carry,
        }
        self._latest_diagnostics.update(diagnostics)
        return adjusted_ret, diagnostics

    def current_diagnostics(self) -> dict[str, object]:
        return dict(self._latest_diagnostics)

    def reset(self) -> None:
        pass


# ============================================================
# A1: Fixed Allocator
# ============================================================

class FixedAllocator(BaseController):
    """Constant invested fraction — no dynamic adjustment."""

    label = 'fixed'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.invested_fraction = config.fixed_invested_fraction

    def compute_invested_fraction(self, state: ControlState) -> float:
        return self.invested_fraction


# ============================================================
# A2: Volatility Targeting
# ============================================================

class VolTargetController(BaseController):
    """Scale exposure to maintain a target portfolio volatility."""

    label = 'vol_target'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.vol_target = config.vol_target_annual

    def compute_invested_fraction(self, state: ControlState) -> float:
        sigma_hat = max(state.recent_vol, 0.01)
        return float(np.clip(self.vol_target / sigma_hat, 0.30, 1.0))


# ============================================================
# A3: Drawdown-Based Deleveraging
# ============================================================

class DDDeleverController(BaseController):
    """Reduce exposure when drawdown exceeds thresholds."""

    label = 'dd_delever'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.thresholds = list(config.dd_thresholds)
        self.min_invested = config.dd_min_invested

    def compute_invested_fraction(self, state: ControlState) -> float:
        dd = state.recent_drawdown
        invested = 1.0
        for dd_thresh, dd_frac in self.thresholds:
            if dd < dd_thresh:
                invested = dd_frac
        return max(self.min_invested, invested)


# ============================================================
# A4: Regime-Conditioned Exposure Rules
# ============================================================

class RegimeRulesController(BaseController):
    """Use HMM regime belief to set exposure level."""

    label = 'regime_rules'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.bull_threshold = config.regime_bull_threshold
        self.bear_threshold = config.regime_bear_threshold
        self.bull_fraction = config.regime_bull_fraction
        self.neutral_fraction = config.regime_neutral_fraction
        self.bear_fraction = config.regime_bear_fraction

    def compute_invested_fraction(self, state: ControlState) -> float:
        belief = state.regime_belief
        if belief > self.bull_threshold:
            return self.bull_fraction
        if belief < self.bear_threshold:
            return self.bear_fraction
        return self.neutral_fraction


# ============================================================
# A5: Simple Ensemble
# ============================================================

class EnsembleController(BaseController):
    """Combine vol-target, DD-delever, and regime rules."""

    label = 'ensemble_rules'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.vol_target = VolTargetController(config)
        self.dd_delever = DDDeleverController(config)
        self.regime_rules = RegimeRulesController(config)
        self.mode = config.ensemble_mode

    def compute_invested_fraction(self, state: ControlState) -> float:
        b_vt = self.vol_target.compute_invested_fraction(state)
        b_dd = self.dd_delever.compute_invested_fraction(state)
        b_rr = self.regime_rules.compute_invested_fraction(state)
        if self.mode == 'min':
            return min(b_vt, b_dd, b_rr)
        return (b_vt + b_dd + b_rr) / 3.0


# ============================================================
# B1: LinUCB Contextual Bandit
# ============================================================

class LinUCBBandit(BaseController):
    """Linear UCB contextual bandit for exposure selection."""

    label = 'linucb'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.n_actions = config.bandit_n_actions
        self.alpha_ucb = config.bandit_alpha_ucb
        self.reward_window = config.bandit_reward_window
        self.feature_lookback = config.bandit_feature_lookback
        self.invest_fractions = np.linspace(0.82, 1.0, self.n_actions)

        # Feature dimension: alpha_strength, recent_drawdown, recent_vol
        self.d = 3
        self.A = [np.eye(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.n_actions)]

        self._feature_history: deque[np.ndarray] = deque(maxlen=config.bandit_feature_lookback)
        self._pending_action: int | None = None
        self._pending_features: np.ndarray | None = None
        self._reward_buffer: deque[float] = deque(maxlen=config.bandit_reward_window)

    def _features(self, state: ControlState) -> np.ndarray:
        raw = np.array([state.alpha_strength, state.recent_drawdown, state.recent_vol])
        self._feature_history.append(raw)
        if len(self._feature_history) < 20:
            return raw
        hist = np.array(self._feature_history)
        mu = hist.mean(axis=0)
        sigma = hist.std(axis=0) + 1e-8
        return (raw - mu) / sigma

    def compute_invested_fraction(self, state: ControlState) -> float:
        x = self._features(state)
        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            ucb_values[a] = float(theta @ x + self.alpha_ucb * np.sqrt(x @ A_inv @ x))
        best = int(np.argmax(ucb_values))
        self._pending_action = best
        self._pending_features = x
        return float(self.invest_fractions[best])

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        self._reward_buffer.append(reward)
        if self._pending_action is not None and self._pending_features is not None:
            a = self._pending_action
            x = self._pending_features
            self.A[a] += np.outer(x, x)
            self.b[a] += reward * x
            self._pending_action = None
            self._pending_features = None

    def reset(self) -> None:
        self.A = [np.eye(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.n_actions)]
        self._feature_history.clear()


# ============================================================
# B2: Thompson Sampling Bandit
# ============================================================

class ThompsonSamplingBandit(BaseController):
    """Bayesian Thompson Sampling contextual bandit."""

    label = 'thompson'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.n_actions = config.bandit_n_actions
        self.reward_window = config.bandit_reward_window
        self.invest_fractions = np.linspace(0.82, 1.0, self.n_actions)

        self.d = 3
        # Bayesian linear regression per arm: N(mu, Sigma)
        self.mu = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.Sigma = [np.eye(self.d) for _ in range(self.n_actions)]
        self.sigma_noise = 1.0  # observation noise

        self._feature_history: deque[np.ndarray] = deque(maxlen=config.bandit_feature_lookback)
        self._pending_action: int | None = None
        self._pending_features: np.ndarray | None = None

    def _features(self, state: ControlState) -> np.ndarray:
        raw = np.array([state.alpha_strength, state.recent_drawdown, state.recent_vol])
        self._feature_history.append(raw)
        if len(self._feature_history) < 20:
            return raw
        hist = np.array(self._feature_history)
        mu = hist.mean(axis=0)
        sigma = hist.std(axis=0) + 1e-8
        return (raw - mu) / sigma

    def compute_invested_fraction(self, state: ControlState) -> float:
        x = self._features(state)
        sampled_rewards = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta_sample = np.random.multivariate_normal(self.mu[a], self.Sigma[a])
            sampled_rewards[a] = float(theta_sample @ x)
        best = int(np.argmax(sampled_rewards))
        self._pending_action = best
        self._pending_features = x
        return float(self.invest_fractions[best])

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        if self._pending_action is not None and self._pending_features is not None:
            a = self._pending_action
            x = self._pending_features
            # Bayesian update for linear regression
            Sigma_inv = np.linalg.inv(self.Sigma[a] + 1e-6 * np.eye(self.d))
            Sigma_new_inv = Sigma_inv + np.outer(x, x) / (self.sigma_noise ** 2)
            self.Sigma[a] = np.linalg.inv(Sigma_new_inv + 1e-6 * np.eye(self.d))
            self.mu[a] = self.Sigma[a] @ (Sigma_inv @ self.mu[a] + x * reward / (self.sigma_noise ** 2))
            self._pending_action = None
            self._pending_features = None


# ============================================================
# B3: Epsilon-Greedy Linear Bandit
# ============================================================

class EpsilonGreedyBandit(BaseController):
    """Simple epsilon-greedy bandit with linear reward model."""

    label = 'epsilon_greedy'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.n_actions = config.bandit_n_actions
        self.epsilon = config.bandit_epsilon
        self.invest_fractions = np.linspace(0.82, 1.0, self.n_actions)

        self.d = 3
        self.weights = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.counts = [0] * self.n_actions
        self.lr = 0.01

        self._feature_history: deque[np.ndarray] = deque(maxlen=config.bandit_feature_lookback)
        self._pending_action: int | None = None
        self._pending_features: np.ndarray | None = None

    def _features(self, state: ControlState) -> np.ndarray:
        raw = np.array([state.alpha_strength, state.recent_drawdown, state.recent_vol])
        self._feature_history.append(raw)
        if len(self._feature_history) < 20:
            return raw
        hist = np.array(self._feature_history)
        mu = hist.mean(axis=0)
        sigma = hist.std(axis=0) + 1e-8
        return (raw - mu) / sigma

    def compute_invested_fraction(self, state: ControlState) -> float:
        x = self._features(state)
        if np.random.random() < self.epsilon:
            best = np.random.randint(self.n_actions)
        else:
            values = [float(self.weights[a] @ x) for a in range(self.n_actions)]
            best = int(np.argmax(values))
        self._pending_action = best
        self._pending_features = x
        return float(self.invest_fractions[best])

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        if self._pending_action is not None and self._pending_features is not None:
            a = self._pending_action
            x = self._pending_features
            pred = float(self.weights[a] @ x)
            self.weights[a] += self.lr * (reward - pred) * x
            self.counts[a] += 1
            self._pending_action = None
            self._pending_features = None


# ============================================================
# C: Supervised Regime-Conditioned Controller
# ============================================================

class SupervisedController(BaseController):
    """Supervised classifier maps state to best exposure bucket."""

    label = 'supervised'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.n_actions = config.bandit_n_actions
        self.invest_fractions = np.linspace(0.82, 1.0, self.n_actions)
        self.model_type = config.supervised_model
        self.retrain_every = config.supervised_retrain_every
        self.label_window = config.supervised_label_window

        self._model = None
        self._feature_buffer: list[np.ndarray] = []
        self._return_buffer: list[float] = []
        self._last_train_t: int = -999
        self._steps_since_train: int = 0

    def _features(self, state: ControlState) -> np.ndarray:
        return np.array([state.alpha_strength, state.recent_drawdown, state.recent_vol])

    def _build_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        if self.model_type == 'decision_tree':
            return DecisionTreeClassifier(max_depth=4, random_state=42)
        return LogisticRegression(max_iter=500, random_state=42)

    def _compute_best_labels(self) -> np.ndarray | None:
        """For each past date, find which invested fraction maximized trailing Sortino."""
        n = len(self._return_buffer)
        if n < self.label_window + 10:
            return None
        labels = np.zeros(n - self.label_window, dtype=int)
        rets = np.array(self._return_buffer)
        for i in range(n - self.label_window):
            fwd_rets = rets[i:i + self.label_window]
            best_sortino = -np.inf
            best_action = self.n_actions // 2
            for a, frac in enumerate(self.invest_fractions):
                scaled = fwd_rets * frac
                downside = scaled[scaled < 0]
                ds_std = float(downside.std()) if len(downside) > 0 else 1e-8
                sortino = float(scaled.mean() / (ds_std + 1e-8))
                if sortino > best_sortino:
                    best_sortino = sortino
                    best_action = a
            labels[i] = best_action
        return labels

    def _try_retrain(self, t: int) -> None:
        if t - self._last_train_t < self.retrain_every:
            return
        labels = self._compute_best_labels()
        if labels is None:
            return
        n_labels = len(labels)
        n_features = len(self._feature_buffer)
        n_usable = min(n_labels, n_features - self.label_window)
        if n_usable < 30:
            return
        X = np.array(self._feature_buffer[:n_usable])
        y = labels[:n_usable]
        if len(np.unique(y)) < 2:
            return
        model = self._build_model()
        model.fit(X, y)
        self._model = model
        self._last_train_t = t

    def compute_invested_fraction(self, state: ControlState) -> float:
        x = self._features(state)
        self._feature_buffer.append(x)
        self._try_retrain(state.t)
        if self._model is None:
            return 0.95  # default until trained
        action = int(self._model.predict(x.reshape(1, -1))[0])
        action = int(np.clip(action, 0, self.n_actions - 1))
        return float(self.invest_fractions[action])

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        self._return_buffer.append(reward)


# ============================================================
# D: CVaR-Aware Robust Optimization
# ============================================================

class CVaRRobustController(BaseController):
    """CVaR-aware robust optimizer that extends the constrained allocator.

    Instead of just adjusting invested fraction, this controller modifies
    the portfolio weights directly by adding a CVaR penalty to the objective.
    It returns invested_fraction=1.0 and provides optimized weights via
    ``optimize_with_cvar()``.
    """

    label = 'cvar_robust'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.confidence = config.cvar_confidence
        self.n_scenarios = config.cvar_n_scenarios
        self.lambda_base = config.cvar_lambda_base
        self.regime_scaling = config.cvar_regime_scaling
        self.dd_budget = config.cvar_dd_budget
        self._current_lambda = config.cvar_lambda_base

    def compute_invested_fraction(self, state: ControlState) -> float:
        # Modulate lambda_cvar based on regime and drawdown
        lam = self.lambda_base
        if self.regime_scaling:
            belief = state.regime_belief
            if belief > 0.70:
                lam = self.lambda_base
            elif belief > 0.30:
                lam = 2.0 * self.lambda_base
            else:
                lam = 4.0 * self.lambda_base
        if self.dd_budget and state.recent_drawdown < -0.05:
            dd_mult = 1.0 + 3.0 * abs(state.recent_drawdown)
            lam *= dd_mult
        self._current_lambda = lam
        return 1.0  # CVaR optimizer produces its own weights

    def uses_direct_weights(self) -> bool:
        return True

    def build_target_weights(
        self,
        allocator,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
        prev_weights: pd.Series | None,
        optimizer_config,
        state: ControlState,
    ) -> pd.Series | None:
        self.compute_invested_fraction(state)
        adapted_optimizer = allocator.adapt_optimizer_config(
            optimizer_config=optimizer_config,
            control_state=state,
        )
        base_target = allocator.optimize_target_book(
            factor_scores=factor_scores,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            optimizer_config=adapted_optimizer,
            adapt_config=False,
        )
        return self.optimize_with_cvar(
            base_target=base_target,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            optimizer_config=adapted_optimizer,
        )

    def optimize_with_cvar(
        self,
        base_target: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
        prev_weights: pd.Series | None,
        optimizer_config,
    ) -> pd.Series:
        """Produce CVaR-optimized weights using Rockafellar-Uryasev formulation."""
        from scipy.optimize import linprog, minimize as sp_minimize

        tickers = base_target.index
        n_assets = len(tickers)
        if n_assets == 0 or recent_returns is None or len(recent_returns) < 40:
            return base_target

        aligned_returns = recent_returns.reindex(columns=tickers).fillna(0.0)
        if aligned_returns.shape[1] < 2:
            return base_target

        try:
            lw = LedoitWolf().fit(aligned_returns.values)
            cov = lw.covariance_
        except Exception:
            return base_target

        # Generate return scenarios from N(alpha, Sigma)
        alpha_view = alpha_scores.reindex(tickers).fillna(0.0).values / 100.0
        rng = np.random.default_rng(42)
        scenarios = rng.multivariate_normal(alpha_view, cov, size=self.n_scenarios)

        # Rockafellar-Uryasev CVaR as part of portfolio optimization
        gamma = self.confidence
        S = self.n_scenarios
        lam_cvar = self._current_lambda

        effective_max_weight = max(optimizer_config.max_weight, 1.0 / max(1, n_assets))

        def objective(params):
            w = params[:n_assets]
            zeta = params[n_assets]
            # Portfolio losses under each scenario
            losses = -scenarios @ w  # (S,)
            excess = np.maximum(losses - zeta, 0.0)
            cvar_term = zeta + excess.mean() / (1.0 - gamma + 1e-8)

            # Standard QP terms
            risk_term = optimizer_config.risk_aversion * float(w @ cov @ w)
            alpha_term = -optimizer_config.alpha_strength * float(alpha_view @ w)
            anchor_term = optimizer_config.anchor_strength * float(
                np.sum((w - base_target.values) ** 2)
            )
            turnover_term = 0.0
            if prev_weights is not None and len(prev_weights) > 0:
                prev = prev_weights.reindex(tickers).fillna(0.0).values
                prev_sum = prev.sum()
                if prev_sum > 1e-8:
                    prev_norm = prev / prev_sum
                    turnover_term = optimizer_config.turnover_penalty * float(
                        np.sum((w - prev_norm) ** 2)
                    )
            return risk_term + alpha_term + anchor_term + turnover_term + lam_cvar * cvar_term

        # Initial guess
        x0 = np.zeros(n_assets + 1)
        x0[:n_assets] = np.clip(base_target.values, 0.0, effective_max_weight)
        x0[:n_assets] /= x0[:n_assets].sum() + 1e-8
        x0[n_assets] = 0.0  # zeta

        bounds = [(0.0, effective_max_weight) for _ in range(n_assets)] + [(-1.0, 1.0)]
        constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:n_assets]) - 1.0}]

        try:
            result = sp_minimize(
                objective, x0=x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8},
            )
            if result.success:
                w = np.clip(result.x[:n_assets], 0.0, effective_max_weight)
                w = w / (w.sum() + 1e-8)
                return pd.Series(w, index=tickers)
        except Exception:
            pass

        return base_target


# ============================================================
# E: Expert-Gated Council
# ============================================================

class CouncilController(BaseController):
    """CVaR-centric council that softly mixes a small set of structured experts.

    The council is intentionally narrow and interpretable. It only gates
    between regime rules, LinUCB, and CVaR-robust optimization, and its
    gate uses a small logistic model over four state variables.
    """

    label = 'council'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.expert_names = tuple(config.council_experts)
        self.gate_model_type = config.council_gate_model
        self.retrain_every = config.council_retrain_every
        self.min_samples = config.council_min_samples
        self.temperature = config.council_temperature
        self.min_weight = config.council_min_weight
        self.default_bias = np.asarray(config.council_default_bias, dtype=float)

        self.experts: dict[str, BaseController] = {
            'regime_rules': RegimeRulesController(config),
            'linucb': LinUCBBandit(config),
            'cvar_robust': CVaRRobustController(config),
        }
        for expert_name in self.expert_names:
            if expert_name not in self.experts:
                raise ValueError(f"Unsupported council expert: {expert_name}")

        self._model = None
        self._feature_buffer: list[np.ndarray] = []
        self._label_buffer: list[int] = []
        self._last_train_t: int = -999
        self._pending_features: np.ndarray | None = None
        self._pending_gate: np.ndarray | None = None
        self._pending_books: dict[str, pd.Series] = {}

    def _features(self, state: ControlState) -> np.ndarray:
        return np.array([
            state.alpha_strength,
            state.recent_drawdown,
            state.recent_vol,
            state.regime_belief,
        ], dtype=float)

    def _build_model(self):
        from sklearn.linear_model import LogisticRegression

        # Keep the gate compatible across scikit-learn versions by relying on
        # the library default multiclass handling instead of forcing an
        # explicit keyword that is not accepted everywhere.
        return LogisticRegression(max_iter=500, random_state=42)

    def _default_gate_weights(self, state: ControlState) -> np.ndarray:
        bias = self.default_bias
        if len(bias) != len(self.expert_names):
            bias = np.ones(len(self.expert_names), dtype=float)
        bias = np.clip(bias, 1e-4, None)
        logits = np.log(bias / bias.sum())

        risk_pressure = (
            6.0 * max(0.0, abs(state.recent_drawdown) - 0.03)
            + 3.0 * max(0.0, state.recent_vol - 0.15)
            + 2.0 * max(0.0, 0.50 - state.regime_belief)
        )
        alpha_pressure = max(0.0, state.alpha_strength - 0.50)
        adjustments: list[float] = []
        for expert_name in self.expert_names:
            if expert_name == 'regime_rules':
                adjustments.append(0.50 * state.regime_belief + 0.20 * alpha_pressure)
            elif expert_name == 'linucb':
                adjustments.append(0.60 * alpha_pressure - 0.20 * risk_pressure)
            elif expert_name == 'cvar_robust':
                adjustments.append(0.80 * risk_pressure - 0.20 * alpha_pressure)
            else:
                adjustments.append(0.0)
        return _stable_softmax(logits + np.asarray(adjustments, dtype=float), self.temperature)

    def _predict_gate_weights(self, state: ControlState) -> tuple[np.ndarray, np.ndarray]:
        x = self._features(state)
        if self._model is None:
            return x, self._default_gate_weights(state)
        probs = np.asarray(self._model.predict_proba(x.reshape(1, -1))[0], dtype=float)
        full = np.zeros(len(self.expert_names), dtype=float)
        classes = np.asarray(getattr(self._model, 'classes_', np.arange(len(self.expert_names))), dtype=int)
        for cls_idx, prob in zip(classes, probs):
            if 0 <= int(cls_idx) < len(full):
                full[int(cls_idx)] = float(prob)
        full = np.clip(full, 1e-8, None)
        full /= full.sum()
        full = np.maximum(full, self.min_weight)
        full /= full.sum()
        return x, full

    def _try_retrain(self, t: int) -> None:
        if self.gate_model_type != 'logistic':
            return
        if t - self._last_train_t < self.retrain_every:
            return
        if len(self._label_buffer) < self.min_samples:
            return
        y = np.asarray(self._label_buffer, dtype=int)
        if len(np.unique(y)) < 2:
            return
        X = np.asarray(self._feature_buffer, dtype=float)
        model = self._build_model()
        model.fit(X, y)
        self._model = model
        self._last_train_t = t

    def uses_direct_weights(self) -> bool:
        return True

    def get_pending_expert_books(self) -> dict[str, pd.Series]:
        return {
            name: weights.copy()
            for name, weights in self._pending_books.items()
        }

    def build_target_weights(
        self,
        allocator,
        factor_scores: pd.Series,
        alpha_scores: pd.Series,
        confidence: pd.Series,
        recent_returns: pd.DataFrame | None,
        prev_weights: pd.Series | None,
        optimizer_config,
        state: ControlState,
    ) -> pd.Series | None:
        constrained_target = allocator.optimize_target_book(
            factor_scores=factor_scores,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            control_state=state,
        ).clip(lower=0.0)
        constrained_target = constrained_target / (constrained_target.sum() + 1e-8)

        expert_books: dict[str, pd.Series] = {}
        for expert_name in self.expert_names:
            expert = self.experts[expert_name]
            if isinstance(expert, CVaRRobustController):
                expert_books[expert_name] = expert.build_target_weights(
                    allocator=allocator,
                    factor_scores=factor_scores,
                    alpha_scores=alpha_scores,
                    confidence=confidence,
                    recent_returns=recent_returns,
                    prev_weights=prev_weights,
                    optimizer_config=optimizer_config,
                    state=state,
                ).clip(lower=0.0)
            else:
                invested_fraction = float(np.clip(expert.compute_invested_fraction(state), 0.0, 1.0))
                expert_books[expert_name] = (constrained_target * invested_fraction).clip(lower=0.0)

        x, gate_weights = self._predict_gate_weights(state)
        combined = pd.Series(0.0, index=constrained_target.index, dtype=float)
        council_weights_map: dict[str, float] = {}
        for expert_name, gate_weight in zip(self.expert_names, gate_weights):
            combined = combined.add(expert_books[expert_name] * float(gate_weight), fill_value=0.0)
            council_weights_map[expert_name] = float(gate_weight)

        combined = combined.clip(lower=0.0)
        if float(combined.sum()) > 1.0:
            combined = combined / float(combined.sum())

        dominant_idx = int(np.argmax(gate_weights))
        dominant_expert = self.expert_names[dominant_idx]
        self._pending_features = x
        self._pending_gate = gate_weights
        self._pending_books = expert_books
        self._latest_diagnostics.update({
            'council_weights': council_weights_map,
            'council_dominant_expert': dominant_expert,
            'council_gate_entropy': float(-np.sum(gate_weights * np.log(gate_weights + 1e-8))),
        })
        return combined

    def update(
        self,
        state: ControlState,
        reward: float,
        next_state: ControlState,
        expert_feedback: dict[str, float] | None = None,
    ) -> None:
        if self._pending_features is None:
            return

        if expert_feedback:
            best_expert = max(expert_feedback.items(), key=lambda item: item[1])[0]
            if best_expert in self.expert_names:
                self._feature_buffer.append(self._pending_features)
                self._label_buffer.append(self.expert_names.index(best_expert))
                self._latest_diagnostics['council_best_expert'] = best_expert
                self._latest_diagnostics['council_expert_rewards'] = {
                    name: float(value) for name, value in expert_feedback.items()
                }
                for expert_name, expert in self.experts.items():
                    expert_reward = float(expert_feedback.get(expert_name, reward))
                    expert.update(state, expert_reward, next_state)
        else:
            dominant_idx = int(np.argmax(self._pending_gate)) if self._pending_gate is not None else 0
            self._feature_buffer.append(self._pending_features)
            self._label_buffer.append(dominant_idx)
            self._latest_diagnostics['council_best_expert'] = self.expert_names[dominant_idx]

        self._try_retrain(next_state.t)
        self._pending_features = None
        self._pending_gate = None
        self._pending_books = {}

    def reset(self) -> None:
        self._model = None
        self._feature_buffer.clear()
        self._label_buffer.clear()
        self._pending_features = None
        self._pending_gate = None
        self._pending_books = {}
        for expert in self.experts.values():
            expert.reset()


def _extended_controller_registry() -> dict[str, type[BaseController]]:
    """Load the heavier controller families lazily to keep the core module small."""
    from .controllers_extended import (
        AdaptiveAllocatorController,
        CMDPLagrangianController,
        MLPMetaController,
        MPCController,
        QLearningController,
    )

    return {
        'mlp_meta': MLPMetaController,
        'mpc': MPCController,
        'adaptive_allocator': AdaptiveAllocatorController,
        'cmdp_lagrangian': CMDPLagrangianController,
        'q_learning': QLearningController,
    }


# ============================================================
# Factory
# ============================================================

def build_controller(config: ControlConfig) -> BaseController:
    """Instantiate the appropriate controller from config."""
    registry = {
        'none': lambda c: FixedAllocator(ControlConfig(fixed_invested_fraction=1.0)),
        'fixed': FixedAllocator,
        'vol_target': VolTargetController,
        'dd_delever': DDDeleverController,
        'regime_rules': RegimeRulesController,
        'ensemble_rules': EnsembleController,
        'linucb': LinUCBBandit,
        'thompson': ThompsonSamplingBandit,
        'epsilon_greedy': EpsilonGreedyBandit,
        'supervised': SupervisedController,
        'cvar_robust': CVaRRobustController,
        'council': CouncilController,
    }
    registry.update(_extended_controller_registry())
    factory = registry.get(config.method)
    if factory is None:
        raise ValueError(f"Unknown control method: {config.method!r}. "
                         f"Choose from: {list(registry.keys())}")
    return factory(config)
