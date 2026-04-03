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


# ============================================================
# G: MLP-Gated Meta-Controller (Environment-Adaptive Selection)
# ============================================================

def _build_attention_gate_network(input_dim: int, n_experts: int, hidden: tuple[int, ...]):
    """Build a PyTorch gate network with self-attention and residual blocks.

    Architecture:
        Input → LayerNorm → Linear projection
        → Self-Attention (learned feature interactions)
        → Residual MLP blocks with GELU + LayerNorm + Dropout
        → Output logits (n_experts)

    The self-attention layer lets the network learn which environment
    features are most relevant conditioned on the current regime, so
    e.g. vol-of-vol matters more during stress than during calm.
    """
    import torch
    import torch.nn as nn

    class _FeatureAttention(nn.Module):
        """Single-head self-attention over the feature dimension."""

        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = dim ** -0.5

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, dim) → treat each feature as a "token" of size 1
            # Reshape to (batch, n_features, 1) for attention
            b, d = x.shape
            qkv = self.qkv(x)                      # (b, 3*d)
            q, k, v = qkv.chunk(3, dim=-1)          # each (b, d)
            # Feature-wise attention: which features attend to which
            attn = (q.unsqueeze(-1) @ k.unsqueeze(-2)) * self.scale  # (b, d, d)
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v.unsqueeze(-1)).squeeze(-1)  # (b, d)
            return self.proj(out)

    class _ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.net(x)

    class AttentionGateNet(nn.Module):
        def __init__(self, input_dim: int, n_experts: int, hidden: tuple[int, ...]):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_proj = nn.Linear(input_dim, hidden[0])
            self.attention = _FeatureAttention(hidden[0], dropout=0.1)
            self.attn_norm = nn.LayerNorm(hidden[0])

            blocks = []
            dims = list(hidden)
            for i in range(len(dims)):
                blocks.append(_ResidualBlock(dims[i], dropout=0.15))
                if i < len(dims) - 1 and dims[i] != dims[i + 1]:
                    blocks.append(nn.Linear(dims[i], dims[i + 1]))
            self.residual_blocks = nn.Sequential(*blocks)

            self.head = nn.Sequential(
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], n_experts),
            )
            # Learnable temperature for softmax gating
            self.log_temperature = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return logits (pre-softmax) for each expert."""
            h = self.input_proj(self.input_norm(x))
            h = h + self.attention(h)
            h = self.attn_norm(h)
            h = self.residual_blocks(h)
            logits = self.head(h)
            temp = self.log_temperature.exp().clamp(min=0.1, max=5.0)
            return logits / temp

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Return softmax probabilities."""
            with torch.no_grad():
                return torch.softmax(self.forward(x), dim=-1)

    return AttentionGateNet(input_dim, n_experts, hidden)


class MLPMetaController(BaseController):
    """MLP-gated meta-controller for environment-adaptive controller selection.

    Addresses the algorithm-selection question: given current environment
    characteristics (alpha quality, regime structure, volatility, market
    stress), which sub-controller should be weighted most heavily?

    Uses a PyTorch neural network with self-attention over environment
    features and residual MLP blocks to gate between sub-controllers.
    Falls back to heuristic default weights until enough training data
    has been collected.
    """

    label = 'mlp_meta'

    # ---- feature names (for diagnostics) ----
    _FEATURE_NAMES = (
        'alpha_mean', 'alpha_std', 'alpha_current_z',
        'vol_mean', 'vol_of_vol', 'downside_semivol',
        'dd_current', 'dd_mean_depth', 'dd_frequency',
        'regime_mean', 'regime_switch_freq', 'regime_entropy', 'regime_risk_off_frac',
        'trend_current', 'trend_mean',
        'concentration_mean',
    )

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.expert_names = tuple(config.mlp_meta_experts)
        self.hidden_layers = tuple(config.mlp_meta_hidden_layers)
        self.retrain_every = config.mlp_meta_retrain_every
        self.min_samples = config.mlp_meta_min_samples
        self.feature_lookback = config.mlp_meta_feature_lookback
        self.temperature = config.mlp_meta_temperature
        self.min_weight = config.mlp_meta_min_weight
        self.default_bias = np.asarray(config.mlp_meta_default_bias, dtype=float)
        self.learning_rate = config.mlp_meta_learning_rate
        self.alpha_reg = config.mlp_meta_alpha_reg

        # Sub-controllers
        self.experts: dict[str, BaseController] = {
            'regime_rules': RegimeRulesController(config),
            'linucb': LinUCBBandit(config),
            'cvar_robust': CVaRRobustController(config),
        }
        for name in self.expert_names:
            if name not in self.experts:
                raise ValueError(f"Unsupported mlp_meta expert: {name}")

        # PyTorch gate model (built lazily on first retrain)
        self._model = None
        self._feature_dim = len(self._FEATURE_NAMES)
        self._scaler_mu: np.ndarray | None = None
        self._scaler_sigma: np.ndarray | None = None

        # Trailing state history for environment feature extraction
        self._state_history: deque[ControlState] = deque(maxlen=max(self.feature_lookback, 63))

        # Training data buffers (replay buffer for mini-batch training)
        self._feature_buffer: list[np.ndarray] = []
        self._label_buffer: list[int] = []
        self._last_train_t: int = -999

        # Per-step bookkeeping
        self._pending_features: np.ndarray | None = None
        self._pending_gate: np.ndarray | None = None
        self._pending_books: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Environment feature extraction
    # ------------------------------------------------------------------

    def _compute_environment_features(self, state: ControlState) -> np.ndarray:
        """Compute a rich environment feature vector from trailing state history.

        Returns a 16-dimensional vector covering alpha quality,
        risk/volatility, regime structure, trend, and concentration.
        """
        self._state_history.append(state)
        hist = list(self._state_history)
        n = len(hist)

        # --- Alpha quality features ---
        alphas = np.array([s.alpha_strength for s in hist])
        alpha_mean = float(alphas.mean())
        alpha_std = float(alphas.std()) if n > 1 else 0.0
        alpha_current_z = float((state.alpha_strength - alpha_mean) / (alpha_std + 1e-8)) if n > 5 else 0.0

        # --- Risk / volatility features ---
        vols = np.array([s.recent_vol for s in hist])
        vol_mean = float(vols.mean())
        vol_of_vol = float(vols.std()) if n > 1 else 0.0
        above_mean = vols[vols > vol_mean]
        downside_semivol = float(above_mean.std()) if len(above_mean) > 1 else 0.0

        # --- Drawdown / market stress features ---
        dds = np.array([s.recent_drawdown for s in hist])
        dd_current = float(state.recent_drawdown)
        dd_mean_depth = float(dds[dds < 0].mean()) if np.any(dds < 0) else 0.0
        dd_frequency = float(np.mean(dds < -0.02))

        # --- Regime features ---
        beliefs = np.array([s.regime_belief for s in hist])
        regime_mean = float(beliefs.mean())
        if n > 1:
            crossings = np.abs(np.diff((beliefs > 0.5).astype(float)))
            regime_switch_freq = float(crossings.mean())
        else:
            regime_switch_freq = 0.0
        b_clipped = np.clip(beliefs, 1e-8, 1.0 - 1e-8)
        regime_entropy = float(np.mean(-b_clipped * np.log(b_clipped) - (1 - b_clipped) * np.log(1 - b_clipped)))
        regime_risk_off_frac = float(np.mean(beliefs < 0.30))

        # --- Trend features ---
        trends = np.array([s.trend for s in hist])
        trend_current = float(state.trend)
        trend_mean = float(trends.mean())

        # --- Concentration features ---
        concentrations = np.array([s.concentration for s in hist])
        concentration_mean = float(concentrations.mean())

        return np.array([
            alpha_mean, alpha_std, alpha_current_z,
            vol_mean, vol_of_vol, downside_semivol,
            dd_current, dd_mean_depth, dd_frequency,
            regime_mean, regime_switch_freq, regime_entropy, regime_risk_off_frac,
            trend_current, trend_mean,
            concentration_mean,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Gate logic
    # ------------------------------------------------------------------

    def _default_gate_weights(self, state: ControlState) -> np.ndarray:
        """Heuristic default gate used before the network has enough data."""
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
        for name in self.expert_names:
            if name == 'regime_rules':
                adjustments.append(0.50 * state.regime_belief + 0.20 * alpha_pressure)
            elif name == 'linucb':
                adjustments.append(0.60 * alpha_pressure - 0.20 * risk_pressure)
            elif name == 'cvar_robust':
                adjustments.append(0.80 * risk_pressure - 0.20 * alpha_pressure)
            else:
                adjustments.append(0.0)
        return _stable_softmax(logits + np.asarray(adjustments, dtype=float), self.temperature)

    def _predict_gate_weights(self, state: ControlState) -> tuple[np.ndarray, np.ndarray]:
        """Return (feature_vector, gate_weights) using neural net or default."""
        import torch

        x = self._compute_environment_features(state)
        if self._model is None or self._scaler_mu is None:
            return x, self._default_gate_weights(state)

        x_scaled = (x - self._scaler_mu) / self._scaler_sigma
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        probs = self._model.predict_proba(x_tensor).squeeze(0).numpy()
        probs = np.clip(probs, 1e-8, None)
        probs = np.maximum(probs, self.min_weight)
        probs /= probs.sum()
        return x, probs

    # ------------------------------------------------------------------
    # Neural network training
    # ------------------------------------------------------------------

    def _try_retrain(self, t: int) -> None:
        import torch
        import torch.nn as nn

        if t - self._last_train_t < self.retrain_every:
            return
        if len(self._label_buffer) < self.min_samples:
            return
        y = np.asarray(self._label_buffer, dtype=int)
        if len(np.unique(y)) < 2:
            return

        X = np.asarray(self._feature_buffer, dtype=float)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        X_scaled = (X - mu) / sigma

        n_experts = len(self.expert_names)
        model = _build_attention_gate_network(self._feature_dim, n_experts, self.hidden_layers)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.alpha_reg,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        n = len(X_t)

        # Train/val split for early stopping
        val_size = max(1, int(0.15 * n))
        train_size = n - val_size
        X_train, X_val = X_t[:train_size], X_t[train_size:]
        y_train, y_val = y_t[:train_size], y_t[train_size:]

        batch_size = min(64, train_size)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        model.train()
        for epoch in range(200):
            perm = torch.randperm(train_size)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, train_size, batch_size):
                idx = perm[i:i + batch_size]
                logits = model(X_train[idx])
                loss = criterion(logits, y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Early stopping on validation
            if val_size > 0 and (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_loss = criterion(val_logits, y_val).item()
                model.train()
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._scaler_mu = mu
        self._scaler_sigma = sigma
        self._last_train_t = t

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def compute_invested_fraction(self, state: ControlState) -> float:
        return 1.0  # meta-controller produces direct weights

    def uses_direct_weights(self) -> bool:
        return True

    def get_pending_expert_books(self) -> dict[str, pd.Series]:
        return {name: w.copy() for name, w in self._pending_books.items()}

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
                frac = float(np.clip(expert.compute_invested_fraction(state), 0.0, 1.0))
                expert_books[expert_name] = (constrained_target * frac).clip(lower=0.0)

        x, gate_weights = self._predict_gate_weights(state)

        combined = pd.Series(0.0, index=constrained_target.index, dtype=float)
        gate_weights_map: dict[str, float] = {}
        for expert_name, gw in zip(self.expert_names, gate_weights):
            combined = combined.add(expert_books[expert_name] * float(gw), fill_value=0.0)
            gate_weights_map[expert_name] = float(gw)

        combined = combined.clip(lower=0.0)
        if float(combined.sum()) > 1.0:
            combined = combined / float(combined.sum())

        dominant_idx = int(np.argmax(gate_weights))
        dominant_expert = self.expert_names[dominant_idx]
        self._pending_features = x
        self._pending_gate = gate_weights
        self._pending_books = expert_books
        self._latest_diagnostics.update({
            'mlp_meta_gate_weights': gate_weights_map,
            'mlp_meta_dominant_expert': dominant_expert,
            'mlp_meta_gate_entropy': float(-np.sum(gate_weights * np.log(gate_weights + 1e-8))),
            'mlp_meta_gate_source': 'pytorch' if self._model is not None else 'heuristic',
            'mlp_meta_n_training_samples': len(self._label_buffer),
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
                self._latest_diagnostics['mlp_meta_best_expert'] = best_expert
                self._latest_diagnostics['mlp_meta_expert_rewards'] = {
                    name: float(v) for name, v in expert_feedback.items()
                }
                for expert_name, expert in self.experts.items():
                    expert_reward = float(expert_feedback.get(expert_name, reward))
                    expert.update(state, expert_reward, next_state)
        else:
            dominant_idx = int(np.argmax(self._pending_gate)) if self._pending_gate is not None else 0
            self._feature_buffer.append(self._pending_features)
            self._label_buffer.append(dominant_idx)
            self._latest_diagnostics['mlp_meta_best_expert'] = self.expert_names[dominant_idx]

        self._try_retrain(next_state.t)
        self._pending_features = None
        self._pending_gate = None
        self._pending_books = {}

    def reset(self) -> None:
        self._model = None
        self._scaler_mu = None
        self._scaler_sigma = None
        self._state_history.clear()
        self._feature_buffer.clear()
        self._label_buffer.clear()
        self._pending_features = None
        self._pending_gate = None
        self._pending_books = {}
        for expert in self.experts.values():
            expert.reset()


# ============================================================
# H: Model-Predictive Controller
# ============================================================

class MPCController(BaseController):
    """Receding-horizon controller over exposure and stabilization.

    The MPC keeps the constrained allocator as the alpha-book engine and
    optimizes a short horizon of two controls:
    - invested fraction
    - stabilizer mix into a minimum-variance core

    This yields a structured controller that can respond to alpha strength,
    volatility, drawdown, and regime stress without relying on learned value
    functions.
    """

    label = 'mpc'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.horizon = max(2, int(config.mpc_horizon))
        self.replan_every = max(1, int(config.mpc_replan_every))
        self.discount = float(config.mpc_discount)
        self.alpha_decay = float(config.mpc_alpha_decay)
        self.stress_reversion = float(config.mpc_stress_reversion)
        self.min_invested = float(config.mpc_min_invested)
        self.max_stabilizer = float(config.mpc_max_stabilizer)
        self.risk_penalty = float(config.mpc_risk_penalty)
        self.turnover_penalty = float(config.mpc_turnover_penalty)
        self.drawdown_penalty = float(config.mpc_drawdown_penalty)
        self.stress_penalty = float(config.mpc_stress_penalty)
        self.terminal_penalty = float(config.mpc_terminal_penalty)
        self.max_daily_change = float(config.mpc_max_daily_change)

        self._plan_start_t = -999
        self._plan: list[tuple[float, float]] = []
        self._plan_source: str = 'none'
        self._last_plan_objective: float = 0.0

    def uses_direct_weights(self) -> bool:
        return True

    def compute_invested_fraction(self, state: ControlState) -> float:
        if self._plan:
            step = int(np.clip(state.t - self._plan_start_t, 0, len(self._plan) - 1))
            return float(self._plan[step][0])
        invested, _ = self._initial_controls(state)
        return invested

    def _stress_score(self, state: ControlState) -> float:
        vol_term = max(0.0, state.recent_vol - 0.14) / 0.16
        dd_term = max(0.0, -state.recent_drawdown - 0.04) / 0.10
        regime_term = max(0.0, 0.50 - state.regime_belief) / 0.25
        trend_term = max(0.0, -state.trend) / 0.20
        return float(np.clip(
            0.9 * vol_term + 1.1 * dd_term + 0.8 * regime_term + 0.3 * trend_term,
            0.0,
            3.0,
        ))

    def _initial_controls(self, state: ControlState) -> tuple[float, float]:
        stress = self._stress_score(state)
        alpha_boost = float(np.tanh(6.0 * state.alpha_strength))
        invested = np.clip(
            0.94
            + 0.05 * alpha_boost
            + 0.08 * (state.regime_belief - 0.50)
            - 0.18 * stress
            - 0.10 * max(0.0, -state.recent_drawdown - 0.05),
            self.min_invested,
            1.0,
        )
        stabilizer_mix = np.clip(
            0.08
            + 0.18 * stress
            + 0.30 * max(0.0, -state.recent_drawdown - 0.05)
            + 0.10 * max(0.0, 0.45 - state.regime_belief),
            0.0,
            self.max_stabilizer,
        )
        return float(invested), float(stabilizer_mix)

    def _compose_book(
        self,
        alpha_book: pd.Series,
        stabilizer_book: pd.Series,
        invested_fraction: float,
        stabilizer_mix: float,
    ) -> pd.Series:
        core = (1.0 - stabilizer_mix) * alpha_book + stabilizer_mix * stabilizer_book
        core = core.clip(lower=0.0)
        core_sum = float(core.sum())
        if core_sum < 1e-8:
            core = alpha_book if float(alpha_book.sum()) > 1e-8 else stabilizer_book
            core_sum = float(core.sum())
        core = core / (core_sum + 1e-8)
        return float(invested_fraction) * core

    def _plan_controls(
        self,
        alpha_book: pd.Series,
        stabilizer_book: pd.Series,
        alpha_view: pd.Series,
        cov: np.ndarray,
        prev_book: pd.Series,
        state: ControlState,
    ) -> tuple[list[tuple[float, float]], str, float]:
        from scipy.optimize import minimize as sp_minimize

        horizon = self.horizon
        alpha_vec = alpha_view.reindex(alpha_book.index).fillna(0.0).to_numpy(dtype=float)
        alpha_base = alpha_book.reindex(alpha_book.index).fillna(0.0)
        stabilizer_base = stabilizer_book.reindex(alpha_book.index).fillna(0.0)
        prev_vec = prev_book.reindex(alpha_book.index).fillna(0.0).to_numpy(dtype=float)
        stress0 = self._stress_score(state)
        invested0, stabilizer0 = self._initial_controls(state)

        if self._plan and self._plan_start_t >= 0:
            prior = list(self._plan)
            if len(prior) < horizon:
                prior.extend([prior[-1]] * (horizon - len(prior)))
            shifted = prior[1:horizon] + [prior[min(len(prior) - 1, horizon - 1)]]
            x0 = np.array(
                [step[0] for step in shifted] + [step[1] for step in shifted],
                dtype=float,
            )
        else:
            x0 = np.array([invested0] * horizon + [stabilizer0] * horizon, dtype=float)

        bounds: list[tuple[float, float]] = []
        current_invested = float(np.clip(state.invested_fraction, 0.0, 1.0))
        first_low = max(self.min_invested, current_invested - self.max_daily_change)
        first_high = min(1.0, current_invested + self.max_daily_change)
        bounds.append((first_low, first_high))
        bounds.extend([(self.min_invested, 1.0)] * (horizon - 1))
        bounds.extend([(0.0, self.max_stabilizer)] * horizon)

        def objective(params: np.ndarray) -> float:
            invest_seq = np.clip(params[:horizon], self.min_invested, 1.0)
            mix_seq = np.clip(params[horizon:], 0.0, self.max_stabilizer)
            total = 0.0
            prev = prev_vec.copy()
            for h in range(horizon):
                invest = float(invest_seq[h])
                mix = float(mix_seq[h])
                weights = self._compose_book(alpha_base, stabilizer_base, invest, mix).to_numpy(dtype=float)
                alpha_scale = float(self.alpha_decay ** h)
                stress_scale = 1.0 + 1.25 * stress0 * float(self.stress_reversion ** h)
                expected_ret = alpha_scale * float(alpha_vec @ weights)
                risk = float(weights @ cov @ weights)
                turnover = float(np.sum((weights - prev) ** 2))
                dd_proxy = float(
                    max(0.0, -state.recent_drawdown)
                    * (invest ** 2)
                    * max(0.15, 1.0 - 1.5 * mix)
                    * (self.stress_reversion ** h)
                )
                stress_cap = float(np.clip(
                    0.92 - 0.10 * stress0 + 0.04 * max(0.0, state.regime_belief - 0.50),
                    self.min_invested,
                    1.0,
                ))
                exposure_penalty = float(
                    self.stress_penalty * stress_scale * max(0.0, invest - stress_cap) ** 2
                )
                smooth = 0.0
                if h > 0:
                    smooth = float(
                        (invest - invest_seq[h - 1]) ** 2
                        + 0.5 * (mix - mix_seq[h - 1]) ** 2
                    )
                stage_value = (
                    expected_ret
                    - self.risk_penalty * stress_scale * risk
                    - self.turnover_penalty * turnover
                    - self.drawdown_penalty * dd_proxy
                    - 0.5 * self.stress_penalty * smooth
                    - exposure_penalty
                )
                total -= float((self.discount ** h) * stage_value)
                prev = weights
            terminal_target = self._compose_book(
                alpha_base,
                stabilizer_base,
                float(invest_seq[-1]),
                float(mix_seq[-1]),
            ).to_numpy(dtype=float)
            total += self.terminal_penalty * float(np.sum((prev - terminal_target) ** 2))
            return float(total)

        try:
            result = sp_minimize(
                objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 120, 'ftol': 1e-7},
            )
            if result.success:
                invest_seq = np.clip(result.x[:horizon], self.min_invested, 1.0)
                mix_seq = np.clip(result.x[horizon:], 0.0, self.max_stabilizer)
                plan = [(float(b), float(s)) for b, s in zip(invest_seq, mix_seq)]
                return plan, 'optimized', float(result.fun)
        except Exception:
            pass

        fallback_plan = [(invested0, stabilizer0) for _ in range(horizon)]
        return fallback_plan, 'heuristic', float(objective(x0))

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
        tickers = factor_scores.index
        alpha_book = allocator.optimize_target_book(
            factor_scores=factor_scores,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            control_state=state,
        ).reindex(tickers).fillna(0.0).clip(lower=0.0)
        alpha_book = alpha_book / (float(alpha_book.sum()) + 1e-8)
        stabilizer_book = allocator.estimate_min_var_core(
            tickers, recent_returns, confidence,
        ).reindex(tickers).fillna(0.0)
        stabilizer_book = stabilizer_book / (float(stabilizer_book.sum()) + 1e-8)

        if recent_returns is None or len(recent_returns) < 40:
            invested, stabilizer_mix = self._initial_controls(state)
            weights = self._compose_book(alpha_book, stabilizer_book, invested, stabilizer_mix)
            self._latest_diagnostics.update({
                'mpc_invested_target': invested,
                'mpc_stabilizer_mix': stabilizer_mix,
                'mpc_plan_step': 0,
                'mpc_plan_source': 'heuristic',
                'mpc_plan_objective': 0.0,
                'overlay_size': stabilizer_mix,
            })
            return weights

        aligned_returns = recent_returns.reindex(columns=tickers).fillna(0.0)
        try:
            cov = LedoitWolf().fit(aligned_returns.values).covariance_
        except Exception:
            invested, stabilizer_mix = self._initial_controls(state)
            weights = self._compose_book(alpha_book, stabilizer_book, invested, stabilizer_mix)
            self._latest_diagnostics.update({
                'mpc_invested_target': invested,
                'mpc_stabilizer_mix': stabilizer_mix,
                'mpc_plan_step': 0,
                'mpc_plan_source': 'fallback_cov',
                'mpc_plan_objective': 0.0,
                'overlay_size': stabilizer_mix,
            })
            return weights

        alpha_view = (alpha_scores * confidence).reindex(tickers).fillna(0.0)
        alpha_scale = float(alpha_view.abs().quantile(0.90))
        if alpha_scale > 1e-8:
            alpha_view = 0.0015 * alpha_view / alpha_scale
        else:
            alpha_view = alpha_view * 0.0
        alpha_view = alpha_view.clip(lower=-0.003, upper=0.003)

        prev_book = (
            prev_weights.reindex(tickers).fillna(0.0)
            if prev_weights is not None
            else alpha_book * state.invested_fraction
        )

        needs_replan = (
            not self._plan
            or state.t - self._plan_start_t >= self.replan_every
            or state.t - self._plan_start_t >= len(self._plan)
        )
        if needs_replan:
            self._plan, self._plan_source, self._last_plan_objective = self._plan_controls(
                alpha_book=alpha_book,
                stabilizer_book=stabilizer_book,
                alpha_view=alpha_view,
                cov=cov,
                prev_book=prev_book,
                state=state,
            )
            self._plan_start_t = state.t

        plan_step = int(np.clip(state.t - self._plan_start_t, 0, len(self._plan) - 1))
        invested, stabilizer_mix = self._plan[plan_step]
        weights = self._compose_book(alpha_book, stabilizer_book, invested, stabilizer_mix)
        self._latest_diagnostics.update({
            'mpc_invested_target': float(invested),
            'mpc_stabilizer_mix': float(stabilizer_mix),
            'mpc_plan_step': int(plan_step),
            'mpc_plan_source': self._plan_source,
            'mpc_plan_objective': float(self._last_plan_objective),
            'overlay_size': float(stabilizer_mix),
        })
        return weights

    def reset(self) -> None:
        self._plan_start_t = -999
        self._plan = []
        self._plan_source = 'none'
        self._last_plan_objective = 0.0


# ============================================================
# I: Decision-Aware Allocator
# ============================================================

class AdaptiveAllocatorController(BaseController):
    """Controller that endogenizes allocator parameters from state.

    Instead of only scaling a fixed allocator output, this controller maps
    the current control state to a small allocator parameter vector
    ``theta_t`` and then solves the constrained target-book optimization
    with those adapted settings:

        w_t = Allocator(alpha_t, theta_t),   theta_t = pi(x_t)

    The policy is intentionally compact and interpretable. It adjusts risk,
    anchor, turnover, alpha strength, position caps, and group caps as a
    function of stress, alpha quality, concentration, and trend.
    """

    label = 'adaptive_allocator'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.min_invested = float(config.adaptive_allocator_min_invested)
        self.param_smoothing = float(np.clip(config.adaptive_allocator_param_smoothing, 0.0, 0.95))
        self.risk_mult_range = tuple(config.adaptive_allocator_risk_mult_range)
        self.anchor_mult_range = tuple(config.adaptive_allocator_anchor_mult_range)
        self.turnover_mult_range = tuple(config.adaptive_allocator_turnover_mult_range)
        self.alpha_mult_range = tuple(config.adaptive_allocator_alpha_mult_range)
        self.cap_scale_range = tuple(config.adaptive_allocator_cap_scale_range)
        self.group_cap_scale_range = tuple(config.adaptive_allocator_group_cap_scale_range)
        self.policy_version = int(config.adaptive_allocator_policy_version)
        self._smoothed_theta: dict[str, float] | None = None

    @staticmethod
    def _interp(bounds: tuple[float, float], signal: float) -> float:
        low, high = map(float, bounds)
        clipped = float(np.clip(signal, 0.0, 1.0))
        return low + (high - low) * clipped

    def _stress_score(self, state: ControlState) -> float:
        dd_component = float(np.clip((-state.recent_drawdown - 0.02) / 0.16, 0.0, 1.0))
        vol_component = float(np.clip((state.recent_vol - 0.14) / 0.20, 0.0, 1.0))
        regime_component = float(np.clip((0.52 - state.regime_belief) / 0.42, 0.0, 1.0))
        concentration_component = float(np.clip((state.concentration - 0.16) / 0.20, 0.0, 1.0))
        return float(np.clip(
            0.40 * dd_component
            + 0.28 * vol_component
            + 0.22 * regime_component
            + 0.10 * concentration_component,
            0.0,
            1.0,
        ))

    def _policy(self, state: ControlState, optimizer_config) -> tuple[object, float]:
        stress = self._stress_score(state)
        alpha_boost = float(np.tanh(6.0 * state.alpha_strength))
        trend_signal = float(np.tanh(2.0 * state.trend))
        concentration_penalty = float(np.clip((state.concentration - 0.16) / 0.18, 0.0, 1.0))

        risk_signal = float(np.clip(0.35 + 0.60 * stress + 0.15 * concentration_penalty - 0.18 * alpha_boost, 0.0, 1.0))
        anchor_signal = float(np.clip(0.25 + 0.55 * stress + 0.20 * concentration_penalty - 0.10 * trend_signal, 0.0, 1.0))
        turnover_signal = float(np.clip(0.20 + 0.65 * stress + 0.15 * concentration_penalty, 0.0, 1.0))
        alpha_signal = float(np.clip(0.35 + 0.28 * alpha_boost + 0.12 * trend_signal - 0.20 * stress, 0.0, 1.0))
        cap_signal = float(np.clip(0.62 - 0.35 * stress + 0.28 * alpha_boost - 0.18 * concentration_penalty, 0.0, 1.0))
        group_signal = float(np.clip(0.70 - 0.32 * stress + 0.10 * alpha_boost, 0.0, 1.0))

        raw_theta = {
            'risk_mult': self._interp(self.risk_mult_range, risk_signal),
            'anchor_mult': self._interp(self.anchor_mult_range, anchor_signal),
            'turnover_mult': self._interp(self.turnover_mult_range, turnover_signal),
            'alpha_mult': self._interp(self.alpha_mult_range, alpha_signal),
            'cap_scale': self._interp(self.cap_scale_range, cap_signal),
            'group_cap_scale': self._interp(self.group_cap_scale_range, group_signal),
        }

        if self._smoothed_theta is None:
            theta = raw_theta
        else:
            theta = {
                key: float(
                    self.param_smoothing * self._smoothed_theta[key]
                    + (1.0 - self.param_smoothing) * raw_theta[key]
                )
                for key in raw_theta
            }
        self._smoothed_theta = dict(theta)

        invested = float(np.clip(
            0.94
            + 0.05 * alpha_boost
            + 0.04 * trend_signal
            - 0.22 * stress
            - 0.10 * max(0.0, -state.recent_drawdown - 0.05),
            self.min_invested,
            1.0,
        ))

        max_weight = float(np.clip(
            optimizer_config.max_weight * theta['cap_scale'],
            0.05,
            0.30,
        ))
        group_caps = {
            group: float(np.clip(cap * theta['group_cap_scale'], 0.10, 1.0))
            for group, cap in optimizer_config.group_caps.items()
        }
        adapted_config = replace(
            optimizer_config,
            risk_aversion=float(optimizer_config.risk_aversion * theta['risk_mult']),
            anchor_strength=float(optimizer_config.anchor_strength * theta['anchor_mult']),
            turnover_penalty=float(optimizer_config.turnover_penalty * theta['turnover_mult']),
            alpha_strength=float(optimizer_config.alpha_strength * theta['alpha_mult']),
            max_weight=max_weight,
            group_caps=group_caps,
        )

        self._latest_diagnostics.update({
            'adaptive_allocator_policy_version': self.policy_version,
            'adaptive_allocator_stress_score': float(stress),
            'adaptive_allocator_invested_target': float(invested),
            'adaptive_allocator_risk_mult': float(theta['risk_mult']),
            'adaptive_allocator_anchor_mult': float(theta['anchor_mult']),
            'adaptive_allocator_turnover_mult': float(theta['turnover_mult']),
            'adaptive_allocator_alpha_mult': float(theta['alpha_mult']),
            'adaptive_allocator_cap_scale': float(theta['cap_scale']),
            'adaptive_allocator_group_cap_scale': float(theta['group_cap_scale']),
            'adaptive_allocator_max_weight': float(max_weight),
            'overlay_size': 0.0,
        })
        return adapted_config, invested

    def compute_invested_fraction(self, state: ControlState) -> float:
        return float(np.clip(state.invested_fraction, self.min_invested, 1.0))

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
        adapted_config, invested = self._policy(state, optimizer_config)
        dynamic_target = allocator.optimize_target_book(
            factor_scores=factor_scores,
            alpha_scores=alpha_scores,
            confidence=confidence,
            recent_returns=recent_returns,
            prev_weights=prev_weights,
            optimizer_config=adapted_config,
            adapt_config=False,
        ).reindex(factor_scores.index).fillna(0.0).clip(lower=0.0)
        target_sum = float(dynamic_target.sum())
        if target_sum > 1e-8:
            dynamic_target = dynamic_target / target_sum
        return float(invested) * dynamic_target

    def reset(self) -> None:
        self._smoothed_theta = None


# Q-Learning Controller (wraps existing PortfolioConstructionRL)
# ============================================================

class QLearningController(BaseController):
    """Tabular Q-learning controller — portfolio RL only, minimal state."""

    label = 'q_learning'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.n_actions = 5
        self.alpha = config.ql_alpha
        self.gamma_rl = config.ql_gamma
        self.epsilon = config.ql_epsilon
        self.invest_fractions = [0.82, 0.90, 0.95, 0.98, 1.00]

        self.Q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))
        self._pending_state: tuple | None = None
        self._pending_action: int | None = None

    def _discretize(self, state: ControlState) -> tuple[int, int, int]:
        alpha_bin = int(np.clip(np.digitize(state.alpha_strength, [0.40, 0.80, 1.20, 1.80]), 0, 4))
        dd_bin = int(np.clip(np.digitize(-state.recent_drawdown, [0.03, 0.06, 0.10, 0.16]), 0, 4))
        vol_bin = int(np.clip(np.digitize(state.recent_vol, [0.10, 0.15, 0.20, 0.30]), 0, 4))
        return (alpha_bin, dd_bin, vol_bin)

    def compute_invested_fraction(self, state: ControlState) -> float:
        s = self._discretize(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.Q[s]))
        self._pending_state = s
        self._pending_action = action
        return float(self.invest_fractions[action])

    def update(self, state: ControlState, reward: float, next_state: ControlState) -> None:
        if self._pending_state is not None and self._pending_action is not None:
            s = self._pending_state
            a = self._pending_action
            s_next = self._discretize(next_state)
            best_next = float(np.max(self.Q[s_next]))
            td = reward + self.gamma_rl * best_next - self.Q[s][a]
            self.Q[s][a] += self.alpha * td
            self._pending_state = None
            self._pending_action = None
            # Decay exploration
            self.epsilon = max(0.01, self.epsilon * 0.9998)


# ============================================================
# F: CMDP-Style Lagrangian Controller
# ============================================================

class CMDPLagrangianController(QLearningController):
    """Lightweight constrained-MDP controller with a dual penalty.

    The controller intentionally mirrors the minimal tabular RL setup:
    same discretized state, same invested-fraction actions, and a simple
    Lagrangian update for a downside-risk cost.
    """

    label = 'cmdp_lagrangian'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.constraint_type = config.cmdp_constraint_type
        self.constraint_kappa = float(config.cmdp_constraint_kappa)
        self.lambda_penalty = float(config.cmdp_lambda_init)
        self.lambda_lr = float(config.cmdp_lambda_lr)
        self.tail_loss_threshold = float(config.cmdp_tail_loss_threshold)

    def update(
        self,
        state: ControlState,
        reward: float,
        next_state: ControlState,
        constraint_cost: float | None = None,
        realized_return: float | None = None,
    ) -> None:
        if constraint_cost is None:
            if self.constraint_type == 'tail_loss':
                realized = float(realized_return or 0.0)
                constraint_cost = max(0.0, -realized - self.tail_loss_threshold)
            else:
                constraint_cost = max(0.0, -float(next_state.recent_drawdown))

        violation = float(constraint_cost - self.constraint_kappa)
        penalized_reward = float(reward - self.lambda_penalty * violation)
        super().update(state, penalized_reward, next_state)
        self.lambda_penalty = max(0.0, self.lambda_penalty + self.lambda_lr * violation)
        self._latest_diagnostics.update({
            'cmdp_lambda': float(self.lambda_penalty),
            'cmdp_constraint_cost': float(constraint_cost),
            'cmdp_violation': float(violation),
            'cmdp_constraint_type': self.constraint_type,
        })


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
        'mlp_meta': MLPMetaController,
        'mpc': MPCController,
        'adaptive_allocator': AdaptiveAllocatorController,
        'cmdp_lagrangian': CMDPLagrangianController,
        'q_learning': QLearningController,
    }
    factory = registry.get(config.method)
    if factory is None:
        raise ValueError(f"Unknown control method: {config.method!r}. "
                         f"Choose from: {list(registry.keys())}")
    return factory(config)
