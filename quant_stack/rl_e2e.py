"""End-to-end RL baseline kept separate from controller-side RL helpers."""

from __future__ import annotations

from collections import deque
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class EndToEndTradingEnv(gym.Env):
    """Gymnasium environment for the PPO end-to-end trading baseline."""

    metadata = {'render_modes': []}

    def __init__(
        self,
        returns: pd.DataFrame,
        tickers: list[str],
        feature_fn: Callable[[int], np.ndarray],
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
        n_features = feature_fn(start_idx).shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
        )
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
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - action.max())
        weights = exp_a / (exp_a.sum() + 1e-8)

        daily_ret = self.returns[self.tickers].iloc[self._t].values
        portfolio_ret = float(np.dot(weights, daily_ret))

        turnover = float(np.abs(weights - self._prev_weights).sum())
        tx_cost = turnover * self.cost_bps / 10000
        portfolio_ret -= tx_cost

        self._wealth *= (1 + portfolio_ret)
        self._peak = max(self._peak, self._wealth)
        self._recent_returns.append(portfolio_ret)
        self._prev_weights = weights.copy()

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
        elif reward_mode == 'mean_variance':
            if len(self._recent_returns) > 5:
                rets_arr = np.array(self._recent_returns)
                lam = 2.0
                reward = float(portfolio_ret * 100.0 - lam * float(np.var(rets_arr)) * 100.0)
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
    factor_scores_fn: Callable[[int], np.ndarray],
    garch_vols_fn: Callable[[int], np.ndarray],
    regime_belief_fn: Callable[[int], float],
    macro_belief_fn: Callable[[int], float],
    option_features_fn: Callable[[int], dict[str, float]] | None = None,
) -> Callable[[int], np.ndarray]:
    """Build the feature function for the end-to-end PPO baseline."""
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
        factor_scores = _factor_scores(t)
        garch_vols = _garch_vols(t)

        ret_5d = returns[tickers].iloc[max(0, t - 5):t].mean().values
        ret_20d = returns[tickers].iloc[max(0, t - 20):t].mean().values
        vol_20d = returns[tickers].iloc[max(0, t - 20):t].std().values

        regime = regime_belief_fn(t)
        macro = macro_belief_fn(t)
        option_feats = option_features_fn(t) if option_features_fn is not None else {
            'iv_annualized': 0.20,
            'iv_percentile': 0.50,
            'iv_realized_spread': 0.0,
            'iv_regime_score': 0.50,
        }
        p = float(np.clip(regime, 1e-6, 1.0 - 1e-6))
        regime_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))
        alpha_dispersion = float(np.tanh(np.std(factor_scores) / 2.0))
        market_ret_20d = returns[tickers].iloc[max(0, t - 20):t].mean(axis=1).sum()
        market_vol = returns[tickers].iloc[max(0, t - 60):t].mean(axis=1).std() * np.sqrt(252)

        features = np.concatenate([
            factor_scores,
            garch_vols,
            ret_5d,
            ret_20d,
            vol_20d,
            np.array([
                regime,
                macro,
                market_ret_20d,
                market_vol,
                alpha_dispersion,
                regime_entropy,
                float(ic_instability[int(np.clip(t, 0, n_obs - 1))]),
                float(option_feats.get('iv_annualized', 0.20)),
                float(option_feats.get('iv_percentile', 0.50)),
                float(option_feats.get('iv_realized_spread', 0.0)),
                float(option_feats.get('iv_regime_score', 0.50)),
            ]),
        ])
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    return feature_fn


def run_e2e_baseline(
    returns: pd.DataFrame,
    tickers: list[str],
    feature_fn: Callable[[int], np.ndarray],
    train_start: int,
    train_end: int,
    test_end: int,
    cost_bps: float = 5.0,
    risk_free_rate: float = 0.035,
    reward_mode: str = 'differential_sharpe',
    total_timesteps: int = 50_000,
    verbose: int = 0,
    log_interval: int = 10,
) -> dict[str, object]:
    """Train PPO on the training period and evaluate on the test period."""
    from stable_baselines3 import PPO

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

    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=min(256, train_end - train_start - 1),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, log_interval=max(1, log_interval))

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
        obs, _reward, terminated, truncated, info = test_env.step(action)
        wealth_path.append(info['wealth'])
        daily_returns.append(info['portfolio_ret'])
        done = terminated or truncated

    return {
        'wealth': wealth_path,
        'daily_returns': daily_returns,
        'model': model,
    }
