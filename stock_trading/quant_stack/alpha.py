"""Alpha models and signal-combination components for the quant trading pipeline."""

from collections import deque

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from .config import PAIRS_CANDIDATES, UNIVERSE

class FamaFrenchFactors:
    """
    Compute factor exposures and expected returns from:
    - Momentum (12-1 month price momentum)
    - Value (price-to-52w-high as proxy for B/M since we lack fundamentals)
    - Quality (earnings stability = return stability proxy)
    - Low Volatility (inverse realized vol)
    """
    def __init__(self, prices, returns, lookback=252, sec_quality_scores=None):
        self.prices = prices
        self.returns = returns
        self.lookback = lookback
        self.tickers = [t for t in UNIVERSE if t in prices.columns]
        self.sec_quality_scores = sec_quality_scores if sec_quality_scores is not None else pd.Series(dtype=float)

    def compute_momentum(self, date_idx):
        """12-month momentum, skip most recent month (classic Jegadeesh-Titman)."""
        if date_idx < 252:
            return pd.Series(0.0, index=self.tickers)
        ret_12m = self.prices.iloc[date_idx] / self.prices.iloc[date_idx - 252] - 1
        ret_1m = self.prices.iloc[date_idx] / self.prices.iloc[date_idx - 21] - 1
        mom = ret_12m - ret_1m  # skip recent month
        return mom[self.tickers]

    def compute_value(self, date_idx):
        """Price relative to 52-week high (proxy for cheapness)."""
        if date_idx < 252:
            return pd.Series(0.0, index=self.tickers)
        window = self.prices.iloc[max(0, date_idx - 252):date_idx + 1]
        high_52w = window.max()
        current = self.prices.iloc[date_idx]
        # Lower ratio = cheaper = higher value score (invert)
        value = 1 - current / high_52w
        return value[self.tickers]

    def compute_quality(self, date_idx):
        """Return stability as quality proxy (lower vol of returns = higher quality)."""
        if date_idx < 126:
            return pd.Series(0.0, index=self.tickers)
        window = self.returns.iloc[max(0, date_idx - 126):date_idx + 1]
        # Stability = negative skewness of drawdowns (less left-tail = better)
        stability_quality = -window[self.tickers].skew()
        if self.sec_quality_scores.empty:
            return stability_quality.fillna(0)

        sec_quality = self.sec_quality_scores.reindex(self.tickers).fillna(0.0)
        quality = 0.6 * stability_quality.fillna(0) + 0.4 * self._zscore(sec_quality)
        return quality.fillna(0)

    def compute_low_vol(self, date_idx):
        """Low volatility factor: inverse of realized vol."""
        if date_idx < 60:
            return pd.Series(0.0, index=self.tickers)
        window = self.returns.iloc[max(0, date_idx - 60):date_idx + 1]
        vol = window[self.tickers].std()
        low_vol = 1.0 / (vol + 1e-8)
        return low_vol

    def get_factor_scores(self, date_idx):
        """Combine all factors into z-scored alpha signal."""
        mom = self._zscore(self.compute_momentum(date_idx))
        val = self._zscore(self.compute_value(date_idx))
        qual = self._zscore(self.compute_quality(date_idx))
        lvol = self._zscore(self.compute_low_vol(date_idx))

        # Equal-weighted combination
        composite = 0.3 * mom + 0.25 * val + 0.25 * qual + 0.2 * lvol
        return composite, {'momentum': mom, 'value': val, 'quality': qual, 'low_vol': lvol}

    @staticmethod
    def _zscore(series):
        s = series.fillna(0)
        mu, sigma = s.mean(), s.std()
        if sigma < 1e-8:
            return s * 0
        return (s - mu) / sigma


# --- 2B. Statistical Arbitrage (Pairs Trading) ---

class PairsTrading:
    """
    Cointegration-based pairs trading.
    - Test for cointegration using Engle-Granger
    - Trade the spread when it deviates from equilibrium
    - Z-score of spread as the signal
    """
    def __init__(self, prices, lookback=252, refit_every=21):
        self.prices = prices
        self.lookback = lookback
        self.refit_every = refit_every
        self._cached_pairs: dict = {}
        self._last_coint_idx: int | None = None

    def find_cointegrated_pairs(self, date_idx):
        """Test all candidate pairs for cointegration. Refits only every `refit_every` days."""
        if date_idx < self.lookback:
            return {}

        window = self.prices.iloc[date_idx - self.lookback:date_idx + 1]

        # Only re-run expensive cointegration tests periodically
        need_refit = (self._last_coint_idx is None
                      or date_idx - self._last_coint_idx >= self.refit_every)

        if need_refit:
            self._cached_pairs = {}
            for t1, t2 in PAIRS_CANDIDATES:
                if t1 not in window.columns or t2 not in window.columns:
                    continue
                p1, p2 = window[t1].values, window[t2].values
                score, pvalue, _ = coint(p1, p2)
                if pvalue < 0.05:
                    X = add_constant(p2)
                    model = OLS(p1, X).fit()
                    hedge_ratio = model.params[1]
                    spread = p1 - hedge_ratio * p2
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    zscore = (spread[-1] - spread_mean) / (spread_std + 1e-8)
                    self._cached_pairs[(t1, t2)] = {
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'zscore': zscore,
                        'spread': spread,
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                    }
            self._last_coint_idx = date_idx
        else:
            # Only update z-scores using cached hedge ratios (cheap)
            updated = {}
            for (t1, t2), info in self._cached_pairs.items():
                if t1 not in window.columns or t2 not in window.columns:
                    continue
                p1, p2 = window[t1].values, window[t2].values
                spread = p1 - info['hedge_ratio'] * p2
                zscore = (spread[-1] - info['spread_mean']) / (info['spread_std'] + 1e-8)
                updated[(t1, t2)] = {**info, 'zscore': zscore, 'spread': spread}
            self._cached_pairs = updated

        return self._cached_pairs

    def get_pairs_signals(self, date_idx):
        """Convert cointegration z-scores into trading signals."""
        pairs = self.find_cointegrated_pairs(date_idx)
        signals = {}
        for (t1, t2), info in pairs.items():
            z = info['zscore']
            # Mean-reversion: short spread when z > 1.5, long when z < -1.5
            if z > 1.5:
                signals[t1] = -1.0  # short the expensive one
                signals[t2] = info['hedge_ratio']  # long the cheap one
            elif z < -1.5:
                signals[t1] = 1.0
                signals[t2] = -info['hedge_ratio']
            elif abs(z) < 0.5:
                signals[t1] = 0.0  # close position near equilibrium
                signals[t2] = 0.0
        return signals, pairs


# --- 2C. GARCH Volatility Forecasting ---

class GARCHForecaster:
    """
    GARCH(1,1) for volatility forecasting.
    Predicts tomorrow's vol — used for position sizing and risk budgeting.
    """
    def __init__(self, returns, refit_every=21):
        self.returns = returns
        self.refit_every = refit_every
        self.models = {}
        self.last_fit = {}

    def forecast_vol(self, ticker, date_idx, horizon=1):
        """Forecast next-day volatility for a ticker."""
        if date_idx < 252:
            return self.returns[ticker].iloc[:date_idx + 1].std() if date_idx > 10 else 0.01

        # Refit periodically
        need_fit = (ticker not in self.last_fit or
                    date_idx - self.last_fit[ticker] >= self.refit_every)

        if need_fit:
            try:
                rets = self.returns[ticker].iloc[max(0, date_idx - 504):date_idx + 1] * 100
                am = arch_model(rets, vol='Garch', p=1, q=1, dist='normal')
                res = am.fit(disp='off', show_warning=False)
                self.models[ticker] = res
                self.last_fit[ticker] = date_idx
            except Exception:
                return self.returns[ticker].iloc[max(0, date_idx - 60):date_idx + 1].std()

        if ticker in self.models:
            try:
                forecasts = self.models[ticker].forecast(horizon=horizon)
                vol = np.sqrt(forecasts.variance.values[-1, 0]) / 100
                return vol
            except Exception:
                pass

        return self.returns[ticker].iloc[max(0, date_idx - 60):date_idx + 1].std()

    def forecast_all(self, date_idx):
        """Forecast vol for all universe tickers."""
        vols = {}
        for ticker in UNIVERSE:
            if ticker in self.returns.columns:
                vols[ticker] = self.forecast_vol(ticker, date_idx)
        return pd.Series(vols)

    def __init_vol_history(self) -> None:
        if not hasattr(self, '_avg_vol_history'):
            self._avg_vol_history: deque[float] = deque(maxlen=21)

    def record_avg_vol(self, avg_vol: float) -> None:
        """Record the cross-sectional average vol for uncertainty tracking."""
        self.__init_vol_history()
        if np.isfinite(avg_vol) and avg_vol > 0:
            self._avg_vol_history.append(float(avg_vol))

    def forecast_vol_uncertainty(self, date_idx: int, lookback: int = 21) -> float:
        """
        Measure uncertainty in the GARCH vol forecast as the coefficient of
        variation of recent cross-sectional average vol forecasts.
        Uses incrementally recorded values instead of re-calling forecast_vol.
        """
        self.__init_vol_history()
        if len(self._avg_vol_history) < 5:
            return 0.5
        arr = np.array(self._avg_vol_history, dtype=float)
        cv = arr.std() / (arr.mean() + 1e-8)
        return float(np.clip(cv / 0.5, 0.0, 1.0))


# --- 2D. HMM Regime Detection ---

class HMMRegimeDetector:
    """
    Simple 2-state Hidden Markov Model for bull/bear regime detection.
    Uses EM algorithm on market returns.
    """
    def __init__(self, n_states=2, lookback=252, refit_every=5):
        self.n_states = n_states
        self.lookback = lookback
        self.refit_every = refit_every
        self.means = None
        self.stds = None
        self.transition = None
        self.beliefs = None
        self._last_fit_idx: int | None = None
        self._cached_belief: float = 0.5

    def fit(self, returns_series):
        """Fit HMM using simple EM on a window of returns."""
        data = returns_series.values
        n = len(data)
        K = self.n_states

        # Initialize: bull (high mean, low vol), bear (low mean, high vol)
        self.means = np.array([data.mean() + data.std(), data.mean() - data.std()])
        self.stds = np.array([data.std() * 0.8, data.std() * 1.5])
        self.transition = np.array([[0.95, 0.05], [0.10, 0.90]])

        # EM iterations
        for _ in range(20):
            # E-step: vectorized forward pass
            gamma = self._forward_backward(data)

            # M-step (vectorized)
            for k in range(K):
                w = gamma[:, k]
                w_sum = w.sum() + 1e-8
                self.means[k] = (w * data).sum() / w_sum
                self.stds[k] = np.sqrt((w * (data - self.means[k]) ** 2).sum() / w_sum + 1e-8)

                # Vectorized transition update
                emission_next = self._emission_vec(data[1:])  # (n-1, K)
                self.transition[k] = (gamma[:-1, k][:, None] * emission_next).sum(axis=0) + 1e-8
                self.transition[k] /= self.transition[k].sum()

        # Sort so state 0 = bull (higher mean)
        if self.means[0] < self.means[1]:
            self.means = self.means[::-1]
            self.stds = self.stds[::-1]
            self.transition = self.transition[::-1, ::-1]

        self.beliefs = gamma[-1]
        return self

    def _emission_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorized emission: returns (len(x), K) array of emission probs."""
        K = self.n_states
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)
        result = np.empty((len(x), K))
        for k in range(K):
            result[:, k] = sp_stats.norm.pdf(x, self.means[k], self.stds[k]) + 1e-300
        return result

    def _forward_backward(self, data):
        n = len(data)
        K = self.n_states
        alpha = np.zeros((n, K))

        # Precompute all emissions at once: (n, K)
        emissions = self._emission_vec(data)

        # Forward pass — vectorized (no inner Python loop over states)
        alpha[0] = emissions[0] / K
        alpha[0] /= alpha[0].sum() + 1e-300

        for t in range(1, n):
            alpha[t] = emissions[t] * (alpha[t - 1] @ self.transition)
            alpha[t] /= alpha[t].sum() + 1e-300

        return alpha  # use forward probs as approximate posterior

    def get_regime_belief(self, date_idx, market_returns):
        """Return P(bull) at a given date. Refits only every `refit_every` days."""
        if date_idx < self.lookback:
            return 0.5

        # Use cached result if refit not due
        if (self._last_fit_idx is not None
                and date_idx - self._last_fit_idx < self.refit_every
                and self.beliefs is not None):
            return self._cached_belief

        window = market_returns.iloc[max(0, date_idx - self.lookback):date_idx + 1]
        self.fit(window)
        self._last_fit_idx = date_idx
        self._cached_belief = float(self.beliefs[0])
        return self._cached_belief


# --- 2E. LSTM Return Predictor (Pure Numpy) ---

class SimpleLSTM:
    """
    Minimal LSTM implementation in pure numpy for return prediction.
    No PyTorch dependency — educational and portable.
    """
    def __init__(self, input_size, hidden_size=16, lr=0.001):
        self.hidden_size = hidden_size
        self.lr = lr
        scale = 0.1

        # LSTM gates: input, forget, cell, output
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros(hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros(hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros(hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros(hidden_size)

        # Output layer
        self.Wy = np.random.randn(1, hidden_size) * scale
        self.by = np.zeros(1)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward_sequence(self, X):
        """Forward pass through sequence. X shape: (seq_len, input_size)."""
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for t in range(seq_len):
            xh = np.concatenate([X[t], h])
            f = self._sigmoid(self.Wf @ xh + self.bf)
            i = self._sigmoid(self.Wi @ xh + self.bi)
            c_tilde = np.tanh(self.Wc @ xh + self.bc)
            o = self._sigmoid(self.Wo @ xh + self.bo)
            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        y = self.Wy @ h + self.by
        return y[0], h

    def train_step(self, X, target):
        """Simple gradient step using finite differences (educational)."""
        pred, _ = self.forward_sequence(X)
        loss = (pred - target) ** 2

        # Simplified gradient update: perturb output layer only
        # (full BPTT is complex — this keeps it tractable)
        grad_y = 2 * (pred - target)
        _, h = self.forward_sequence(X)

        self.Wy -= self.lr * grad_y * h.reshape(1, -1)
        self.by -= self.lr * grad_y

        return loss

    def predict(self, X):
        pred, _ = self.forward_sequence(X)
        return pred


class LSTMAlpha:
    """LSTM-based return predictor using rolling features."""
    def __init__(self, seq_len=20, hidden_size=16, n_features=5):
        self.seq_len = seq_len
        self.model = SimpleLSTM(input_size=n_features, hidden_size=hidden_size)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, returns, date_idx, ticker):
        """Build feature matrix for one ticker."""
        if date_idx < self.seq_len + 20:
            return None

        ret = returns[ticker].iloc[date_idx - self.seq_len - 20:date_idx + 1]
        features = pd.DataFrame(index=ret.index)
        features['ret'] = ret
        features['ret_5d'] = ret.rolling(5).mean()
        features['vol_10d'] = ret.rolling(10).std()
        features['mom_20d'] = ret.rolling(20).sum()
        features['zscore'] = (ret - ret.rolling(20).mean()) / (ret.rolling(20).std() + 1e-8)
        features = features.dropna()

        if len(features) < self.seq_len:
            return None
        return features.values[-self.seq_len:]

    def train_on_window(self, returns, date_idx, ticker, window=252):
        """Train on a rolling window."""
        start = max(self.seq_len + 20, date_idx - window)
        losses = []

        for t in range(start, date_idx - 1):
            X = self.prepare_features(returns, t, ticker)
            if X is None:
                continue
            target = returns[ticker].iloc[t + 1]
            X_scaled = self.scaler.fit_transform(X)
            loss = self.model.train_step(X_scaled, target * 100)
            losses.append(loss)

        self.is_trained = True
        return np.mean(losses) if losses else float('inf')

    def predict_return(self, returns, date_idx, ticker):
        """Predict next-day return."""
        X = self.prepare_features(returns, date_idx, ticker)
        if X is None or not self.is_trained:
            return 0.0
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled) / 100
        return np.clip(pred, -0.05, 0.05)


# ============================================================
# 3. SIGNAL COMBINATION
# ============================================================

class AlphaCombiner:
    """
    Combine multiple alpha signals into a single expected return vector.
    Adaptively reweights sources based on recent realized signal quality.
    """
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.base_weights = {'factor': 0.55, 'pairs': 0.25, 'lstm': 0.20}
        self.source_skill = {source: 0.0 for source in self.base_weights}
        self.ic_history = {source: deque(maxlen=lookback) for source in self.base_weights}

    def get_source_weights(self, adaptive=True):
        raw = {}
        for source, base_weight in self.base_weights.items():
            skill_scale = np.clip(1.0 + 1.5 * self.source_skill[source], 0.25, 2.0) if adaptive else 1.0
            raw[source] = base_weight * skill_scale
        total = sum(raw.values()) + 1e-8
        return {source: weight / total for source, weight in raw.items()}

    def update_signal_quality(self, source_signals, realized_returns):
        for source, signals in source_signals.items():
            aligned = pd.concat([signals, realized_returns], axis=1).dropna()
            if source != 'factor':
                aligned = aligned[aligned.iloc[:, 0].abs() > 1e-8]
            if len(aligned) < 4:
                continue
            sig = aligned.iloc[:, 0]
            rets = aligned.iloc[:, 1]
            if sig.std() < 1e-8 or rets.std() < 1e-8:
                continue
            ic = sig.corr(rets, method='spearman')
            if np.isnan(ic):
                continue
            ic = float(np.clip(ic, -0.5, 0.5))
            self.ic_history[source].append(ic)
            self.source_skill[source] = 0.85 * self.source_skill[source] + 0.15 * ic

    def get_ic_instability(self) -> float:
        """Average rolling IC volatility across alpha sleeves."""
        instabilities: list[float] = []
        for hist in self.ic_history.values():
            if len(hist) < 5:
                continue
            instabilities.append(float(np.std(np.array(hist, dtype=float))))
        if not instabilities:
            return 0.0
        return float(np.mean(instabilities))

    def combine(
        self,
        factor_scores,
        pairs_signals,
        garch_vols,
        regime_belief,
        lstm_preds,
        tickers,
        use_factor=True,
        use_pairs=True,
        use_lstm=True,
        adaptive=True,
    ):
        """
        Weighted combination of all alpha sources.
        Returns: alpha score per ticker (z-score scale), confidence per ticker.
        """
        alpha = pd.Series(0.0, index=tickers)
        confidence = pd.Series(1.0, index=tickers)
        source_signals = {
            'factor': factor_scores.reindex(tickers).fillna(0.0) if use_factor else pd.Series(0.0, index=tickers),
            'pairs': pd.Series(pairs_signals, dtype=float).reindex(tickers).fillna(0.0) if use_pairs else pd.Series(0.0, index=tickers),
            'lstm': pd.Series(lstm_preds, dtype=float).reindex(tickers).fillna(0.0) * 100 if use_lstm else pd.Series(0.0, index=tickers),
        }
        source_weights = self.get_source_weights(adaptive=adaptive)

        for ticker in tickers:
            signals = []
            weights = []

            # Factor model — z-score directly as alpha
            if use_factor and ticker in factor_scores.index:
                signals.append(source_signals['factor'][ticker])
                weights.append(source_weights['factor'])

            # Pairs signal — already directional (-1, 0, +1 scale)
            if use_pairs and ticker in pairs_signals:
                signals.append(source_signals['pairs'][ticker])
                weights.append(source_weights['pairs'])

            # LSTM — rescale small return prediction to z-score scale
            if use_lstm and ticker in lstm_preds:
                signals.append(source_signals['lstm'][ticker])  # e.g. 0.003 → 0.3
                weights.append(source_weights['lstm'])

            # Regime adjustment: scale down in bear, up in bull
            regime_scale = 0.6 + 0.4 * regime_belief  # 0.6 in bear, 1.0 in bull

            if signals:
                w = np.array(weights)
                w /= w.sum()
                raw_alpha = sum(s * ww for s, ww in zip(signals, w))
                alpha[ticker] = raw_alpha * regime_scale

                # Confidence: inverse-vol weighting (vol-target each name)
                if ticker in garch_vols.index and garch_vols[ticker] > 1e-6:
                    target_vol = 0.15 / np.sqrt(252)  # target ~15% ann. vol per name
                    confidence[ticker] = target_vol / garch_vols[ticker]
                    confidence[ticker] = np.clip(confidence[ticker], 0.3, 3.0)

        return alpha, confidence, source_signals, source_weights


def compute_alpha_opportunity(alpha_scores, confidence, top_k=3):
    """
    Summarize how attractive the current cross-section looks.
    Mean alpha is usually near zero after z-scoring, so use spread instead.
    """
    effective_alpha = (alpha_scores * confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_assets = len(effective_alpha)
    if n_assets == 0:
        return 0.0

    k = max(1, min(top_k, n_assets))
    top_bucket = effective_alpha.nlargest(k).mean()
    bottom_bucket = effective_alpha.nsmallest(k).mean()
    return float(top_bucket - bottom_bucket)
