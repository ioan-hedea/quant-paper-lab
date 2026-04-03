"""Extended controllers kept separate from the core controller module."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ControlConfig
from .controllers import (
    BaseController,
    CVaRRobustController,
    ControlState,
    LinUCBBandit,
    RegimeRulesController,
    _stable_softmax,
)


def _build_attention_gate_network(input_dim: int, n_experts: int, hidden: tuple[int, ...]):
    """Build a PyTorch gate network with self-attention and residual blocks."""
    import torch
    import torch.nn as nn

    class _FeatureAttention(nn.Module):
        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = dim ** -0.5

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, d = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            attn = (q.unsqueeze(-1) @ k.unsqueeze(-2)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v.unsqueeze(-1)).squeeze(-1)
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
            self.log_temperature = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_proj(self.input_norm(x))
            h = h + self.attention(h)
            h = self.attn_norm(h)
            h = self.residual_blocks(h)
            logits = self.head(h)
            temp = self.log_temperature.exp().clamp(min=0.1, max=5.0)
            return logits / temp

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return torch.softmax(self.forward(x), dim=-1)

    return AttentionGateNet(input_dim, n_experts, hidden)


class MLPMetaController(BaseController):
    """MLP-gated meta-controller for environment-adaptive controller selection."""

    label = 'mlp_meta'
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
        self.experts: dict[str, BaseController] = {
            'regime_rules': RegimeRulesController(config),
            'linucb': LinUCBBandit(config),
            'cvar_robust': CVaRRobustController(config),
        }
        for expert_name in self.expert_names:
            if expert_name not in self.experts:
                raise ValueError(f"Unsupported MLP meta expert: {expert_name}")
        self._model = None
        self._feature_history: list[np.ndarray] = []
        self._feature_buffer: list[np.ndarray] = []
        self._label_buffer: list[int] = []
        self._last_train_t: int = -999
        self._pending_features: np.ndarray | None = None
        self._pending_gate: np.ndarray | None = None
        self._pending_books: dict[str, pd.Series] = {}

    def _env_features(self, state: ControlState) -> np.ndarray:
        history = np.asarray(self._feature_history[-self.feature_lookback:], dtype=float)
        if history.size == 0:
            history = np.asarray([
                [
                    state.alpha_strength,
                    state.recent_vol,
                    state.recent_drawdown,
                    state.regime_belief,
                    state.trend,
                    state.concentration,
                ]
            ], dtype=float)
        alpha_hist = history[:, 0]
        vol_hist = history[:, 1]
        dd_hist = history[:, 2]
        regime_hist = history[:, 3]
        trend_hist = history[:, 4]
        conc_hist = history[:, 5]
        alpha_mean = float(alpha_hist.mean())
        alpha_std = float(alpha_hist.std())
        alpha_current_z = float((state.alpha_strength - alpha_mean) / (alpha_std + 1e-8))
        vol_mean = float(vol_hist.mean())
        vol_of_vol = float(vol_hist.std())
        downside_semivol = float(np.sqrt(np.mean(np.square(np.minimum(vol_hist - vol_mean, 0.0))))) if len(vol_hist) > 0 else 0.0
        dd_current = float(state.recent_drawdown)
        dd_mean_depth = float(dd_hist.mean())
        dd_frequency = float((dd_hist < -0.05).mean())
        regime_mean = float(regime_hist.mean())
        regime_switch_freq = float(np.mean(np.abs(np.diff((regime_hist > 0.5).astype(float))))) if len(regime_hist) > 1 else 0.0
        regime_entropy = float(-(regime_mean * np.log(np.clip(regime_mean, 1e-6, 1.0)) + (1.0 - regime_mean) * np.log(np.clip(1.0 - regime_mean, 1e-6, 1.0))) / np.log(2.0))
        regime_risk_off_frac = float((regime_hist < 0.4).mean())
        trend_current = float(state.trend)
        trend_mean = float(trend_hist.mean())
        concentration_mean = float(conc_hist.mean())
        return np.array([
            alpha_mean, alpha_std, alpha_current_z,
            vol_mean, vol_of_vol, downside_semivol,
            dd_current, dd_mean_depth, dd_frequency,
            regime_mean, regime_switch_freq, regime_entropy, regime_risk_off_frac,
            trend_current, trend_mean,
            concentration_mean,
        ], dtype=float)

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
                adjustments.append(0.40 * state.regime_belief + 0.10 * alpha_pressure)
            elif expert_name == 'linucb':
                adjustments.append(0.50 * alpha_pressure - 0.15 * risk_pressure)
            elif expert_name == 'cvar_robust':
                adjustments.append(0.75 * risk_pressure - 0.15 * alpha_pressure)
            else:
                adjustments.append(0.0)
        weights = _stable_softmax(logits + np.asarray(adjustments, dtype=float), self.temperature)
        weights = np.maximum(weights, self.min_weight)
        return weights / (weights.sum() + 1e-8)

    def _predict_gate_weights(self, state: ControlState) -> tuple[np.ndarray, np.ndarray, str]:
        x = self._env_features(state)
        if self._model is None:
            return x, self._default_gate_weights(state), 'heuristic'
        import torch
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        probs = self._model.predict_proba(x_tensor).cpu().numpy()[0]
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        probs = np.maximum(probs, self.min_weight)
        probs /= probs.sum()
        return x, probs, 'mlp'

    def _try_retrain(self, t: int) -> None:
        if t - self._last_train_t < self.retrain_every or len(self._label_buffer) < self.min_samples:
            return
        y = np.asarray(self._label_buffer, dtype=int)
        if len(np.unique(y)) < 2:
            return
        X = np.asarray(self._feature_buffer, dtype=float)
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset, random_split
        model = _build_attention_gate_network(X.shape[1], len(self.expert_names), self.hidden_layers)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        n_val = max(8, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=min(32, n_train), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(64, n_val), shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.alpha_reg)
        best_state = None
        best_val = float('inf')
        patience = 0
        for _ in range(100):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                logits = model(xb)
                n_classes = logits.shape[-1]
                smoothing = 0.05
                true_dist = torch.full_like(logits, smoothing / max(n_classes - 1, 1))
                true_dist.scatter_(1, yb.unsqueeze(1), 1.0 - smoothing)
                loss = torch.sum(-true_dist * F.log_softmax(logits, dim=-1), dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    val_losses.append(float(F.cross_entropy(logits, yb).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model.eval()
        self._last_train_t = t

    def uses_direct_weights(self) -> bool:
        return True

    def get_pending_expert_books(self) -> dict[str, pd.Series]:
        return {name: weights.copy() for name, weights in self._pending_books.items()}

    def build_target_weights(self, allocator, factor_scores, alpha_scores, confidence, recent_returns, prev_weights, optimizer_config, state: ControlState) -> pd.Series | None:
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
        x, gate_weights, gate_source = self._predict_gate_weights(state)
        combined = pd.Series(0.0, index=constrained_target.index, dtype=float)
        gate_map: dict[str, float] = {}
        for expert_name, gate_weight in zip(self.expert_names, gate_weights):
            combined = combined.add(expert_books[expert_name] * float(gate_weight), fill_value=0.0)
            gate_map[expert_name] = float(gate_weight)
        combined = combined.clip(lower=0.0)
        if float(combined.sum()) > 1.0:
            combined = combined / float(combined.sum())
        dominant_idx = int(np.argmax(gate_weights))
        dominant_expert = self.expert_names[dominant_idx]
        self._pending_features = x
        self._pending_gate = gate_weights
        self._pending_books = expert_books
        self._feature_history.append(np.array([
            state.alpha_strength,
            state.recent_vol,
            state.recent_drawdown,
            state.regime_belief,
            state.trend,
            state.concentration,
        ], dtype=float))
        self._latest_diagnostics.update({
            'mlp_meta_gate_weights': gate_map,
            'mlp_meta_dominant_expert': dominant_expert,
            'mlp_meta_gate_entropy': float(-np.sum(gate_weights * np.log(gate_weights + 1e-8))),
            'mlp_meta_gate_source': gate_source,
            'mlp_meta_n_training_samples': len(self._label_buffer),
        })
        return combined

    def update(self, state: ControlState, reward: float, next_state: ControlState, expert_feedback: dict[str, float] | None = None) -> None:
        if self._pending_features is None:
            return
        if expert_feedback:
            best_expert = max(expert_feedback.items(), key=lambda item: item[1])[0]
            if best_expert in self.expert_names:
                self._feature_buffer.append(self._pending_features)
                self._label_buffer.append(self.expert_names.index(best_expert))
                self._latest_diagnostics['mlp_meta_best_expert'] = best_expert
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
        self._feature_history.clear()
        self._feature_buffer.clear()
        self._label_buffer.clear()
        self._pending_features = None
        self._pending_gate = None
        self._pending_books = {}
        for expert in self.experts.values():
            expert.reset()


class MPCController(BaseController):
    label = 'mpc'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.horizon = int(config.mpc_horizon)
        self.replan_every = int(config.mpc_replan_every)
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
        self._plan_source = 'none'
        self._last_plan_objective = 0.0

    def uses_direct_weights(self) -> bool:
        return True

    def _stress_score(self, state: ControlState) -> float:
        dd_term = max(0.0, (-state.recent_drawdown - 0.04) / 0.18)
        vol_term = max(0.0, (state.recent_vol - 0.14) / 0.20)
        regime_term = max(0.0, (0.50 - state.regime_belief) / 0.40)
        trend_term = max(0.0, (-state.trend) / 0.25)
        return float(np.clip(0.9 * vol_term + 1.1 * dd_term + 0.8 * regime_term + 0.3 * trend_term, 0.0, 3.0))

    def _initial_controls(self, state: ControlState) -> tuple[float, float]:
        stress = self._stress_score(state)
        alpha_boost = float(np.tanh(6.0 * state.alpha_strength))
        invested = np.clip(
            0.94 + 0.05 * alpha_boost + 0.08 * (state.regime_belief - 0.50)
            - 0.18 * stress - 0.10 * max(0.0, -state.recent_drawdown - 0.05),
            self.min_invested, 1.0,
        )
        stabilizer_mix = np.clip(
            0.08 + 0.18 * stress + 0.30 * max(0.0, -state.recent_drawdown - 0.05)
            + 0.10 * max(0.0, 0.45 - state.regime_belief),
            0.0, self.max_stabilizer,
        )
        return float(invested), float(stabilizer_mix)

    def _compose_book(self, alpha_book: pd.Series, stabilizer_book: pd.Series, invested_fraction: float, stabilizer_mix: float) -> pd.Series:
        core = (1.0 - stabilizer_mix) * alpha_book + stabilizer_mix * stabilizer_book
        core = core.clip(lower=0.0)
        core_sum = float(core.sum())
        if core_sum < 1e-8:
            core = alpha_book if float(alpha_book.sum()) > 1e-8 else stabilizer_book
            core_sum = float(core.sum())
        core = core / (core_sum + 1e-8)
        return float(invested_fraction) * core

    def _plan_controls(self, alpha_book: pd.Series, stabilizer_book: pd.Series, alpha_view: pd.Series, cov: np.ndarray, prev_book: pd.Series, state: ControlState) -> tuple[list[tuple[float, float]], str, float]:
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
            x0 = np.array([step[0] for step in shifted] + [step[1] for step in shifted], dtype=float)
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
                dd_proxy = float(max(0.0, -state.recent_drawdown) * (invest ** 2) * max(0.15, 1.0 - 1.5 * mix) * (self.stress_reversion ** h))
                stress_cap = float(np.clip(0.92 - 0.10 * stress0 + 0.04 * max(0.0, state.regime_belief - 0.50), self.min_invested, 1.0))
                exposure_penalty = float(self.stress_penalty * stress_scale * max(0.0, invest - stress_cap) ** 2)
                smooth = 0.0
                if h > 0:
                    smooth = float((invest - invest_seq[h - 1]) ** 2 + 0.5 * (mix - mix_seq[h - 1]) ** 2)
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
            terminal_target = self._compose_book(alpha_base, stabilizer_base, float(invest_seq[-1]), float(mix_seq[-1])).to_numpy(dtype=float)
            total += self.terminal_penalty * float(np.sum((prev - terminal_target) ** 2))
            return float(total)

        try:
            result = sp_minimize(objective, x0=x0, method='SLSQP', bounds=bounds, options={'maxiter': 120, 'ftol': 1e-7})
            if result.success:
                invest_seq = np.clip(result.x[:horizon], self.min_invested, 1.0)
                mix_seq = np.clip(result.x[horizon:], 0.0, self.max_stabilizer)
                plan = [(float(b), float(s)) for b, s in zip(invest_seq, mix_seq)]
                return plan, 'optimized', float(result.fun)
        except Exception:
            pass
        fallback_plan = [(invested0, stabilizer0) for _ in range(horizon)]
        return fallback_plan, 'heuristic', float(objective(x0))

    def build_target_weights(self, allocator, factor_scores, alpha_scores, confidence, recent_returns, prev_weights, optimizer_config, state: ControlState) -> pd.Series | None:
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
        stabilizer_book = allocator.estimate_min_var_core(tickers, recent_returns, confidence).reindex(tickers).fillna(0.0)
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
        prev_book = prev_weights.reindex(tickers).fillna(0.0) if prev_weights is not None else alpha_book * state.invested_fraction
        needs_replan = (not self._plan or state.t - self._plan_start_t >= self.replan_every or state.t - self._plan_start_t >= len(self._plan))
        if needs_replan:
            self._plan, self._plan_source, self._last_plan_objective = self._plan_controls(alpha_book, stabilizer_book, alpha_view, cov, prev_book, state)
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


class AdaptiveAllocatorController(BaseController):
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
        return float(np.clip(0.40 * dd_component + 0.28 * vol_component + 0.22 * regime_component + 0.10 * concentration_component, 0.0, 1.0))

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
            theta = {key: float(self.param_smoothing * self._smoothed_theta[key] + (1.0 - self.param_smoothing) * raw_theta[key]) for key in raw_theta}
        self._smoothed_theta = dict(theta)
        invested = float(np.clip(
            0.94 + 0.05 * alpha_boost + 0.04 * trend_signal - 0.22 * stress - 0.10 * max(0.0, -state.recent_drawdown - 0.05),
            self.min_invested,
            1.0,
        ))
        max_weight = float(np.clip(optimizer_config.max_weight * theta['cap_scale'], 0.05, 0.30))
        group_caps = {group: float(np.clip(cap * theta['group_cap_scale'], 0.10, 1.0)) for group, cap in optimizer_config.group_caps.items()}
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

    def build_target_weights(self, allocator, factor_scores, alpha_scores, confidence, recent_returns, prev_weights, optimizer_config, state: ControlState) -> pd.Series | None:
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


class QLearningController(BaseController):
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
            self.epsilon = max(0.01, self.epsilon * 0.9998)


class CMDPLagrangianController(QLearningController):
    label = 'cmdp_lagrangian'

    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.constraint_type = config.cmdp_constraint_type
        self.constraint_kappa = float(config.cmdp_constraint_kappa)
        self.lambda_penalty = float(config.cmdp_lambda_init)
        self.lambda_lr = float(config.cmdp_lambda_lr)
        self.tail_loss_threshold = float(config.cmdp_tail_loss_threshold)

    def update(self, state: ControlState, reward: float, next_state: ControlState, constraint_cost: float | None = None, realized_return: float | None = None) -> None:
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
