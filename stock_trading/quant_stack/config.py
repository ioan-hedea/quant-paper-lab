"""Configuration constants for the quant trading pipeline."""

import os
from dataclasses import dataclass, field

# ============================================================
# Configuration
# ============================================================

# Broader cross-section: growth, cyclicals, defensives, and diversifiers.
# This gives the alpha layer more relative-value opportunity and gives the
# portfolio layer lower-correlation assets to rotate into when risk rises.
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
    'JPM', 'GS', 'XOM', 'CVX',
    'JNJ', 'PG', 'KO', 'PEP', 'WMT',
    'GLD', 'TLT', 'XLU', 'VNQ',
]
BENCHMARK = 'SPY'
PAIRS_CANDIDATES = [
    ('KO', 'PEP'),
    ('AAPL', 'MSFT'),
    ('GOOGL', 'META'),
    ('JPM', 'GS'),
    ('XOM', 'CVX'),
]
LSTM_TICKERS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
DATA_PERIOD = '5y'
RISK_FREE_RATE = 0.035  # approximate
FRED_SERIES = {
    'rate_10y': 'DGS10',
    'rate_2y': 'DGS2',
    'fed_funds': 'FEDFUNDS',
    'unrate': 'UNRATE',
    'vix': 'VIXCLS',
    'hy_oas': 'BAMLH0A0HYM2',
    'dxy': 'DTWEXBGS',
}
TREASURY_SECURITIES = {
    'treasury_bill_rate': 'Treasury Bills',
    'treasury_note_rate': 'Treasury Notes',
    'treasury_bond_rate': 'Treasury Bonds',
}
BLS_SERIES = {
    'unemployment_rate': 'LNS14000000',
    'cpi_all_items': 'CUUR0000SA0',
}
ASSET_GROUPS = {
    'growth': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    'cyclical': ['JPM', 'GS', 'XOM', 'CVX'],
    'defensive': ['JNJ', 'PG', 'KO', 'PEP', 'WMT'],
    'diversifier': ['GLD', 'TLT', 'XLU', 'VNQ'],
}
TICKER_TO_GROUP = {
    ticker: group
    for group, tickers in ASSET_GROUPS.items()
    for ticker in tickers
}
SEC_USER_AGENT = os.getenv(
    'SEC_USER_AGENT',
    'sequential-decision-making/1.0 research@local'
)


@dataclass
class FeatureAvailabilityConfig:
    """Simple research-time availability assumptions for external features."""

    macro_lag_days: int = 3
    allow_static_sec_quality: bool = True
    sec_quality_note: str = 'Static SEC snapshot used as research prior, not point-in-time safe.'


@dataclass
class CostModelConfig:
    """Transaction-cost and slippage assumptions for backtests."""

    base_cost_bps: float = 5.0
    turnover_vol_multiplier: float = 0.20
    size_penalty_bps: float = 7.5


@dataclass
class OptimizerConfig:
    """Controls for the intermediate constrained allocator."""

    use_optimizer: bool = True
    max_weight: float = 0.18
    risk_aversion: float = 4.0
    alpha_strength: float = 1.5
    anchor_strength: float = 10.0
    turnover_penalty: float = 2.0
    group_caps: dict[str, float] = field(
        default_factory=lambda: {
            'growth': 0.50,
            'cyclical': 0.32,
            'defensive': 0.38,
            'diversifier': 0.38,
        }
    )


@dataclass
class ExperimentConfig:
    """Feature toggles used for ablation studies."""

    label: str = 'full_pipeline'
    use_factor: bool = True
    use_pairs: bool = True
    use_lstm: bool = True
    adaptive_combiner: bool = True
    use_portfolio_rl: bool = True
    use_hedge_rl: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for a single backtest experiment."""

    train_frac: float = 0.5
    rebalance_band: float = 0.015
    min_turnover: float = 0.08
    hedge_ratios: tuple[float, ...] = (0.0, 0.03, 0.08, 0.15)
    portfolio_reward_mode: str = 'differential_sharpe'
    hedge_reward_mode: str = 'asymmetric_return'
    e2e_reward_mode: str = 'differential_sharpe'
    feature_availability: FeatureAvailabilityConfig = field(default_factory=FeatureAvailabilityConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass
class EvaluationConfig:
    """Grid settings for the research evaluation engine."""

    train_fracs: tuple[float, ...] = (0.4, 0.5, 0.6)
    rolling_train_frac: float = 0.5
    rolling_window_days: int = 504
    rolling_step_days: int = 126
    min_rolling_windows: int = 4
    max_rolling_windows: int = 6
    cost_bps_grid: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
    rebalance_band_grid: tuple[float, ...] = (0.005, 0.015, 0.03)
    hedge_scale_grid: tuple[float, ...] = (0.75, 1.0, 1.25)
    macro_lag_grid: tuple[int, ...] = (1, 3, 5)
    reward_mode_grid: tuple[str, ...] = ('differential_sharpe', 'return', 'sortino')
    bootstrap_samples: int = 400
    bootstrap_block_size: int = 20
    bootstrap_seed: int = 7
