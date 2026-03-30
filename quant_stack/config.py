"""Configuration constants for the quant trading pipeline."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

# ============================================================
# Universe Definitions
# ============================================================
# UNIVERSE_CORE: original 19-stock universe (fast runs, development)
# UNIVERSE_EXPANDED: ~75 stocks across 11 GICS sectors (publication runs)
# Switch by setting UNIVERSE = UNIVERSE_CORE or UNIVERSE_EXPANDED below.

UNIVERSE_CORE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
    'JPM', 'GS', 'XOM', 'CVX',
    'JNJ', 'PG', 'KO', 'PEP', 'WMT',
    'GLD', 'TLT', 'XLU', 'VNQ',
]

UNIVERSE_EXPANDED = list(dict.fromkeys([
    # Technology (8)
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO',
    # Financials (8)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
    # Healthcare (8)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'BMY',
    # Consumer Staples (8)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS',
    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    # Industrials (8)
    'CAT', 'BA', 'HON', 'UNP', 'UPS', 'GE', 'MMM', 'LMT',
    # Materials (4)
    'LIN', 'APD', 'SHW', 'FCX',
    # Real Estate (4)
    'AMT', 'PLD', 'PSA', 'O',
    # Utilities (4)
    'NEE', 'DUK', 'SO', 'XEL',
    # Communications (6)
    'META', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX',
    # Consumer Discretionary (7)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX',
    # Cross-asset diversifiers (3)
    'GLD', 'TLT', 'VNQ',
]))

# ---- Second universe (B) for cross-universe validation ----
# Deliberately minimal overlap with UNIVERSE_EXPANDED to test
# whether controller rankings transfer to a different stock set.
# Tilts: mid-cap growth, biotech, REITs, commodity-linked, small-cap value.
UNIVERSE_B = list(dict.fromkeys([
    # Technology — mid-cap / different names
    'CRM', 'ADBE', 'NOW', 'PANW', 'SNPS', 'CDNS', 'MRVL', 'FTNT',
    # Biotech / Pharma — higher-vol healthcare
    'GILD', 'REGN', 'VRTX', 'ISRG', 'DXCM', 'IDXX', 'ZTS', 'MRNA',
    # Financials — insurance + regional / fintech
    'AIG', 'TRV', 'MET', 'ALL', 'BK', 'SCHW', 'ICE', 'CME',
    # Consumer — different mix
    'TGT', 'ROST', 'ORLY', 'AZO', 'DG', 'DLTR', 'YUM', 'CMG',
    # Industrials — mid-cap, transport, aerospace
    'FDX', 'NSC', 'CSX', 'WM', 'RSG', 'ITW', 'EMR', 'ROK',
    # Energy — midstream + services
    'WMB', 'KMI', 'OKE', 'ET', 'HAL', 'BKR', 'DVN', 'FANG',
    # Materials / Mining
    'NEM', 'GOLD', 'NUE', 'STLD', 'CF', 'MOS', 'ALB', 'ECL',
    # REITs — broader than EXPANDED
    'EQIX', 'DLR', 'SPG', 'WELL', 'AVB', 'EQR', 'VICI', 'IRM',
    # Utilities — different names
    'AEP', 'D', 'SRE', 'ED', 'WEC', 'ES', 'EXC', 'PCG',
    # Cross-asset diversifiers
    'SLV', 'IEF', 'HYG',
]))

PAIRS_CANDIDATES_B = [
    # Tech
    ('CRM', 'NOW'), ('ADBE', 'CRM'), ('PANW', 'FTNT'), ('SNPS', 'CDNS'),
    # Biotech
    ('GILD', 'REGN'), ('VRTX', 'MRNA'), ('ISRG', 'IDXX'),
    # Financials
    ('AIG', 'MET'), ('TRV', 'ALL'), ('ICE', 'CME'), ('BK', 'SCHW'),
    # Consumer
    ('TGT', 'ROST'), ('ORLY', 'AZO'), ('DG', 'DLTR'), ('YUM', 'CMG'),
    # Industrials
    ('FDX', 'NSC'), ('CSX', 'NSC'), ('WM', 'RSG'), ('ITW', 'EMR'),
    # Energy
    ('WMB', 'KMI'), ('OKE', 'ET'), ('HAL', 'BKR'), ('DVN', 'FANG'),
    # Materials
    ('NEM', 'GOLD'), ('NUE', 'STLD'), ('CF', 'MOS'),
    # REITs
    ('EQIX', 'DLR'), ('SPG', 'WELL'), ('AVB', 'EQR'),
    # Cross-asset
    ('SLV', 'IEF'),
]

LSTM_TICKERS_B = [
    'CRM', 'NOW', 'PANW', 'GILD', 'REGN',
    'TRV', 'CME', 'SCHW',
    'DVN', 'FANG',
    'NEM', 'EQIX',
    'SLV', 'IEF',
]

ASSET_GROUPS_B = {
    'technology': ['CRM', 'ADBE', 'NOW', 'PANW', 'SNPS', 'CDNS', 'MRVL', 'FTNT'],
    'biotech': ['GILD', 'REGN', 'VRTX', 'ISRG', 'DXCM', 'IDXX', 'ZTS', 'MRNA'],
    'financials': ['AIG', 'TRV', 'MET', 'ALL', 'BK', 'SCHW', 'ICE', 'CME'],
    'consumer': ['TGT', 'ROST', 'ORLY', 'AZO', 'DG', 'DLTR', 'YUM', 'CMG'],
    'industrials': ['FDX', 'NSC', 'CSX', 'WM', 'RSG', 'ITW', 'EMR', 'ROK'],
    'energy': ['WMB', 'KMI', 'OKE', 'ET', 'HAL', 'BKR', 'DVN', 'FANG'],
    'materials': ['NEM', 'GOLD', 'NUE', 'STLD', 'CF', 'MOS', 'ALB', 'ECL'],
    'reits': ['EQIX', 'DLR', 'SPG', 'WELL', 'AVB', 'EQR', 'VICI', 'IRM'],
    'utilities': ['AEP', 'D', 'SRE', 'ED', 'WEC', 'ES', 'EXC', 'PCG'],
    'diversifier': ['SLV', 'IEF', 'HYG'],
}

# ---- Active universe (change this line to switch) ----
UNIVERSE = UNIVERSE_EXPANDED

BENCHMARK = 'SPY'

# ---- Pairs candidates ----
PAIRS_CANDIDATES_CORE = [
    ('KO', 'PEP'), ('AAPL', 'MSFT'), ('GOOGL', 'META'),
    ('JPM', 'GS'), ('XOM', 'CVX'),
]

PAIRS_CANDIDATES_EXPANDED = [
    # Tech
    ('AAPL', 'MSFT'), ('GOOGL', 'META'), ('INTC', 'AMD'), ('QCOM', 'AVGO'),
    ('NVDA', 'AMD'),
    # Financials
    ('JPM', 'GS'), ('BAC', 'WFC'), ('MS', 'GS'), ('USB', 'PNC'), ('JPM', 'BAC'),
    # Healthcare
    ('JNJ', 'PFE'), ('ABBV', 'BMY'), ('UNH', 'TMO'), ('MRK', 'LLY'),
    # Consumer Staples
    ('KO', 'PEP'), ('PG', 'CL'), ('WMT', 'COST'), ('KMB', 'GIS'),
    # Energy
    ('XOM', 'CVX'), ('COP', 'EOG'), ('MPC', 'PSX'), ('MPC', 'VLO'), ('SLB', 'EOG'),
    # Industrials
    ('CAT', 'HON'), ('UNP', 'UPS'), ('BA', 'LMT'), ('GE', 'HON'),
    # Communications
    ('VZ', 'T'), ('DIS', 'CMCSA'), ('META', 'NFLX'),
    # Consumer Discretionary
    ('HD', 'LOW'), ('MCD', 'SBUX'), ('NKE', 'SBUX'),
    # Cross-sector
    ('GLD', 'TLT'), ('XOM', 'GLD'),
]

PAIRS_CANDIDATES = (
    PAIRS_CANDIDATES_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else PAIRS_CANDIDATES_CORE
)

# ---- LSTM tickers ----
LSTM_TICKERS_CORE = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
LSTM_TICKERS_EXPANDED = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META',
    'JPM', 'GS', 'BAC',
    'XOM', 'CVX',
    'JNJ', 'UNH', 'LLY',
    'GLD', 'TLT',
]
LSTM_TICKERS = (
    LSTM_TICKERS_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else LSTM_TICKERS_CORE
)

# ---- Backtest window ----
DATA_PERIOD = '13y'  # 2013–2026: taper tantrum, Brexit, COVID, inflation shock

RISK_FREE_RATE = 0.035

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

# ---- Sector groups ----
ASSET_GROUPS_CORE = {
    'growth': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    'cyclical': ['JPM', 'GS', 'XOM', 'CVX'],
    'defensive': ['JNJ', 'PG', 'KO', 'PEP', 'WMT'],
    'diversifier': ['GLD', 'TLT', 'XLU', 'VNQ'],
}

ASSET_GROUPS_EXPANDED = {
    'technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO'],
    'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'BMY'],
    'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
    'industrials': ['CAT', 'BA', 'HON', 'UNP', 'UPS', 'GE', 'MMM', 'LMT'],
    'materials': ['LIN', 'APD', 'SHW', 'FCX'],
    'real_estate': ['AMT', 'PLD', 'PSA', 'O'],
    'utilities': ['NEE', 'DUK', 'SO', 'XEL'],
    'communications': ['META', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX'],
    'consumer_disc': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX'],
    'diversifier': ['GLD', 'TLT', 'VNQ'],
}

ASSET_GROUPS = (
    ASSET_GROUPS_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else ASSET_GROUPS_CORE
)
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
    """Transaction-cost and slippage assumptions for backtests.

    When ``use_almgren_chriss`` is True the cost model uses a permanent +
    temporary market-impact formula calibrated from Almgren & Chriss (2000).
    Otherwise falls back to the original base-bps + vol-scaling model.
    """

    base_cost_bps: float = 5.0
    turnover_vol_multiplier: float = 0.20
    size_penalty_bps: float = 7.5
    use_almgren_chriss: bool = True
    ac_permanent_beta: float = 0.10
    ac_temporary_eta: float = 0.50


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
            group: 0.40 for group in (
                ASSET_GROUPS_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
                else ASSET_GROUPS_CORE
            )
        }
    )


@dataclass
class OptionOverlayConfig:
    """Controls for the option-based hedge sleeve."""

    use_option_overlay: bool = True
    hedge_types: tuple[str, ...] = ('protective_put', 'collar', 'put_spread')
    target_dte_days: int = 21
    put_strike_otm: float = 0.04
    call_strike_otm: float = 0.05
    spread_width: float = 0.10
    max_effective_hedge: float = 0.35
    theta_premium_scale: float = 0.55
    convexity_scale: float = 1.15
    collar_financing_ratio: float = 0.65


# ============================================================
# Control Method Registry
# ============================================================
# All control candidates from architecture revision v2.
CONTROL_METHODS = (
    'none',                  # No control — factor-only baseline
    'fixed',                 # A1: Fixed allocator (constant invested fraction)
    'vol_target',            # A2: Volatility targeting
    'dd_delever',            # A3: Drawdown-based deleveraging
    'regime_rules',          # A4: Regime-conditioned exposure rules
    'ensemble_rules',        # A5: Simple ensemble of A2–A4
    'linucb',                # B1: Contextual bandit — LinUCB
    'thompson',              # B2: Contextual bandit — Thompson Sampling
    'epsilon_greedy',        # B3: Contextual bandit — epsilon-greedy linear
    'supervised',            # C:  Supervised regime-conditioned controller
    'cvar_robust',           # D:  CVaR-aware robust optimization
    'council',               # E:  Expert-gated council meta-controller
    'mlp_meta',              # G:  MLP-gated meta-controller (environment-adaptive)
    'mpc',                   # H:  Model-predictive control over allocator exposure and stabilization
    'cmdp_lagrangian',       # F:  Simple constrained-MDP controller
    'q_learning',            # RL: Tabular Q-learning (portfolio only)
    'ppo',                   # RL: End-to-end PPO
)


@dataclass
class ControlConfig:
    """Configuration for the pluggable control layer."""

    method: str = 'none'
    # A1: Fixed allocator
    fixed_invested_fraction: float = 0.95
    # A2: Vol-target
    vol_target_annual: float = 0.15
    vol_lookback: int = 63
    # A3: DD-delever
    dd_thresholds: tuple[tuple[float, float], ...] = ((-0.05, 1.0), (-0.08, 0.70), (-0.12, 0.40))
    dd_min_invested: float = 0.30
    # A4: Regime rules
    regime_bull_threshold: float = 0.70
    regime_bear_threshold: float = 0.30
    regime_bull_fraction: float = 1.00
    regime_neutral_fraction: float = 0.90
    regime_bear_fraction: float = 0.75
    # A5: Ensemble
    ensemble_mode: str = 'mean'  # 'mean' or 'min'
    # B: Bandit
    bandit_n_actions: int = 5
    bandit_reward_window: int = 5
    bandit_alpha_ucb: float = 1.0
    bandit_epsilon: float = 0.10
    bandit_feature_lookback: int = 252
    # C: Supervised controller
    supervised_model: str = 'logistic'  # 'logistic', 'random_forest', 'decision_tree'
    supervised_retrain_every: int = 63
    supervised_label_window: int = 21
    # D: CVaR-robust
    cvar_confidence: float = 0.95
    cvar_n_scenarios: int = 500
    cvar_lambda_base: float = 1.0
    cvar_regime_scaling: bool = True
    cvar_dd_budget: bool = True
    # E: Expert-gated council
    council_experts: tuple[str, ...] = ('regime_rules', 'linucb', 'cvar_robust')
    council_gate_model: str = 'logistic'
    council_retrain_every: int = 63
    council_min_samples: int = 40
    council_temperature: float = 1.0
    council_min_weight: float = 0.10
    council_default_bias: tuple[float, ...] = (0.20, 0.20, 0.60)
    # H: Model-predictive controller
    mpc_horizon: int = 5
    mpc_replan_every: int = 5
    mpc_discount: float = 0.92
    mpc_alpha_decay: float = 0.88
    mpc_stress_reversion: float = 0.82
    mpc_min_invested: float = 0.35
    mpc_max_stabilizer: float = 0.45
    mpc_risk_penalty: float = 8.0
    mpc_turnover_penalty: float = 2.5
    mpc_drawdown_penalty: float = 6.0
    mpc_stress_penalty: float = 1.5
    mpc_terminal_penalty: float = 0.75
    mpc_max_daily_change: float = 0.18
    mpc_objective_version: int = 2
    # F: CMDP-style constrained controller
    cmdp_constraint_type: str = 'drawdown'  # 'drawdown' or 'tail_loss'
    cmdp_constraint_kappa: float = 0.12
    cmdp_lambda_init: float = 1.0
    cmdp_lambda_lr: float = 0.05
    cmdp_tail_loss_threshold: float = 0.01
    # Convexity-aware payoff shaping
    convexity_enabled: bool = False
    convexity_threshold: float = 0.0
    convexity_mode_carries: tuple[float, float, float] = (0.0, 0.00015, 0.00040)
    convexity_mode_lambdas: tuple[float, float, float] = (0.0, 1.50, 4.00)
    convexity_mild_drawdown: float = -0.05
    convexity_strong_drawdown: float = -0.10
    convexity_mild_vol: float = 0.18
    convexity_strong_vol: float = 0.26
    convexity_mild_regime: float = 0.45
    convexity_strong_regime: float = 0.30
    # G: MLP Meta-controller (environment-adaptive controller selection)
    mlp_meta_experts: tuple[str, ...] = ('regime_rules', 'linucb', 'cvar_robust')
    mlp_meta_hidden_layers: tuple[int, ...] = (32, 16)
    mlp_meta_retrain_every: int = 63
    mlp_meta_min_samples: int = 60
    mlp_meta_feature_lookback: int = 63
    mlp_meta_min_weight: float = 0.10
    mlp_meta_default_bias: tuple[float, ...] = (0.20, 0.20, 0.60)
    mlp_meta_learning_rate: float = 0.001
    mlp_meta_alpha_reg: float = 0.001
    mlp_meta_temperature: float = 1.0
    # Q-learning (portfolio only)
    ql_alpha: float = 0.03
    ql_gamma: float = 0.95
    ql_epsilon: float = 0.15


@dataclass
class ExperimentConfig:
    """Feature toggles used for ablation studies."""

    label: str = 'full_pipeline'
    use_factor: bool = True
    use_pairs: bool = False
    use_lstm: bool = False
    adaptive_combiner: bool = True
    use_portfolio_rl: bool = True
    use_hedge_rl: bool = False
    # State-feature ablation toggles
    use_uncertainty_state: bool = False
    use_regime_state: bool = False
    use_vol_state: bool = False
    # Control method (new architecture v2)
    control_method: str = 'none'


@dataclass
class PipelineConfig:
    """Top-level configuration for a single backtest experiment."""

    train_frac: float = 0.5
    rebalance_band: float = 0.015
    min_turnover: float = 0.08
    hedge_ratios: tuple[float, ...] = (0.0, 0.03, 0.08, 0.15)
    portfolio_reward_mode: str = 'sortino'
    hedge_reward_mode: str = 'asymmetric_return'
    e2e_reward_mode: str = 'differential_sharpe'
    enable_e2e_baseline: bool = True
    e2e_ppo_verbose: int = 0
    e2e_ppo_log_interval: int = 10
    feature_availability: FeatureAvailabilityConfig = field(default_factory=FeatureAvailabilityConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    option_overlay: OptionOverlayConfig = field(default_factory=OptionOverlayConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


@dataclass
class EvaluationConfig:
    """Grid settings for the research evaluation engine."""

    output_dir: str = 'results'
    timestamp_outputs: bool = True
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
    reward_mode_grid: tuple[str, ...] = ('differential_sharpe', 'return', 'sortino', 'mean_variance')
    research_e2e_scope: str = 'baseline_only'
    enable_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints/research_runs'
    checkpoint_match_mode: str = 'compatible'
    bootstrap_samples: int = 400
    bootstrap_block_size: int = 20
    bootstrap_seed: int = 7
    # Cross-universe meta-learning evaluation
    meta_learning_enabled: bool = False
    meta_learning_universes: tuple[str, ...] = ('A', 'B')
    # Time-series cross-validation
    ts_cv_folds: int = 5
    enable_ts_cv: bool = True


@dataclass
class UniverseProfile:
    """Bundles all universe-specific settings for cross-universe evaluation."""

    label: str
    tickers: list[str]
    pairs: list[tuple[str, str]]
    lstm_tickers: list[str]
    asset_groups: dict[str, list[str]]


def get_universe_profile(universe_id: str) -> UniverseProfile:
    """Return a complete universe profile by ID ('A' or 'B')."""
    if universe_id == 'A':
        return UniverseProfile(
            label='A',
            tickers=list(UNIVERSE_EXPANDED),
            pairs=list(PAIRS_CANDIDATES_EXPANDED),
            lstm_tickers=list(LSTM_TICKERS_EXPANDED),
            asset_groups=dict(ASSET_GROUPS_EXPANDED),
        )
    if universe_id == 'B':
        return UniverseProfile(
            label='B',
            tickers=list(UNIVERSE_B),
            pairs=list(PAIRS_CANDIDATES_B),
            lstm_tickers=list(LSTM_TICKERS_B),
            asset_groups=dict(ASSET_GROUPS_B),
        )
    raise ValueError(f"Unknown universe ID: {universe_id!r}. Choose 'A' or 'B'.")


@contextmanager
def use_universe(universe_id: str):
    """Temporarily swap the active universe across all quant_stack modules.

    This context manager patches the module-level UNIVERSE,
    PAIRS_CANDIDATES, LSTM_TICKERS, ASSET_GROUPS, and TICKER_TO_GROUP
    constants in every module that imports them, so that
    ``load_market_data()`` and ``run_full_pipeline()`` operate on the
    requested universe without permanent side effects.

    Usage::

        with use_universe('B'):
            prices, volumes, returns, macro, sec = load_market_data()
            results = run_full_pipeline(prices, volumes, returns, ...)
    """
    import importlib
    import quant_stack.config as _cfg
    import quant_stack.data as _data
    import quant_stack.pipeline as _pipeline
    import quant_stack.alpha as _alpha

    profile = get_universe_profile(universe_id)
    new_ticker_to_group = {
        ticker: group
        for group, tickers in profile.asset_groups.items()
        for ticker in tickers
    }

    # Save originals
    saved = {
        'cfg_UNIVERSE': _cfg.UNIVERSE,
        'cfg_PAIRS': getattr(_cfg, 'PAIRS_CANDIDATES', None),
        'cfg_LSTM': getattr(_cfg, 'LSTM_TICKERS', None),
        'cfg_ASSET_GROUPS': getattr(_cfg, 'ASSET_GROUPS', None),
        'cfg_TICKER_TO_GROUP': getattr(_cfg, 'TICKER_TO_GROUP', None),
        'data_UNIVERSE': _data.UNIVERSE,
        'pipeline_UNIVERSE': _pipeline.UNIVERSE,
        'alpha_UNIVERSE': _alpha.UNIVERSE,
        'alpha_PAIRS': _alpha.PAIRS_CANDIDATES,
        'alpha_TICKER_TO_GROUP': _alpha.TICKER_TO_GROUP,
    }

    try:
        # Patch config module
        _cfg.UNIVERSE = profile.tickers
        _cfg.PAIRS_CANDIDATES = profile.pairs
        _cfg.LSTM_TICKERS = profile.lstm_tickers
        _cfg.ASSET_GROUPS = profile.asset_groups
        _cfg.TICKER_TO_GROUP = new_ticker_to_group

        # Patch downstream modules that imported at module level
        _data.UNIVERSE = profile.tickers
        _pipeline.UNIVERSE = profile.tickers
        _alpha.UNIVERSE = profile.tickers
        _alpha.PAIRS_CANDIDATES = profile.pairs
        _alpha.TICKER_TO_GROUP = new_ticker_to_group

        yield profile
    finally:
        # Restore originals
        _cfg.UNIVERSE = saved['cfg_UNIVERSE']
        _cfg.PAIRS_CANDIDATES = saved['cfg_PAIRS']
        _cfg.LSTM_TICKERS = saved['cfg_LSTM']
        _cfg.ASSET_GROUPS = saved['cfg_ASSET_GROUPS']
        _cfg.TICKER_TO_GROUP = saved['cfg_TICKER_TO_GROUP']
        _data.UNIVERSE = saved['data_UNIVERSE']
        _pipeline.UNIVERSE = saved['pipeline_UNIVERSE']
        _alpha.UNIVERSE = saved['alpha_UNIVERSE']
        _alpha.PAIRS_CANDIDATES = saved['alpha_PAIRS']
        _alpha.TICKER_TO_GROUP = saved['alpha_TICKER_TO_GROUP']
