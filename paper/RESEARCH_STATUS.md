# Quant Pipeline — Research Plan Implementation Status

---

## Phase 1: Original Research Plan (COMPLETE)

### RQ1 — Controller RL vs End-to-End RL

| Item | Status | Where |
|------|--------|-------|
| E2E Gymnasium environment (`EndToEndTradingEnv`) | Done | `rl.py` |
| E2E feature builder (`build_e2e_features`) | Done | `rl.py` |
| PPO training + eval via Stable-Baselines3 (`run_e2e_baseline`) | Done | `rl.py` |
| E2E equity curve on main performance plot | Done | `plots.py – plot_performance()` |
| E2E equity curve on research eval plot | Done | `plots.py – plot_research_evaluation()` |
| `enable_e2e_baseline` config flag + PPO verbosity controls | Done | `config.py – PipelineConfig` |
| `research_e2e_scope` config (`baseline_only` / `all`) | Done | `config.py – EvaluationConfig` |

### RQ2 — RL vs Rule-Based Baselines

| Item | Status | Where |
|------|--------|-------|
| Vol-targeting baseline (`vt_invested = min(1, 0.15/σ)`) | Done | `pipeline.py` |
| Drawdown-deleveraging baseline (3-tier ladder) | Done | `pipeline.py` |
| Risk-parity baseline (LedoitWolf inverse-variance) | Done | `pipeline.py` |
| All three baselines tracked in wealth paths | Done | `pipeline.py` |
| Baselines included in plots, metrics, rolling references | Done | `plots.py`, `evaluation.py` |
| Block-bootstrap CIs (400 samples, block=20) | Done | `evaluation.py` |
| Pairwise significance: Full Pipeline vs each baseline | Done | `evaluation.py` |
| LaTeX table: bootstrap significance | Done | `evaluation.py – _write_research_tables()` |

### RQ3 — Component Attribution (Ablation)

| Item | Status | Where |
|------|--------|-------|
| `factor_only` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `factor_plus_pairs` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `factor_plus_lstm` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `alpha_stack_no_rl` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `alpha_plus_portfolio_rl` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `alpha_plus_hedge_rl` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| `full_pipeline` ablation config | Done | `evaluation.py – build_ablation_suite()` |
| Ablation ordering / display labels | Done | `evaluation.py – _display_label()` |
| LaTeX ablation table | Done | `evaluation.py – _write_research_tables()` |

### Uncertainty Features (§4.2)

| Item | Status | Where |
|------|--------|-------|
| Alpha dispersion (weighted std of alpha signals) | Done | `pipeline.py` |
| Regime entropy (binary entropy of HMM belief) | Done | `pipeline.py – _regime_entropy()` |
| IC instability (std of rolling Spearman ICs) | Done | `alpha.py – AlphaCombiner.get_ic_instability()` |
| GARCH vol forecast CV (cross-sectional, 21-day) | Done | `alpha.py – GARCHForecaster.forecast_vol_uncertainty()` |
| Combined `uncertainty_score` (35/30/20/15 weights) | Done | `pipeline.py` |
| `uncertainty_bin` passed to `PortfolioConstructionRL.get_state()` | Done | `rl.py`, `pipeline.py` |

### Reward Function Ablation (§8)

| Item | Status | Where |
|------|--------|-------|
| `differential_sharpe` reward mode | Done | `pipeline.py`, `rl.py` |
| `return` reward mode | Done | `pipeline.py`, `rl.py` |
| `sortino` reward mode | Done | `pipeline.py`, `rl.py` |
| `mean_variance` reward mode (λ=2.0) | Done | `pipeline.py`, `rl.py` |
| Reward modes on `DynamicHedgingRL.compute_reward()` | Done | `rl.py` |
| Reward modes on `EndToEndTradingEnv.step()` | Done | `rl.py` |
| Reward ablation sweep in evaluation engine | Done | `evaluation.py` |
| Figure 6: Reward ablation bar chart | Done | `plots.py – plot_reward_ablation()` |

### Rolling-Window Robustness (§5.2)

| Item | Status | Where |
|------|--------|-------|
| Adaptive window generation (4–6 windows, 504-day, 126-day step) | Done | `evaluation.py – _rolling_starts()` |
| Rolling metrics for full pipeline | Done | `evaluation.py` |
| Rolling reference rows: SPY, factor, vol-target, dd-delever | Done | `evaluation.py` |
| Rolling Sharpe line plot | Done | `evaluation.py – plot_research_evaluation()` |
| Figure 4: Rolling-window box plots (Sharpe + Calmar) | Done | `plots.py – plot_rolling_windows()` |

### Macro Regime Signal (§2.2)

| Item | Status | Where |
|------|--------|-------|
| VIX, HY OAS, DXY added to `FRED_SERIES` | Done | `config.py` |
| Weighted composite: term_spread 24%, unrate 20%, fed_funds 14%, vix 20%, hy_oas 16%, dxy 6% | Done | `data.py` |

### Checkpointing (§6)

| Item | Status | Where |
|------|--------|-------|
| Pickle checkpoint per run, keyed by data+config fingerprint | Done | `evaluation.py` |
| Schema versioning (`CHECKPOINT_SCHEMA_VERSION = 1`) | Done | `evaluation.py` |
| Progress manifest JSON (`research_progress.json`) | Done | `evaluation.py` |
| `enable_checkpoints` + `checkpoint_dir` config | Done | `config.py – EvaluationConfig` |

### Performance Optimizations

| Item | Status | Where |
|------|--------|-------|
| HMM forward-backward vectorized (numpy matrix ops) | Done | `alpha.py` |
| HMM refit cached (every 5 days) | Done | `alpha.py` |
| Pairs cointegration cached (every 21 days, z-scores updated daily) | Done | `alpha.py` |
| Risk parity LedoitWolf cached (every 5 days) | Done | `pipeline.py` |
| Vol percentile precomputed (`np.searchsorted` on rolling vol) | Done | `pipeline.py` |
| Portfolio vol precomputed (rolling 20d std) | Done | `pipeline.py` |
| LedoitWolf shared between `estimate_min_var_core` / `optimize_target_book` | Done | `rl.py` |
| `forecast_vol_uncertainty` incremental (no backward re-calls) | Done | `alpha.py` |
| Macro regime signal cached (every 5 days) | Done | `pipeline.py` |

---

## Phase 2: Extended Research Upgrades (TODO)

### 1. Data & Realism Upgrades (High Impact)

#### A. Extend Universe to 50–100 Stocks
| Item | Status | Priority |
|------|--------|----------|
| Define `UNIVERSE_EXPANDED` with 11 GICS sectors (~80 names) | TODO | High |
| Update `PAIRS_CANDIDATES` for expanded universe | TODO | High |
| Update `TICKER_TO_GROUP` mapping for optimizer group caps | TODO | High |
| Update `LSTM_TICKERS` selection for expanded universe | TODO | Medium |
| Re-run full evaluation suite on expanded universe | TODO | High |

#### B. Extend Backtest Window to 10+ Years (2013–2026)
| Item | Status | Priority |
|------|--------|----------|
| Verify yfinance data availability for all tickers back to 2013 | TODO | High |
| Handle ticker renames / delistings (e.g., META was FB pre-2022) | TODO | High |
| Adjust LSTM training window for longer history | TODO | Medium |
| Re-run full evaluation suite on extended window | TODO | High |

#### C. Point-in-Time Fundamentals
| Item | Status | Priority |
|------|--------|----------|
| Implement filing-date-aware SEC data loader (OpenBB or sec-edgar-downloader) | TODO | Medium |
| Parse XBRL for quarterly metrics (ROE, leverage, EPS) with filing timestamps | TODO | Medium |
| Replace static `sec_quality_scores` with point-in-time panel | TODO | Medium |
| Add ablation: static prior vs point-in-time fundamentals | TODO | Medium |

#### D. Macro Data with Vintage/Revision History (FRED ALFRED)
| Item | Status | Priority |
|------|--------|----------|
| Use FRED ALFRED API (`get_series_all_releases`) for vintage data | TODO | Medium |
| Replace current FRED fetch with vintage-aware loader | TODO | Medium |
| Add `realtime_start` filtering so backtest only sees published values | TODO | Medium |

### 2. Methodological Enhancements (Medium-High Impact)

#### E. Multi-Asset Extension (Bonds, Commodities, Currencies)
| Item | Status | Priority |
|------|--------|----------|
| Add defensive assets (TLT, AGG, GLD, UUP) to universe | TODO | Medium |
| Extend hedge RL action space to include asset-class rotation | TODO | Medium |
| Cross-asset regime detection using expanded macro features | TODO | Medium |

#### F. Options-Based Hedging
| Item | Status | Priority |
|------|--------|----------|
| Fetch SPY options chain data (yfinance or CBOE) | TODO | Low |
| Replace stylized convex hedge payoff with real put pricing | TODO | Low |
| Backtest protective put strategy as additional baseline | TODO | Low |

#### G. Transaction Cost Model Upgrade (Almgren-Chriss)
| Item | Status | Priority |
|------|--------|----------|
| Implement permanent + temporary market impact model | TODO | Medium |
| Calibrate impact coefficients (β, η) from literature | TODO | Medium |
| Replace current 3-component cost model in `_compute_transaction_cost()` | TODO | Medium |

### 3. Baseline & Ablation Expansion (Critical for Publication)

#### H. Expanded Ablations
| Item | Status | Priority |
|------|--------|----------|
| RL state feature ablation: without uncertainty features | TODO | High |
| RL state feature ablation: without regime information | TODO | High |
| RL state feature ablation: without volatility forecasts | TODO | High |
| Alpha component ablation: factor + GARCH + HMM only (no pairs/LSTM) | TODO | Medium |
| Calmar-style reward mode (penalize drawdowns explicitly) | TODO | Medium |

#### I. Sample Complexity Analysis
| Item | Status | Priority |
|------|--------|----------|
| Train RL on varying data lengths (1–5 years) | TODO | Medium |
| Plot training days vs out-of-sample Sharpe | TODO | Medium |
| Fit power law to sample complexity curve | TODO | Medium |

### 4. Statistical Rigor Upgrades

#### J. Formal Hypothesis Testing
| Item | Status | Priority |
|------|--------|----------|
| Jobson-Korkie test for Sharpe ratio equality (with Memmel correction) | TODO | High |
| Report p-values for all pairwise strategy comparisons | TODO | High |
| Add to LaTeX tables alongside bootstrap CIs | TODO | High |

#### K. Out-of-Sample Validation (Time-Series CV)
| Item | Status | Priority |
|------|--------|----------|
| Implement blocked time-series cross-validation (5-fold) | TODO | High |
| Collect per-fold Sharpe, Calmar, MaxDD distributions | TODO | High |
| Report mean ± std across folds | TODO | High |

### 5. Presentation Upgrades

#### L. Interactive Results Dashboard (Streamlit)
| Item | Status | Priority |
|------|--------|----------|
| Build Streamlit app with interactive equity curves (Plotly) | TODO | Low |
| Add component selector, rolling window slider | TODO | Low |
| Deploy to Streamlit Cloud, link from paper | TODO | Low |

#### M. Video Abstract
| Item | Status | Priority |
|------|--------|----------|
| Script 3–5 min video (problem, solution, results, impact) | TODO | Low |
| Record with OBS Studio + slides | TODO | Low |

### 6. Theoretical Contributions (Stretch Goals)

#### N. Regret Bounds for Factor-Anchored Control
| Item | Status | Priority |
|------|--------|----------|
| Formalize factor-anchored MDP assumptions (A1–A3) | TODO | Stretch |
| Derive O(√T log T) regret bound | TODO | Stretch |
| Write theorem + proof for paper appendix | TODO | Stretch |

#### O. Sample Complexity Analysis (Theoretical)
| Item | Status | Priority |
|------|--------|----------|
| Empirical learning curve: training length vs OOS Sharpe | TODO | Medium |
| Fit power law N^(-β) to characterize data efficiency | TODO | Medium |

---

## Outputs

| File | Description |
|------|-------------|
| `research_metrics.csv` | All run metrics (ablation, rolling, sensitivity, reward) |
| `research_regime_summary.csv` | Regime-conditional controller behavior |
| `research_rolling_references.csv` | Per-window metrics for SPY / factor / baselines |
| `research_ablation_summary.csv` | Mean metrics per ablation component |
| `research_bootstrap_cis.csv` | Bootstrap CIs for main baselines |
| `research_bootstrap_significance.csv` | Pairwise p-values (Full Pipeline vs baselines) |
| `paper/2col/research_paper_tables.tex` | LaTeX tables: ablation, robustness, significance |
| `research_summary.json` | Full summary + best Sharpe per suite |
| `pipeline_research_eval.png` | Figure 1: main result, ablation bar, rolling Sharpe, regime behavior |
| `pipeline_rolling_windows.png` | Figure 4: Sharpe + Calmar box plots across rolling windows |
| `pipeline_reward_ablation.png` | Figure 6: reward mode ablation bar charts |
| `pipeline_performance.png` | Full performance dashboard (7 strategies) |
| `pipeline_alpha_models.png` | Alpha model decomposition |
| `pipeline_rl_analysis.png` | RL component deep dive |

## Dependencies

```
gymnasium>=1.2.3
stable-baselines3>=2.7.1
torch>=2.11
scikit-learn  (LedoitWolf)
arch          (GARCH)
```

Install: `.venv/bin/pip install gymnasium torch stable-baselines3`
