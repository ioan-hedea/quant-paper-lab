# Quant Pipeline — Research Plan Implementation Status

All items from `quant_pipeline_research_plan.md` have been implemented.

---

## RQ1 — Controller RL vs End-to-End RL

| Item | Status | Where |
|------|--------|-------|
| E2E Gymnasium environment (`EndToEndTradingEnv`) | Done | `rl.py` |
| E2E feature builder (`build_e2e_features`) | Done | `rl.py` |
| PPO training + eval via Stable-Baselines3 (`run_e2e_baseline`) | Done | `rl.py` |
| E2E equity curve on main performance plot | Done | `plots.py – plot_performance()` |
| E2E equity curve on research eval plot | Done | `plots.py – plot_research_evaluation()` |
| `enable_e2e_baseline` config flag + PPO verbosity controls | Done | `config.py – PipelineConfig` |
| `research_e2e_scope` config (`baseline_only` / `all`) | Done | `config.py – EvaluationConfig` |

---

## RQ2 — RL vs Rule-Based Baselines

| Item | Status | Where |
|------|--------|-------|
| Vol-targeting baseline (`vt_invested = min(1, 0.15/σ)`) | Done | `pipeline.py` |
| Drawdown-deleveraging baseline (3-tier ladder) | Done | `pipeline.py` |
| Risk-parity baseline (LedoitWolf inverse-variance) | Done | `pipeline.py` |
| All three baselines tracked in wealth paths (`voltarget`, `ddlever`, `risk_parity`) | Done | `pipeline.py` |
| Baselines included in plots, metrics, rolling references | Done | `plots.py`, `evaluation.py` |
| Block-bootstrap CIs (400 samples, block=20) for all baselines | Done | `evaluation.py` |
| Pairwise significance: Full Pipeline vs each baseline | Done | `evaluation.py` |
| LaTeX table: bootstrap significance | Done | `evaluation.py – _write_research_tables()` |

---

## RQ3 — Component Attribution (Ablation)

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

---

## Uncertainty Features (§4.2)

| Item | Status | Where |
|------|--------|-------|
| Alpha dispersion (weighted std of alpha signals) | Done | `pipeline.py` |
| Regime entropy (binary entropy of HMM belief) | Done | `pipeline.py – _regime_entropy()` |
| IC instability (std of rolling Spearman ICs) | Done | `alpha.py – AlphaCombiner.get_ic_instability()` |
| GARCH vol forecast CV (cross-sectional, 21-day) | Done | `alpha.py – GARCHForecaster.forecast_vol_uncertainty()` |
| Combined `uncertainty_score` (35/30/20/15 weights) | Done | `pipeline.py` |
| `uncertainty_bin` passed to `PortfolioConstructionRL.get_state()` | Done | `rl.py`, `pipeline.py` |

---

## Reward Function Ablation (§8)

| Item | Status | Where |
|------|--------|-------|
| `differential_sharpe` reward mode | Done | `pipeline.py`, `rl.py` |
| `return` reward mode | Done | `pipeline.py`, `rl.py` |
| `sortino` reward mode | Done | `pipeline.py`, `rl.py` |
| `mean_variance` reward mode (λ=2.0) | Done | `pipeline.py`, `rl.py` |
| Reward modes on `DynamicHedgingRL.compute_reward()` | Done | `rl.py` |
| Reward modes on `EndToEndTradingEnv.step()` | Done | `rl.py` |
| Reward ablation sweep in evaluation engine | Done | `evaluation.py – run_research_evaluation()` |
| **Figure 6: Reward ablation bar chart** | Done | `plots.py – plot_reward_ablation()` |

---

## Rolling-Window Robustness (§5.2)

| Item | Status | Where |
|------|--------|-------|
| Adaptive window generation (4–6 windows, 504-day, 126-day step) | Done | `evaluation.py – _rolling_starts()` |
| Rolling metrics for full pipeline | Done | `evaluation.py` |
| Rolling reference rows: SPY, factor, vol-target, dd-delever | Done | `evaluation.py` |
| Rolling Sharpe line plot (existing) | Done | `evaluation.py – plot_research_evaluation()` |
| **Figure 4: Rolling-window box plots (Sharpe + Calmar)** | Done | `plots.py – plot_rolling_windows()` |

---

## Macro Regime Signal (§2.2)

| Item | Status | Where |
|------|--------|-------|
| VIX, HY OAS, DXY added to `FRED_SERIES` | Done | `config.py` |
| Weighted composite: term_spread 24%, unrate 20%, fed_funds 14%, vix 20%, hy_oas 16%, dxy 6% | Done | `data.py – compute_macro_regime_signal()` |

---

## Checkpointing (§6)

| Item | Status | Where |
|------|--------|-------|
| Pickle checkpoint per run, keyed by data+config fingerprint | Done | `evaluation.py` |
| Schema versioning (`CHECKPOINT_SCHEMA_VERSION = 1`) | Done | `evaluation.py` |
| Progress manifest JSON (`research_progress.json`) | Done | `evaluation.py` |
| `enable_checkpoints` + `checkpoint_dir` config | Done | `config.py – EvaluationConfig` |

---

## Outputs

| File | Description |
|------|-------------|
| `stock_trading/research_metrics.csv` | All run metrics (ablation, rolling, sensitivity, reward) |
| `stock_trading/research_regime_summary.csv` | Regime-conditional controller behavior |
| `stock_trading/research_rolling_references.csv` | Per-window metrics for SPY / factor / baselines |
| `stock_trading/research_ablation_summary.csv` | Mean metrics per ablation component |
| `stock_trading/research_bootstrap_cis.csv` | Bootstrap CIs for main baselines |
| `stock_trading/research_bootstrap_significance.csv` | Pairwise p-values (Full Pipeline vs baselines) |
| `stock_trading/research_paper_tables.tex` | LaTeX tables: ablation, robustness, significance |
| `stock_trading/research_summary.json` | Full summary + best Sharpe per suite |
| `stock_trading/pipeline_research_eval.png` | Figure 1: main result, ablation bar, rolling Sharpe, regime behavior |
| `stock_trading/pipeline_rolling_windows.png` | Figure 4: Sharpe + Calmar box plots across rolling windows |
| `stock_trading/pipeline_reward_ablation.png` | Figure 6: reward mode ablation bar charts |
| `stock_trading/pipeline_performance.png` | Full performance dashboard (7 strategies) |
| `stock_trading/pipeline_alpha_models.png` | Alpha model decomposition |
| `stock_trading/pipeline_rl_analysis.png` | RL component deep dive |

---

## Dependencies

```
gymnasium>=1.2.3
stable-baselines3>=2.7.1
torch>=2.11
scikit-learn  (LedoitWolf)
arch          (GARCH)
```

Install: `.venv/bin/pip install gymnasium torch stable-baselines3`
