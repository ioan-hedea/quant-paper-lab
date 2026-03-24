# Quant Trading Pipeline

This folder contains a research-style quantitative trading pipeline that combines classical cross-sectional alpha models with reinforcement-learning-based portfolio sizing, execution, and hedging.

The code now lives in a small package under [`quant_stack`](quant_stack), while [`quant_pipeline.py`](quant_pipeline.py) remains a thin compatibility wrapper so the original run command still works.

## Run

Baseline run:

```bash
MPLCONFIGDIR=/tmp/matplotlib uv run python stock_trading/quant_pipeline.py
```

With optional macro and SEC enrichment:

```bash
FRED_API_KEY="your_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
uv run python stock_trading/quant_pipeline.py
```

Research evaluation run:

```bash
MPLCONFIGDIR=/tmp/matplotlib uv run python stock_trading/quant_research.py
```

## Output Files

Running the pipeline regenerates:

- [`pipeline_alpha_models.png`](pipeline_alpha_models.png): factor, volatility, regime, turnover, and allocation diagnostics.
- [`pipeline_performance.png`](pipeline_performance.png): cumulative returns, drawdowns, rolling Sharpe, distributions, and metrics.
- [`pipeline_rl_analysis.png`](pipeline_rl_analysis.png): policy behavior for the portfolio and hedging agents.
- [`pipeline_execution_rl.png`](pipeline_execution_rl.png): standalone execution RL demo.

Running the research engine also writes:

- [`research_metrics.csv`](research_metrics.csv): aggregated ablation and robustness metrics.
- [`research_ablation_summary.csv`](research_ablation_summary.csv): split-aggregated ablation summary used by the paper.
- [`research_robustness_summary.csv`](research_robustness_summary.csv): rolling-window medians and benchmark beat fractions.
- [`research_regime_summary.csv`](research_regime_summary.csv): regime-conditional action, hedge, cash, and return diagnostics.
- [`research_rolling_references.csv`](research_rolling_references.csv): rolling-window SPY and factor-benchmark reference metrics.
- [`research_summary.json`](research_summary.json): serialized experiment settings plus best Sharpe configuration per suite.
- [`research_paper_tables.tex`](research_paper_tables.tex): paper-ready ablation and robustness tables.
- [`pipeline_research_eval.png`](pipeline_research_eval.png): four-panel summary figure for the paper story.

## Architecture

Pipeline flow:

```text
Market Data + Macro + SEC
        ↓
Feature Engineering
        ↓
Alpha Models
  - Factor composite
  - Pairs trading
  - GARCH volatility
  - HMM regime belief
  - LSTM return sleeve
        ↓
Adaptive Alpha Combiner
        ↓
Constrained Intermediate Allocator
  - Factor anchor
  - Risk / turnover penalty
  - Asset-group caps
        ↓
Portfolio RL
  - RL exposure budget
  - RL active overlay size
  - No-trade rebalance band
        ↓
Hedging RL + Vol Target + Cash Carry
        ↓
Walk-Forward Backtest + Diagnostics
        ↓
Research Evaluation Engine
  - Ablations
  - Rolling windows
  - Cost / lag / hedge / band sensitivity
  - Regime-conditional summaries
```

## Code Layout

- [`quant_stack/config.py`](quant_stack/config.py): asset universe, data settings, macro series IDs, and runtime constants.
- [`quant_stack/data.py`](quant_stack/data.py): `yfinance`, FRED, BLS, Treasury, and SEC loading helpers.
- [`quant_stack/alpha.py`](quant_stack/alpha.py): factor model, statistical arbitrage, GARCH, HMM, LSTM, and adaptive signal combiner.
- [`quant_stack/rl.py`](quant_stack/rl.py): portfolio construction RL, execution RL, and dynamic hedging RL.
- [`quant_stack/pipeline.py`](quant_stack/pipeline.py): walk-forward orchestration and benchmark computation.
- [`quant_stack/evaluation.py`](quant_stack/evaluation.py): ablation suite, rolling-window robustness engine, and research artifact export.
- [`quant_stack/plots.py`](quant_stack/plots.py): figure generation and metrics tables.
- [`quant_stack/main.py`](quant_stack/main.py): command-line entrypoint used by the wrapper.

## Modeling Summary

### Alpha

- The factor sleeve combines momentum, value proxy, quality proxy, and low-volatility signals into a z-scored cross-sectional composite.
- The pairs sleeve uses Engle-Granger cointegration tests and spread z-scores for mean-reversion signals.
- The GARCH sleeve forecasts name-level volatility for confidence scaling.
- The HMM sleeve produces a bull-vs-bear regime belief from market returns.
- The LSTM sleeve provides a lightweight nonlinear return forecast for a small subset of tickers.
- The combiner adaptively reweights factor, pairs, and LSTM sources based on recent realized signal quality.

### RL Control

- The portfolio RL agent does not select stocks from scratch. It starts from a factor-anchored target book, then chooses how much capital to deploy and how large the active overlay around that book should be.
- Between alpha and RL, a constrained optimizer converts signals into a long-only target portfolio with turnover control and simple asset-group caps.
- A no-trade band suppresses small reallocations and reduces turnover drag.
- The hedge RL agent chooses a hedge intensity based on drawdown, volatility regime, and recent momentum, with a volatility-targeting and convex-stress overlay approximation.
- The execution RL demo is kept separate from the main backtest and illustrates order-splitting logic under market-impact costs.

## Research Evaluation

The research engine is designed to answer the main architectural claim of the project:
RL is more useful as a controller layered on top of finance signals than as an end-to-end return predictor.

It currently runs:

- Component ablations for `factor_only`, `alpha_stack_no_rl`, `alpha_plus_portfolio_rl`, `alpha_plus_hedge_rl`, and `full_pipeline`.
- Multiple train/test splits.
- Rolling-window robustness checks.
- Sensitivity tests for transaction costs, rebalance bands, hedge intensity, and macro-lag assumptions.
- A stricter timing-discipline run that disables the static SEC quality snapshot and forces lagged macro inputs.
- Regime-conditioned summaries of portfolio action, hedge ratio, cash weight, turnover, and returns.

## Data Sources

- Prices and volumes: `yfinance`
- Macro, preferred: FRED when `FRED_API_KEY` is available
- Macro, no-key fallback: BLS + U.S. Treasury Fiscal Data
- Fundamentals: SEC company facts when `SEC_USER_AGENT` is provided

## Research Caveats

- This is research code, not a live execution system.
- Macro series are shifted forward by a configurable trading-day lag in the backtest, but they are not yet true vintage datasets with historical revisions.
- SEC quality features can be included as a static research prior, but they are not yet a fully point-in-time fundamentals database.
- Transaction costs now include base bps, turnover-volatility interaction, and a large-trade penalty, but they are still stylized rather than venue-specific.
- The hedge sleeve combines a light hedge ladder, volatility targeting, and a crash overlay, but it still approximates convex protection without live options data.

## Report

The formal write-up for the pipeline is in [`quant_pipeline_report.tex`](quant_pipeline_report.tex). It documents the architecture, mathematical framing, data assumptions, and current empirical interpretation.
