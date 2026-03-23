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

## Output Files

Running the pipeline regenerates:

- [`pipeline_alpha_models.png`](pipeline_alpha_models.png): factor, volatility, regime, turnover, and allocation diagnostics.
- [`pipeline_performance.png`](pipeline_performance.png): cumulative returns, drawdowns, rolling Sharpe, distributions, and metrics.
- [`pipeline_rl_analysis.png`](pipeline_rl_analysis.png): policy behavior for the portfolio and hedging agents.
- [`pipeline_execution_rl.png`](pipeline_execution_rl.png): standalone execution RL demo.

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
Portfolio RL
  - Factor-anchored target book
  - RL exposure budget
  - RL active overlay size
  - No-trade rebalance band
        ↓
Hedging RL + Cash Carry
        ↓
Walk-Forward Backtest + Diagnostics
```

## Code Layout

- [`quant_stack/config.py`](quant_stack/config.py): asset universe, data settings, macro series IDs, and runtime constants.
- [`quant_stack/data.py`](quant_stack/data.py): `yfinance`, FRED, BLS, Treasury, and SEC loading helpers.
- [`quant_stack/alpha.py`](quant_stack/alpha.py): factor model, statistical arbitrage, GARCH, HMM, LSTM, and adaptive signal combiner.
- [`quant_stack/rl.py`](quant_stack/rl.py): portfolio construction RL, execution RL, and dynamic hedging RL.
- [`quant_stack/pipeline.py`](quant_stack/pipeline.py): walk-forward orchestration and benchmark computation.
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
- A no-trade band suppresses small reallocations and reduces turnover drag.
- The hedge RL agent chooses a hedge intensity based on drawdown, volatility regime, and recent momentum, with a light convex payoff approximation.
- The execution RL demo is kept separate from the main backtest and illustrates order-splitting logic under market-impact costs.

## Data Sources

- Prices and volumes: `yfinance`
- Macro, preferred: FRED when `FRED_API_KEY` is available
- Macro, no-key fallback: BLS + U.S. Treasury Fiscal Data
- Fundamentals: SEC company facts when `SEC_USER_AGENT` is provided

## Research Caveats

- This is research code, not a live execution system.
- Macro series are not yet modeled with full release-lag or vintage-awareness.
- SEC quality features are simple cross-sectional snapshots, not a fully point-in-time fundamentals database.
- Transaction costs are stylized and do not model market impact, queue position, or borrow constraints.
- The hedge sleeve approximates an options-like payoff without a live options chain.

## Report

The formal write-up for the pipeline is in [`quant_pipeline_report.tex`](quant_pipeline_report.tex). It documents the architecture, mathematical framing, data assumptions, and current empirical interpretation.
