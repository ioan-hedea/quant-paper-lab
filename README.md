# Sequential Decision-Making

This repository collects course and research-style implementations across reinforcement learning, planning, control, verification, and quantitative trading. The strongest end-to-end artifact in the repo is the modular quant pipeline in [`stock_trading`](stock_trading), which combines classical alpha models with reinforcement-learning-based portfolio control, execution, and hedging.

## Repository Map

- [`stock_trading`](stock_trading): quantitative trading experiments, including the modular `quant_stack` pipeline and generated figures.
- [`depp_rl`](depp_rl): tabular and deep RL assignments, skeletons, and recorded training artifacts.
- [`mdp_dp`](mdp_dp): dynamic-programming visualizations for value iteration and policy iteration.
- [`pomdp`](pomdp): belief-space POMDP examples and visualizations.
- [`pomcp`](pomcp): online Monte Carlo planning examples such as Tiger and RockSample.
- [`model_based_rl`](model_based_rl): model-based RL visualization material.
- [`bayesian_rl`](bayesian_rl): Bayesian bandits and Thompson sampling illustrations.
- [`safe_rl`](safe_rl): safe RL and robust/constrained MDP visualizations.
- [`verification`](verification): reachability, BMC, and verification-oriented figures.
- [`mpc`](mpc): MPC coursework, figures, and report artifacts.
- [`irl`](irl): inverse RL, neural certificates, and SMT-based components.

## Quant Pipeline Quick Start

The cleaned trading pipeline keeps the existing run command:

```bash
MPLCONFIGDIR=/tmp/matplotlib uv run python stock_trading/quant_pipeline.py
```

Optional data enrichments:

```bash
FRED_API_KEY="your_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
uv run python stock_trading/quant_pipeline.py
```

More detailed usage, architecture notes, and file-level documentation live in [`stock_trading/README.md`](stock_trading/README.md).

## Quant Pipeline Structure

The large trading script has been split into a small package under [`stock_trading/quant_stack`](stock_trading/quant_stack):

- [`config.py`](stock_trading/quant_stack/config.py): universe, macro series, and runtime constants.
- [`data.py`](stock_trading/quant_stack/data.py): market, macro, and SEC loaders.
- [`alpha.py`](stock_trading/quant_stack/alpha.py): factor, pairs, GARCH, HMM, LSTM, and alpha-combination logic.
- [`rl.py`](stock_trading/quant_stack/rl.py): portfolio, execution, and hedging RL agents.
- [`pipeline.py`](stock_trading/quant_stack/pipeline.py): walk-forward backtest orchestration.
- [`plots.py`](stock_trading/quant_stack/plots.py): performance, diagnostics, and execution figures.
- [`main.py`](stock_trading/quant_stack/main.py): CLI entrypoint used by the thin wrapper in [`quant_pipeline.py`](stock_trading/quant_pipeline.py).

## Notes

- The repo mixes lightweight demos and richer research prototypes; not every folder follows the same packaging style.
- The quant pipeline is backtest-oriented and should still be treated as research code, not production trading software.
- Generated plots in [`stock_trading`](stock_trading) document the latest run, but they are only as realistic as the underlying data and lag assumptions.
