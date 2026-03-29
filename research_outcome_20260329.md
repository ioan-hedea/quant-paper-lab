# Research Outcome: Architecture Revision v2

Date: 2026-03-29

This note summarizes the completed frozen-bundle evaluation produced by the
revised control-method comparison architecture. It is intended as a paper-facing
summary of what the finished run actually shows before updating the manuscript.

## Scope

The completed run now compares control mechanisms on top of the same
finance-first alpha stack:

- Alpha layer: factor + GARCH + HMM
- Shared allocator: long-only, capped, turnover-penalized
- Control candidates:
  - `factor_only`
  - `A1_fixed`
  - `A2_vol_target`
  - `A3_dd_delever`
  - `A4_regime_rules`
  - `A5_ensemble_mean`
  - `A5_ensemble_min`
  - `B1_linucb`
  - `B2_thompson`
  - `B3_epsilon_greedy`
  - `C_supervised`
  - `D_cvar_robust`
  - `RL_q_learning`

The root-level artifacts were refreshed at `2026-03-29 20:09`.

## Headline Result

The strongest control candidate in the new suite is `D_cvar_robust`.

Mean control-comparison results across the three train/test splits:

| Method | Mean Return | Mean Vol | Mean Sharpe | Mean Max DD | Mean Calmar |
|---|---:|---:|---:|---:|---:|
| factor_only | 9.24% | 13.40% | 0.402 | -19.8% | 0.460 |
| A1_fixed | 8.99% | 12.73% | 0.404 | -18.6% | 0.476 |
| A2_vol_target | 6.90% | 10.02% | 0.335 | -13.3% | 0.511 |
| A3_dd_delever | 6.64% | 11.71% | 0.251 | -18.4% | 0.362 |
| A4_regime_rules | 8.75% | 11.94% | 0.415 | -16.1% | 0.539 |
| A5_ensemble_mean | 7.36% | 11.49% | 0.320 | -15.3% | 0.484 |
| A5_ensemble_min | 6.60% | 10.15% | 0.297 | -12.7% | 0.515 |
| B1_linucb | 9.02% | 12.29% | 0.424 | -18.0% | 0.498 |
| B2_thompson | 7.91% | 12.31% | 0.334 | -18.6% | 0.417 |
| B3_epsilon_greedy | 8.44% | 12.30% | 0.373 | -18.0% | 0.460 |
| C_supervised | 8.87% | 12.45% | 0.395 | -19.2% | 0.454 |
| **D_cvar_robust** | **17.64%** | **17.32%** | **0.805** | **-26.4%** | **0.681** |
| RL_q_learning | 7.91% | 12.10% | 0.342 | -18.3% | 0.424 |

Best single run in the control-comparison suite:

- `D_cvar_robust_tf0.50`
- Sharpe: `0.911`
- Annualized return: `20.68%`
- Max drawdown: `-30.5%`

## What The Revised Bundle Says

### 1. Robust optimization currently wins

The new evidence strongly favors `D_cvar_robust` over the other control
candidates on point estimates. It is the only candidate that clearly separates
itself from the rest of the control-comparison table.

### 2. Simple rules are competitive

Among simple rule-based methods, `A4_regime_rules` is the most attractive
practical baseline:

- Sharpe `0.415`, slightly above `factor_only` at `0.402`
- Max drawdown improves from `-19.8%` to `-16.1%`
- Calmar improves from `0.460` to `0.539`

`A1_fixed` is also competitive, while `A2_vol_target` offers the best drawdown
control among the simple rules but gives up more return and Sharpe.

### 3. Bandits are mixed, not dominant

The contextual bandits do not collapse, but they also do not dominate:

- `B1_linucb` is the best of the learned lightweight controllers
- `B2_thompson` and `B3_epsilon_greedy` are weaker
- None of the bandits approach `D_cvar_robust`

### 4. Supervised control is respectable but not the winner

`C_supervised` is broadly competitive with the better simple controls, but it
does not establish a clear edge over `A4_regime_rules`, `A1_fixed`, or `B1`.

### 5. RL remains weak in this setting

`RL_q_learning` is not compelling in the revised suite:

- Mean Sharpe `0.342`
- Mean return `7.91%`
- Mean max drawdown `-18.3%`

This is better framed as a negative result for RL in this daily portfolio
control problem than as an implementation accident.

## Relation To The Legacy Full Pipeline

The legacy `full_pipeline` remains weak in the fresh root-level artifacts:

- Annualized return point estimate: `8.98%`
- Sharpe point estimate: `0.423`
- Calmar point estimate: `0.459`
- Max drawdown point estimate: `-19.6%`

Bootstrap significance versus key baselines:

| Comparison | Delta Sharpe | 95% CI | p-value |
|---|---:|---:|---:|
| full_pipeline - SPY | -0.160 | [-0.908, 0.510] | 0.615 |
| full_pipeline - factor_benchmark | -0.580 | [-1.142, -0.151] | 0.005 |
| full_pipeline - vol_target | -0.469 | [-0.980, -0.029] | 0.035 |
| full_pipeline - dd_delever | -0.500 | [-1.040, 0.062] | 0.070 |
| full_pipeline - e2e_rl | -0.139 | [-1.187, 0.812] | 0.820 |

Interpretation:

- the legacy full pipeline is significantly worse than the factor benchmark
- it is also worse than vol-targeting on Sharpe
- it is not clearly distinguishable from SPY or end-to-end PPO on Sharpe

## Mapping To The Revised Research Questions

### RQ1: Which control mechanism adds the most value?

Current answer:

- `D_cvar_robust` is the strongest candidate by a wide margin on point
  estimates
- the nearest challengers are `B1_linucb`, `A4_regime_rules`, and `A1_fixed`,
  but none are close to the robust optimizer

### RQ2: Can a simple control layer improve downside without degrading Sharpe too much?

Current answer:

- Yes, especially `A4_regime_rules`
- `A4_regime_rules` improves both drawdown and Sharpe relative to `factor_only`
- `A2_vol_target` also improves drawdown materially, but with a bigger Sharpe
  trade-off

### RQ3: Which sources of complexity are justified?

Current answer:

- The old RL-heavy complexity is not justified
- The factor core remains the strongest retained alpha engine
- The one complex control layer that currently looks justified is
  `D_cvar_robust`
- Lightweight bandits and supervised control are acceptable comparators, but
  they are not the main winner

## Robustness Snapshot

From the refreshed robustness summary:

- Rolling windows: `6`
- Median full-pipeline Sharpe: `0.589`
- Median full-pipeline Calmar: `1.086`
- Median full-pipeline max drawdown: `-8.9%`
- Fraction of windows full pipeline beats SPY on Sharpe: `17%`
- Fraction of windows full pipeline beats factor benchmark on drawdown: `67%`

This still supports a narrow downside-shaping interpretation for the old full
pipeline, but not a broad performance advantage.

## Important Caveats

1. The `research_progress.json` manifest still reports a stale `total_runs`
   field even though the completed run count is higher. This is a progress
   accounting bug, not a result bug.
2. The current bootstrap-significance table is centered on the legacy
   `full_pipeline` comparisons. It does not yet provide pairwise significance
   tests for `D_cvar_robust` versus `A4_regime_rules`, `B1_linucb`, or
   `factor_only`.
3. The paper should therefore treat `D_cvar_robust` as the strongest current
   point-estimate winner, while avoiding overclaiming significance until those
   pairwise tests are added.

## Recommended Paper Direction

The finished run supports a revised manuscript narrative:

- The paper is no longer about defending RL.
- The factor engine remains the foundation.
- The key scientific question is which control complexity is actually justified.
- The strongest current evidence favors:
  - simple regime-aware rules as practical baselines
  - CVaR-aware robust optimization as the most effective advanced controller
  - RL as a weak comparator rather than the main contribution

In short: the project has evolved from an RL-centric trading paper into a much
stronger control-method comparison study over a finance-first alpha engine.
