# Quant Paper Lab

This repository contains a research-oriented portfolio-management system and the paper artifacts built around it.

The project started as an RL-heavy modular trading pipeline and then pivoted, based on archived negative results, into a broader comparison of control mechanisms on top of a finance-first alpha engine. The current codebase supports both that legacy branch and the revised control-comparison study.

## Project Focus

The current paper direction is:

- finance-first alpha generation
- systematic comparison of control layers
- frozen-bundle evaluation and checkpointed research artifacts
- transparent treatment of negative results

The revised control study asks:

1. Which control mechanism adds the most value on top of a strong alpha engine?
2. Can simple rules provide competitive downside control?
3. Do learning-based methods beat optimization-based methods?

In the current archived story, no single method dominates every metric:

- `D_cvar_robust` is strongest on Sharpe-based point estimates
- `A4_regime_rules` is strongest on drawdown protection among the top candidates
- simple rules remain competitive
- tabular RL is a weak comparator in this setting

## Current Architecture

The active research stack is defined in [architecture_revision_v2.md](/quant-paper-lab%20/architecture_revision_v2.md).

### 1. Alpha Layer

- cross-sectional factor model
- GARCH volatility forecasts
- 2-state HMM regime belief
- adaptive signal combiner

This is a finance-first alpha engine. Pairs, LSTM sleeves, and hedge RL were removed from the main path because the archived ablations did not justify them.

### 2. Allocator

- long-only constrained optimizer
- capped weights
- turnover penalty
- factor-anchored target book
- no-trade band
- 3-part transaction cost model

### 3. Control Candidates

Implemented controller families include:

- `none`: no control over the shared alpha engine
- `fixed`: fixed invested fraction
- `vol_target`: volatility targeting
- `dd_delever`: drawdown-based deleveraging
- `regime_rules`: regime-aware exposure rules
- `ensemble_rules`: simple rule ensemble
- `linucb`, `thompson`, `epsilon_greedy`: contextual bandits
- `supervised`: offline-learned exposure classifier
- `cvar_robust`: CVaR-aware robust optimization
- `council`: small expert-gated meta-controller
- `cmdp_lagrangian`: simple constrained-MDP style controller
- `q_learning`: tabular RL controller
- `ppo`: end-to-end RL baseline

The extension layer also supports:

- convexity-aware payoff shaping
- council + convexity combinations

These are intended as extensions of the strong robust baseline, not replacements for it.

## Repository Layout

- `quant_stack/`: main research package
  - `config.py`: all experiment and controller configuration
  - `data.py`: market, macro, and SEC enrichment
  - `alpha.py`: factor, GARCH, and regime logic
  - `controllers.py`: rule, bandit, supervised, robust, council, CMDP, and RL controllers
  - `pipeline.py`: end-to-end backtest engine
  - `evaluation.py`: checkpointed research suite and artifact generation
  - `plots.py`: legacy and paper-facing plots
- `quant_pipeline.py`: single pipeline / strategy run
- `quant_research.py`: full evaluation driver
- `scripts/bootstrap_venv.sh`: environment bootstrap
- `scripts/generate_control_story_plots.py`: higher-level paper plots from saved CSV artifacts
- `paper/1col/`: single-column paper source and PDF
- `paper/2col/`: two-column paper source, auto-generated tables, and PDF
- `checkpoints/research_runs/`: per-run checkpoint cache
- `logs/`: experiment logs
- `research_*.csv`, `research_*.json`: root-level summary artifacts

## Environment Setup

### Option A: Standard venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-e2e.txt
```

`requirements-e2e.txt` already includes `requirements.txt`, so you usually only need the one install command above.

### Option B: Bootstrap Script

```bash
bash scripts/bootstrap_venv.sh --with-e2e
```

### Option C: uv

If you prefer `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-e2e.txt
```

## Environment Variables

Optional but useful:

- `FRED_API_KEY`: enables higher-quality macro feature loading
- `SEC_USER_AGENT`: enables SEC company-facts enrichment
- `MPLCONFIGDIR=/tmp/matplotlib`: avoids Matplotlib cache permission issues in some environments

Example:

```bash
FRED_API_KEY="your_fred_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
python quant_research.py
```

## Running the Code

### Single Pipeline Run

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_pipeline.py
```

This produces a one-off backtest plus diagnostic plots such as:

- `pipeline_alpha_models.png`
- `pipeline_performance.png`
- `pipeline_rl_analysis.png`
- `pipeline_execution_rl.png`

### Full Research Evaluation

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_research.py
```

This runs the checkpointed research suite:

- legacy ablations
- control-method comparison
- strict-timing discipline check
- rolling-window robustness
- cost sensitivity
- rebalance sensitivity
- macro-lag sensitivity
- reward ablation

The wrapper prints the key generated artifacts when it completes.

### Plot Refresh Only

If the CSV artifacts already exist and you only want the higher-level summary plots:

```bash
source .venv/bin/activate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib python scripts/generate_control_story_plots.py
```

This regenerates:

- `control_method_overview.png`
- `control_split_heatmaps.png`
- `control_pareto_frontier.png`
- `legacy_pruning_story.png`

## Checkpoints and Artifact Reuse

The research engine is designed to reuse checkpointed runs aggressively.

- per-run results are stored in `checkpoints/research_runs/*.pkl`
- progress is tracked in `checkpoints/research_runs/research_progress.json`
- checkpoint compatibility is based on canonicalized config metadata
- unchanged runs are loaded from cache instead of recomputed

This is especially useful when:

- you add a new controller candidate
- you tweak paper-facing summaries or plots
- you rerun after a partial interruption

Recent examples of extension-only labels include:

- `D_plus_convexity`
- `E_council`
- `E_plus_convexity`
- `F_cmdp_lagrangian`

If a new experiment only adds one of those labels, the older controller checkpoints should still be reused.

### Q-Learning Split Note

The tabular Q-learning controller is now scheduled with a `75/25` split in place of the old `50/50` split inside the control comparison grid. Other controller families keep the shared `(0.4, 0.5, 0.6)` train-fraction schedule.

## Main Research Artifacts

Core tabular outputs:

- `research_metrics.csv`
- `research_ablation_summary.csv`
- `research_control_comparison.csv`
- `research_robustness_summary.csv`
- `research_regime_summary.csv`
- `research_rolling_references.csv`
- `research_bootstrap_cis.csv`
- `research_bootstrap_significance.csv`
- `research_control_significance.csv`
- `research_jobson_korkie.csv`
- `research_ts_cv.csv`
- `research_summary.json`

Core figures:

- `pipeline_research_eval.png`
- `pipeline_reward_ablation.png`
- `pipeline_rolling_windows.png`
- `control_method_overview.png`
- `control_split_heatmaps.png`
- `control_pareto_frontier.png`
- `legacy_pruning_story.png`

Paper-support artifact:

- `paper/2col/research_paper_tables.tex`

## Paper Files

Main manuscript sources:

- [paper/1col/quant_pipeline_report.tex](/quant-paper-lab%20/paper/1col/quant_pipeline_report.tex)
- [paper/2col/quant_pipeline_report_2col.tex](paper/2col/quant_pipeline_report_2col.tex)

Built PDFs:

- [paper/1col/quant_pipeline_report.pdf](/quant-paper-lab%20/paper/1col/quant_pipeline_report.pdf)
- [paper/2col/quant_pipeline_report_2col.pdf](/quant-paper-lab%20/paper/2col/quant_pipeline_report_2col.pdf)

The paper currently presents:

- the frozen-bundle evaluation protocol as the methodological contribution
- the control-method comparison as the empirical contribution

## Data Sources

- prices and volumes: `yfinance`
- macro series: FRED when available, with fallbacks for core public macro inputs
- SEC fundamentals: company-facts enrichment when `SEC_USER_AGENT` is provided

## Important Caveats

- This is research code, not production trading infrastructure.
- Backtests are sensitive to cost assumptions, split design, and universe definition.
- Some artifact files in the repo root are snapshots from previous experiment bundles; the paper should always be aligned with the latest frozen bundle you intend to report.
- The legacy RL-heavy branch is kept for transparency and negative-result analysis, not because it is the recommended architecture.

## Recommended Workflow

For everyday work, the simplest sequence is:

1. Update code or experiment configs.
2. Run `quant_research.py` and let checkpoint reuse skip old runs.
3. Regenerate the control-story plots if needed.
4. Update the paper sources in `paper/1col/` and `paper/2col/`.
5. Rebuild the PDFs.

If the study direction changes, update [architecture_revision_v2.md](/quant-paper-lab%20/architecture_revision_v2.md) first, then align code, artifacts, and manuscript text to that decision.
