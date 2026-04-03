# Quant Paper Lab

Research code and paper artifacts for **Systematic Evaluation of Control Mechanisms for Factor-Based Portfolio Management**.

Public repository:
- <https://github.com/ioan-hedea/quant-paper-lab>

This project began as an RL-heavy modular trading prototype and evolved into a broader, more disciplined study of **portfolio control as risk shaping** on top of a finance-first alpha engine. The current repository contains:

- the active two-universe research code
- checkpointed frozen-bundle evaluations
- paper-ready figures and tables
- the one-column and two-column manuscript sources and PDFs

## Current Paper Claim

The current paper is no longer an “RL paper.” Its central claim is:

> Portfolio control can be understood as **risk shaping**. In the reported frozen bundle, explicit downside-aware optimization dominates the learned alternatives, convex payoff shaping is highly valuable when tail asymmetry is strong, and learning contributes most as **meta-control** rather than as the primary decision layer.

The latest archived manuscript evidence is based on **two disjoint equity universes**:

- `Universe A`: 77 tickers
- `Universe B`: 75 tickers with zero overlap with A

Both are evaluated on daily data from **2013-04-02 through 2026-03-30**.

## Current Empirical Picture

The finished two-universe bundle supports a nuanced but clean story:

- the **robust family** is the overall frontier across both universes
- `D_plus_convexity` is the strongest overall controller in **Universe A**
- plain `D_cvar_robust` is the strongest and more portable robust baseline in **Universe B**
- `E_council` is the strongest learned controller across both universes
- `G_mlp_meta` is the strongest expressive learned gate, but does not improve on `E_council`
- `H_mpc` is the strongest low-drawdown / defensive frontier point
- tabular RL remains a negative or secondary reference, not the recommended controller

In short:

- **optimization wins**
- **convexity helps when left-tail asymmetry is strong**
- **learning helps most as meta-control**

## Active Architecture

The active study design is summarized in [architecture_revision_v2.md](architecture_revision_v2.md).

### Alpha Layer

The alpha engine is intentionally finance-first:

- cross-sectional factor model
- GARCH volatility sleeve
- 2-state HMM regime sleeve
- adaptive IC-weighted alpha combiner

Pairs, LSTM sleeves, and hedge RL remain in the repo only as legacy or negative-reference context, not as the main paper path.

### Allocator

The intermediate allocator turns alpha into a constrained target book:

- long-only
- capped positions
- turnover-penalized
- group-aware
- factor-anchored
- no-trade band
- three-part transaction-cost model

### Control Layer

Implemented controller families currently include:

- `none`: no additional control beyond the shared alpha engine
- `fixed`: fixed invested fraction
- `vol_target`: volatility targeting
- `dd_delever`: drawdown-based deleveraging
- `regime_rules`: regime-conditioned rules
- `ensemble_rules`: simple rule ensemble
- `linucb`, `thompson`, `epsilon_greedy`: contextual bandits
- `supervised`: offline-trained control policy
- `cvar_robust`: CVaR-aware robust optimization
- `council`: logistic expert-gated meta-controller
- `mlp_meta`: PyTorch MLP/attention-gated meta-controller
- `mpc`: model-predictive control
- `cmdp_lagrangian`: constrained-MDP style controller
- `q_learning`: tabular RL controller
- `ppo`: end-to-end PPO baseline

Extensions built on top of strong baselines:

- convexity-aware payoff shaping
- council + convexity
- MLP meta + convexity

## Repository Layout

- `quant_stack/`
  - `config.py`: experiment grids, universes, controller configs, checkpoint settings
  - `data.py`: market, macro, and SEC enrichment
  - `alpha.py`: factor, GARCH, and HMM alpha sleeves
  - `controllers.py`: rules, bandits, supervised, robust, council, MLP meta, MPC, CMDP, and RL controllers
  - `pipeline.py`: shared walk-forward backtest engine
  - `evaluation.py`: research suite, checkpoints, statistics, and artifact generation
  - `plots.py`: pipeline-level plotting helpers
  - `main.py`: one-off pipeline entrypoint
- `quant_pipeline.py`: thin wrapper for a single pipeline run
- `quant_research.py`: full research runner
- `scripts/generate_control_story_plots.py`: paper-facing plot generation from archived results bundles
- `paper/2col/`: final two-column manuscript source, generated tables, and PDF
- `checkpoints/research_runs/universe_A/`: Universe A cached run PKLs and progress manifest
- `checkpoints/research_runs/universe_B/`: Universe B cached run PKLs and progress manifest
- `results/`: timestamped research bundles
- `logs/`: timestamped run logs

## Environment Setup

### Option A: Standard `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-e2e.txt
```

`requirements-e2e.txt` already includes `requirements.txt`.

### Option B: Bootstrap Script

```bash
bash scripts/bootstrap_venv.sh --with-e2e
```

### Option C: `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-e2e.txt
```

## Useful Environment Variables

- `FRED_API_KEY`: enables better macro-data retrieval
- `SEC_USER_AGENT`: enables SEC company-facts enrichment
- `MPLCONFIGDIR=/tmp/matplotlib`: avoids Matplotlib cache permission problems

Example:

```bash
FRED_API_KEY="your_fred_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
python quant_research.py
```

## Running the Code

### 1. Single Pipeline Run

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_pipeline.py
```

This runs a single end-to-end backtest on the default active universe and writes pipeline-level plots in the repository root, including:

- `pipeline_alpha_models.png`
- `pipeline_performance.png`
- `pipeline_rl_analysis.png`
- `pipeline_execution_rl.png`

### 2. Full Research Evaluation

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_research.py
```

By default, this now runs:

1. the full frozen-bundle research suite for **Universe A**
2. the full frozen-bundle research suite for **Universe B**
3. a cross-universe transfer / meta-learning evaluation over the controller results

The runner prints the output bundle path for each universe and for the transfer stage.

### 3. Plot Refresh from an Existing Results Bundle

```bash
source .venv/bin/activate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib python scripts/generate_control_story_plots.py
```

This regenerates the higher-level paper figures inside the selected or latest results bundle, including:

- `control_method_overview.png`
- `control_split_heatmaps.png`
- `control_pareto_frontier.png`
- `control_interpretability.png`
- `control_mpc_diagnostic.png`
- `control_tail_diagnostic.png`
- `legacy_pruning_story.png`
- 
## Universes

Universe handling is now a first-class part of the repo.

- `quant_stack.config.get_universe_profile('A'|'B')` returns the full universe definition
- `quant_stack.config.use_universe('A'|'B')` temporarily swaps the active universe across the main modules
- `quant_research.py` uses that mechanism internally so both universes run through the same code path

The current paper bundles use:

- `results/20260330_221820_universe_A/`
- `results/20260331_002747_universe_B/`

## Results and Artifact Layout

Each primary research run writes a timestamped per-universe bundle:

- `results/<timestamp>_universe_A/`
- `results/<timestamp>_universe_B/`

Cross-universe transfer writes:

- `results/<timestamp>_transfer_A-B/`

Typical per-universe outputs include:

- `research_metrics.csv`
- `research_control_comparison.csv`
- `research_ablation_summary.csv`
- `research_robustness_summary.csv`
- `research_regime_summary.csv`
- `research_rolling_references.csv`
- `research_bootstrap_cis.csv`
- `research_bootstrap_significance.csv`
- `research_control_significance.csv`
- `research_jobson_korkie.csv`
- `research_ts_cv.csv`
- `research_summary.json`
- `research_paper_tables.tex`
- `control_method_overview.png`
- `control_split_heatmaps.png`
- `control_pareto_frontier.png`
- `control_interpretability.png`
- `control_mpc_diagnostic.png`
- `control_tail_diagnostic.png`
- `legacy_pruning_story.png`
- `pipeline_rolling_windows.png`
- `pipeline_reward_ablation.png`

Typical transfer outputs include:

- `meta_learning_dataset.csv`
- `meta_learning_transfer.json`

## Checkpoints and Reuse

Checkpoint reuse is a core part of the workflow.

Current layout:

- `checkpoints/research_runs/universe_A/`
- `checkpoints/research_runs/universe_B/`

Key properties:

- checkpoints are stored per run as `.pkl`
- progress manifests are now universe-local
- checkpoint lookup is backward-compatible with older naming schemes
- the default match mode is `compatible`, not brittle exact equality

That means old results are still reused when the important parts are unchanged:

- same universe
- same controller / experiment
- same effective suite
- same ticker columns
- same macro feature set

Small patch-level metadata changes or tiny trailing-date drifts no longer destroy checkpoint reuse unnecessarily.

## Cross-Universe Transfer / Meta-Learning

The repo now includes a transfer-style evaluation layer over controller behavior.

Implemented pieces:

- environment-level feature extraction from completed runs
- environment × controller dataset assembly
- cross-environment controller-transfer evaluation

Reported transfer outputs include:

- Kendall’s tau for ranking transfer
- top-1 / top-2 accuracy
- average regret

This is intended to support the broader question:

> can environment-aware meta-selection learn which controller family is best in a new setting?

## Paper Files

Built PDF:

- [paper/2col/quant_pipeline_paper_2col.pdf](paper/2col/quant_pipeline_paper_2col.pdf)

The repo now keeps only the two-column submission-style paper track.

## Reproducibility

The methodological contribution of the repo is the **frozen-bundle** workflow:

- shared alpha engine
- shared timing contract
- shared cost assumptions
- archived result bundles
- checkpointed reruns
- paper tables generated from the same archived outputs

The intent is not to make trading claims from ad hoc plots. It is to make controller comparisons reproducible under a shared empirical contract.

## Data Sources

- market prices and volumes: `yfinance`
- macro series: FRED when available
- SEC enrichment: company-facts API when `SEC_USER_AGENT` is set

These sources are convenient and reproducible, but they are still research-grade rather than full institutional point-in-time data feeds.

## Licensing

The repo uses a split-license setup:

- code: [LICENSE](LICENSE) (`MIT`)
- paper text, documentation, and repo-authored figures: [LICENSE-docs](LICENSE-docs) (`CC BY 4.0`)
- exclusions and notes: [NOTICE](NOTICE)

Important:

- external data are not relicensed by this repository
- third-party dependencies keep their own licenses
- generated artifacts derived from upstream data still need to respect provider terms

## Caveats

- This is research code, not production trading infrastructure.
- Results are sensitive to universe design, transaction-cost assumptions, and evaluation splits.
- The robust family is strongest in the current frozen bundle, but the repo does **not** claim universal optimality.
- The legacy RL-heavy branch is retained for transparency and negative-result analysis, not as the recommended architecture.

## Recommended Workflow

For day-to-day work:

1. Make the code or config change.
2. Run `quant_research.py`.
3. Let checkpoint reuse skip unchanged runs.
4. Regenerate the plot bundle if needed.
