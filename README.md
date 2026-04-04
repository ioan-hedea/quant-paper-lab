# Quant Paper Lab

Research code and reproducible artifacts for:

> **Systematic Evaluation of Control Mechanisms for Factor-Based Portfolio Management**

---

## TL;DR

Holding the alpha engine fixed, portfolio control behaves as **risk shaping**:

- **optimization-based control dominates** across universes  
- **convex payoff shaping** is valuable under left-tail asymmetry  
- **learning contributes most as meta-control**, not as the primary decision layer  

All results are evaluated under a **shared, frozen empirical contract**:
same alpha, same timing, same costs, fully reproducible bundles.

---

## Why This Repo Exists

This project isolates one question:

> *If we hold alpha fixed, which control layer produces the strongest risk-adjusted outcomes?*

Rather than mixing signal discovery and execution, this repo treats control as a **first-class research object**.

---

## What Makes This Different

- fixed **finance-first alpha engine**
- explicit **alpha → allocator → control** decomposition
- **two disjoint universes** (zero ticker overlap)
- **frozen result bundles** with reproducible artifacts
- checkpoint-aware reruns for fast iteration
- paper tables and plots generated from archived outputs

---

## Current Empirical Picture

Across two universes (2013–2026):

- the **robust (CVaR-based) family** defines the overall frontier  
- **convexity-aware control** dominates under asymmetric downside  
- **meta-control (council / MLP gating)** improves adaptivity  
- **end-to-end RL does not outperform structured control**  

In short:

> **optimization wins → convexity helps → learning refines**

---
## Key Features

- two-universe evaluation protocol (`Universe A` and `Universe B`, zero ticker overlap)
- reproducible research bundles (`results/<timestamp>_universe_*`)
- checkpoint-aware reruns (`checkpoints/research_runs/...`)
- cross-universe transfer/meta-learning evaluation
- paper-ready plots and table exports

## Quickstart

### 1. Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install package

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[dev,e2e]'
```

Alternative bootstrap script:

```bash
bash scripts/bootstrap_venv.sh           # base + dev tools
bash scripts/bootstrap_venv.sh --with-e2e
```

## Command Reference

After install, these console commands are available:

- `quant-pipeline`: single walk-forward pipeline run
- `quant-research`: full research suite (both universes + transfer, by default)
- `quant-plots`: regenerate control-story figures for an existing results bundle

Equivalent direct script entrypoints remain available:

- `python quant_pipeline.py`
- `python quant_research.py`
- `python scripts/generate_control_story_plots.py`

## Typical Workflow

### 1) Run a single pipeline sanity check

```bash
MPLCONFIGDIR=/tmp/matplotlib quant-pipeline
```

### 2) Run the full frozen-bundle research suite

```bash
MPLCONFIGDIR=/tmp/matplotlib quant-research
```

### 3) Regenerate paper-facing control-story plots

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib quant-plots
```

## Output Layout

Per-universe outputs:

- `results/<timestamp>_universe_A/`
- `results/<timestamp>_universe_B/`

Transfer outputs:

- `results/<timestamp>_transfer_A-B/`

Common artifacts include:

- `research_metrics.csv`
- `research_control_comparison.csv`
- `research_ablation_summary.csv`
- `research_robustness_summary.csv`
- `research_bootstrap_cis.csv`
- `research_bootstrap_significance.csv`
- `research_control_significance.csv`
- `research_jobson_korkie.csv`
- `research_summary.json`
- `research_paper_tables.tex`
- `control_method_overview.png`
- `control_pareto_frontier.png`
- `control_split_heatmaps.png`
- `control_tail_diagnostic.png`

## Reproducibility Model

Reproducibility in this repo is built around frozen bundles plus checkpoint reuse:

- checkpoints are universe-local
- compatibility mode reuses runs when material metadata still matches
- manifest files track provenance and config hashes

This makes iterative research fast while preserving a stable comparison contract.

## Repository Structure

- `quant_stack/`: core package (data, alpha, controls, pipeline, evaluation)
- `quant_pipeline.py`: single-pipeline wrapper
- `quant_research.py`: full research runner
- `scripts/`: helper scripts (bootstrap, plotting, cross-universe)
- `paper/2col/`: manuscript source + generated PDF
- `checkpoints/`: reusable cached run payloads
- `results/`: timestamped archived artifacts
- `tests/`: regression + reliability tests

## Environment Variables

- `FRED_API_KEY`: enable richer macro features from FRED
- `SEC_USER_AGENT`: enable SEC company-facts enrichment
- `MPLCONFIGDIR=/tmp/matplotlib`: avoid Matplotlib cache permission issues in constrained environments
- `QUANT_LOG_FILE=/path/to/log.txt`: redirect tee logging output

Example:

```bash
FRED_API_KEY="your_fred_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
quant-research
```

## Paper and Latest Frozen Bundles

Paper files:

- [paper/2col/quant_pipeline_paper_2col.tex](paper/2col/quant_pipeline_paper_2col.tex)
- [paper/2col/quant_pipeline_paper_2col.pdf](paper/2col/quant_pipeline_paper_2col.pdf)

Latest archived frozen-bundle directories currently tracked in the repo:

- `results/20260330_221820_universe_A/`
- `results/20260331_002747_universe_B/`

Those bundles are based on daily data through **March 30, 2026**.

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
flake8 .
```

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Data and Scope Notes

- Data sources are research-grade convenience feeds (`yfinance`, FRED, SEC).
- This codebase is for research and reproducibility, not production trading deployment.
- Results remain sensitive to universe choice, transaction costs, and split design.

## License

- code: [MIT](LICENSE)
- docs/paper/repo-authored figures: [CC BY 4.0](LICENSE-docs)
- additional notes and exclusions: [NOTICE](NOTICE)

## Citation

If this repository contributes to your work, please cite the manuscript in [`paper/2col/`](paper/2col/).
