# Quant Paper Lab

This repository contains a research-style quantitative trading pipeline and the associated paper artifacts.

## Repository Layout

- `quant_stack/`: modular pipeline package (`data`, `alpha`, `rl`, `pipeline`, `evaluation`, `plots`).
- `quant_pipeline.py`: main pipeline entrypoint.
- `quant_research.py`: ablation and robustness evaluation entrypoint.
- `paper/1col/`: single-column paper source and build artifacts.
- `paper/2col/`: two-column paper source, paper-ready tables, and build artifacts.
- `results/`, `logs/`, `checkpoints/`: run artifacts and progress state.

## Environment Setup

### Option A: Manual venv creation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional end-to-end RL extras:

```bash
python -m pip install -r requirements-e2e.txt
```

### Option B: One-command bootstrap

```bash
bash scripts/bootstrap_venv.sh
```

With optional E2E extras:

```bash
bash scripts/bootstrap_venv.sh --with-e2e
```

## Run

Baseline pipeline run:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_pipeline.py
```

With optional macro and SEC enrichment:

```bash
FRED_API_KEY="your_key" \
SEC_USER_AGENT="Your Name your_email@example.com" \
MPLCONFIGDIR=/tmp/matplotlib \
python quant_pipeline.py
```

Research evaluation run:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python quant_research.py
```

Follow latest research log:

```bash
tail -f logs/latest_quant_research.log
```

## Outputs

Pipeline run generates, among others:

- `pipeline_alpha_models.png`
- `pipeline_performance.png`
- `pipeline_rl_analysis.png`
- `pipeline_execution_rl.png`

Research run generates, among others:

- `research_metrics.csv`
- `research_ablation_summary.csv`
- `research_robustness_summary.csv`
- `research_regime_summary.csv`
- `research_rolling_references.csv`
- `research_summary.json`
- `paper/2col/research_paper_tables.tex`
- `pipeline_research_eval.png`

## Paper Files

- 1-column source: `paper/1col/quant_pipeline_report.tex`
- 2-column source: `paper/2col/quant_pipeline_report_2col.tex`
- 2-column PDF: `paper/2col/quant_pipeline_paper.pdf`

## Data Sources

- Prices and volumes: `yfinance`
- Macro (preferred): FRED (`FRED_API_KEY`)
- Macro fallback: BLS + U.S. Treasury Fiscal Data
- Fundamentals: SEC company facts (`SEC_USER_AGENT`)

## Caveat

This is research code and backtest infrastructure, not production trading software.
