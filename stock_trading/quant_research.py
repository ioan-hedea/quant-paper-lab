"""Thin wrapper for the research evaluation engine."""

from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_trading.quant_stack.data import load_market_data
from stock_trading.quant_stack.evaluation import run_research_evaluation


def main() -> None:
    warnings.filterwarnings("ignore")

    prices, volumes, returns, macro_data, sec_quality_scores = load_market_data()
    artifacts = run_research_evaluation(
        prices,
        volumes,
        returns,
        macro_data=macro_data,
        sec_quality_scores=sec_quality_scores,
    )

    print("\n" + "=" * 60)
    print("RESEARCH EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Metric rows: {artifacts['summary']['n_metric_rows']}")
    print(f"Regime rows: {artifacts['summary']['n_regime_rows']}")
    print("Generated research artifacts in stock_trading/:")
    print("  - research_metrics.csv")
    print("  - research_ablation_summary.csv")
    print("  - research_robustness_summary.csv")
    print("  - research_regime_summary.csv")
    print("  - research_rolling_references.csv")
    print("  - research_bootstrap_cis.csv")
    print("  - research_bootstrap_significance.csv")
    print("  - research_summary.json")
    print("  - research_paper_tables.tex")
    print("  - pipeline_research_eval.png")


if __name__ == "__main__":
    main()
