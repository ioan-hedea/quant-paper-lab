"""Thin wrapper for the research evaluation engine."""

from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_stack.data import load_market_data
from quant_stack.evaluation import run_research_evaluation
from logging_utils import tee_output_from_env


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
    output_dir = Path(artifacts['output_dir'])
    print(f"Output directory: {output_dir}")
    print("Generated research artifacts:")
    print(f"  - {output_dir / 'research_metrics.csv'}")
    print(f"  - {output_dir / 'research_ablation_summary.csv'}")
    print(f"  - {output_dir / 'research_robustness_summary.csv'}")
    print(f"  - {output_dir / 'research_regime_summary.csv'}")
    print(f"  - {output_dir / 'research_rolling_references.csv'}")
    print(f"  - {output_dir / 'research_bootstrap_cis.csv'}")
    print(f"  - {output_dir / 'research_bootstrap_significance.csv'}")
    print(f"  - {output_dir / 'research_control_significance.csv'}")
    print(f"  - {output_dir / 'research_summary.json'}")
    print(f"  - {output_dir / 'pipeline_research_eval.png'}")
    print(f"  - {output_dir / 'control_pareto_frontier.png'}")
    print("  - paper/2col/research_paper_tables.tex")


if __name__ == "__main__":
    with tee_output_from_env("quant_research"):
        main()
