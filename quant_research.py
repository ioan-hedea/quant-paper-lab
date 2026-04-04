"""Thin wrapper for the research evaluation engine."""

from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_stack.config import EvaluationConfig, use_universe
from quant_stack.data import load_market_data
from quant_stack.evaluation import run_research_evaluation, run_cross_universe_evaluation
from logging_utils import tee_output_from_env

SEPARATOR_WIDTH = 60
SUMMARY_ARTIFACTS = (
    "research_metrics.csv",
    "research_ablation_summary.csv",
    "research_robustness_summary.csv",
    "research_regime_summary.csv",
    "research_rolling_references.csv",
    "research_bootstrap_cis.csv",
    "research_bootstrap_significance.csv",
    "research_control_significance.csv",
    "research_summary.json",
    "pipeline_research_eval.png",
    "control_pareto_frontier.png",
)


def _print_stage_header(title: str) -> None:
    print("\n" + "=" * SEPARATOR_WIDTH)
    print(title)
    print("=" * SEPARATOR_WIDTH)


def _print_artifact_summary(universe_id: str, primary_universe: str, artifacts: dict[str, object]) -> None:
    summary = artifacts["summary"]
    output_dir = Path(artifacts["output_dir"])

    _print_stage_header(f"RESEARCH EVALUATION COMPLETE: Universe {universe_id}")
    print(f"Metric rows: {summary['n_metric_rows']}")
    print(f"Regime rows: {summary['n_regime_rows']}")
    print(f"Output directory: {output_dir}")
    print("Generated research artifacts:")
    for name in SUMMARY_ARTIFACTS:
        print(f"  - {output_dir / name}")
    if universe_id == primary_universe:
        print("  - paper/2col/research_paper_tables.tex")


def main() -> None:
    warnings.filterwarnings("ignore")

    evaluation_config = EvaluationConfig()
    universe_ids = evaluation_config.meta_learning_universes or ("A",)
    primary_universe = universe_ids[0]

    for universe_id in universe_ids:
        _print_stage_header(f"PRIMARY RESEARCH EVALUATION: Universe {universe_id}")
        with use_universe(universe_id):
            prices, volumes, returns, macro_data, sec_quality_scores = load_market_data(
                universe_id=universe_id,
            )
            artifacts = run_research_evaluation(
                prices,
                volumes,
                returns,
                macro_data=macro_data,
                sec_quality_scores=sec_quality_scores,
                evaluation_config=evaluation_config,
                universe_id=universe_id,
                export_paper_tables=(universe_id == primary_universe),
            )

        _print_artifact_summary(universe_id, primary_universe, artifacts)

    if len(universe_ids) > 1:
        _print_stage_header("CROSS-UNIVERSE META-LEARNING EVALUATION")
        cross_results = run_cross_universe_evaluation(
            universe_ids=universe_ids,
            evaluation_config=evaluation_config,
        )
        print(f"  Meta-learning output: {cross_results['output_dir']}")


if __name__ == "__main__":
    with tee_output_from_env("quant_research"):
        main()
