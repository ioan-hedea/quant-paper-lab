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


def main() -> None:
    warnings.filterwarnings("ignore")

    evaluation_config = EvaluationConfig()
    universe_ids = evaluation_config.meta_learning_universes or ('A',)
    primary_universe = universe_ids[0]

    for universe_id in universe_ids:
        print("\n" + "=" * 60)
        print(f"PRIMARY RESEARCH EVALUATION: Universe {universe_id}")
        print("=" * 60)
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

        print("\n" + "=" * 60)
        print(f"RESEARCH EVALUATION COMPLETE: Universe {universe_id}")
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
        if universe_id == primary_universe:
            print("  - paper/2col/research_paper_tables.tex")

    if len(universe_ids) > 1:
        print("\n" + "=" * 60)
        print("CROSS-UNIVERSE META-LEARNING EVALUATION")
        print("=" * 60)
        cross_results = run_cross_universe_evaluation(
            universe_ids=universe_ids,
            evaluation_config=evaluation_config,
        )
        print(f"  Meta-learning output: {cross_results['output_dir']}")


if __name__ == "__main__":
    with tee_output_from_env("quant_research"):
        main()
