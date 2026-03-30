"""Run the full controller suite on both universes (A + B) for meta-learning evaluation.

Usage::

    python scripts/run_cross_universe.py          # both universes
    python scripts/run_cross_universe.py --universe B   # single universe
"""

from pathlib import Path
import sys
import warnings

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_stack.config import EvaluationConfig, PipelineConfig
from quant_stack.evaluation import run_cross_universe_evaluation
from logging_utils import tee_output_from_env


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-universe controller evaluation")
    parser.add_argument(
        "--universe", nargs="+", default=["A", "B"],
        help="Universe IDs to evaluate (default: A B)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    universe_ids = tuple(args.universe)
    print(f"Running cross-universe evaluation on: {universe_ids}")

    results = run_cross_universe_evaluation(
        universe_ids=universe_ids,
        base_config=PipelineConfig(),
        evaluation_config=EvaluationConfig(),
    )

    print(f"\nMeta-learning dataset saved to: {results['output_dir']}")


if __name__ == "__main__":
    with tee_output_from_env("cross_universe"):
        main()
