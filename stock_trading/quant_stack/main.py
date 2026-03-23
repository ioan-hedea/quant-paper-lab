"""CLI entrypoint for the quant trading pipeline."""

from __future__ import annotations

import warnings

from .data import load_market_data
from .pipeline import run_full_pipeline
from .plots import plot_alpha_models, plot_execution_demo, plot_performance, plot_rl_analysis


def main() -> None:
    warnings.filterwarnings("ignore")

    prices, volumes, returns, macro_data, sec_quality_scores = load_market_data()

    results = run_full_pipeline(
        prices, volumes, returns,
        macro_data=macro_data,
        sec_quality_scores=sec_quality_scores,
    )

    print("\n" + "=" * 60)
    print("STAGE 3: Generating Visualizations")
    print("=" * 60)

    print("  Plotting alpha model decomposition...")
    plot_alpha_models(results, prices, returns)

    print("  Plotting performance comparison...")
    plot_performance(results)

    print("  Plotting RL component analysis...")
    plot_rl_analysis(results)

    plot_execution_demo()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Generated plots in stock_trading/:")
    print("  - pipeline_alpha_models.png  (alpha model decomposition)")
    print("  - pipeline_performance.png   (performance vs benchmarks)")
    print("  - pipeline_rl_analysis.png   (RL component deep dive)")
    print("  - pipeline_execution_rl.png  (execution optimization)")


if __name__ == "__main__":
    main()
