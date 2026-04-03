"""Result publishing helpers for research evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation_helpers import _display_label, _latex_pct, _table_label
from .plots import plot_research_evaluation, plot_reward_ablation, plot_rolling_windows


def _write_research_tables(
    ablation_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    output_path: Path,
    bootstrap_significance: pd.DataFrame | None = None,
) -> None:
    lines: list[str] = []

    if not ablation_summary.empty:
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Ablation summary across train/test split settings.}",
            r"\label{tab:ablation_summary}",
            r"\begin{tabular}{@{} l r r r r r @{}}",
            r"\toprule",
            r"Component & Mean Return & Mean Vol. & Mean Sharpe & Mean Max DD & Mean Calmar \\",
            r"\midrule",
        ])
        for _, row in ablation_summary.iterrows():
            lines.append(
                f"{_table_label(str(row['component_label']))} & "
                f"{_latex_pct(float(row['mean_return']))} & "
                f"{_latex_pct(float(row['mean_vol']))} & "
                f"{row['mean_sharpe']:.2f} & "
                f"${_latex_pct(float(row['mean_max_drawdown']))}$ & "
                f"{row['mean_calmar']:.2f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    if not robustness_summary.empty:
        row = robustness_summary.iloc[0]
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Rolling-window robustness summary for the full pipeline.}",
            r"\label{tab:robustness_summary}",
            r"\begin{tabular}{@{} l r @{}}",
            r"\toprule",
            r"Statistic & Value \\",
            r"\midrule",
            f"Rolling windows & {int(row['rolling_window_count'])} \\\\",
            f"Median full-pipeline Sharpe & {row['median_full_sharpe']:.2f} \\\\",
            f"Median full-pipeline Calmar & {row['median_full_calmar']:.2f} \\\\",
            f"Median full-pipeline Max DD & ${_latex_pct(float(row['median_full_max_drawdown']))}$ \\\\",
        ])
        if 'frac_full_beats_factor_drawdown' in row:
            lines.append(f"Frac. windows full beats factor on drawdown & {_latex_pct(float(row['frac_full_beats_factor_drawdown']), digits=0)} \\\\")
        if 'frac_full_beats_factor_calmar' in row:
            lines.append(f"Frac. windows full beats factor on Calmar & {_latex_pct(float(row['frac_full_beats_factor_calmar']), digits=0)} \\\\")
        if 'frac_full_beats_spy_sharpe' in row:
            lines.append(f"Frac. windows full beats SPY on Sharpe & {_latex_pct(float(row['frac_full_beats_spy_sharpe']), digits=0)} \\\\")
        if 'frac_full_beats_spy_calmar' in row:
            lines.append(f"Frac. windows full beats SPY on Calmar & {_latex_pct(float(row['frac_full_beats_spy_calmar']), digits=0)} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    if bootstrap_significance is not None and not bootstrap_significance.empty:
        lines.extend([
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Block-bootstrap significance for full pipeline versus baselines (Sharpe deltas).}",
            r"\label{tab:bootstrap_significance}",
            r"\begin{tabular}{@{} l r r r @{}}",
            r"\toprule",
            r"Comparison & Delta Sharpe & 95\% CI & p-value \\",
            r"\midrule",
        ])
        for _, row in bootstrap_significance.iterrows():
            delta = float(row['delta_sharpe_point'])
            low = float(row['delta_sharpe_ci_low'])
            high = float(row['delta_sharpe_ci_high'])
            pval = float(row['delta_sharpe_pvalue_two_sided'])
            comp = f"Full - {_table_label(str(row['compare_label']))}"
            lines.append(f"{comp} & {delta:.2f} & [{low:.2f}, {high:.2f}] & {pval:.3f} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding='utf-8')


def publish_research_outputs(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    rolling_references: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    control_comparison_summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    bootstrap_cis: pd.DataFrame,
    bootstrap_significance: pd.DataFrame,
    control_significance: pd.DataFrame,
    baseline_results: dict[str, object] | None,
    summary: dict[str, object],
    export_paper_tables: bool,
) -> None:
    metrics.to_csv(output_dir / 'research_metrics.csv', index=False)
    regime_summary.to_csv(output_dir / 'research_regime_summary.csv', index=False)
    rolling_references.to_csv(output_dir / 'research_rolling_references.csv', index=False)
    ablation_summary.to_csv(output_dir / 'research_ablation_summary.csv', index=False)
    if not control_comparison_summary.empty:
        control_comparison_summary.to_csv(output_dir / 'research_control_comparison.csv', index=False)
        print("\n  Control Comparison Summary:")
        for _, row in control_comparison_summary.iterrows():
            print(f"    {row['component_label']:25s}  Sharpe={row['mean_sharpe']:.2f}  "
                  f"Calmar={row['mean_calmar']:.2f}  MaxDD={row['mean_max_drawdown']:.1%}")
    if not execution_summary.empty:
        execution_summary.to_csv(output_dir / 'research_execution_summary.csv', index=False)
    robustness_summary.to_csv(output_dir / 'research_robustness_summary.csv', index=False)
    if not bootstrap_cis.empty:
        bootstrap_cis.to_csv(output_dir / 'research_bootstrap_cis.csv', index=False)
    if not bootstrap_significance.empty:
        bootstrap_significance.to_csv(output_dir / 'research_bootstrap_significance.csv', index=False)
    if not control_significance.empty:
        control_significance.to_csv(output_dir / 'research_control_significance.csv', index=False)

    if export_paper_tables:
        paper_output = Path(__file__).resolve().parents[1] / 'paper' / '2col' / 'research_paper_tables.tex'
        _write_research_tables(ablation_summary, robustness_summary, bootstrap_significance=bootstrap_significance, output_path=paper_output)
    _write_research_tables(ablation_summary, robustness_summary, bootstrap_significance=bootstrap_significance, output_path=output_dir / 'research_paper_tables.tex')

    plot_research_evaluation(
        metrics, regime_summary, baseline_results=baseline_results,
        rolling_references=rolling_references if not rolling_references.empty else None,
        output_path=output_dir / 'pipeline_research_eval.png',
        frontier_output_path=output_dir / 'control_pareto_frontier.png',
    )
    plot_rolling_windows(
        metrics,
        rolling_references_df=rolling_references if not rolling_references.empty else None,
        output_path=output_dir / 'pipeline_rolling_windows.png',
    )
    plot_reward_ablation(metrics, output_path=output_dir / 'pipeline_reward_ablation.png')

    with (output_dir / 'research_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    try:
        from scripts.generate_control_story_plots import generate_plot_set

        generate_plot_set(output_dir)
    except Exception as exc:
        universe_id = summary.get('universe_id', '?')
        print(f"Story-plot generation failed for Universe {universe_id}: {exc}")
