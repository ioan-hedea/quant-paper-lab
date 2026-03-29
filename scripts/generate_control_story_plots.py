from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]


DISPLAY_LABELS = {
    "alpha_engine_no_control": "No control\n(alpha engine)",
    "factor_only": "Factor-only",
    "A1_fixed": "A1 Fixed",
    "A2_vol_target": "A2 Vol-target",
    "A3_dd_delever": "A3 DD-delever",
    "A4_regime_rules": "A4 Regime rules",
    "A5_ensemble_mean": "A5 Ensemble mean",
    "A5_ensemble_min": "A5 Ensemble min",
    "B1_linucb": "B1 LinUCB",
    "B2_thompson": "B2 Thompson",
    "B3_epsilon_greedy": "B3 Eps-greedy",
    "C_supervised": "C Supervised",
    "D_cvar_robust": "D CVaR robust",
    "D_plus_convexity": "D+ Convexity",
    "E_council": "E Council",
    "E_plus_convexity": "E+ Council + convexity",
    "F_cmdp_lagrangian": "F CMDP-Lagrangian",
    "RL_q_learning": "RL Q-learning",
    "alpha_stack_fixed_weights": "Alpha stack fixed",
    "alpha_stack_no_rl": "Alpha stack adaptive",
    "portfolio_rl_fixed_weights": "Portfolio RL fixed alpha",
    "full_pipeline_fixed_weights": "Full pipeline fixed alpha",
    "full_pipeline": "Full pipeline",
}


FAMILY_LABELS = {
    "baseline": "Baseline",
    "rule": "Rule-based",
    "bandit": "Bandit",
    "supervised": "Supervised",
    "robust": "Robust",
    "meta": "Meta-control",
    "safe_rl": "Safe RL",
    "rl": "RL",
    "legacy": "Legacy",
}


FAMILY_COLORS = {
    "baseline": "#4D4D4D",
    "rule": "#2A9D8F",
    "bandit": "#457B9D",
    "supervised": "#E9C46A",
    "robust": "#D1495B",
    "meta": "#3B8EA5",
    "safe_rl": "#8C6D31",
    "rl": "#7B2CBF",
    "legacy": "#6C757D",
}


LEGACY_PATH = [
    "factor_only",
    "alpha_stack_fixed_weights",
    "alpha_stack_no_rl",
    "portfolio_rl_fixed_weights",
    "full_pipeline_fixed_weights",
    "full_pipeline",
]


SELECTED_METHODS = [
    "alpha_engine_no_control",
    "A4_regime_rules",
    "B1_linucb",
    "C_supervised",
    "D_cvar_robust",
    "F_cmdp_lagrangian",
    "RL_q_learning",
]


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " "))


def family_of(label: str) -> str:
    if label in {"factor_only", "alpha_engine_no_control"}:
        return "baseline"
    if label.startswith("A"):
        return "rule"
    if label.startswith("B"):
        return "bandit"
    if label.startswith("C"):
        return "supervised"
    if label.startswith("D"):
        return "robust"
    if label.startswith("E"):
        return "meta"
    if label.startswith("F"):
        return "safe_rl"
    if label.startswith("RL"):
        return "rl"
    return "legacy"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    control = pd.read_csv(ROOT / "research_control_comparison.csv")
    ablation = pd.read_csv(ROOT / "research_ablation_summary.csv")
    metrics = pd.read_csv(ROOT / "research_metrics.csv")
    with open(ROOT / "research_summary.json") as f:
        summary = json.load(f)

    control["component_label"] = control["component_label"].replace({"factor_only": "alpha_engine_no_control"})
    detail = metrics[metrics["suite"] == "control_comparison"].copy()
    detail["base_label"] = detail["label"].str.rsplit("_tf", n=1).str[0]
    detail["base_label"] = detail["base_label"].replace({"factor_only": "alpha_engine_no_control"})
    detail["train_frac"] = detail["param_value"].astype(float)
    detail["family"] = detail["base_label"].map(family_of)

    control["family"] = control["component_label"].map(family_of)
    ablation["family"] = "legacy"
    return control, ablation, detail, summary


def add_family_legend(ax) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS[key],
               markeredgecolor="black", markersize=8, label=FAMILY_LABELS[key])
        for key in ["baseline", "rule", "bandit", "supervised", "robust", "meta", "safe_rl", "rl"]
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["drawdown_abs"] = ranked["mean_max_drawdown"].abs()
    ranked = ranked.sort_values(["drawdown_abs", "mean_return"], ascending=[True, False])
    frontier_rows: list[dict] = []
    best_return = -np.inf
    for _, row in ranked.iterrows():
        value = float(row["mean_return"])
        if value > best_return + 1e-12:
            frontier_rows.append(row.to_dict())
            best_return = value
    return pd.DataFrame(frontier_rows)


def plot_control_overview(
    control: pd.DataFrame,
    detail: pd.DataFrame,
    summary: dict,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Control Comparison Overview: Winner, Trade-offs, and Stability",
        fontsize=18,
        fontweight="bold",
    )

    ranked = control.sort_values("mean_sharpe", ascending=True).reset_index(drop=True)

    ax = axes[0, 0]
    colors = [FAMILY_COLORS[family_of(label)] for label in ranked["component_label"]]
    ax.barh([display_label(x) for x in ranked["component_label"]], ranked["mean_sharpe"], color=colors, alpha=0.9)
    ax.axvline(
        float(control.loc[control["component_label"] == "alpha_engine_no_control", "mean_sharpe"].iloc[0]),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    for y, val in enumerate(ranked["mean_sharpe"]):
        ax.text(val + 0.01, y, f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Mean Sharpe by Control Method", fontweight="bold")
    ax.set_xlabel("Mean Sharpe")
    add_family_legend(ax)

    ax = axes[0, 1]
    offsets = [(7, 5), (7, -10), (7, 11), (7, -13), (7, 3), (7, -4)]
    scatter_source = control.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    for idx, row in scatter_source.iterrows():
        label = str(row["component_label"])
        x = abs(float(row["mean_max_drawdown"])) * 100
        y = float(row["mean_return"]) * 100
        s = 80 + float(row["mean_sharpe"]) * 220
        ax.scatter(
            x,
            y,
            s=s,
            color=FAMILY_COLORS[family_of(label)],
            edgecolors="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(display_label(label), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8)
    frontier = pareto_frontier(control)
    if not frontier.empty and len(frontier) >= 2:
        ax.plot(
            frontier["mean_max_drawdown"].abs() * 100,
            frontier["mean_return"] * 100,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )
    ax.set_title("Return vs Drawdown Trade-off", fontweight="bold")
    ax.set_xlabel("Absolute mean max drawdown (%)")
    ax.set_ylabel("Mean annualized return (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.text(0.98, 0.03, "Better", ha="right", va="bottom", transform=ax.transAxes, fontsize=9, alpha=0.7)

    ax = axes[1, 0]
    representative_rows = []
    representative_specs = [
        ("Alpha engine", control["component_label"] == "alpha_engine_no_control"),
        ("Best rule", control["component_label"].str.startswith("A")),
        ("Best bandit", control["component_label"].str.startswith("B")),
        ("Supervised", control["component_label"] == "C_supervised"),
        ("Robust", control["component_label"].str.startswith("D")),
        ("Meta-control", control["component_label"].str.startswith("E")),
        ("Safe RL", control["component_label"].str.startswith("F")),
        ("RL", control["component_label"] == "RL_q_learning"),
    ]
    for plot_label, mask in representative_specs:
        subset = control[mask].sort_values("mean_sharpe", ascending=False)
        if subset.empty:
            continue
        row = subset.iloc[0].copy()
        row["plot_label"] = plot_label
        representative_rows.append(row)
    family_representatives = pd.DataFrame(representative_rows)
    bar_colors = [FAMILY_COLORS[family_of(x)] for x in family_representatives["component_label"]]
    ax.bar(family_representatives["plot_label"], family_representatives["mean_sharpe"], color=bar_colors, alpha=0.92)
    for x, val in enumerate(family_representatives["mean_sharpe"]):
        ax.text(x, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_title("Best Method in Each Family", fontweight="bold")
    ax.set_ylabel("Mean Sharpe")
    ax.tick_params(axis="x", rotation=12)

    ax = axes[1, 1]
    selected = detail[detail["base_label"].isin(SELECTED_METHODS)].copy()
    for label in SELECTED_METHODS:
        group = selected[selected["base_label"] == label].sort_values("train_frac")
        if group.empty:
            continue
        ax.plot(
            group["train_frac"],
            group["sharpe"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=FAMILY_COLORS[family_of(label)],
            label=display_label(label),
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Sharpe Across Train Fractions", fontweight="bold")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Sharpe")
    xticks = sorted(selected["train_frac"].dropna().unique().tolist())
    if xticks:
        ax.set_xticks(xticks)
    ax.legend(fontsize=8, ncol=2)

    best = summary["best_sharpe_by_suite"]["control_comparison"]
    fig.text(
        0.5,
        0.015,
        (
            f"Best single control run: {best['label']} | "
            f"Sharpe {best['sharpe']:.3f} | "
            f"Annualized return {best['ann_return'] * 100:.2f}% | "
            f"Max drawdown {best['max_drawdown'] * 100:.1f}%"
        ),
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_control_heatmaps(control: pd.DataFrame, detail: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("default")
    ordered_labels = control.sort_values("mean_sharpe", ascending=False)["component_label"].tolist()
    ordered_names = [display_label(x) for x in ordered_labels]

    sharpe = (
        detail.pivot_table(index="base_label", columns="train_frac", values="sharpe", aggfunc="mean")
        .reindex(ordered_labels)
    )
    drawdown = (
        detail.assign(drawdown_abs=detail["max_drawdown"].abs())
        .pivot_table(index="base_label", columns="train_frac", values="drawdown_abs", aggfunc="mean")
        .reindex(ordered_labels)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(
        f"Split-by-Split Stability Across the {len(ordered_labels)} Control Methods",
        fontsize=18,
        fontweight="bold",
    )

    sharpe_values = sharpe.values
    im = axes[0].imshow(sharpe_values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.95)
    axes[0].set_title("Sharpe by Train Fraction", fontweight="bold")
    axes[0].set_xticks(range(sharpe.shape[1]), [f"{c:.1f}" for c in sharpe.columns])
    axes[0].set_yticks(range(len(ordered_names)), ordered_names)
    axes[0].set_xlabel("Train fraction")
    for i in range(sharpe.shape[0]):
        for j in range(sharpe.shape[1]):
            val = sharpe_values[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    dd_values = drawdown.values * 100
    im2 = axes[1].imshow(dd_values, aspect="auto", cmap="RdYlGn_r", vmin=10, vmax=32)
    axes[1].set_title("Absolute Max Drawdown by Train Fraction", fontweight="bold")
    axes[1].set_xticks(range(drawdown.shape[1]), [f"{c:.1f}" for c in drawdown.columns])
    axes[1].set_yticks(range(len(ordered_names)), ordered_names)
    axes[1].set_xlabel("Train fraction")
    for i in range(drawdown.shape[0]):
        for j in range(drawdown.shape[1]):
            val = dd_values[i, j]
            axes[1].text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_legacy_pruning(ablation: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    path_df = ablation.set_index("component_label").loc[LEGACY_PATH].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.suptitle("Legacy Pruning Story: Why the Architecture Was Simplified", fontsize=18, fontweight="bold")

    ax = axes[0]
    ranked = ablation.sort_values("mean_sharpe", ascending=True)
    ax.barh(
        [display_label(x) for x in ranked["component_label"]],
        ranked["mean_sharpe"],
        color=FAMILY_COLORS["legacy"],
        alpha=0.88,
    )
    for y, val in enumerate(ranked["mean_sharpe"]):
        ax.text(val + 0.01, y, f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Legacy Stack Components by Mean Sharpe", fontweight="bold")
    ax.set_xlabel("Mean Sharpe")

    ax = axes[1]
    x = path_df["mean_max_drawdown"].abs().to_numpy() * 100
    y = path_df["mean_return"].to_numpy() * 100
    ax.plot(x, y, color="#6C757D", linewidth=2, alpha=0.8)
    for i, row in path_df.iterrows():
        ax.scatter(
            abs(float(row["mean_max_drawdown"])) * 100,
            float(row["mean_return"]) * 100,
            s=130,
            color="#6C757D",
            edgecolors="black",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            display_label(str(row["component_label"])),
            (abs(float(row["mean_max_drawdown"])) * 100, float(row["mean_return"]) * 100),
            xytext=(6, 5 if i % 2 == 0 else -11),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Legacy Path: Return Sacrificed for Modest Risk Relief", fontweight="bold")
    ax.set_xlabel("Absolute mean max drawdown (%)")
    ax.set_ylabel("Mean annualized return (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_control_pareto(control: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    frontier = pareto_frontier(control)
    ranked = control.copy()
    ranked["drawdown_abs"] = ranked["mean_max_drawdown"].abs() * 100
    ranked["return_pct"] = ranked["mean_return"] * 100

    for _, row in ranked.iterrows():
        label = str(row["component_label"])
        ax.scatter(
            float(row["drawdown_abs"]),
            float(row["return_pct"]),
            s=120,
            color=FAMILY_COLORS[family_of(label)],
            edgecolors="black",
            linewidth=0.7,
            alpha=0.92,
        )
        ax.annotate(
            display_label(label),
            (float(row["drawdown_abs"]), float(row["return_pct"])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    if not frontier.empty:
        ax.plot(
            frontier["mean_max_drawdown"].abs() * 100,
            frontier["mean_return"] * 100,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Pareto frontier",
        )
        ax.legend(loc="lower right", fontsize=9)

    ax.set_title("Pareto Frontier in Return-Drawdown Space", fontweight="bold")
    ax.set_xlabel("Absolute mean max drawdown (%)")
    ax.set_ylabel("Mean annualized return (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    control, ablation, detail, summary = load_data()

    outputs = {
        "control_overview": ROOT / "control_method_overview.png",
        "control_heatmaps": ROOT / "control_split_heatmaps.png",
        "control_pareto": ROOT / "control_pareto_frontier.png",
        "legacy_pruning": ROOT / "legacy_pruning_story.png",
    }

    plot_control_overview(control, detail, summary, outputs["control_overview"])
    plot_control_heatmaps(control, detail, outputs["control_heatmaps"])
    plot_control_pareto(control, outputs["control_pareto"])
    plot_legacy_pruning(ablation, outputs["legacy_pruning"])

    print("Saved plot set:")
    for path in outputs.values():
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
