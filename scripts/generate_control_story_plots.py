from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    "G_mlp_meta": "G MLP meta",
    "G_plus_convexity": "G+ MLP meta + convexity",
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
    "D_cvar_robust",
    "D_plus_convexity",
    "E_council",
    "G_mlp_meta",
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
    if label.startswith("E") or label.startswith("G"):
        return "meta"
    if label.startswith("F"):
        return "safe_rl"
    if label.startswith("RL"):
        return "rl"
    return "legacy"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def resolve_results_dir() -> Path:
    results_root = ROOT / "results"
    candidates = sorted(
        [path for path in results_root.iterdir() if path.is_dir() and (path / "research_summary.json").exists()],
        key=lambda path: path.name,
    ) if results_root.exists() else []
    if candidates:
        return candidates[-1]
    return ROOT


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    results_dir = resolve_results_dir()
    control = pd.read_csv(results_dir / "research_control_comparison.csv")
    ablation = pd.read_csv(results_dir / "research_ablation_summary.csv")
    metrics = pd.read_csv(results_dir / "research_metrics.csv")
    with open(results_dir / "research_summary.json") as f:
        summary = json.load(f)

    control["component_label"] = control["component_label"].replace({"factor_only": "alpha_engine_no_control"})
    detail = metrics[metrics["suite"] == "control_comparison"].copy()
    detail["base_label"] = detail["label"].str.rsplit("_tf", n=1).str[0]
    detail["base_label"] = detail["base_label"].replace({"factor_only": "alpha_engine_no_control"})
    detail["train_frac"] = detail["param_value"].astype(float)
    detail["family"] = detail["base_label"].map(family_of)

    control["family"] = control["component_label"].map(family_of)
    ablation["family"] = "legacy"
    summary["results_dir"] = str(results_dir)
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


def _spread_positions(values: list[float], low: float, high: float, min_gap: float) -> list[float]:
    if not values:
        return []
    placed = []
    for value in values:
        if not placed:
            placed.append(max(value, low))
        else:
            placed.append(max(value, placed[-1] + min_gap))
    overflow = placed[-1] - high
    if overflow > 0:
        placed = [value - overflow for value in placed]
    if placed[0] < low:
        shift = low - placed[0]
        placed = [value + shift for value in placed]
    for idx in range(1, len(placed)):
        placed[idx] = max(placed[idx], placed[idx - 1] + min_gap)
    return [min(value, high) for value in placed]


def _annotate_scatter_columns(
    ax,
    rows: list[dict],
    x_key: str,
    y_key: str,
    x_left: float,
    x_right: float,
    y_low: float,
    y_high: float,
    min_gap: float,
) -> None:
    if not rows:
        return
    x_mid = float(np.median([float(row[x_key]) for row in rows]))
    left_rows = sorted([row for row in rows if float(row[x_key]) <= x_mid], key=lambda row: float(row[y_key]))
    right_rows = sorted([row for row in rows if float(row[x_key]) > x_mid], key=lambda row: float(row[y_key]))

    for side_rows, x_text, ha in ((left_rows, x_left, "left"), (right_rows, x_right, "right")):
        y_targets = [float(row[y_key]) for row in side_rows]
        y_positions = _spread_positions(y_targets, y_low, y_high, min_gap)
        for row, y_text in zip(side_rows, y_positions):
            x_point = float(row[x_key])
            y_point = float(row[y_key])
            label = display_label(str(row["component_label"]))
            ax.annotate(
                label,
                xy=(x_point, y_point),
                xytext=(x_text, y_text),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="0.75"),
                arrowprops=dict(arrowstyle="-", color="0.45", linewidth=0.8, shrinkA=3, shrinkB=4),
            )


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
    scatter_source = control.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    scatter_rows = []
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
        scatter_rows.append({"component_label": label, "x": x, "y": y})
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
    x_vals = [row["x"] for row in scatter_rows]
    y_vals = [row["y"] for row in scatter_rows]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_pad = 1.8
    y_pad = 1.2
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    _annotate_scatter_columns(
        ax,
        [{"component_label": row["component_label"], "drawdown_abs": row["x"], "return_pct": row["y"]} for row in scatter_rows],
        x_key="drawdown_abs",
        y_key="return_pct",
        x_left=x_min - x_pad + 0.4,
        x_right=x_max + x_pad - 0.4,
        y_low=y_min - y_pad + 0.6,
        y_high=y_max + y_pad - 0.6,
        min_gap=2.2,
    )
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
        (
            "Meta-control",
            control["component_label"].str.startswith("E")
            | control["component_label"].str.startswith("G"),
        ),
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

    label_rows = []
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
        label_rows.append(row.to_dict())

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
    x_min = float(ranked["drawdown_abs"].min())
    x_max = float(ranked["drawdown_abs"].max())
    y_min = float(ranked["return_pct"].min())
    y_max = float(ranked["return_pct"].max())
    x_pad = 1.6
    y_pad = 0.7
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    _annotate_scatter_columns(
        ax,
        label_rows,
        x_key="drawdown_abs",
        y_key="return_pct",
        x_left=x_min - x_pad + 0.25,
        x_right=x_max + x_pad - 0.25,
        y_low=y_min - y_pad + 0.5,
        y_high=y_max + y_pad - 0.3,
        min_gap=1.6,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_checkpoint_results(label: str) -> dict | None:
    path = ROOT / "checkpoints" / "research_runs" / f"{label}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj.get("results", obj)


def plot_interpretability(out_path: Path) -> None:
    meta = _load_checkpoint_results("control_G_mlp_meta_tf0.50")
    meta_prefix = "mlp_meta"
    meta_title = "MLP Meta Weights Over Time (G, tf=0.50)"
    meta_note = "Mean weights:\nCVaR {cvar:.2f}\nRules {rules:.2f}\nLinUCB {linucb:.2f}"
    if meta is None:
        meta = _load_checkpoint_results("control_E_council_tf0.50")
        meta_prefix = "council"
        meta_title = "Council Weights Over Time (E, tf=0.50)"
    convex = _load_checkpoint_results("control_D_plus_convexity_tf0.50")
    if meta is None or convex is None:
        print("plot_interpretability: missing cached checkpoints; skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Interpretability: Learned Meta-Control and Convexity Activation", fontsize=17, fontweight="bold")

    ax = axes[0]
    dates = pd.to_datetime(meta.get("dates", []))
    w_regime = np.asarray(meta.get(f"{meta_prefix}_weight_regime_rules", []), dtype=float)
    w_linucb = np.asarray(meta.get(f"{meta_prefix}_weight_linucb", []), dtype=float)
    w_cvar = np.asarray(meta.get(f"{meta_prefix}_weight_cvar_robust", []), dtype=float)
    n_obs = min(len(dates), len(w_regime), len(w_linucb), len(w_cvar))
    dates = dates[:n_obs]
    w_regime = w_regime[:n_obs]
    w_linucb = w_linucb[:n_obs]
    w_cvar = w_cvar[:n_obs]

    ax.stackplot(
        dates,
        w_regime,
        w_linucb,
        w_cvar,
        labels=["Regime rules", "LinUCB", "CVaR robust"],
        colors=["#2A9D8F", "#457B9D", "#D1495B"],
        alpha=0.92,
    )
    ax.set_title(meta_title, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(
        0.98,
        0.05,
        meta_note.format(cvar=w_cvar.mean(), rules=w_regime.mean(), linucb=w_linucb.mean()),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.75"),
    )

    ax = axes[1]
    beliefs = np.asarray(convex.get("regime_beliefs", []), dtype=float)
    modes = np.asarray(convex.get("convexity_mode_names", []), dtype=object)
    n_obs = min(len(beliefs), len(modes))
    beliefs = beliefs[:n_obs]
    modes = modes[:n_obs]
    regime_masks = {
        "Bull": beliefs > 0.60,
        "Neutral": (beliefs >= 0.40) & (beliefs <= 0.60),
        "Bear": beliefs < 0.40,
    }
    mode_order = ["none", "mild", "strong"]
    mode_colors = {"none": "#BDBDBD", "mild": "#E9C46A", "strong": "#D1495B"}
    bottoms = np.zeros(len(regime_masks), dtype=float)
    regime_names = list(regime_masks.keys())
    for mode in mode_order:
        shares = []
        for _, mask in regime_masks.items():
            if mask.sum() == 0:
                shares.append(0.0)
            else:
                shares.append(float(np.mean(modes[mask] == mode)))
        ax.bar(regime_names, shares, bottom=bottoms, color=mode_colors[mode], label=mode.title(), alpha=0.92)
        bottoms += np.asarray(shares)
    ax.set_title("Convexity Mode Usage by Regime (D+, tf=0.50)", fontweight="bold")
    ax.set_ylabel("Share of days")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(
        0.98,
        0.05,
        "Bear states are mostly\nstrong convexity;\nbull states are mostly none.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.75"),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tail_diagnostic(out_path: Path) -> None:
    factor = _load_checkpoint_results("control_factor_only_tf0.50")
    cvar = _load_checkpoint_results("control_D_cvar_robust_tf0.50")
    convex = _load_checkpoint_results("control_D_plus_convexity_tf0.50")
    if factor is None or cvar is None or convex is None:
        print("plot_tail_diagnostic: missing cached checkpoints; skipping.")
        return

    series = {
        "Factor-only": np.asarray(factor.get("portfolio_returns", []), dtype=float),
        "D CVaR": np.asarray(cvar.get("portfolio_returns", []), dtype=float),
        "D+ Convexity": np.asarray(convex.get("portfolio_returns", []), dtype=float),
    }
    if any(len(values) == 0 for values in series.values()):
        print("plot_tail_diagnostic: empty return series; skipping.")
        return

    all_returns = np.concatenate(list(series.values()))
    x_min = float(np.nanpercentile(all_returns, 0.5))
    x_max = min(0.0, float(np.nanpercentile(all_returns, 20)))
    grid = np.linspace(x_min, x_max, 300)
    colors = {
        "Factor-only": "#4D4D4D",
        "D CVaR": "#D1495B",
        "D+ Convexity": "#1D6F8C",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    for name, values in series.items():
        values = values[np.isfinite(values)]
        cdf = np.array([np.mean(values <= x) for x in grid], dtype=float)
        var5 = float(np.nanpercentile(values, 5))
        cvar5 = float(values[values <= var5].mean())
        ax.plot(grid * 100, cdf, linewidth=2.2, color=colors[name], label=f"{name}  VaR5={var5*100:.2f}%  CVaR5={cvar5*100:.2f}%")
        ax.axvline(var5 * 100, color=colors[name], linewidth=1.0, alpha=0.25)

    ax.set_title("Left-Tail Distribution Diagnostic (Reference Split tf=0.50)", fontweight="bold")
    ax.set_xlabel("Daily return threshold")
    ax.set_ylabel("Empirical probability of return below threshold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(grid[0] * 100, grid[-1] * 100)
    ax.set_ylim(0, 0.22)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(
        0.98,
        0.04,
        "D+ shifts the left tail inward:\nits worst-day mass is lower than both\nfactor-only and plain CVaR.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.75"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    control, ablation, detail, summary = load_data()

    outputs = {
        "control_overview": Path(summary["results_dir"]) / "control_method_overview.png",
        "control_heatmaps": Path(summary["results_dir"]) / "control_split_heatmaps.png",
        "control_pareto": Path(summary["results_dir"]) / "control_pareto_frontier.png",
        "legacy_pruning": Path(summary["results_dir"]) / "legacy_pruning_story.png",
        "interpretability": Path(summary["results_dir"]) / "control_interpretability.png",
        "tail_diagnostic": Path(summary["results_dir"]) / "control_tail_diagnostic.png",
    }

    plot_control_overview(control, detail, summary, outputs["control_overview"])
    plot_control_heatmaps(control, detail, outputs["control_heatmaps"])
    plot_control_pareto(control, outputs["control_pareto"])
    plot_legacy_pruning(ablation, outputs["legacy_pruning"])
    plot_interpretability(outputs["interpretability"])
    plot_tail_diagnostic(outputs["tail_diagnostic"])

    print("Saved plot set:")
    for path in outputs.values():
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
