"""Smoke-test universe-aware checkpoint compatibility.

Usage::

    python scripts/checkpoint_smoke.py
    python scripts/checkpoint_smoke.py --universes C D E
    python scripts/checkpoint_smoke.py --checkpoint-dir /tmp/quant-checkpoint-smoke

The script does not download market data or run the full backtest. It creates
small synthetic panels, writes checkpoint payloads using the same metadata
helpers as the research engine, and then verifies that:

1. strict reload works within the same universe contract
2. compatible reload still rejects cross-universe benchmark/timing mismatches
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logging_utils import tee_output_from_env
from quant_stack.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_metadata,
    _checkpoint_path,
    _load_checkpoint_results,
    _scope_universe_run_key,
    _universe_checkpoint_dir,
)
from quant_stack.config import PipelineConfig, get_universe_profile, use_universe


def _synthetic_inputs(universe_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    profile = get_universe_profile(universe_id)
    benchmark_tickers = [ticker for ticker, weight in profile.benchmark_components if float(weight) > 0.0]
    tickers = list(dict.fromkeys(profile.tickers[:3] + benchmark_tickers))
    index = pd.date_range("2025-01-02", periods=12, freq="B")
    base = np.linspace(100.0, 112.0, len(index))

    prices = pd.DataFrame(
        {
            ticker: base * (1.0 + 0.01 * idx)
            for idx, ticker in enumerate(tickers)
        },
        index=index,
    )
    volumes = pd.DataFrame(
        {
            ticker: np.linspace(1_000_000 + 10_000 * idx, 1_120_000 + 10_000 * idx, len(index))
            for idx, ticker in enumerate(tickers)
        },
        index=index,
    )
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({"term_spread": np.linspace(0.5, 1.0, len(returns))}, index=returns.index)
    sec_quality = pd.Series(dtype=float)
    return prices, volumes, returns, macro, sec_quality


def _write_checkpoint(
    checkpoint_root: Path,
    universe_id: str,
    run_key: str,
    metadata: dict[str, object],
) -> Path:
    checkpoint_dir = _universe_checkpoint_dir(checkpoint_root, universe_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(checkpoint_dir, _scope_universe_run_key(run_key, universe_id))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata,
        "results": {
            "wealth": [1.0, 1.01, 1.02],
            "benchmark_label": metadata.get("benchmark_label"),
            "universe_id": universe_id,
        },
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return checkpoint_path


def run_smoke(universe_ids: tuple[str, ...], checkpoint_root: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_key = "checkpoint_smoke_control_H_mpc_tf0.50"
    metadata_by_universe: dict[str, dict[str, object]] = {}
    path_by_universe: dict[str, Path] = {}

    print(f"Checkpoint smoke root: {checkpoint_root}")
    print(f"Universes: {universe_ids}")

    for universe_id in universe_ids:
        with use_universe(universe_id) as profile:
            prices, volumes, returns, macro, sec_quality = _synthetic_inputs(universe_id)
            config = PipelineConfig()
            config.experiment.label = f"smoke_{universe_id}"
            metadata = _checkpoint_metadata(
                prices,
                volumes,
                returns,
                macro,
                sec_quality,
                config,
                suite="checkpoint_smoke",
                include_e2e=False,
                run_key=_scope_universe_run_key(run_key, universe_id),
            )
            checkpoint_path = _write_checkpoint(checkpoint_root, universe_id, run_key, metadata)
            loaded = _load_checkpoint_results(checkpoint_path, metadata, match_mode="strict")
            if loaded is None:
                raise RuntimeError(f"Strict checkpoint reload failed for universe {universe_id}.")
            metadata_by_universe[universe_id] = metadata
            path_by_universe[universe_id] = checkpoint_path
            print(
                f"[ok] {universe_id}: {profile.benchmark_label} "
                f"{profile.benchmark_components} -> {checkpoint_path}"
            )

    for source_id, checkpoint_path in path_by_universe.items():
        for target_id, expected_metadata in metadata_by_universe.items():
            if source_id == target_id:
                continue
            loaded = _load_checkpoint_results(
                checkpoint_path,
                expected_metadata,
                match_mode="compatible",
            )
            if loaded is not None:
                raise RuntimeError(
                    f"Compatible checkpoint reload incorrectly matched {source_id} to {target_id}."
                )
            print(f"[ok] mismatch rejected: {source_id} -> {target_id}")

    print("Checkpoint smoke passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Universe-aware checkpoint smoke test")
    parser.add_argument(
        "--universes",
        nargs="+",
        default=["C", "D", "E"],
        help="Universe IDs to verify (default: C D E)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory to persist smoke checkpoints. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        run_smoke(tuple(args.universes), args.checkpoint_dir)
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="quant-checkpoint-smoke-"))
    try:
        run_smoke(tuple(args.universes), temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    with tee_output_from_env("checkpoint_smoke"):
        main()
