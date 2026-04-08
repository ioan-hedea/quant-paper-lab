"""Focused reliability tests for execution realism, checkpoints, and manifests."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_stack.config import EvaluationConfig, PipelineConfig, use_universe
from quant_stack.evaluation import (
    CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_metadata,
    _load_checkpoint_results,
    _write_run_manifest,
)
from quant_stack.checkpointing import _scope_universe_run_key, _universe_checkpoint_dir
from quant_stack.execution import _apply_execution_constraints, _compute_transaction_cost


class ExecutionRealismTests(unittest.TestCase):
    def test_adv_cap_scales_trades_down(self) -> None:
        config = PipelineConfig()
        config.cost_model.adv_participation_cap = 0.05

        target = pd.Series({'AAPL': 1.0, 'MSFT': 0.0})
        prev = pd.Series({'AAPL': 0.0, 'MSFT': 1.0})
        prices_row = pd.Series({'AAPL': 100.0, 'MSFT': 100.0})
        volume_window = pd.DataFrame(
            [{'AAPL': 1_000.0, 'MSFT': 1_000.0}] * 5,
            columns=['AAPL', 'MSFT'],
        )

        executed, stats = _apply_execution_constraints(
            target, prev, prices_row, volume_window, wealth=10_000_000.0, config=config,
        )

        self.assertLess(stats['liquidity_scale'], 1.0)
        self.assertEqual(stats['adv_cap_hit'], 1.0)
        self.assertLess(float((executed - prev).abs().sum()), float((target - prev).abs().sum()))

    def test_transaction_cost_stress_multiplier_increases_cost(self) -> None:
        base = PipelineConfig()
        stressed = PipelineConfig()
        stressed.cost_model.cost_stress_multiplier = 3.0

        base_cost = _compute_transaction_cost(0.20, 0.10, 0.15, base, 0.01, 0.02, 0.0)
        stressed_cost = _compute_transaction_cost(0.20, 0.10, 0.15, stressed, 0.01, 0.02, 0.0)

        self.assertGreater(stressed_cost, base_cost)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_compatible_checkpoint_reuses_old_window(self) -> None:
        idx_short = pd.date_range('2025-01-01', periods=3, freq='B')
        idx_long = pd.date_range('2025-01-01', periods=5, freq='B')
        prices_short = pd.DataFrame({'AAPL': [100, 101, 102]}, index=idx_short)
        volumes_short = pd.DataFrame({'AAPL': [10, 11, 12]}, index=idx_short)
        returns_short = prices_short.pct_change().dropna()
        prices_long = pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]}, index=idx_long)
        volumes_long = pd.DataFrame({'AAPL': [10, 11, 12, 13, 14]}, index=idx_long)
        returns_long = prices_long.pct_change().dropna()
        cfg = PipelineConfig()
        macro_short = pd.DataFrame(index=returns_short.index)
        macro_long = pd.DataFrame(index=returns_long.index)
        sec = pd.Series(dtype=float)

        payload_metadata = _checkpoint_metadata(
            prices_short, volumes_short, returns_short, macro_short, sec, cfg,
            suite='control_comparison', include_e2e=False, run_key='control_A1_fixed_tf0.50',
        )
        expected_metadata = _checkpoint_metadata(
            prices_long, volumes_long, returns_long, macro_long, sec, cfg,
            suite='control_comparison', include_e2e=False, run_key='control_A1_fixed_tf0.50',
        )

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'sample.pkl'
            with checkpoint_path.open('wb') as handle:
                pickle.dump(
                    {
                        'schema_version': CHECKPOINT_SCHEMA_VERSION,
                        'metadata': payload_metadata,
                        'results': {'wealth': [1.0, 1.1]},
                    },
                    handle,
                )

            strict = _load_checkpoint_results(checkpoint_path, expected_metadata, match_mode='strict')
            compatible = _load_checkpoint_results(checkpoint_path, expected_metadata, match_mode='compatible')

        self.assertIsNone(strict)
        self.assertEqual(compatible, {'wealth': [1.0, 1.1]})

    def test_compatible_checkpoint_rejects_benchmark_contract_mismatch(self) -> None:
        idx = pd.date_range('2025-01-01', periods=5, freq='B')
        prices = pd.DataFrame({'AAPL': [100, 101, 102, 103, 104], 'SPY': [200, 201, 202, 203, 204], 'TLT': [90, 91, 91, 92, 93]}, index=idx)
        volumes = pd.DataFrame({'AAPL': [10, 11, 12, 13, 14], 'SPY': [20, 20, 21, 22, 22], 'TLT': [30, 30, 31, 31, 32]}, index=idx)
        returns = prices.pct_change().dropna()
        cfg = PipelineConfig()
        macro = pd.DataFrame(index=returns.index)
        sec = pd.Series(dtype=float)

        with use_universe('A'):
            payload_metadata = _checkpoint_metadata(
                prices, volumes, returns, macro, sec, cfg,
                suite='control_comparison', include_e2e=False, run_key='control_H_mpc_tf0.50',
            )
        with use_universe('E'):
            expected_metadata = _checkpoint_metadata(
                prices, volumes, returns, macro, sec, cfg,
                suite='control_comparison', include_e2e=False, run_key='control_H_mpc_tf0.50',
            )

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'sample.pkl'
            with checkpoint_path.open('wb') as handle:
                pickle.dump(
                    {
                        'schema_version': CHECKPOINT_SCHEMA_VERSION,
                        'metadata': payload_metadata,
                        'results': {'wealth': [1.0, 1.1]},
                    },
                    handle,
                )

            compatible = _load_checkpoint_results(checkpoint_path, expected_metadata, match_mode='compatible')

        self.assertIsNone(compatible)

    def test_checkpoint_paths_scope_extended_universes(self) -> None:
        base_dir = Path('/tmp/checkpoint-root')
        scoped_dir = _universe_checkpoint_dir(base_dir, 'E')
        scoped_key = _scope_universe_run_key('control_H_mpc_tf0.50', 'E')

        self.assertEqual(scoped_dir, base_dir / 'universe_E')
        self.assertEqual(scoped_key, 'universe_E_control_H_mpc_tf0.50')


class RunManifestTests(unittest.TestCase):
    def test_run_manifest_contains_provenance_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_run_manifest(
                output_dir,
                run_type='research_evaluation',
                base_config=PipelineConfig(),
                evaluation_config=EvaluationConfig(),
                universe_id='A',
                run_timestamp='20260403_210000',
                status='completed',
                summary={'n_metric_rows': 12},
            )

            payload = json.loads((output_dir / 'run_manifest.json').read_text(encoding='utf-8'))

        self.assertEqual(payload['run_type'], 'research_evaluation')
        self.assertEqual(payload['status'], 'completed')
        self.assertEqual(payload['universe_id'], 'A')
        self.assertIn('config_hash', payload)
        self.assertIn('package_versions', payload)
        self.assertIn('git_commit', payload)
