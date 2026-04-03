"""Regression tests for the modular quant trading pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_stack.data import (
    _parse_bls_monthly_series,
    _sanitize_ohlcv_download,
    compute_macro_regime_signal,
)
from quant_stack.evaluation import build_ablation_suite, build_control_comparison_suite, run_research_evaluation
from quant_stack.execution import _compute_transaction_cost
from quant_stack.pipeline import _apply_macro_lag
from quant_stack.config import ControlConfig, EvaluationConfig, PipelineConfig
from quant_stack.controllers import build_controller, ControlState
from quant_stack.rl import PortfolioConstructionRL


class DataLayerTests(unittest.TestCase):
    def test_parse_bls_monthly_series_skips_placeholder_values(self) -> None:
        series = _parse_bls_monthly_series(
            [
                {"year": "2025", "period": "M01", "value": "-"},
                {"year": "2025", "period": "M02", "value": "."},
                {"year": "2025", "period": "M03", "value": "4.1"},
            ]
        )

        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(float(series.iloc[0]), 4.1)

    def test_apply_macro_lag_shifts_values_forward_in_trading_time(self) -> None:
        index = pd.date_range("2025-01-01", periods=5, freq="B")
        macro = pd.DataFrame({"macro": [10, 11, 12, 13, 14]}, index=index)

        lagged = _apply_macro_lag(macro, 2)

        self.assertTrue(np.isnan(lagged.iloc[0, 0]))
        self.assertTrue(np.isnan(lagged.iloc[1, 0]))
        self.assertEqual(float(lagged.iloc[2, 0]), 10.0)

    def test_transaction_cost_grows_with_turnover_and_trade_size(self) -> None:
        config = PipelineConfig()
        small = _compute_transaction_cost(0.05, 0.02, 0.01, config)
        large = _compute_transaction_cost(0.25, 0.10, 0.02, config)

        self.assertGreater(large, small)

    def test_compute_macro_regime_signal_returns_bounded_score(self) -> None:
        index = pd.date_range("2024-01-01", periods=40, freq="B")
        macro = pd.DataFrame(
            {
                "term_spread": np.linspace(0.5, 1.5, 40),
                "unrate": np.linspace(5.0, 3.5, 40),
                "fed_funds": np.linspace(5.5, 4.0, 40),
            },
            index=index,
        )

        score = compute_macro_regime_signal(macro)

        self.assertGreaterEqual(score, 0.05)
        self.assertLessEqual(score, 0.95)

    def test_sanitize_ohlcv_download_keeps_partial_success_panel(self) -> None:
        index = pd.date_range("2025-01-01", periods=4, freq="B")
        columns = pd.MultiIndex.from_product(
            [["Close", "Volume"], ["AAPL", "MSFT", "JNJ"]],
            names=["field", "ticker"],
        )
        raw = pd.DataFrame(index=index, columns=columns, dtype=float)
        raw[("Close", "AAPL")] = [100.0, 101.0, 102.0, 103.0]
        raw[("Close", "MSFT")] = [200.0, 201.0, 202.0, 203.0]
        raw[("Close", "JNJ")] = [np.nan, np.nan, np.nan, np.nan]
        raw[("Volume", "AAPL")] = [10.0, 11.0, 12.0, 13.0]
        raw[("Volume", "MSFT")] = [20.0, 21.0, 22.0, 23.0]
        raw[("Volume", "JNJ")] = [np.nan, np.nan, np.nan, np.nan]

        prices, volumes, returns, dropped = _sanitize_ohlcv_download(
            raw,
            ["AAPL", "MSFT", "JNJ"],
        )

        self.assertListEqual(list(prices.columns), ["AAPL", "MSFT"])
        self.assertListEqual(list(volumes.columns), ["AAPL", "MSFT"])
        self.assertIn("JNJ", dropped)
        self.assertEqual(len(returns), 3)


class PortfolioConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = PortfolioConstructionRL()
        self.factor = pd.Series([1.2, 0.8, -0.1], index=["AAPL", "MSFT", "GLD"])
        self.alpha = pd.Series([1.0, 0.5, -0.2], index=["AAPL", "MSFT", "GLD"])
        self.confidence = pd.Series([1.1, 1.0, 0.8], index=["AAPL", "MSFT", "GLD"])
        self.recent = pd.DataFrame(
            np.random.default_rng(0).normal(0.0, 0.01, size=(80, 3)),
            columns=["AAPL", "MSFT", "GLD"],
        )

    def test_construct_portfolio_preserves_full_budget(self) -> None:
        weights, cash = self.rl.construct_portfolio(
            self.factor,
            self.alpha,
            self.confidence,
            action=3,
            recent_returns=self.recent,
        )

        self.assertAlmostEqual(float(weights.sum() + cash), 1.0, places=7)
        self.assertGreater(float(weights["AAPL"]), float(weights["GLD"]))

    def test_rebalance_band_keeps_previous_weights_when_turnover_is_tiny(self) -> None:
        previous = pd.Series([0.40, 0.35, 0.20], index=["AAPL", "MSFT", "GLD"])
        target = pd.Series([0.405, 0.345, 0.20], index=["AAPL", "MSFT", "GLD"])

        adjusted = self.rl.apply_rebalance_band(target, previous)

        pd.testing.assert_series_equal(adjusted, previous)

    def test_rebalance_band_rescales_to_target_budget(self) -> None:
        previous = pd.Series([0.55, 0.25, 0.10], index=["AAPL", "MSFT", "GLD"])
        target = pd.Series([0.30, 0.40, 0.25], index=["AAPL", "MSFT", "GLD"])

        adjusted = self.rl.apply_rebalance_band(target, previous)

        self.assertAlmostEqual(float(adjusted.sum()), float(target.sum()), places=8)

    def test_construct_portfolio_respects_feasible_weight_cap(self) -> None:
        recent = pd.DataFrame(
            np.random.default_rng(1).normal(0.0, 0.01, size=(100, 4)),
            columns=["AAPL", "MSFT", "GLD", "TLT"],
        )
        factor = pd.Series([1.0, 0.8, 0.2, -0.1], index=recent.columns)
        alpha = pd.Series([1.2, 0.5, 0.1, -0.2], index=recent.columns)
        confidence = pd.Series([1.1, 0.9, 0.8, 0.7], index=recent.columns)

        weights, cash = self.rl.construct_portfolio(
            factor,
            alpha,
            confidence,
            action=3,
            recent_returns=recent,
        )

        self.assertAlmostEqual(float(weights.sum() + cash), 1.0, places=7)
        self.assertLessEqual(float(weights.max()), 0.2450001)


class EvaluationTests(unittest.TestCase):
    def test_ablation_suite_labels_are_unique_and_stable(self) -> None:
        configs = build_ablation_suite(PipelineConfig())
        labels = [config.experiment.label for config in configs]

        self.assertEqual(
            labels,
            [
                "factor_only",
                "alpha_stack_fixed_weights",
                "alpha_stack_no_rl",
                "portfolio_rl_fixed_weights",
                "full_pipeline",
                "full_pipeline_fixed_weights",
            ],
        )
        self.assertEqual(len(labels), len(set(labels)))

    def test_control_comparison_suite_covers_all_candidates(self) -> None:
        configs = build_control_comparison_suite(PipelineConfig())
        labels = [config.experiment.label for config in configs]

        expected_labels = [
            'factor_only',
            'A1_fixed',
            'A2_vol_target',
            'A3_dd_delever',
            'A4_regime_rules',
            'A5_ensemble_mean',
            'B1_linucb',
            'B2_thompson',
            'B3_epsilon_greedy',
            'C_supervised',
            'D_cvar_robust',
            'D_plus_convexity',
            'H_mpc',
            'E_council',
            'E_plus_convexity',
            'G_mlp_meta',
            'G_plus_convexity',
            'F_cmdp_lagrangian',
            'RL_q_learning',
        ]
        self.assertEqual(labels, expected_labels)
        self.assertEqual(len(labels), len(set(labels)))

    def test_controllers_return_valid_invested_fraction(self) -> None:
        state = ControlState(
            alpha_strength=0.5,
            recent_drawdown=-0.03,
            recent_vol=0.15,
            regime_belief=0.6,
            trend=0.05,
            concentration=0.15,
            invested_fraction=0.95,
            t=500,
        )
        for method in ['fixed', 'vol_target', 'dd_delever', 'regime_rules',
                       'ensemble_rules', 'linucb', 'thompson', 'epsilon_greedy',
                       'q_learning']:
            ctrl = build_controller(ControlConfig(method=method))
            frac = ctrl.compute_invested_fraction(state)
            self.assertGreaterEqual(frac, 0.0, msg=f"{method} returned negative fraction")
            self.assertLessEqual(frac, 1.5, msg=f"{method} returned unreasonable fraction")

    def test_research_defers_single_e2e_baseline_to_end(self) -> None:
        index = pd.date_range("2025-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": np.linspace(100, 109, 10), "AAPL": np.linspace(50, 59, 10)}, index=index)
        volumes = pd.DataFrame({"SPY": np.linspace(1000, 1009, 10), "AAPL": np.linspace(500, 509, 10)}, index=index)
        returns = prices.pct_change().dropna()
        call_log: list[tuple[str, bool]] = []

        def fake_run_full_pipeline(*args, **kwargs):
            config = kwargs["config"]
            call_log.append((config.experiment.label, config.enable_e2e_baseline))
            return {
                "wealth": [1.0, 1.01],
                "spy": [1.0, 1.0],
                "factor": [1.0, 1.0],
                "voltarget": [1.0, 1.0],
                "ddlever": [1.0, 1.0],
                "e2e_rl": [1.0, 1.0],
                "experiment_label": config.experiment.label,
                "turnover": [0.0],
                "transaction_costs": [0.0],
            }

        eval_config = EvaluationConfig(
            train_fracs=(0.4, 0.5),
            rolling_window_days=20,
            rolling_step_days=5,
            min_rolling_windows=1,
            max_rolling_windows=1,
            cost_bps_grid=(5.0,),
            rebalance_band_grid=(0.015,),
            hedge_scale_grid=(1.0,),
            macro_lag_grid=(3,),
            reward_mode_grid=("differential_sharpe",),
            bootstrap_samples=10,
            enable_checkpoints=False,
        )

        with patch("quant_stack.evaluation.run_full_pipeline", side_effect=fake_run_full_pipeline), \
             patch("pandas.DataFrame.to_csv", return_value=None), \
             patch("quant_stack.evaluation._write_research_tables", return_value=None), \
             patch("quant_stack.evaluation.plot_research_evaluation", return_value=None):
            run_research_evaluation(
                prices=prices,
                volumes=volumes,
                returns=returns,
                macro_data=pd.DataFrame(index=returns.index),
                sec_quality_scores=pd.Series(dtype=float),
                base_config=PipelineConfig(),
                evaluation_config=eval_config,
            )

        e2e_calls = [entry for entry in call_log if entry[1]]
        self.assertEqual(len(e2e_calls), 1)
        self.assertEqual(call_log[-1], e2e_calls[0])
        self.assertEqual(call_log[-1][0], "full_pipeline_tf0.50")

    def test_research_resume_uses_cached_run_results(self) -> None:
        index = pd.date_range("2025-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": np.linspace(100, 109, 10), "AAPL": np.linspace(50, 59, 10)}, index=index)
        volumes = pd.DataFrame({"SPY": np.linspace(1000, 1009, 10), "AAPL": np.linspace(500, 509, 10)}, index=index)
        returns = prices.pct_change().dropna()
        eval_config = EvaluationConfig(
            train_fracs=(0.4,),
            rolling_window_days=20,
            rolling_step_days=5,
            min_rolling_windows=1,
            max_rolling_windows=1,
            cost_bps_grid=(5.0,),
            rebalance_band_grid=(0.015,),
            hedge_scale_grid=(1.0,),
            macro_lag_grid=(3,),
            reward_mode_grid=("differential_sharpe",),
            bootstrap_samples=10,
        )

        with TemporaryDirectory() as tmpdir:
            eval_config.checkpoint_dir = tmpdir
            call_count = 0

            def fake_run_full_pipeline(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                config = kwargs["config"]
                return {
                    "wealth": [1.0, 1.01],
                    "spy": [1.0, 1.0],
                    "factor": [1.0, 1.0],
                    "voltarget": [1.0, 1.0],
                    "ddlever": [1.0, 1.0],
                    "e2e_rl": [1.0, 1.0],
                    "dates": list(returns.index),
                    "experiment_label": config.experiment.label,
                    "turnover": [0.0],
                    "transaction_costs": [0.0],
                    "live_agent": lambda: None,
                }

            common_patches = (
                patch("pandas.DataFrame.to_csv", return_value=None),
                patch("quant_stack.evaluation._write_research_tables", return_value=None),
                patch("quant_stack.evaluation.plot_research_evaluation", return_value=None),
            )

            with patch("quant_stack.evaluation.run_full_pipeline", side_effect=fake_run_full_pipeline), \
                 common_patches[0], common_patches[1], common_patches[2]:
                run_research_evaluation(
                    prices=prices,
                    volumes=volumes,
                    returns=returns,
                    macro_data=pd.DataFrame(index=returns.index),
                    sec_quality_scores=pd.Series(dtype=float),
                    base_config=PipelineConfig(),
                    evaluation_config=eval_config,
                )

            self.assertGreater(call_count, 0)
            first_call_count = call_count
            call_count = 0

            with patch("quant_stack.evaluation.run_full_pipeline", side_effect=fake_run_full_pipeline), \
                 common_patches[0], common_patches[1], common_patches[2]:
                run_research_evaluation(
                    prices=prices,
                    volumes=volumes,
                    returns=returns,
                    macro_data=pd.DataFrame(index=returns.index),
                    sec_quality_scores=pd.Series(dtype=float),
                    base_config=PipelineConfig(),
                    evaluation_config=eval_config,
                )

            self.assertEqual(call_count, 0)
            self.assertGreater(first_call_count, 0)
            progress_path = Path(tmpdir) / "universe_A" / "research_progress.json"
            self.assertTrue(progress_path.exists())
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["status"], "completed")
            self.assertGreater(progress_payload["completed_runs"], 0)

    def test_research_ignores_legacy_checkpoint_payloads(self) -> None:
        index = pd.date_range("2025-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": np.linspace(100, 109, 10), "AAPL": np.linspace(50, 59, 10)}, index=index)
        volumes = pd.DataFrame({"SPY": np.linspace(1000, 1009, 10), "AAPL": np.linspace(500, 509, 10)}, index=index)
        returns = prices.pct_change().dropna()
        eval_config = EvaluationConfig(
            train_fracs=(0.4,),
            rolling_window_days=20,
            rolling_step_days=5,
            min_rolling_windows=1,
            max_rolling_windows=1,
            cost_bps_grid=(5.0,),
            rebalance_band_grid=(0.015,),
            hedge_scale_grid=(1.0,),
            macro_lag_grid=(3,),
            reward_mode_grid=("differential_sharpe",),
            bootstrap_samples=10,
        )

        with TemporaryDirectory() as tmpdir:
            eval_config.checkpoint_dir = tmpdir
            legacy_path = Path(tmpdir) / "ablation_factor_only_tf0.40.pkl"
            with legacy_path.open("wb") as handle:
                pickle.dump({"wealth": [1.0, 9.9], "spy": [1.0, 1.0]}, handle)

            call_count = 0

            def fake_run_full_pipeline(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                config = kwargs["config"]
                return {
                    "wealth": [1.0, 1.01],
                    "spy": [1.0, 1.0],
                    "factor": [1.0, 1.0],
                    "voltarget": [1.0, 1.0],
                    "ddlever": [1.0, 1.0],
                    "e2e_rl": [1.0, 1.0],
                    "dates": list(returns.index),
                    "experiment_label": config.experiment.label,
                    "turnover": [0.0],
                    "transaction_costs": [0.0],
                }

            with patch("quant_stack.evaluation.run_full_pipeline", side_effect=fake_run_full_pipeline), \
                 patch("pandas.DataFrame.to_csv", return_value=None), \
                 patch("quant_stack.evaluation._write_research_tables", return_value=None), \
                 patch("quant_stack.evaluation.plot_research_evaluation", return_value=None):
                run_research_evaluation(
                    prices=prices,
                    volumes=volumes,
                    returns=returns,
                    macro_data=pd.DataFrame(index=returns.index),
                    sec_quality_scores=pd.Series(dtype=float),
                    base_config=PipelineConfig(),
                    evaluation_config=eval_config,
                )

            self.assertGreater(call_count, 0)


if __name__ == "__main__":
    unittest.main()
