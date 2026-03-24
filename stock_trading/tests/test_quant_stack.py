"""Regression tests for the modular quant trading pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_trading.quant_stack.data import _parse_bls_monthly_series, compute_macro_regime_signal
from stock_trading.quant_stack.evaluation import build_ablation_suite
from stock_trading.quant_stack.pipeline import _apply_macro_lag, _compute_transaction_cost
from stock_trading.quant_stack.config import PipelineConfig
from stock_trading.quant_stack.rl import PortfolioConstructionRL


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
                "alpha_stack_no_rl",
                "alpha_plus_portfolio_rl",
                "alpha_plus_hedge_rl",
                "full_pipeline",
            ],
        )
        self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":
    unittest.main()
