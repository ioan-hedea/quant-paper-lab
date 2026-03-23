"""Regression tests for the modular quant trading pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_trading.quant_stack.data import _parse_bls_monthly_series
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


if __name__ == "__main__":
    unittest.main()
