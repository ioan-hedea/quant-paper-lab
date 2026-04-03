"""Validation tests for research-time configuration objects."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_stack.config import ControlConfig, CostModelConfig, EvaluationConfig, PipelineConfig


class ConfigValidationTests(unittest.TestCase):
    def test_cost_model_rejects_negative_base_cost(self) -> None:
        with self.assertRaises(ValueError):
            CostModelConfig(base_cost_bps=-1.0)

    def test_control_config_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            ControlConfig(method='totally_unknown')

    def test_pipeline_config_rejects_invalid_train_fraction(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(train_frac=1.2)

    def test_evaluation_config_rejects_unknown_checkpoint_mode(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationConfig(checkpoint_match_mode='mystery')
