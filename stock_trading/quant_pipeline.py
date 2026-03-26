"""Thin compatibility wrapper for the modular quant trading pipeline."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_trading.logging_utils import tee_output_from_env
from stock_trading.quant_stack.main import main


if __name__ == "__main__":
    with tee_output_from_env("quant_pipeline"):
        main()
