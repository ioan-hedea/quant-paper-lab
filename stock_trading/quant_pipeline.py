"""Thin compatibility wrapper for the modular quant trading pipeline."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_trading.quant_stack.main import main


if __name__ == "__main__":
    main()
