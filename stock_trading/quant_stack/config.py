"""Configuration constants for the quant trading pipeline."""

import os

# ============================================================
# Configuration
# ============================================================

# Broader cross-section: growth, cyclicals, defensives, and diversifiers.
# This gives the alpha layer more relative-value opportunity and gives the
# portfolio layer lower-correlation assets to rotate into when risk rises.
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
    'JPM', 'GS', 'XOM', 'CVX',
    'JNJ', 'PG', 'KO', 'PEP', 'WMT',
    'GLD', 'TLT', 'XLU', 'VNQ',
]
BENCHMARK = 'SPY'
PAIRS_CANDIDATES = [
    ('KO', 'PEP'),
    ('AAPL', 'MSFT'),
    ('GOOGL', 'META'),
    ('JPM', 'GS'),
    ('XOM', 'CVX'),
]
LSTM_TICKERS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
DATA_PERIOD = '5y'
RISK_FREE_RATE = 0.035  # approximate
FRED_SERIES = {
    'rate_10y': 'DGS10',
    'rate_2y': 'DGS2',
    'fed_funds': 'FEDFUNDS',
    'unrate': 'UNRATE',
}
TREASURY_SECURITIES = {
    'treasury_bill_rate': 'Treasury Bills',
    'treasury_note_rate': 'Treasury Notes',
    'treasury_bond_rate': 'Treasury Bonds',
}
BLS_SERIES = {
    'unemployment_rate': 'LNS14000000',
    'cpi_all_items': 'CUUR0000SA0',
}
SEC_USER_AGENT = os.getenv(
    'SEC_USER_AGENT',
    'sequential-decision-making/1.0 research@local'
)
