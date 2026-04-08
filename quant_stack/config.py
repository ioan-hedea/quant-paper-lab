"""Configuration constants for the quant trading pipeline."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable


def _validate_non_negative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _validate_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _validate_int_min(name: str, value: int, minimum: int) -> None:
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")


def _validate_half_open_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must lie in [0, 1), got {value!r}")


def _validate_choice(name: str, value: str, allowed: Iterable[str]) -> None:
    allowed_tuple = tuple(str(item) for item in allowed)
    if str(value) not in allowed_tuple:
        raise ValueError(f"{name} must be one of {allowed_tuple}, got {value!r}")


def _validate_range_tuple(name: str, bounds: tuple[float, float], *, lower_bound: float | None = None) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a length-2 tuple, got {bounds!r}")
    low, high = map(float, bounds)
    if lower_bound is not None and low < lower_bound:
        raise ValueError(f"{name} lower bound must be >= {lower_bound}, got {bounds!r}")
    if high < low:
        raise ValueError(f"{name} upper bound must be >= lower bound, got {bounds!r}")

# ============================================================
# Universe Definitions
# ============================================================
# UNIVERSE_CORE: original 19-stock universe (fast runs, development)
# UNIVERSE_EXPANDED: ~75 stocks across 11 GICS sectors (publication runs)
# Switch by setting UNIVERSE = UNIVERSE_CORE or UNIVERSE_EXPANDED below.

UNIVERSE_CORE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
    'JPM', 'GS', 'XOM', 'CVX',
    'JNJ', 'PG', 'KO', 'PEP', 'WMT',
    'GLD', 'TLT', 'XLU', 'VNQ',
]

UNIVERSE_EXPANDED = list(dict.fromkeys([
    # Technology (8)
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO',
    # Financials (8)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
    # Healthcare (8)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'BMY',
    # Consumer Staples (8)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS',
    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    # Industrials (8)
    'CAT', 'BA', 'HON', 'UNP', 'UPS', 'GE', 'MMM', 'LMT',
    # Materials (4)
    'LIN', 'APD', 'SHW', 'FCX',
    # Real Estate (4)
    'AMT', 'PLD', 'PSA', 'O',
    # Utilities (4)
    'NEE', 'DUK', 'SO', 'XEL',
    # Communications (6)
    'META', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX',
    # Consumer Discretionary (7)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX',
    # Cross-asset diversifiers (3)
    'GLD', 'TLT', 'VNQ',
]))

# ---- Second universe (B) for cross-universe validation ----
# Deliberately minimal overlap with UNIVERSE_EXPANDED to test
# whether controller rankings transfer to a different stock set.
# Tilts: mid-cap growth, biotech, REITs, commodity-linked, small-cap value.
UNIVERSE_B = list(dict.fromkeys([
    # Technology — mid-cap / different names
    'CRM', 'ADBE', 'NOW', 'PANW', 'SNPS', 'CDNS', 'MRVL', 'FTNT',
    # Biotech / Pharma — higher-vol healthcare
    'GILD', 'REGN', 'VRTX', 'ISRG', 'DXCM', 'IDXX', 'ZTS', 'MRNA',
    # Financials — insurance + regional / fintech
    'AIG', 'TRV', 'MET', 'ALL', 'BK', 'SCHW', 'ICE', 'CME',
    # Consumer — different mix
    'TGT', 'ROST', 'ORLY', 'AZO', 'DG', 'DLTR', 'YUM', 'CMG',
    # Industrials — mid-cap, transport, aerospace
    'FDX', 'NSC', 'CSX', 'WM', 'RSG', 'ITW', 'EMR', 'ROK',
    # Energy — midstream + services
    'WMB', 'KMI', 'OKE', 'ET', 'HAL', 'BKR', 'DVN', 'FANG',
    # Materials / Mining
    'NEM', 'GOLD', 'NUE', 'STLD', 'CF', 'MOS', 'ALB', 'ECL',
    # REITs — broader than EXPANDED
    'EQIX', 'DLR', 'SPG', 'WELL', 'AVB', 'EQR', 'VICI', 'IRM',
    # Utilities — different names
    'AEP', 'D', 'SRE', 'ED', 'WEC', 'ES', 'EXC', 'PCG',
    # Cross-asset diversifiers
    'SLV', 'IEF', 'HYG',
]))

PAIRS_CANDIDATES_B = [
    # Tech
    ('CRM', 'NOW'), ('ADBE', 'CRM'), ('PANW', 'FTNT'), ('SNPS', 'CDNS'),
    # Biotech
    ('GILD', 'REGN'), ('VRTX', 'MRNA'), ('ISRG', 'IDXX'),
    # Financials
    ('AIG', 'MET'), ('TRV', 'ALL'), ('ICE', 'CME'), ('BK', 'SCHW'),
    # Consumer
    ('TGT', 'ROST'), ('ORLY', 'AZO'), ('DG', 'DLTR'), ('YUM', 'CMG'),
    # Industrials
    ('FDX', 'NSC'), ('CSX', 'NSC'), ('WM', 'RSG'), ('ITW', 'EMR'),
    # Energy
    ('WMB', 'KMI'), ('OKE', 'ET'), ('HAL', 'BKR'), ('DVN', 'FANG'),
    # Materials
    ('NEM', 'GOLD'), ('NUE', 'STLD'), ('CF', 'MOS'),
    # REITs
    ('EQIX', 'DLR'), ('SPG', 'WELL'), ('AVB', 'EQR'),
    # Cross-asset
    ('SLV', 'IEF'),
]

LSTM_TICKERS_B = [
    'CRM', 'NOW', 'PANW', 'GILD', 'REGN',
    'TRV', 'CME', 'SCHW',
    'DVN', 'FANG',
    'NEM', 'EQIX',
    'SLV', 'IEF',
]

ASSET_GROUPS_B = {
    'technology': ['CRM', 'ADBE', 'NOW', 'PANW', 'SNPS', 'CDNS', 'MRVL', 'FTNT'],
    'biotech': ['GILD', 'REGN', 'VRTX', 'ISRG', 'DXCM', 'IDXX', 'ZTS', 'MRNA'],
    'financials': ['AIG', 'TRV', 'MET', 'ALL', 'BK', 'SCHW', 'ICE', 'CME'],
    'consumer': ['TGT', 'ROST', 'ORLY', 'AZO', 'DG', 'DLTR', 'YUM', 'CMG'],
    'industrials': ['FDX', 'NSC', 'CSX', 'WM', 'RSG', 'ITW', 'EMR', 'ROK'],
    'energy': ['WMB', 'KMI', 'OKE', 'ET', 'HAL', 'BKR', 'DVN', 'FANG'],
    'materials': ['NEM', 'GOLD', 'NUE', 'STLD', 'CF', 'MOS', 'ALB', 'ECL'],
    'reits': ['EQIX', 'DLR', 'SPG', 'WELL', 'AVB', 'EQR', 'VICI', 'IRM'],
    'utilities': ['AEP', 'D', 'SRE', 'ED', 'WEC', 'ES', 'EXC', 'PCG'],
    'diversifier': ['SLV', 'IEF', 'HYG'],
}

# ---- Universe C: European equities (zero overlap with A/B) ----
UNIVERSE_C = list(dict.fromkeys([
    'ADS.DE', 'ALV.DE', 'ASML.AS', 'PRX.AS', 'INGA.AS', 'SAP.DE', 'SIE.DE', 'DTE.DE', 'IFX.DE',
    'MUV2.DE', 'RWE.DE', 'BMW.DE', 'MBG.DE', 'VOW3.DE', 'BAS.DE', 'BAYN.DE', 'DHL.DE',
    'BEI.DE', 'AIR.PA', 'AI.PA', 'BNP.PA', 'CAP.PA', 'CS.PA', 'DG.PA', 'ENGI.PA',
    'MC.PA', 'OR.PA', 'RI.PA', 'SAN.PA', 'SU.PA', 'TTE.PA', 'VIE.PA', 'AZN.L',
    'BARC.L', 'BATS.L', 'BP.L', 'DGE.L', 'GSK.L', 'HSBA.L', 'LSEG.L', 'NG.L',
    'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L', 'ULVR.L', 'VOD.L', 'BBVA.MC', 'IBE.MC',
    'ITX.MC', 'REP.MC', 'SAN.MC', 'ENEL.MI', 'ENI.MI', 'ISP.MI', 'STLAM.MI',
    'UCG.MI', 'ABBN.SW', 'ALC.SW', 'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW',
    'ZURN.SW', 'EWG', 'EWU', 'EWQ', 'EWI', 'EWP',
]))

PAIRS_CANDIDATES_C = [
    ('ASML.AS', 'SAP.DE'), ('INGA.AS', 'ABBN.SW'), ('ALV.DE', 'MUV2.DE'),
    ('BMW.DE', 'MBG.DE'), ('VOW3.DE', 'STLAM.MI'), ('AIR.PA', 'VIE.PA'),
    ('MC.PA', 'OR.PA'), ('SAN.PA', 'NOVN.SW'), ('GSK.L', 'AZN.L'),
    ('BP.L', 'SHEL.L'), ('ENEL.MI', 'IBE.MC'), ('BBVA.MC', 'SAN.MC'),
    ('BNP.PA', 'CS.PA'), ('REL.L', 'LSEG.L'), ('EWG', 'EWQ'),
]

LSTM_TICKERS_C = [
    'ASML.AS', 'SAP.DE', 'SIE.DE', 'AIR.PA', 'MC.PA',
    'AZN.L', 'SHEL.L', 'HSBA.L', 'BBVA.MC', 'ENEL.MI',
    'NESN.SW', 'NOVN.SW', 'UBSG.SW', 'EWG', 'EWU',
]

ASSET_GROUPS_C = {
    'technology': ['ASML.AS', 'SAP.DE', 'SIE.DE', 'IFX.DE', 'CAP.PA', 'DTE.DE', 'PRX.AS', 'ADS.DE'],
    'financials': ['ALV.DE', 'MUV2.DE', 'INGA.AS', 'HSBA.L', 'BARC.L', 'BNP.PA', 'CS.PA', 'BBVA.MC', 'SAN.MC', 'UCG.MI', 'ISP.MI', 'UBSG.SW', 'ZURN.SW'],
    'healthcare': ['AZN.L', 'GSK.L', 'SAN.PA', 'NOVN.SW', 'ROG.SW', 'ALC.SW', 'BAYN.DE'],
    'consumer': ['MC.PA', 'OR.PA', 'RI.PA', 'ULVR.L', 'DGE.L', 'BATS.L', 'ITX.MC', 'NESN.SW', 'BEI.DE', 'BMW.DE', 'MBG.DE', 'VOW3.DE'],
    'industrials': ['AIR.PA', 'AI.PA', 'DHL.DE', 'VIE.PA', 'REL.L', 'LSEG.L', 'RWE.DE', 'ENGI.PA'],
    'energy': ['BP.L', 'SHEL.L', 'TTE.PA', 'ENI.MI', 'REP.MC', 'NG.L', 'IBE.MC', 'ENEL.MI'],
    'diversifier': ['EWG', 'EWU', 'EWQ', 'EWI', 'EWP'],
}

# ---- Universe D: Emerging markets (zero overlap with A/B/C/E) ----
UNIVERSE_D = list(dict.fromkeys([
    '0700.HK', '9988.HK', '3690.HK', '9618.HK', '9888.HK', '1810.HK', '1211.HK',
    '0941.HK', '1299.HK', '2318.HK', '1398.HK', '3988.HK', '0939.HK', '0883.HK',
    '2628.HK', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS', 'ITC.NS', 'HINDUNILVR.NS', 'AXISBANK.NS',
    'KOTAKBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS', '2330.TW',
    '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2882.TW', '005930.KS', '000660.KS',
    '035420.KS', '051910.KS', 'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA',
    'B3SA3.SA', 'ABEV3.SA', 'WEGE3.SA', 'WALMEX.MX', 'AMXB.MX', 'NPN.JO',
    'CFR.JO', 'SOL.JO', 'HDB', 'IBN', 'PBR', 'NU', 'MELI',
]))

PAIRS_CANDIDATES_D = [
    ('0700.HK', '9988.HK'), ('3690.HK', '9618.HK'), ('1398.HK', '0939.HK'),
    ('RELIANCE.NS', 'TCS.NS'), ('HDFCBANK.NS', 'ICICIBANK.NS'),
    ('AXISBANK.NS', 'KOTAKBANK.NS'), ('2330.TW', '2317.TW'),
    ('005930.KS', '000660.KS'), ('VALE3.SA', 'PETR4.SA'),
    ('ITUB4.SA', 'BBDC4.SA'), ('WALMEX.MX', 'AMXB.MX'),
    ('HDB', 'IBN'), ('PBR', 'PETR4.SA'), ('NPN.JO', 'CFR.JO'),
]

LSTM_TICKERS_D = [
    '0700.HK', '9988.HK', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS',
    '2330.TW', '005930.KS', 'VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
    'WALMEX.MX', 'NPN.JO', 'HDB', 'IBN', 'MELI',
]

ASSET_GROUPS_D = {
    'greater_china': ['0700.HK', '9988.HK', '3690.HK', '9618.HK', '9888.HK', '1810.HK', '1211.HK', '0941.HK', '1299.HK', '2318.HK', '1398.HK', '3988.HK', '0939.HK', '0883.HK', '2628.HK'],
    'india': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS', 'ITC.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS', 'HDB', 'IBN'],
    'north_asia': ['2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2882.TW', '005930.KS', '000660.KS', '035420.KS', '051910.KS'],
    'latin_america': ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'B3SA3.SA', 'ABEV3.SA', 'WEGE3.SA', 'WALMEX.MX', 'AMXB.MX', 'PBR', 'NU', 'MELI'],
    'emea': ['NPN.JO', 'CFR.JO', 'SOL.JO'],
}

# ---- Universe E: multi-asset stress test ----
UNIVERSE_E = list(dict.fromkeys([
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLU', 'XLRE', 'XLP', 'XLY', 'XLB', 'XLC',
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'EMB', 'AGG', 'TIP', 'BIL',
    'GLD', 'SLV', 'USO', 'DBA', 'PDBC', 'CPER', 'UNG', 'CORN', 'WEAT',
    'VNQ', 'SCHH', 'VNQI', 'REM',
    'EFA', 'EWJ', 'FXI', 'EWZ', 'EWC', 'EWA', 'INDA', 'EWT', 'EIDO', 'EWS',
    'VIXY', 'UUP', 'FXE', 'FXY',
]))

PAIRS_CANDIDATES_E = [
    ('XLK', 'XLF'), ('XLV', 'XLI'), ('TLT', 'IEF'), ('HYG', 'LQD'),
    ('GLD', 'SLV'), ('USO', 'DBA'), ('VNQ', 'SCHH'), ('EFA', 'EWJ'),
    ('FXI', 'EWZ'), ('UUP', 'FXE'),
]

LSTM_TICKERS_E = [
    'XLK', 'XLF', 'TLT', 'IEF', 'HYG',
    'GLD', 'USO', 'VNQ', 'EFA', 'FXI',
    'EWZ', 'VIXY', 'UUP', 'FXE', 'INDA',
]

ASSET_GROUPS_E = {
    'us_sector_equity': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLU', 'XLRE', 'XLP', 'XLY', 'XLB', 'XLC'],
    'fixed_income': ['TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'EMB', 'AGG', 'TIP', 'BIL'],
    'commodities': ['GLD', 'SLV', 'USO', 'DBA', 'PDBC', 'CPER', 'UNG', 'CORN', 'WEAT'],
    'real_assets': ['VNQ', 'SCHH', 'VNQI', 'REM'],
    'international_equity': ['EFA', 'EWJ', 'FXI', 'EWZ', 'EWC', 'EWA', 'INDA', 'EWT', 'EIDO', 'EWS'],
    'hedges_fx': ['VIXY', 'UUP', 'FXE', 'FXY'],
}

# ---- Universe A-Liquid: weekly robustness slice ----
UNIVERSE_A_LIQUID = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'AVGO', 'AMZN', 'META',
    'JPM', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'COP', 'SLB',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'PG', 'KO', 'PEP', 'WMT', 'COST',
    'CAT', 'HON', 'UNP', 'UPS', 'GE', 'HD', 'MCD', 'NKE', 'DIS',
    'CMCSA', 'GLD', 'TLT', 'VNQ', 'XLU', 'TSLA',
]

PAIRS_CANDIDATES_A_LIQUID = [
    ('AAPL', 'MSFT'), ('GOOGL', 'META'), ('NVDA', 'AMD'), ('JPM', 'GS'),
    ('BAC', 'WFC'), ('XOM', 'CVX'), ('COP', 'SLB'), ('JNJ', 'PFE'),
    ('PG', 'KO'), ('WMT', 'COST'), ('CAT', 'HON'), ('GLD', 'TLT'),
]

LSTM_TICKERS_A_LIQUID = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'JPM',
    'XOM', 'JNJ', 'PG', 'CAT', 'GLD',
]

ASSET_GROUPS_A_LIQUID = {
    group: [ticker for ticker in tickers if ticker in UNIVERSE_A_LIQUID]
    for group, tickers in ASSET_GROUPS_EXPANDED.items()
}

# ---- Active universe (change this line to switch) ----
UNIVERSE = UNIVERSE_EXPANDED

BENCHMARK = 'SPY'
BENCHMARK_COMPONENTS: tuple[tuple[str, float], ...] = (('SPY', 1.0),)
BENCHMARK_REBALANCE = 'buyhold'
DATA_START = '2013-04-01'
DATA_END = '2026-04-01'

# ---- Pairs candidates ----
PAIRS_CANDIDATES_CORE = [
    ('KO', 'PEP'), ('AAPL', 'MSFT'), ('GOOGL', 'META'),
    ('JPM', 'GS'), ('XOM', 'CVX'),
]

PAIRS_CANDIDATES_EXPANDED = [
    # Tech
    ('AAPL', 'MSFT'), ('GOOGL', 'META'), ('INTC', 'AMD'), ('QCOM', 'AVGO'),
    ('NVDA', 'AMD'),
    # Financials
    ('JPM', 'GS'), ('BAC', 'WFC'), ('MS', 'GS'), ('USB', 'PNC'), ('JPM', 'BAC'),
    # Healthcare
    ('JNJ', 'PFE'), ('ABBV', 'BMY'), ('UNH', 'TMO'), ('MRK', 'LLY'),
    # Consumer Staples
    ('KO', 'PEP'), ('PG', 'CL'), ('WMT', 'COST'), ('KMB', 'GIS'),
    # Energy
    ('XOM', 'CVX'), ('COP', 'EOG'), ('MPC', 'PSX'), ('MPC', 'VLO'), ('SLB', 'EOG'),
    # Industrials
    ('CAT', 'HON'), ('UNP', 'UPS'), ('BA', 'LMT'), ('GE', 'HON'),
    # Communications
    ('VZ', 'T'), ('DIS', 'CMCSA'), ('META', 'NFLX'),
    # Consumer Discretionary
    ('HD', 'LOW'), ('MCD', 'SBUX'), ('NKE', 'SBUX'),
    # Cross-sector
    ('GLD', 'TLT'), ('XOM', 'GLD'),
]

PAIRS_CANDIDATES = (
    PAIRS_CANDIDATES_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else PAIRS_CANDIDATES_CORE
)

# ---- LSTM tickers ----
LSTM_TICKERS_CORE = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
LSTM_TICKERS_EXPANDED = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META',
    'JPM', 'GS', 'BAC',
    'XOM', 'CVX',
    'JNJ', 'UNH', 'LLY',
    'GLD', 'TLT',
]
LSTM_TICKERS = (
    LSTM_TICKERS_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else LSTM_TICKERS_CORE
)

# ---- Backtest window ----
DATA_PERIOD = '13y'  # Legacy fallback when explicit start/end dates are absent

RISK_FREE_RATE = 0.035

FRED_SERIES = {
    'rate_10y': 'DGS10',
    'rate_2y': 'DGS2',
    'fed_funds': 'FEDFUNDS',
    'unrate': 'UNRATE',
    'vix': 'VIXCLS',
    'hy_oas': 'BAMLH0A0HYM2',
    'dxy': 'DTWEXBGS',
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

# ---- Sector groups ----
ASSET_GROUPS_CORE = {
    'growth': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    'cyclical': ['JPM', 'GS', 'XOM', 'CVX'],
    'defensive': ['JNJ', 'PG', 'KO', 'PEP', 'WMT'],
    'diversifier': ['GLD', 'TLT', 'XLU', 'VNQ'],
}

ASSET_GROUPS_EXPANDED = {
    'technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO'],
    'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'BMY'],
    'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
    'industrials': ['CAT', 'BA', 'HON', 'UNP', 'UPS', 'GE', 'MMM', 'LMT'],
    'materials': ['LIN', 'APD', 'SHW', 'FCX'],
    'real_estate': ['AMT', 'PLD', 'PSA', 'O'],
    'utilities': ['NEE', 'DUK', 'SO', 'XEL'],
    'communications': ['META', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX'],
    'consumer_disc': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX'],
    'diversifier': ['GLD', 'TLT', 'VNQ'],
}

ASSET_GROUPS = (
    ASSET_GROUPS_EXPANDED if UNIVERSE is UNIVERSE_EXPANDED
    else ASSET_GROUPS_CORE
)
TICKER_TO_GROUP = {
    ticker: group
    for group, tickers in ASSET_GROUPS.items()
    for ticker in tickers
}
SEC_USER_AGENT = os.getenv(
    'SEC_USER_AGENT',
    'sequential-decision-making/1.0 research@local'
)


@dataclass
class FeatureAvailabilityConfig:
    """Simple research-time availability assumptions for external features."""

    macro_lag_days: int = 3
    allow_static_sec_quality: bool = True
    sec_quality_note: str = 'Static SEC snapshot used as research prior, not point-in-time safe.'

    def __post_init__(self) -> None:
        _validate_int_min('macro_lag_days', self.macro_lag_days, 0)
        if not str(self.sec_quality_note).strip():
            raise ValueError("sec_quality_note must be non-empty")


@dataclass
class CostModelConfig:
    """Transaction-cost and slippage assumptions for backtests.

    When ``use_almgren_chriss`` is True the cost model uses a permanent +
    temporary market-impact formula calibrated from Almgren & Chriss (2000).
    Otherwise falls back to the original base-bps + vol-scaling model.
    """

    base_cost_bps: float = 5.0
    turnover_vol_multiplier: float = 0.20
    size_penalty_bps: float = 7.5
    use_almgren_chriss: bool = True
    ac_permanent_beta: float = 0.10
    ac_temporary_eta: float = 0.50
    cost_stress_multiplier: float = 1.0
    adv_lookback_days: int = 20
    adv_participation_cap: float = 0.05
    adv_penalty_bps: float = 10.0
    execution_delay_days: int = 0

    def __post_init__(self) -> None:
        for name in (
            'base_cost_bps', 'turnover_vol_multiplier', 'size_penalty_bps',
            'ac_permanent_beta', 'ac_temporary_eta', 'cost_stress_multiplier',
            'adv_penalty_bps',
        ):
            _validate_non_negative(name, getattr(self, name))
        _validate_int_min('adv_lookback_days', self.adv_lookback_days, 1)
        _validate_unit_interval('adv_participation_cap', self.adv_participation_cap)
        _validate_int_min('execution_delay_days', self.execution_delay_days, 0)


@dataclass
class OptimizerConfig:
    """Controls for the intermediate constrained allocator."""

    use_optimizer: bool = True
    max_weight: float = 0.18
    risk_aversion: float = 4.0
    alpha_strength: float = 1.5
    anchor_strength: float = 10.0
    turnover_penalty: float = 2.0
    adaptive_allocator: bool = True
    adaptive_allocator_min_invested: float = 0.60
    adaptive_allocator_param_smoothing: float = 0.55
    adaptive_allocator_risk_mult_range: tuple[float, float] = (0.70, 2.40)
    adaptive_allocator_anchor_mult_range: tuple[float, float] = (0.60, 1.90)
    adaptive_allocator_turnover_mult_range: tuple[float, float] = (0.75, 2.50)
    adaptive_allocator_alpha_mult_range: tuple[float, float] = (0.75, 1.35)
    adaptive_allocator_cap_scale_range: tuple[float, float] = (0.75, 1.20)
    adaptive_allocator_group_cap_scale_range: tuple[float, float] = (0.85, 1.05)
    adaptive_allocator_policy_version: int = 1
    group_caps: dict[str, float] = field(
        default_factory=lambda: {group: 0.40 for group in ASSET_GROUPS}
    )

    def __post_init__(self) -> None:
        _validate_unit_interval('max_weight', self.max_weight)
        if self.max_weight == 0.0:
            raise ValueError("max_weight must be > 0")
        for name in ('risk_aversion', 'alpha_strength', 'anchor_strength', 'turnover_penalty'):
            _validate_non_negative(name, getattr(self, name))
        _validate_unit_interval('adaptive_allocator_min_invested', self.adaptive_allocator_min_invested)
        _validate_half_open_unit_interval('adaptive_allocator_param_smoothing', self.adaptive_allocator_param_smoothing)
        for name in (
            'adaptive_allocator_risk_mult_range',
            'adaptive_allocator_anchor_mult_range',
            'adaptive_allocator_turnover_mult_range',
            'adaptive_allocator_alpha_mult_range',
            'adaptive_allocator_cap_scale_range',
            'adaptive_allocator_group_cap_scale_range',
        ):
            _validate_range_tuple(name, getattr(self, name), lower_bound=0.0)
        _validate_int_min('adaptive_allocator_policy_version', self.adaptive_allocator_policy_version, 1)
        for group, cap in self.group_caps.items():
            _validate_unit_interval(f'group_caps[{group}]', cap)
            if float(cap) == 0.0:
                raise ValueError(f"group_caps[{group}] must be > 0")


@dataclass
class AlphaFeedbackConfig:
    """Closed-loop alpha adaptation using prior actions and portfolio state."""

    enabled: bool = False
    exposure_reweight_strength: float = 0.18
    cash_garch_boost: float = 0.18
    cash_hmm_boost: float = 0.24
    crowded_name_penalty_strength: float = 0.08
    action_regime_feedback_strength: float = 0.12
    shrink_in_stress_strength: float = 0.08
    policy_version: int = 2

    def __post_init__(self) -> None:
        for name in (
            'exposure_reweight_strength', 'cash_garch_boost', 'cash_hmm_boost',
            'crowded_name_penalty_strength', 'action_regime_feedback_strength',
            'shrink_in_stress_strength',
        ):
            _validate_non_negative(name, getattr(self, name))
        _validate_int_min('policy_version', self.policy_version, 1)


@dataclass
class OptionOverlayConfig:
    """Controls for the option-based hedge sleeve."""

    use_option_overlay: bool = True
    hedge_types: tuple[str, ...] = ('protective_put', 'collar', 'put_spread')
    target_dte_days: int = 21
    put_strike_otm: float = 0.04
    call_strike_otm: float = 0.05
    spread_width: float = 0.10
    max_effective_hedge: float = 0.35
    theta_premium_scale: float = 0.55
    convexity_scale: float = 1.15
    collar_financing_ratio: float = 0.65

    def __post_init__(self) -> None:
        _validate_int_min('target_dte_days', self.target_dte_days, 1)
        for name in ('put_strike_otm', 'call_strike_otm', 'spread_width', 'max_effective_hedge', 'collar_financing_ratio'):
            _validate_unit_interval(name, getattr(self, name))
        for name in ('theta_premium_scale', 'convexity_scale'):
            _validate_non_negative(name, getattr(self, name))


# ============================================================
# Control Method Registry
# ============================================================
# All control candidates from architecture revision v2.
CONTROL_METHODS = (
    'none',                  # No control — factor-only baseline
    'fixed',                 # A1: Fixed allocator (constant invested fraction)
    'vol_target',            # A2: Volatility targeting
    'dd_delever',            # A3: Drawdown-based deleveraging
    'regime_rules',          # A4: Regime-conditioned exposure rules
    'ensemble_rules',        # A5: Simple ensemble of A2–A4
    'linucb',                # B1: Contextual bandit — LinUCB
    'thompson',              # B2: Contextual bandit — Thompson Sampling
    'epsilon_greedy',        # B3: Contextual bandit — epsilon-greedy linear
    'supervised',            # C:  Supervised regime-conditioned controller
    'cvar_robust',           # D:  CVaR-aware robust optimization
    'council',               # E:  Expert-gated council meta-controller
    'mlp_meta',              # G:  MLP-gated meta-controller (environment-adaptive)
    'mpc',                   # H:  Model-predictive control over allocator exposure and stabilization
    'cmdp_lagrangian',       # F:  Simple constrained-MDP controller
    'q_learning',            # RL: Tabular Q-learning (portfolio only)
    'ppo',                   # RL: End-to-end PPO
)
SUPPORTED_CONTROL_METHODS = CONTROL_METHODS + ('adaptive_allocator',)


@dataclass
class ControlConfig:
    """Configuration for the pluggable control layer."""

    method: str = 'none'
    # A1: Fixed allocator
    fixed_invested_fraction: float = 0.95
    # A2: Vol-target
    vol_target_annual: float = 0.15
    vol_lookback: int = 63
    # A3: DD-delever
    dd_thresholds: tuple[tuple[float, float], ...] = ((-0.05, 1.0), (-0.08, 0.70), (-0.12, 0.40))
    dd_min_invested: float = 0.30
    # A4: Regime rules
    regime_bull_threshold: float = 0.70
    regime_bear_threshold: float = 0.30
    regime_bull_fraction: float = 1.00
    regime_neutral_fraction: float = 0.90
    regime_bear_fraction: float = 0.75
    # A5: Ensemble
    ensemble_mode: str = 'mean'  # 'mean' or 'min'
    # B: Bandit
    bandit_n_actions: int = 5
    bandit_reward_window: int = 5
    bandit_alpha_ucb: float = 1.0
    bandit_epsilon: float = 0.10
    bandit_feature_lookback: int = 252
    # C: Supervised controller
    supervised_model: str = 'logistic'  # 'logistic', 'random_forest', 'decision_tree'
    supervised_retrain_every: int = 63
    supervised_label_window: int = 21
    # D: CVaR-robust
    cvar_confidence: float = 0.95
    cvar_n_scenarios: int = 500
    cvar_lambda_base: float = 1.0
    cvar_regime_scaling: bool = True
    cvar_dd_budget: bool = True
    # E: Expert-gated council
    council_experts: tuple[str, ...] = ('regime_rules', 'linucb', 'cvar_robust')
    council_gate_model: str = 'logistic'
    council_retrain_every: int = 63
    council_min_samples: int = 40
    council_temperature: float = 1.0
    council_min_weight: float = 0.10
    council_default_bias: tuple[float, ...] = (0.20, 0.20, 0.60)
    # H: Model-predictive controller
    mpc_horizon: int = 5
    mpc_replan_every: int = 5
    mpc_discount: float = 0.92
    mpc_alpha_decay: float = 0.88
    mpc_stress_reversion: float = 0.82
    mpc_min_invested: float = 0.35
    mpc_max_stabilizer: float = 0.45
    mpc_risk_penalty: float = 8.0
    mpc_turnover_penalty: float = 2.5
    mpc_drawdown_penalty: float = 6.0
    mpc_stress_penalty: float = 1.5
    mpc_terminal_penalty: float = 0.75
    mpc_max_daily_change: float = 0.18
    mpc_objective_version: int = 2
    mpc_joint_convexity: bool = False
    mpc_convexity_tail_scale: float = 1.65
    # Legacy decision-aware allocator controller (kept for backwards compatibility;
    # the active architecture now applies these rules directly inside the allocator)
    adaptive_allocator_min_invested: float = 0.60
    adaptive_allocator_param_smoothing: float = 0.55
    adaptive_allocator_risk_mult_range: tuple[float, float] = (0.70, 2.40)
    adaptive_allocator_anchor_mult_range: tuple[float, float] = (0.60, 1.90)
    adaptive_allocator_turnover_mult_range: tuple[float, float] = (0.75, 2.50)
    adaptive_allocator_alpha_mult_range: tuple[float, float] = (0.75, 1.35)
    adaptive_allocator_cap_scale_range: tuple[float, float] = (0.75, 1.20)
    adaptive_allocator_group_cap_scale_range: tuple[float, float] = (0.85, 1.05)
    adaptive_allocator_policy_version: int = 1
    # F: CMDP-style constrained controller
    cmdp_constraint_type: str = 'drawdown'  # 'drawdown' or 'tail_loss'
    cmdp_constraint_kappa: float = 0.12
    cmdp_lambda_init: float = 1.0
    cmdp_lambda_lr: float = 0.05
    cmdp_tail_loss_threshold: float = 0.01
    # Convexity-aware payoff shaping
    convexity_enabled: bool = False
    convexity_threshold: float = 0.0
    convexity_mode_carries: tuple[float, float, float] = (0.0, 0.00015, 0.00040)
    convexity_mode_lambdas: tuple[float, float, float] = (0.0, 1.50, 4.00)
    convexity_mild_drawdown: float = -0.05
    convexity_strong_drawdown: float = -0.10
    convexity_mild_vol: float = 0.18
    convexity_strong_vol: float = 0.26
    convexity_mild_regime: float = 0.45
    convexity_strong_regime: float = 0.30
    # G: MLP Meta-controller (environment-adaptive controller selection)
    mlp_meta_experts: tuple[str, ...] = ('regime_rules', 'linucb', 'cvar_robust')
    mlp_meta_hidden_layers: tuple[int, ...] = (32, 16)
    mlp_meta_retrain_every: int = 63
    mlp_meta_min_samples: int = 60
    mlp_meta_feature_lookback: int = 63
    mlp_meta_min_weight: float = 0.10
    mlp_meta_default_bias: tuple[float, ...] = (0.20, 0.20, 0.60)
    mlp_meta_learning_rate: float = 0.001
    mlp_meta_alpha_reg: float = 0.001
    mlp_meta_temperature: float = 1.0
    # Q-learning (portfolio only)
    ql_alpha: float = 0.03
    ql_gamma: float = 0.95
    ql_epsilon: float = 0.15

    def __post_init__(self) -> None:
        _validate_choice('method', self.method, SUPPORTED_CONTROL_METHODS)
        _validate_non_negative('fixed_invested_fraction', self.fixed_invested_fraction)
        _validate_positive('vol_target_annual', self.vol_target_annual)
        _validate_int_min('vol_lookback', self.vol_lookback, 2)
        _validate_non_negative('dd_min_invested', self.dd_min_invested)
        _validate_choice('ensemble_mode', self.ensemble_mode, ('mean', 'min'))
        _validate_int_min('bandit_n_actions', self.bandit_n_actions, 2)
        _validate_int_min('bandit_reward_window', self.bandit_reward_window, 1)
        _validate_positive('bandit_alpha_ucb', self.bandit_alpha_ucb)
        _validate_unit_interval('bandit_epsilon', self.bandit_epsilon)
        _validate_int_min('bandit_feature_lookback', self.bandit_feature_lookback, 1)
        _validate_choice('supervised_model', self.supervised_model, ('logistic', 'random_forest', 'decision_tree'))
        _validate_int_min('supervised_retrain_every', self.supervised_retrain_every, 1)
        _validate_int_min('supervised_label_window', self.supervised_label_window, 1)
        _validate_unit_interval('cvar_confidence', self.cvar_confidence)
        if self.cvar_confidence == 1.0:
            raise ValueError("cvar_confidence must be < 1")
        _validate_int_min('cvar_n_scenarios', self.cvar_n_scenarios, 1)
        _validate_non_negative('cvar_lambda_base', self.cvar_lambda_base)
        if not self.council_experts:
            raise ValueError("council_experts must be non-empty")
        if not self.mlp_meta_experts:
            raise ValueError("mlp_meta_experts must be non-empty")
        for expert in self.council_experts:
            _validate_choice('council_experts', expert, ('regime_rules', 'linucb', 'cvar_robust'))
        for expert in self.mlp_meta_experts:
            _validate_choice('mlp_meta_experts', expert, ('regime_rules', 'linucb', 'cvar_robust'))
        _validate_choice('council_gate_model', self.council_gate_model, ('logistic',))
        _validate_int_min('council_retrain_every', self.council_retrain_every, 1)
        _validate_int_min('council_min_samples', self.council_min_samples, 1)
        _validate_positive('council_temperature', self.council_temperature)
        _validate_unit_interval('council_min_weight', self.council_min_weight)
        if len(self.council_default_bias) != len(self.council_experts):
            raise ValueError("council_default_bias must match council_experts length")
        _validate_int_min('mpc_horizon', self.mpc_horizon, 1)
        _validate_int_min('mpc_replan_every', self.mpc_replan_every, 1)
        _validate_unit_interval('mpc_discount', self.mpc_discount)
        _validate_unit_interval('mpc_alpha_decay', self.mpc_alpha_decay)
        _validate_unit_interval('mpc_stress_reversion', self.mpc_stress_reversion)
        _validate_unit_interval('mpc_min_invested', self.mpc_min_invested)
        _validate_unit_interval('mpc_max_stabilizer', self.mpc_max_stabilizer)
        for name in ('mpc_risk_penalty', 'mpc_turnover_penalty', 'mpc_drawdown_penalty', 'mpc_stress_penalty', 'mpc_terminal_penalty'):
            _validate_non_negative(name, getattr(self, name))
        _validate_unit_interval('mpc_max_daily_change', self.mpc_max_daily_change)
        _validate_int_min('mpc_objective_version', self.mpc_objective_version, 1)
        _validate_non_negative('mpc_convexity_tail_scale', self.mpc_convexity_tail_scale)
        _validate_unit_interval('adaptive_allocator_min_invested', self.adaptive_allocator_min_invested)
        _validate_half_open_unit_interval('adaptive_allocator_param_smoothing', self.adaptive_allocator_param_smoothing)
        for name in (
            'adaptive_allocator_risk_mult_range',
            'adaptive_allocator_anchor_mult_range',
            'adaptive_allocator_turnover_mult_range',
            'adaptive_allocator_alpha_mult_range',
            'adaptive_allocator_cap_scale_range',
            'adaptive_allocator_group_cap_scale_range',
        ):
            _validate_range_tuple(name, getattr(self, name), lower_bound=0.0)
        _validate_int_min('adaptive_allocator_policy_version', self.adaptive_allocator_policy_version, 1)
        _validate_choice('cmdp_constraint_type', self.cmdp_constraint_type, ('drawdown', 'tail_loss'))
        _validate_non_negative('cmdp_constraint_kappa', self.cmdp_constraint_kappa)
        _validate_non_negative('cmdp_lambda_init', self.cmdp_lambda_init)
        _validate_non_negative('cmdp_lambda_lr', self.cmdp_lambda_lr)
        _validate_non_negative('cmdp_tail_loss_threshold', self.cmdp_tail_loss_threshold)
        if len(self.convexity_mode_carries) != 3 or len(self.convexity_mode_lambdas) != 3:
            raise ValueError("convexity_mode_carries and convexity_mode_lambdas must both have length 3")
        for idx, value in enumerate(self.convexity_mode_carries):
            _validate_non_negative(f'convexity_mode_carries[{idx}]', value)
        for idx, value in enumerate(self.convexity_mode_lambdas):
            _validate_non_negative(f'convexity_mode_lambdas[{idx}]', value)
        _validate_int_min('mlp_meta_retrain_every', self.mlp_meta_retrain_every, 1)
        _validate_int_min('mlp_meta_min_samples', self.mlp_meta_min_samples, 1)
        _validate_int_min('mlp_meta_feature_lookback', self.mlp_meta_feature_lookback, 1)
        _validate_unit_interval('mlp_meta_min_weight', self.mlp_meta_min_weight)
        if len(self.mlp_meta_default_bias) != len(self.mlp_meta_experts):
            raise ValueError("mlp_meta_default_bias must match mlp_meta_experts length")
        _validate_positive('mlp_meta_learning_rate', self.mlp_meta_learning_rate)
        _validate_non_negative('mlp_meta_alpha_reg', self.mlp_meta_alpha_reg)
        _validate_positive('mlp_meta_temperature', self.mlp_meta_temperature)
        _validate_positive('ql_alpha', self.ql_alpha)
        _validate_unit_interval('ql_gamma', self.ql_gamma)
        _validate_unit_interval('ql_epsilon', self.ql_epsilon)


@dataclass
class ExperimentConfig:
    """Feature toggles used for ablation studies."""

    label: str = 'full_pipeline'
    use_factor: bool = True
    use_pairs: bool = False
    use_lstm: bool = False
    adaptive_combiner: bool = True
    use_portfolio_rl: bool = True
    use_hedge_rl: bool = False
    # State-feature ablation toggles
    use_uncertainty_state: bool = False
    use_regime_state: bool = False
    use_vol_state: bool = False
    # Control method (new architecture v2)
    control_method: str = 'none'

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("experiment label must be non-empty")
        _validate_choice('control_method', self.control_method, SUPPORTED_CONTROL_METHODS)


@dataclass
class PipelineConfig:
    """Top-level configuration for a single backtest experiment."""

    train_frac: float = 0.5
    rebalance_band: float = 0.015
    min_turnover: float = 0.08
    hedge_ratios: tuple[float, ...] = (0.0, 0.03, 0.08, 0.15)
    portfolio_reward_mode: str = 'sortino'
    hedge_reward_mode: str = 'asymmetric_return'
    e2e_reward_mode: str = 'differential_sharpe'
    enable_e2e_baseline: bool = True
    e2e_ppo_verbose: int = 0
    e2e_ppo_log_interval: int = 10
    feature_availability: FeatureAvailabilityConfig = field(default_factory=FeatureAvailabilityConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    alpha_feedback: AlphaFeedbackConfig = field(default_factory=AlphaFeedbackConfig)
    option_overlay: OptionOverlayConfig = field(default_factory=OptionOverlayConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.train_frac) < 1.0:
            raise ValueError(f"train_frac must lie in (0, 1), got {self.train_frac!r}")
        _validate_unit_interval('rebalance_band', self.rebalance_band)
        _validate_unit_interval('min_turnover', self.min_turnover)
        for idx, value in enumerate(self.hedge_ratios):
            _validate_unit_interval(f'hedge_ratios[{idx}]', value)
        _validate_choice('portfolio_reward_mode', self.portfolio_reward_mode, ('differential_sharpe', 'return', 'sortino', 'mean_variance', 'asymmetric_return'))
        _validate_choice('hedge_reward_mode', self.hedge_reward_mode, ('differential_sharpe', 'return', 'sortino', 'mean_variance', 'asymmetric_return'))
        _validate_choice('e2e_reward_mode', self.e2e_reward_mode, ('differential_sharpe', 'return', 'sortino', 'mean_variance'))


@dataclass
class EvaluationConfig:
    """Grid settings for the research evaluation engine."""

    output_dir: str = 'results'
    timestamp_outputs: bool = True
    train_fracs: tuple[float, ...] = (0.4, 0.5, 0.6)
    rolling_train_frac: float = 0.5
    rolling_window_days: int = 504
    rolling_step_days: int = 126
    min_rolling_windows: int = 4
    max_rolling_windows: int = 6
    cost_bps_grid: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 20.0, 35.0)
    cost_stress_multiplier_grid: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    rebalance_band_grid: tuple[float, ...] = (0.005, 0.015, 0.03)
    hedge_scale_grid: tuple[float, ...] = (0.75, 1.0, 1.25)
    adv_participation_cap_grid: tuple[float, ...] = (0.02, 0.05, 0.10)
    execution_delay_grid: tuple[int, ...] = (0, 1, 2, 5)
    macro_lag_grid: tuple[int, ...] = (1, 3, 5)
    reward_mode_grid: tuple[str, ...] = ('differential_sharpe', 'return', 'sortino', 'mean_variance')
    research_e2e_scope: str = 'baseline_only'
    enable_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints/research_runs'
    checkpoint_match_mode: str = 'compatible'
    bootstrap_samples: int = 400
    bootstrap_block_size: int = 20
    bootstrap_seed: int = 7
    # Cross-universe meta-learning evaluation
    meta_learning_enabled: bool = False
    meta_learning_universes: tuple[str, ...] = ('A', 'B')
    # Time-series cross-validation
    ts_cv_folds: int = 5
    enable_ts_cv: bool = True

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must be non-empty")
        if not self.train_fracs:
            raise ValueError("train_fracs must be non-empty")
        for idx, value in enumerate(self.train_fracs):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"train_fracs[{idx}] must lie in (0, 1), got {value!r}")
        if not 0.0 < float(self.rolling_train_frac) < 1.0:
            raise ValueError(f"rolling_train_frac must lie in (0, 1), got {self.rolling_train_frac!r}")
        _validate_int_min('rolling_window_days', self.rolling_window_days, 2)
        _validate_int_min('rolling_step_days', self.rolling_step_days, 1)
        _validate_int_min('min_rolling_windows', self.min_rolling_windows, 1)
        _validate_int_min('max_rolling_windows', self.max_rolling_windows, self.min_rolling_windows)
        for name in ('cost_bps_grid', 'cost_stress_multiplier_grid', 'rebalance_band_grid', 'hedge_scale_grid', 'adv_participation_cap_grid', 'execution_delay_grid', 'macro_lag_grid', 'reward_mode_grid'):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must be non-empty")
        for idx, value in enumerate(self.cost_bps_grid):
            _validate_non_negative(f'cost_bps_grid[{idx}]', value)
        for idx, value in enumerate(self.cost_stress_multiplier_grid):
            _validate_non_negative(f'cost_stress_multiplier_grid[{idx}]', value)
        for idx, value in enumerate(self.rebalance_band_grid):
            _validate_unit_interval(f'rebalance_band_grid[{idx}]', value)
        for idx, value in enumerate(self.hedge_scale_grid):
            _validate_non_negative(f'hedge_scale_grid[{idx}]', value)
        for idx, value in enumerate(self.adv_participation_cap_grid):
            _validate_unit_interval(f'adv_participation_cap_grid[{idx}]', value)
        for idx, value in enumerate(self.execution_delay_grid):
            _validate_int_min(f'execution_delay_grid[{idx}]', value, 0)
        for idx, value in enumerate(self.macro_lag_grid):
            _validate_int_min(f'macro_lag_grid[{idx}]', value, 0)
        for idx, value in enumerate(self.reward_mode_grid):
            _validate_choice(f'reward_mode_grid[{idx}]', value, ('differential_sharpe', 'return', 'sortino', 'mean_variance'))
        _validate_choice('research_e2e_scope', self.research_e2e_scope, ('baseline_only', 'all', 'disabled'))
        _validate_choice('checkpoint_match_mode', self.checkpoint_match_mode, ('strict', 'compatible', 'config_only'))
        _validate_int_min('bootstrap_samples', self.bootstrap_samples, 1)
        _validate_int_min('bootstrap_block_size', self.bootstrap_block_size, 1)
        _validate_int_min('ts_cv_folds', self.ts_cv_folds, 2)
        if not self.meta_learning_universes:
            raise ValueError("meta_learning_universes must be non-empty")
        for idx, value in enumerate(self.meta_learning_universes):
            _validate_choice(f'meta_learning_universes[{idx}]', value, ('A', 'B', 'C', 'D', 'E', 'A_LIQUID'))


@dataclass
class UniverseProfile:
    """Bundles all universe-specific settings for cross-universe evaluation."""

    label: str
    tickers: list[str]
    pairs: list[tuple[str, str]]
    lstm_tickers: list[str]
    asset_groups: dict[str, list[str]]
    benchmark_label: str
    benchmark_components: tuple[tuple[str, float], ...]
    benchmark_rebalance: str = 'buyhold'
    data_start: str | None = None
    data_end: str | None = None
    rebalance_frequency: str = 'daily'
    role: str = ''


def get_active_benchmark_label() -> str:
    """Return the currently active benchmark label."""
    return str(BENCHMARK)


def get_universe_profile(universe_id: str) -> UniverseProfile:
    """Return a complete universe profile by ID."""
    if universe_id == 'A':
        return UniverseProfile(
            label='A',
            tickers=list(UNIVERSE_EXPANDED),
            pairs=list(PAIRS_CANDIDATES_EXPANDED),
            lstm_tickers=list(LSTM_TICKERS_EXPANDED),
            asset_groups=dict(ASSET_GROUPS_EXPANDED),
            benchmark_label='SPY',
            benchmark_components=(('SPY', 1.0),),
            data_start='2013-04-01',
            data_end='2026-04-01',
            rebalance_frequency='daily',
            role='Primary U.S. large/mid-cap equity universe',
        )
    if universe_id == 'B':
        return UniverseProfile(
            label='B',
            tickers=list(UNIVERSE_B),
            pairs=list(PAIRS_CANDIDATES_B),
            lstm_tickers=list(LSTM_TICKERS_B),
            asset_groups=dict(ASSET_GROUPS_B),
            benchmark_label='SPY',
            benchmark_components=(('SPY', 1.0),),
            data_start='2013-04-01',
            data_end='2026-04-01',
            rebalance_frequency='daily',
            role='Zero-overlap U.S. mid-cap / biotech / REIT tilt robustness universe',
        )
    if universe_id == 'C':
        return UniverseProfile(
            label='C',
            tickers=list(UNIVERSE_C),
            pairs=list(PAIRS_CANDIDATES_C),
            lstm_tickers=list(LSTM_TICKERS_C),
            asset_groups=dict(ASSET_GROUPS_C),
            benchmark_label='VGK',
            benchmark_components=(('VGK', 1.0),),
            data_start='2014-01-01',
            data_end='2026-04-01',
            rebalance_frequency='daily',
            role='European equity generalization universe',
        )
    if universe_id == 'D':
        return UniverseProfile(
            label='D',
            tickers=list(UNIVERSE_D),
            pairs=list(PAIRS_CANDIDATES_D),
            lstm_tickers=list(LSTM_TICKERS_D),
            asset_groups=dict(ASSET_GROUPS_D),
            benchmark_label='EEM',
            benchmark_components=(('EEM', 1.0),),
            data_start='2015-01-01',
            data_end='2026-04-01',
            rebalance_frequency='daily',
            role='Emerging-markets high-volatility universe with the sharpest operator prediction',
        )
    if universe_id == 'E':
        return UniverseProfile(
            label='E',
            tickers=list(UNIVERSE_E),
            pairs=list(PAIRS_CANDIDATES_E),
            lstm_tickers=list(LSTM_TICKERS_E),
            asset_groups=dict(ASSET_GROUPS_E),
            benchmark_label='60/40',
            benchmark_components=(('SPY', 0.60), ('TLT', 0.40)),
            benchmark_rebalance='monthly',
            data_start='2014-01-01',
            data_end='2026-04-01',
            rebalance_frequency='daily',
            role='Multi-asset secondary stress-test universe',
        )
    if universe_id == 'A_LIQUID':
        return UniverseProfile(
            label='A-Liquid',
            tickers=list(UNIVERSE_A_LIQUID),
            pairs=list(PAIRS_CANDIDATES_A_LIQUID),
            lstm_tickers=list(LSTM_TICKERS_A_LIQUID),
            asset_groups=dict(ASSET_GROUPS_A_LIQUID),
            benchmark_label='SPY',
            benchmark_components=(('SPY', 1.0),),
            data_start='2013-04-01',
            data_end='2026-04-01',
            rebalance_frequency='weekly',
            role='Weekly-rebalanced liquidity robustness slice of Universe A',
        )
    raise ValueError(f"Unknown universe ID: {universe_id!r}. Choose one of 'A', 'B', 'C', 'D', 'E', or 'A_LIQUID'.")


@contextmanager
def use_universe(universe_id: str):
    """Temporarily swap the active universe across all quant_stack modules.

    This context manager patches the module-level UNIVERSE,
    PAIRS_CANDIDATES, LSTM_TICKERS, ASSET_GROUPS, TICKER_TO_GROUP,
    benchmark settings, and date window
    constants in every module that imports them, so that
    ``load_market_data()`` and ``run_full_pipeline()`` operate on the
    requested universe without permanent side effects.

    Usage::

        with use_universe('B'):
            prices, volumes, returns, macro, sec = load_market_data()
            results = run_full_pipeline(prices, volumes, returns, ...)
    """
    import importlib
    import quant_stack.config as _cfg
    import quant_stack.data as _data
    import quant_stack.pipeline as _pipeline
    import quant_stack.alpha as _alpha

    profile = get_universe_profile(universe_id)
    new_ticker_to_group = {
        ticker: group
        for group, tickers in profile.asset_groups.items()
        for ticker in tickers
    }

    # Save originals
    saved = {
        'cfg_UNIVERSE': _cfg.UNIVERSE,
        'cfg_PAIRS': getattr(_cfg, 'PAIRS_CANDIDATES', None),
        'cfg_LSTM': getattr(_cfg, 'LSTM_TICKERS', None),
        'cfg_ASSET_GROUPS': getattr(_cfg, 'ASSET_GROUPS', None),
        'cfg_TICKER_TO_GROUP': getattr(_cfg, 'TICKER_TO_GROUP', None),
        'cfg_BENCHMARK': getattr(_cfg, 'BENCHMARK', None),
        'cfg_BENCHMARK_COMPONENTS': getattr(_cfg, 'BENCHMARK_COMPONENTS', None),
        'cfg_BENCHMARK_REBALANCE': getattr(_cfg, 'BENCHMARK_REBALANCE', None),
        'cfg_DATA_START': getattr(_cfg, 'DATA_START', None),
        'cfg_DATA_END': getattr(_cfg, 'DATA_END', None),
        'data_UNIVERSE': _data.UNIVERSE,
        'data_BENCHMARK': _data.BENCHMARK,
        'data_BENCHMARK_COMPONENTS': getattr(_data, 'BENCHMARK_COMPONENTS', None),
        'data_DATA_START': getattr(_data, 'DATA_START', None),
        'data_DATA_END': getattr(_data, 'DATA_END', None),
        'pipeline_UNIVERSE': _pipeline.UNIVERSE,
        'pipeline_BENCHMARK': getattr(_pipeline, 'BENCHMARK', None),
        'pipeline_BENCHMARK_COMPONENTS': getattr(_pipeline, 'BENCHMARK_COMPONENTS', None),
        'pipeline_BENCHMARK_REBALANCE': getattr(_pipeline, 'BENCHMARK_REBALANCE', None),
        'alpha_UNIVERSE': _alpha.UNIVERSE,
        'alpha_PAIRS': _alpha.PAIRS_CANDIDATES,
        'alpha_TICKER_TO_GROUP': _alpha.TICKER_TO_GROUP,
    }

    try:
        # Patch config module
        _cfg.UNIVERSE = profile.tickers
        _cfg.PAIRS_CANDIDATES = profile.pairs
        _cfg.LSTM_TICKERS = profile.lstm_tickers
        _cfg.ASSET_GROUPS = profile.asset_groups
        _cfg.TICKER_TO_GROUP = new_ticker_to_group
        _cfg.BENCHMARK = profile.benchmark_label
        _cfg.BENCHMARK_COMPONENTS = tuple(profile.benchmark_components)
        _cfg.BENCHMARK_REBALANCE = profile.benchmark_rebalance
        _cfg.DATA_START = profile.data_start
        _cfg.DATA_END = profile.data_end

        # Patch downstream modules that imported at module level
        _data.UNIVERSE = profile.tickers
        _data.BENCHMARK = profile.benchmark_label
        _data.BENCHMARK_COMPONENTS = tuple(profile.benchmark_components)
        _data.DATA_START = profile.data_start
        _data.DATA_END = profile.data_end
        _pipeline.UNIVERSE = profile.tickers
        _pipeline.BENCHMARK = profile.benchmark_label
        _pipeline.BENCHMARK_COMPONENTS = tuple(profile.benchmark_components)
        _pipeline.BENCHMARK_REBALANCE = profile.benchmark_rebalance
        _alpha.UNIVERSE = profile.tickers
        _alpha.PAIRS_CANDIDATES = profile.pairs
        _alpha.TICKER_TO_GROUP = new_ticker_to_group

        yield profile
    finally:
        # Restore originals
        _cfg.UNIVERSE = saved['cfg_UNIVERSE']
        _cfg.PAIRS_CANDIDATES = saved['cfg_PAIRS']
        _cfg.LSTM_TICKERS = saved['cfg_LSTM']
        _cfg.ASSET_GROUPS = saved['cfg_ASSET_GROUPS']
        _cfg.TICKER_TO_GROUP = saved['cfg_TICKER_TO_GROUP']
        _cfg.BENCHMARK = saved['cfg_BENCHMARK']
        _cfg.BENCHMARK_COMPONENTS = saved['cfg_BENCHMARK_COMPONENTS']
        _cfg.BENCHMARK_REBALANCE = saved['cfg_BENCHMARK_REBALANCE']
        _cfg.DATA_START = saved['cfg_DATA_START']
        _cfg.DATA_END = saved['cfg_DATA_END']
        _data.UNIVERSE = saved['data_UNIVERSE']
        _data.BENCHMARK = saved['data_BENCHMARK']
        _data.BENCHMARK_COMPONENTS = saved['data_BENCHMARK_COMPONENTS']
        _data.DATA_START = saved['data_DATA_START']
        _data.DATA_END = saved['data_DATA_END']
        _pipeline.UNIVERSE = saved['pipeline_UNIVERSE']
        _pipeline.BENCHMARK = saved['pipeline_BENCHMARK']
        _pipeline.BENCHMARK_COMPONENTS = saved['pipeline_BENCHMARK_COMPONENTS']
        _pipeline.BENCHMARK_REBALANCE = saved['pipeline_BENCHMARK_REBALANCE']
        _alpha.UNIVERSE = saved['alpha_UNIVERSE']
        _alpha.PAIRS_CANDIDATES = saved['alpha_PAIRS']
        _alpha.TICKER_TO_GROUP = saved['alpha_TICKER_TO_GROUP']
