"""Data loading and feature enrichment for the quant trading pipeline."""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sp_stats

from .config import (
    BLS_SERIES,
    BENCHMARK,
    DATA_PERIOD,
    FRED_SERIES,
    SEC_USER_AGENT,
    TREASURY_SECURITIES,
    UNIVERSE,
)

def _sanitize_ohlcv_download(
    data: pd.DataFrame,
    requested_tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop fully missing tickers while keeping the rest of the panel usable."""
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), requested_tickers

    close = data['Close'].dropna(axis=1, how='all')
    volume = data['Volume'].dropna(axis=1, how='all')
    common_tickers = close.columns.intersection(volume.columns)
    dropped_tickers = sorted(set(requested_tickers) - set(common_tickers))

    prices = close.loc[:, common_tickers].dropna()
    volumes = volume.loc[:, common_tickers].dropna()
    returns = prices.pct_change().dropna()

    common_idx = prices.index.intersection(volumes.index).intersection(returns.index)
    return prices.loc[common_idx], volumes.loc[common_idx], returns.loc[common_idx], dropped_tickers


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download OHLCV for the full universe + benchmark."""
    print("=" * 60)
    print("STAGE 1: Loading Market Data")
    print("=" * 60)
    all_tickers = list(set(UNIVERSE + [BENCHMARK]))
    data = yf.download(all_tickers, period=DATA_PERIOD, auto_adjust=True)
    prices, volumes, returns, dropped_tickers = _sanitize_ohlcv_download(data, all_tickers)

    if prices.empty or volumes.empty or returns.empty:
        raise RuntimeError(
            "No market data was downloaded from yfinance. "
            "Check network access or the upstream data source and try again."
        )
    if BENCHMARK not in prices.columns:
        raise RuntimeError(
            f"Benchmark {BENCHMARK} was not downloaded from yfinance. "
            "Check network access or try again."
        )
    if dropped_tickers:
        print(f"  Dropped tickers with no downloaded OHLCV: {dropped_tickers}")

    macro_data = load_macro_data(prices.index)
    sec_quality_scores = load_sec_quality_scores([t for t in UNIVERSE if t in prices.columns])

    print(f"  Loaded {len(prices)} trading days for {len(all_tickers)} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    if not macro_data.empty:
        print(f"  Macro features loaded: {list(macro_data.columns)}")
    if not sec_quality_scores.empty:
        print(f"  SEC quality scores loaded for {len(sec_quality_scores)} names")
    return prices, volumes, returns, macro_data, sec_quality_scores


def _fetch_json(
    url: str,
    params: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> dict[str, object]:
    request_url = url
    if params:
        request_url = f"{url}?{urlencode(params)}"
    req = Request(request_url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _fetch_json_post(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> dict[str, object]:
    req = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers or {},
        method='POST',
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _align_macro_to_trading_days(frame: pd.DataFrame, trading_index: pd.Index) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=trading_index)
    aligned = frame.sort_index()
    aligned = aligned.reindex(aligned.index.union(trading_index)).sort_index().ffill()
    return aligned.reindex(trading_index).ffill()


def _parse_bls_monthly_series(series_rows: list[dict[str, str]]) -> pd.Series:
    values = {}
    for item in series_rows:
        period = item.get('period', '')
        if not period.startswith('M') or period == 'M13':
            continue
        raw_value = item.get('value')
        if raw_value in (None, '', '-', '.'):
            continue
        year = int(item['year'])
        month = int(period[1:])
        ts = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        try:
            values[ts] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return pd.Series(values, dtype=float).sort_index()


def load_bls_macro_data(trading_index: pd.Index) -> pd.DataFrame:
    """Load public BLS monthly macro series without requiring an API key."""
    start_year = str((pd.Timestamp(trading_index.min()) - pd.Timedelta(days=400)).year)
    end_year = str(pd.Timestamp(trading_index.max()).year)
    try:
        payload = _fetch_json_post(
            'https://api.bls.gov/publicAPI/v1/timeseries/data/',
            {'seriesid': list(BLS_SERIES.values()), 'startyear': start_year, 'endyear': end_year},
            headers={'Content-Type': 'application/json'},
        )
    except Exception as exc:
        print(f"  BLS macro load failed: {exc}")
        return pd.DataFrame(index=trading_index)

    series_data = {}
    for series in payload.get('Results', {}).get('series', []):
        series_id = series.get('seriesID')
        label = next((k for k, v in BLS_SERIES.items() if v == series_id), None)
        if label is None:
            continue
        parsed = _parse_bls_monthly_series(series.get('data', []))
        if not parsed.empty:
            series_data[label] = parsed

    if not series_data:
        return pd.DataFrame(index=trading_index)

    macro = pd.DataFrame(series_data)
    if 'cpi_all_items' in macro.columns:
        macro['inflation_yoy'] = macro['cpi_all_items'].pct_change(12) * 100
    return _align_macro_to_trading_days(macro, trading_index)


def load_treasury_macro_data(trading_index: pd.Index) -> pd.DataFrame:
    """Load open Treasury rate data without any API key."""
    start_date = (pd.Timestamp(trading_index.min()) - pd.Timedelta(days=400)).date().isoformat()
    try:
        payload = _fetch_json(
            'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates',
            params={
                'fields': 'record_date,security_desc,avg_interest_rate_amt',
                'filter': f'record_date:gte:{start_date}',
                'page[size]': 10000,
            },
        )
    except Exception as exc:
        print(f"  Treasury macro load failed: {exc}")
        return pd.DataFrame(index=trading_index)

    rows = payload.get('data', [])
    if not rows:
        return pd.DataFrame(index=trading_index)

    records = []
    for row in rows:
        security = row.get('security_desc')
        label = next((k for k, v in TREASURY_SECURITIES.items() if v == security), None)
        rate = row.get('avg_interest_rate_amt')
        if label is None or rate in (None, '', 'null'):
            continue
        records.append({
            'date': pd.to_datetime(row['record_date']),
            'label': label,
            'value': float(rate),
        })

    if not records:
        return pd.DataFrame(index=trading_index)

    treasury = pd.DataFrame(records).pivot_table(index='date', columns='label', values='value', aggfunc='last')
    treasury = treasury.sort_index()
    if {'treasury_bond_rate', 'treasury_bill_rate'}.issubset(treasury.columns):
        treasury['term_spread'] = treasury['treasury_bond_rate'] - treasury['treasury_bill_rate']
    return _align_macro_to_trading_days(treasury, trading_index)


def load_macro_data(trading_index: pd.Index) -> pd.DataFrame:
    """
    Prefer FRED when a key is available; otherwise fall back to no-key
    public sources from BLS and Treasury.
    """
    fred_data = load_fred_macro_data(trading_index)
    if not fred_data.empty:
        return fred_data

    print("  Using no-key macro fallback: BLS + Treasury")
    bls_data = load_bls_macro_data(trading_index)
    treasury_data = load_treasury_macro_data(trading_index)
    if bls_data.empty and treasury_data.empty:
        return pd.DataFrame(index=trading_index)
    return pd.concat([bls_data, treasury_data], axis=1)


def load_fred_macro_data(trading_index: pd.Index) -> pd.DataFrame:
    """Fetch a small macro feature set from FRED and align it to trading days."""
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("  FRED_API_KEY not set; skipping FRED macro features")
        return pd.DataFrame(index=trading_index)

    start_date = (pd.Timestamp(trading_index.min()) - pd.Timedelta(days=400)).date().isoformat()
    series_data = {}
    for label, series_id in FRED_SERIES.items():
        try:
            payload = _fetch_json(
                'https://api.stlouisfed.org/fred/series/observations',
                params={
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'observation_start': start_date,
                },
            )
            obs = payload.get('observations', [])
            if not obs:
                continue
            series = pd.Series(
                {
                    pd.to_datetime(item['date']): float(item['value'])
                    for item in obs if item.get('value') not in (None, '.')
                }
            ).sort_index()
            if not series.empty:
                series_data[label] = series
        except Exception as exc:
            print(f"  FRED load failed for {series_id}: {exc}")

    if not series_data:
        return pd.DataFrame(index=trading_index)

    macro = pd.DataFrame(series_data).sort_index()
    macro = macro.reindex(macro.index.union(trading_index)).sort_index().ffill()
    macro = macro.reindex(trading_index).ffill()
    if {'rate_10y', 'rate_2y'}.issubset(macro.columns):
        macro['term_spread'] = macro['rate_10y'] - macro['rate_2y']
    return macro


def _extract_latest_sec_value(
    company_facts: dict[str, object],
    concepts: list[tuple[str, str]],
) -> float | None:
    facts = company_facts.get('facts', {}).get('us-gaap', {})
    for concept in concepts:
        entry = facts.get(concept)
        if not entry:
            continue
        values = []
        for units in entry.get('units', {}).values():
            values.extend(units)
        cleaned = []
        for item in values:
            val = item.get('val')
            form = item.get('form', '')
            if isinstance(val, (int, float)) and form in {'10-K', '10-Q', '20-F', '40-F'}:
                end_date = pd.to_datetime(item.get('end', item.get('fy', '1900-01-01')))
                cleaned.append((end_date, float(val)))
        if cleaned:
            cleaned.sort(key=lambda x: x[0])
            return cleaned[-1][1]
    return np.nan


def load_sec_quality_scores(tickers: list[str]) -> pd.Series:
    """Fetch a simple cross-sectional quality score from SEC company facts."""
    if SEC_USER_AGENT == 'sequential-decision-making/1.0 research@local':
        print("  SEC_USER_AGENT not set; skipping SEC quality features")
        return pd.Series(dtype=float)

    headers = {'User-Agent': SEC_USER_AGENT}
    try:
        mapping_payload = _fetch_json('https://www.sec.gov/files/company_tickers.json', headers=headers)
    except Exception as exc:
        print(f"  SEC ticker map load failed: {exc}. Set SEC_USER_AGENT='Name email@domain.com' to enable SEC features.")
        return pd.Series(dtype=float)

    ticker_to_cik = {
        item['ticker'].upper(): int(item['cik_str'])
        for item in mapping_payload.values()
    }

    concept_sets = {
        'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'],
        'net_income': ['NetIncomeLoss'],
        'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
        'assets': ['Assets'],
        'operating_income': ['OperatingIncomeLoss'],
    }

    raw_scores = {}
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if cik is None:
            continue
        try:
            company_facts = _fetch_json(
                f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',
                headers=headers,
            )
        except Exception:
            continue

        revenue = _extract_latest_sec_value(company_facts, concept_sets['revenue'])
        net_income = _extract_latest_sec_value(company_facts, concept_sets['net_income'])
        equity = _extract_latest_sec_value(company_facts, concept_sets['equity'])
        assets = _extract_latest_sec_value(company_facts, concept_sets['assets'])
        operating_income = _extract_latest_sec_value(company_facts, concept_sets['operating_income'])

        score_parts = []
        if np.isfinite(revenue) and abs(revenue) > 1e-8 and np.isfinite(net_income):
            score_parts.append(net_income / revenue)
        if np.isfinite(equity) and abs(equity) > 1e-8 and np.isfinite(net_income):
            score_parts.append(net_income / equity)
        if np.isfinite(assets) and abs(assets) > 1e-8 and np.isfinite(operating_income):
            score_parts.append(operating_income / assets)

        if score_parts:
            raw_scores[ticker] = float(np.mean(score_parts))

    sec_scores = pd.Series(raw_scores, dtype=float)
    if sec_scores.empty:
        return sec_scores
    return (sec_scores - sec_scores.mean()) / (sec_scores.std() + 1e-8)


def compute_macro_regime_signal(macro_window: pd.DataFrame) -> float:
    """Convert FRED macro data into a bull/bear score between 0 and 1."""
    if macro_window.empty:
        return 0.5

    weighted_components: list[tuple[float, float]] = []

    def _add_component(column: str, higher_is_better: bool, weight: float) -> None:
        if column not in macro_window.columns or macro_window[column].notna().sum() <= 20:
            return
        hist = macro_window[column].dropna()
        if hist.empty:
            return
        current = float(hist.iloc[-1])
        pct = float(sp_stats.percentileofscore(hist, current) / 100.0)
        score = pct if higher_is_better else (1.0 - pct)
        weighted_components.append((score, weight))

    # Core rates/labor structure
    _add_component('term_spread', higher_is_better=True, weight=0.24)
    _add_component('unrate', higher_is_better=False, weight=0.20)
    _add_component('fed_funds', higher_is_better=False, weight=0.14)

    # Credit/spread/volatility structure (added with extended FRED feature set)
    _add_component('vix', higher_is_better=False, weight=0.20)
    _add_component('hy_oas', higher_is_better=False, weight=0.16)
    _add_component('dxy', higher_is_better=False, weight=0.06)

    if not weighted_components:
        return 0.5
    scores = np.array([item[0] for item in weighted_components], dtype=float)
    weights = np.array([item[1] for item in weighted_components], dtype=float)
    regime_score = float(np.average(scores, weights=weights))
    return float(np.clip(regime_score, 0.05, 0.95))
