"""Daily OHLCV bars from Yahoo Finance (via yfinance) for HKEX and KRX tickers."""

from __future__ import annotations

import logging

import pandas as pd

from .config import Asset

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["ticker", "exchange", "date", "open", "high", "low", "close", "volume"]
MARKET_COLUMNS = ["series", "date", "close"]

# Market context known before the Asian session opens (US) or at the Asian close (local index/FX).
MARKET_SERIES = {
    "^HSI": "HKEX_INDEX",
    "^KS11": "KRX_INDEX",
    "^IXIC": "US_NASDAQ",
    "^SOX": "US_SOX",
    "^VIX": "US_VIX",
    "KRW=X": "USDKRW",
    "HKD=X": "USDHKD",
}


def fetch_prices(assets: tuple[Asset, ...], period: str) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for asset in assets:
        bars = yf.Ticker(asset.ticker).history(period=period, auto_adjust=True)
        if bars.empty:
            log.warning("no price data for %s", asset.ticker)
            continue
        df = bars.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        # Keep the exchange-local calendar date so sessions line up with social timestamps.
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df.insert(0, "exchange", asset.exchange)
        df.insert(0, "ticker", asset.ticker)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]


def fetch_market(period: str) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for ticker, series in MARKET_SERIES.items():
        bars = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if bars.empty:
            log.warning("no market data for %s", ticker)
            continue
        df = bars.reset_index()[["Date", "Close"]]
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df.insert(0, "series", series)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return pd.concat(frames, ignore_index=True)[MARKET_COLUMNS]
