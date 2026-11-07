"""Daily OHLCV bars from Yahoo Finance (via yfinance) for HKEX and KRX tickers."""

from __future__ import annotations

import logging

import pandas as pd

from .config import Asset

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["ticker", "exchange", "date", "open", "high", "low", "close", "volume"]


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
