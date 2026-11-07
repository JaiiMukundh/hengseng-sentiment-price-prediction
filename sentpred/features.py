"""Price technicals, the joined model dataset, and the forward target."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_5", "ret_10", "vol_10", "rsi_14", "dist_ma20", "vol_z"]
SENTIMENT_FEATURES = [
    "social_n_log", "social_sent_mean", "social_sent_wmean", "social_pos_share", "social_neg_share",
    "social_eng_log", "news_n_log", "news_sent_mean", "news_sent_wmean", "has_social", "has_news",
]
_AGG_COLUMNS = ["n", "sent_mean", "sent_wmean", "pos_share", "neg_share", "eng_log"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, g in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = g.copy()
        close = g["close"]
        g["ret_1"] = close.pct_change()
        g["ret_5"] = close.pct_change(5)
        g["ret_10"] = close.pct_change(10)
        g["vol_10"] = g["ret_1"].rolling(10).std()
        g["rsi_14"] = rsi(close)
        g["dist_ma20"] = close / close.rolling(20).mean() - 1
        logv = np.log1p(g["volume"])
        g["vol_z"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
        # Target: the return of the NEXT session, known only after this session's close.
        g["y_next_ret"] = close.shift(-1) / close - 1
        g["next_close"] = close.shift(-1)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def build_dataset(prices: pd.DataFrame, social: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    df = price_features(prices)
    df = df.merge(social, on=["ticker", "date"], how="left")
    df = df.merge(news, on=["ticker", "date"], how="left")

    for prefix in ("social", "news"):
        for suffix in _AGG_COLUMNS:
            col = f"{prefix}_{suffix}"
            if col not in df:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
        df[f"has_{prefix}"] = (df[f"{prefix}_n"] > 0).astype(float)
        df[f"{prefix}_n_log"] = np.log1p(df[f"{prefix}_n"])
    return df
