"""Price technicals, the joined model dataset, and the forward target."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_5", "ret_10", "vol_10", "rsi_14", "dist_ma20", "vol_z"]
MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5", "us_ret_1", "sox_ret_1", "vix_level", "vix_chg", "fx_ret_1"]
BASE_FEATURES = PRICE_FEATURES + MARKET_FEATURES
SENTIMENT_FEATURES = [
    "social_n_log", "social_sent_mean", "social_sent_wmean", "social_pos_share", "social_neg_share",
    "social_eng_log", "social_hit_share", "social_sent_mean_hit", "social_sent_wmean_hit",
    "news_n_log", "news_sent_mean", "news_sent_wmean", "news_hit_share", "news_sent_mean_hit",
    "news_sent_wmean_hit", "has_social", "has_news",
]
# Compact, strongly regularized feature set for the short-history sentiment panel.
SENTIMENT_MODEL_FEATURES = [
    "ret_1", "ret_5", "vol_10", "rsi_14",
    "social_sent_wmean", "social_pos_share", "social_neg_share",
    "news_sent_wmean", "social_n_log", "has_social",
]
_AGG_COLUMNS = ["n", "sent_mean", "sent_wmean", "pos_share", "neg_share", "eng_log",
                "hit_share", "sent_mean_hit", "sent_wmean_hit"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def market_features(prices: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    """Attach local index, US overnight, VIX and FX context without look-ahead."""
    out = prices[["ticker", "exchange", "date"]].copy()
    if market is None or market.empty:
        for col in MARKET_FEATURES:
            out[col] = 0.0
        return out

    m = market.sort_values(["series", "date"]).copy()
    m["ret_1"] = m.groupby("series")["close"].pct_change()
    m["ret_5"] = m.groupby("series")["close"].pct_change(5)

    local = pd.concat([
        m[m["series"] == "HKEX_INDEX"][["date", "ret_1", "ret_5"]].assign(exchange="HKEX"),
        m[m["series"] == "KRX_INDEX"][["date", "ret_1", "ret_5"]].assign(exchange="KRX"),
    ]).rename(columns={"ret_1": "mkt_ret_1", "ret_5": "mkt_ret_5"})
    out = out.merge(local, on=["exchange", "date"], how="left")

    us = (m[m["series"].isin(["US_NASDAQ", "US_SOX", "US_VIX"])]
          .pivot(index="date", columns="series", values="close").sort_index())
    if {"US_NASDAQ", "US_SOX", "US_VIX"} <= set(us.columns):
        us = pd.DataFrame({
            "date": us.index,
            "us_ret_1": us["US_NASDAQ"].pct_change().values,
            "sox_ret_1": us["US_SOX"].pct_change().values,
            "vix_level": us["US_VIX"].values,
            "vix_chg": us["US_VIX"].pct_change().values,
        })
        out = pd.merge_asof(out.sort_values("date"), us.sort_values("date"), on="date",
                            direction="backward", allow_exact_matches=False)

    for exchange, series in (("KRX", "USDKRW"), ("HKEX", "USDHKD")):
        s = m[m["series"] == series].set_index("date")["close"].pct_change().rename("fx_ret_1")
        mask = out["exchange"] == exchange
        out.loc[mask, "fx_ret_1"] = out.loc[mask, "date"].map(s)

    for col in MARKET_FEATURES:
        if col not in out:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    return out


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


def _append_pending_sessions(df: pd.DataFrame, social: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Carry the last price features into the next session so after-close/weekend posts reach inference."""
    if social.empty and news.empty:
        return df
    rows = []
    for ticker, group in df.sort_values("date").groupby("ticker", sort=False):
        last = group.iloc[-1]
        future_dates = []
        for feed in (social, news):
            if not feed.empty:
                future = feed[(feed["ticker"] == ticker) & (feed["date"] > last["date"])]
                if not future.empty:
                    future_dates.append(future["date"].max())
        if not future_dates:
            continue
        session = max(future_dates)
        row = last.copy()
        row["date"] = session
        row["y_next_ret"] = np.nan
        row["next_close"] = np.nan
        for prefix, feed in (("social", social), ("news", news)):
            if feed.empty:
                continue
            sub = feed[(feed["ticker"] == ticker) & (feed["date"] == session)]
            for col in feed.columns:
                if col in {"ticker", "date"}:
                    continue
                row[col] = sub[col].iloc[0] if not sub.empty else 0.0
            row[f"{prefix}_n"] = float(sub[f"{prefix}_n"].iloc[0]) if not sub.empty else 0.0
            row[f"has_{prefix}"] = 1.0 if row[f"{prefix}_n"] > 0 else 0.0
            row[f"{prefix}_n_log"] = np.log1p(row[f"{prefix}_n"])
        rows.append(row)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df


def build_dataset(prices: pd.DataFrame, social: pd.DataFrame, news: pd.DataFrame,
                  market: pd.DataFrame | None = None) -> pd.DataFrame:
    df = price_features(prices)
    df = df.merge(market_features(prices, market), on=["ticker", "exchange", "date"], how="left")
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
    return _append_pending_sessions(df, social, news)
