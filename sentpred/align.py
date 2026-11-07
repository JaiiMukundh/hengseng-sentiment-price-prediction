"""Align UTC social/news timestamps to exchange trading sessions and aggregate them per ticker-day.

A post belongs to the session whose window is (previous close, this close]. Posts published after the
close (or on weekends/holidays) roll to the next trading session. Trading sessions come from the price
bars, plus one extra business day so after-close chatter is visible to the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sentiment import score_text


def session_calendar(price_dates: pd.Series) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(price_dates).unique()))
    nxt = pd.bdate_range(dates.max() + pd.Timedelta(days=1), periods=1)
    return dates.append(nxt)


def map_to_session(published_utc: pd.Series, tz: str, close: str, sessions: pd.DatetimeIndex) -> pd.Series:
    local = published_utc.dt.tz_convert(tz).dt.tz_localize(None)
    close_time = pd.Timestamp(close).time()
    day = local.dt.normalize()
    after_close = local.dt.time > close_time
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")

    sess = sessions.values
    idx = np.searchsorted(sess, day.values, side="left")
    valid = idx < len(sess)
    mapped = np.full(len(day), np.datetime64("NaT", "ns"))
    mapped[valid] = sess[idx[valid]]
    return pd.Series(mapped, index=published_utc.index)


def engagement_weight(posts: pd.DataFrame) -> pd.Series:
    """Weight a post by engagement and author reach. Missing metrics count as zero."""
    engagement = posts["likes"].fillna(0) + posts["reposts"].fillna(0) + posts["views"].fillna(0) / 100
    reach = np.log1p(posts["followers"].fillna(0))
    return (1 + np.log1p(engagement)) * (1 + reach / 10)


def aggregate(posts: pd.DataFrame, calendars: dict[str, tuple[str, str, pd.DatetimeIndex]], prefix: str) -> pd.DataFrame:
    """Aggregate posts to one row per (ticker, session date).

    calendars maps exchange -> (tz, close, sessions).
    Columns are prefixed (e.g. "social_", "news_") so the two feeds can be joined.
    """
    empty = pd.DataFrame(columns=["ticker", "date"])
    if posts.empty:
        return empty
    parts = []
    for exchange, (tz, close, sessions) in calendars.items():
        sub = posts[posts["exchange"] == exchange].copy()
        if sub.empty:
            continue
        sub["date"] = map_to_session(sub["published_utc"], tz, close, sessions)
        parts.append(sub.dropna(subset=["date"]))
    if not parts:
        return empty
    df = pd.concat(parts, ignore_index=True)

    scored = df["text"].map(score_text)
    df["sent"] = [s for s, _ in scored]
    df["hits"] = [h for _, h in scored]
    df["hit"] = (df["hits"] > 0).astype(float)
    df["w"] = engagement_weight(df)
    df["ws"] = df["w"] * df["sent"]
    df["pos"] = (df["sent"] > 0).astype(float)
    df["neg"] = (df["sent"] < 0).astype(float)
    df["eng"] = np.log1p(df["likes"].fillna(0) + df["reposts"].fillna(0) + df["views"].fillna(0))

    grouped = df.groupby(["ticker", "date"])
    out = pd.DataFrame({
        f"{prefix}n": grouped.size(),
        f"{prefix}sent_mean": grouped["sent"].mean(),
        f"{prefix}sent_wmean": grouped["ws"].sum() / grouped["w"].sum(),
        f"{prefix}pos_share": grouped["pos"].mean(),
        f"{prefix}neg_share": grouped["neg"].mean(),
        f"{prefix}eng_log": grouped["eng"].sum(),
        f"{prefix}hit_share": grouped["hit"].mean(),
    }).reset_index()

    hit = df[df["hit"] > 0]
    if not hit.empty:
        hg = hit.groupby(["ticker", "date"])
        out = out.merge(pd.DataFrame({
            f"{prefix}sent_mean_hit": hg["sent"].mean(),
            f"{prefix}sent_wmean_hit": hg["ws"].sum() / hg["w"].sum(),
        }).reset_index(), on=["ticker", "date"], how="left")
    for col in (f"{prefix}sent_mean_hit", f"{prefix}sent_wmean_hit"):
        if col not in out:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    return out
