"""News-only edge test on 365 days of Exa news."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.edge_lab import NEWS, add_features, forward_predict, metrics, MODELS
from sentpred import align, features
from sentpred.config import load_config


def rank_test(df, signal, k=2, cost=25):
    rows, ics = [], []
    for _, g in df.groupby("date"):
        if len(g) < 2 * k:
            continue
        g = g.sort_values(signal)
        n = min(k, len(g) // 2)
        rows.append(g.tail(n)["y_next_ret"].mean() - g.head(n)["y_next_ret"].mean())
        ic = spearmanr(g[signal], g["y_next_ret"]).correlation
        if np.isfinite(ic):
            ics.append(ic)
    net = np.array(rows) - 2 * cost / 10_000
    if len(net) < 3:
        return None
    eq = pd.Series(1 + net).cumprod()
    rng = np.random.default_rng(1)
    boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(2000)]
    return {"days": len(net), "gross_pct": np.mean(rows) * 100, "net_pct": net.mean() * 100,
            "sharpe": net.mean() / net.std() * np.sqrt(252), "hit_pct": (net > 0).mean() * 100,
            "cum_pct": (eq.iloc[-1] - 1) * 100, "maxdd_pct": (eq / eq.cummax() - 1).min() * 100,
            "ic": np.mean(ics) if ics else np.nan,
            "ci_low": np.percentile(boot, 2.5) * 100, "ci_high": np.percentile(boot, 97.5) * 100}


def main():
    cfg = load_config()
    prices = pd.read_csv("data/processed/prices.csv", parse_dates=["date"])
    market = pd.read_csv("data/processed/market.csv", parse_dates=["date"])
    news = pd.read_csv("data/processed/news_365.csv")
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, format="mixed")
    calendars = {}
    for exchange in cfg.exchanges:
        ex = cfg.exchange(exchange)
        ex_prices = prices[prices["exchange"] == exchange]
        calendars[exchange] = (ex["tz"], ex["close"], align.session_calendar(ex_prices["date"]))
    agg = align.aggregate(news, calendars, "news_")
    df = features.build_dataset(prices, pd.DataFrame(columns=["ticker", "date"]), agg, market)
    df = df[df["has_news"] > 0]
    df = add_features(df)
    print("rows", len(df), "sessions", df.date.nunique(), "range", df.date.min(), "->", df.date.max())
    print("assets/session:", df.groupby("date").size().describe().round(1).to_dict())

    print("\nRaw signal rank tests (25bps):")
    for signal in ["news_sent_wmean", "news_n_log", "news_hit_share"]:
        for k in (1, 2, 3):
            print(signal, "k=", k, rank_test(df, signal, k))

    print("\nModels (forward chaining, 25bps):")
    for name, factory in MODELS.items():
        for target in ("y_next_ret", "y_rel"):
            p = forward_predict(df, NEWS, factory, target)
            m = metrics(p, 25)
            if m:
                print(name, target, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
