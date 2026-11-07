"""Price-only falsification: 5-year cross-sectional reversal/momentum with costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sentpred.config import load_config
from sentpred.prices import fetch_prices


def main():
    try:
        p = pd.read_csv("data/processed/prices_5y.csv", parse_dates=["date"])
    except FileNotFoundError:
        p = fetch_prices(load_config().assets, "5y")
    p = p.sort_values(["ticker", "date"])
    for n in (1, 5, 20, 60):
        p[f"ret_{n}"] = p.groupby("ticker")["close"].pct_change(n)
    p["y"] = p.groupby("ticker")["close"].shift(-1) / p["close"] - 1

    for signal in ("ret_1", "ret_5", "ret_20", "ret_60"):
        for sign in (1, -1):
            rows, ics = [], []
            for _, g in p.dropna(subset=[signal, "y"]).groupby("date"):
                if len(g) < 6:
                    continue
                g = g.sort_values(signal)
                gross = (g.tail(3)["y"].mean() - g.head(3)["y"].mean()) if sign == 1 else (
                    g.head(3)["y"].mean() - g.tail(3)["y"].mean())
                rows.append(gross)
                ic = spearmanr(g[signal], g["y"]).correlation
                if np.isfinite(ic):
                    ics.append(ic)
            net = np.array(rows) - 2 * 25 / 10_000
            rng = np.random.default_rng(1)
            boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(2000)]
            eq = pd.Series(1 + net).cumprod()
            print(f"{signal} sign={sign:+d}: days={len(net)} gross={np.mean(rows)*100:.3f}% "
                  f"net={net.mean()*100:.3f}% SR={net.mean()/net.std()*np.sqrt(252):.2f} "
                  f"IC={np.mean(ics):.3f} maxdd={(eq/eq.cummax()-1).min()*100:.1f}% "
                  f"CI[{np.percentile(boot, 2.5)*100:.3f},{np.percentile(boot, 97.5)*100:.3f}]")


if __name__ == "__main__":
    main()
