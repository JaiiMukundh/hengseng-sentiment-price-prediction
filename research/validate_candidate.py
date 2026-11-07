"""Falsification checks for the news ElasticNet candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.edge_lab import NEWS, forward_predict, load_data, metrics
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def factory():
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))


def daily_returns(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in pred.groupby("date"):
        if len(g) < 4:
            continue
        g = g.sort_values("pred")
        n = min(2, len(g) // 2)
        gross = g.tail(n)["y_next_ret"].mean() - g.head(n)["y_next_ret"].mean()
        rows.append({"date": g["date"].iloc[0], "gross": gross, "assets": len(g),
                     "long": ",".join(g.tail(n)["ticker"]), "short": ",".join(g.head(n)["ticker"])})
    return pd.DataFrame(rows)


def stats(d: pd.DataFrame, cost_bps: float) -> dict:
    net = d["gross"] - 2 * cost_bps / 10_000
    eq = (1 + net).cumprod()
    return {
        "days": len(d), "mean_gross_pct": d["gross"].mean() * 100,
        "mean_net_pct": net.mean() * 100,
        "sharpe": net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan,
        "hit_pct": (net > 0).mean() * 100,
        "cum_pct": (eq.iloc[-1] - 1) * 100 if len(eq) else np.nan,
        "maxdd_pct": (eq / eq.cummax() - 1).min() * 100 if len(eq) else np.nan,
    }


def main():
    df = load_data()["news120"]
    print("news rows:", len(df), "sessions:", df.date.nunique(), "range:", df.date.min(), "->", df.date.max())
    print("assets/session:\n", df.groupby("date").size().describe().round(1).to_string())

    pred = forward_predict(df, NEWS, factory, "y_next_ret")
    d = daily_returns(pred)
    print("\nDaily long/short:")
    print(d.round(4).to_string(index=False))
    for cost in (0, 10, 25, 50):
        print(f"cost={cost:>2}bps:", {k: round(v, 3) for k, v in stats(d, cost).items()})

    print("\nBootstrap of daily net (25bps):")
    net = (d["gross"] - 2 * 25 / 10_000).to_numpy()
    rng = np.random.default_rng(1)
    boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(2000)]
    print("mean %.4f  95%% CI [%.4f, %.4f]" % (net.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

    print("\nLast-10-session holdout:")
    last_dates = sorted(pred.date.unique())[-10:]
    print(stats(d[d.date.isin(last_dates)], 25))

    print("\nPer exchange (25bps):")
    for ex, g in pred.groupby("exchange"):
        dd = daily_returns(g)
        if len(dd):
            print(ex, {k: round(v, 3) for k, v in stats(dd, 25).items()})

    print("\nFeature ablation (all costs 25bps):")
    subsets = {
        "sentiment_only": ["news_sent_wmean", "news_sent_wmean_rank"],
        "volume_only": ["news_n_log"],
        "sentiment+volume": ["news_sent_wmean", "news_sent_wmean_rank", "news_n_log"],
        "all_news": NEWS,
    }
    for name, cols in subsets.items():
        p = forward_predict(df, cols, factory, "y_next_ret")
        dd = daily_returns(p)
        m = metrics(p, 25)
        print(name, {"days": len(dd), "rank_ic": round(m.get("rank_ic", np.nan), 3),
                     "net_pct": round(stats(dd, 25)["mean_net_pct"], 3),
                     "sharpe": round(stats(dd, 25)["sharpe"], 2)})

    print("\nPlacebo: shuffle news sentiment within each date")
    placebo = df.copy()
    for col in ("news_sent_wmean", "news_sent_wmean_rank", "news_chg", "news_social_div"):
        placebo[col] = placebo.groupby("date")[col].transform(lambda s: s.sample(frac=1, random_state=7).values)
    p = forward_predict(placebo, NEWS, factory, "y_next_ret")
    dd = daily_returns(p)
    print({"days": len(dd), "net_pct": round(stats(dd, 25)["mean_net_pct"], 3),
           "sharpe": round(stats(dd, 25)["sharpe"], 2)})

    print("\nCoefficients on last training window:")
    dates = sorted(df.date.unique())
    train = df[df.date <= dates[-3]]
    model = factory().fit(train[NEWS].fillna(0), train["y_next_ret"])
    coef = pd.DataFrame({"feature": NEWS, "coef": model.named_steps["elasticnet"].coef_})
    print(coef.reindex(coef["coef"].abs().sort_values(ascending=False).index).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
