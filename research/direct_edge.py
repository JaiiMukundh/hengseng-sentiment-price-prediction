"""90-day KRX direct-backfill edge test: social_pos_share, holding periods, costs, bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.edge_lab import SETS, SOCIAL, add_features, forward_predict, metrics
from research.transformer_edge import cls_metrics, forward_classify
from sentpred import align, features
from sentpred.config import load_config


def build():
    cfg = load_config("config.research.yaml")
    prices = pd.read_csv("data/processed/prices_research.csv", parse_dates=["date"])
    prices = prices[prices.exchange == "KRX"]
    market = pd.read_csv("data/processed/market.csv", parse_dates=["date"])
    news = pd.read_csv("data/processed/news_research.csv")
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, format="mixed")
    news = news[news.exchange == "KRX"]
    posts = pd.read_csv("data/processed/naver_direct_all_scored.csv")
    posts["published_utc"] = pd.to_datetime(posts["published_utc"], utc=True, format="mixed")
    ex = cfg.exchange("KRX")
    cal = {"KRX": (ex["tz"], ex["close"], align.session_calendar(prices["date"]))}
    social = align.aggregate(posts, cal, "social_", score_col="sent_score")
    news_agg = align.aggregate(news, cal, "news_")
    df = features.build_dataset(prices, social, news_agg, market)
    df = df[df["has_social"] > 0]
    return add_features(df)


def simulate(df: pd.DataFrame, H: int, k: int = 2, cost: float = 25):
    dates = sorted(df["date"].unique())
    ret = df.set_index(["date", "ticker"])["y_next_ret"]
    daily, held_l, held_s = [], [], []
    for i, date in enumerate(dates):
        if i % H == 0:
            day = df[df["date"] == date]
            if len(day) < 2 * k:
                held_l, held_s = [], []
                continue
            day = day.sort_values("social_pos_share")
            held_l, held_s = list(day.tail(k)["ticker"]), list(day.head(k)["ticker"])
            cost_today = 2 * cost / 10_000
        else:
            cost_today = 0.0
        lr = [ret.loc[(date, t)] for t in held_l if (date, t) in ret.index]
        sr = [ret.loc[(date, t)] for t in held_s if (date, t) in ret.index]
        if not lr or not sr:
            continue
        daily.append(np.mean(lr) - np.mean(sr) - cost_today)
    net = np.array(daily)
    if len(net) < 8:
        return None
    eq = pd.Series(1 + net).cumprod()
    rng = np.random.default_rng(1)
    boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(3000)]
    return {"H": H, "k": k, "days": len(net), "mean_net_pct": net.mean() * 100,
            "sharpe": net.mean() / net.std() * np.sqrt(252),
            "hit_pct": (net > 0).mean() * 100, "cum_pct": (eq.iloc[-1] - 1) * 100,
            "maxdd_pct": (eq / eq.cummax() - 1).min() * 100,
            "ci_low": np.percentile(boot, 2.5) * 100, "ci_high": np.percentile(boot, 97.5) * 100}


def simulate_side(df: pd.DataFrame, H: int, side: str = "long", k: int = 2, cost: float = 25):
    """Long-only or short-only vs the equal-weight universe (pre-registered confirmatory rule)."""
    dates = sorted(df["date"].unique())
    ret = df.set_index(["date", "ticker"])["y_next_ret"]
    daily, held = [], []
    for i, date in enumerate(dates):
        if i % H == 0:
            day = df[df["date"] == date]
            if len(day) < 2 * k:
                held = []
                continue
            day = day.sort_values("social_pos_share")
            held = list(day.tail(k)["ticker"]) if side == "long" else list(day.head(k)["ticker"])
            cost_today = cost / 10_000
        else:
            cost_today = 0.0
        if not held:
            continue
        vals = [ret.loc[(date, t)] for t in held if (date, t) in ret.index]
        if not vals:
            continue
        universe = df[df["date"] == date]["y_next_ret"].mean()
        gross = (np.mean(vals) - universe) if side == "long" else (universe - np.mean(vals))
        daily.append(gross - cost_today)
    net = np.array(daily)
    if len(net) < 8:
        return None
    eq = pd.Series(1 + net).cumprod()
    rng = np.random.default_rng(1)
    boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(3000)]
    return {"side": side, "H": H, "days": len(net), "mean_net_pct": net.mean() * 100,
            "sharpe": net.mean() / net.std() * np.sqrt(252), "hit_pct": (net > 0).mean() * 100,
            "cum_pct": (eq.iloc[-1] - 1) * 100, "maxdd_pct": (eq / eq.cummax() - 1).min() * 100,
            "ci_low": np.percentile(boot, 2.5) * 100, "ci_high": np.percentile(boot, 97.5) * 100}


def main():
    df = build()
    print("rows", len(df), "sessions", df["date"].nunique(), "assets", df.ticker.nunique(),
          "range", df["date"].min().date(), "->", df["date"].max().date())
    rows = []
    for H in (1, 2, 3, 4, 5, 10):
        r = simulate(df, H, 2, 25)
        if r:
            rows.append(r)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    pd.DataFrame(rows).to_csv("reports/research/direct_edge_results.csv", index=False)

    print("\nPre-registered confirmatory rule: long-only, H=4, k=2")
    for side in ("long", "short"):
        r = simulate_side(df, 4, side, 2, 25)
        if r:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})

    print("\nRegression models:")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for name, factory in {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=120, max_depth=3, learning_rate=0.05,
                                                     l2_regularization=1.0, random_state=1),
    }.items():
        for set_name, cols in (("social", SOCIAL), ("all", SETS["all"])):
            for target in ("y_next_ret", "y_rel"):
                p = forward_predict(df, cols, factory, target)
                m = metrics(p, 25)
                print(name, set_name, target, {k: round(v, 3) if isinstance(v, float) else v
                                               for k, v in m.items() if k in ("rank_ic", "ls_net_pct", "ls_sharpe", "direction_acc_pct", "n_days")})

    print("\nClassification models:")
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    for name, factory in {
        "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "rf_cls": lambda: RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, random_state=1),
        "hgb_cls": lambda: HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=0.05,
                                                         l2_regularization=1.0, random_state=1),
    }.items():
        for set_name, cols in (("social", SOCIAL), ("all", SETS["all"])):
            p = forward_classify(df, cols, factory)
            print(name, set_name, {k: round(v, 3) for k, v in cls_metrics(p).items()})


if __name__ == "__main__":
    main()
