"""Transformer sentiment + classification/ensemble edge test on the 15-asset research panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.edge_lab import NEWS, SETS, SOCIAL, TECH, add_features, forward_predict, metrics
from sentpred import align, features
from sentpred.config import load_config


def build():
    cfg = load_config("config.research.yaml")
    prices = pd.read_csv("data/processed/prices_research.csv", parse_dates=["date"])
    market = pd.read_csv("data/processed/market.csv", parse_dates=["date"])
    news = pd.read_csv("data/processed/news_research.csv")
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, format="mixed")
    a = pd.read_csv("data/processed/social_posts_transformer.csv")
    b = pd.read_csv("data/processed/new_assets_posts_transformer.csv")
    posts = pd.concat([a, b], ignore_index=True)
    posts["published_utc"] = pd.to_datetime(posts["published_utc"], utc=True, format="mixed")
    posts = posts.drop_duplicates(subset=["source", "ticker", "text", "published_utc"])
    calendars = {}
    for exchange in cfg.exchanges:
        ex = cfg.exchange(exchange)
        ex_prices = prices[prices["exchange"] == exchange]
        calendars[exchange] = (ex["tz"], ex["close"], align.session_calendar(ex_prices["date"]))
    social = align.aggregate(posts, calendars, "social_", score_col="sent_score")
    news_agg = align.aggregate(news, calendars, "news_")
    df = features.build_dataset(prices, social, news_agg, market)
    df = df[df["has_social"] > 0]
    return add_features(df), posts


def rank_test(df, signal, k=2, cost=25):
    rows, ics = [], []
    for _, g in df.groupby("date"):
        if len(g) < 2 * k:
            continue
        g = g.sort_values(signal)
        rows.append(g.tail(k)["y_next_ret"].mean() - g.head(k)["y_next_ret"].mean())
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


def forward_classify(df, cols, factory, min_train=8):
    dates = sorted(df["date"].unique())
    out = []
    for i, date in enumerate(dates):
        if i < min_train:
            continue
        train = df[df["date"] <= dates[i - 2]]
        test = df[df["date"] == date].copy()
        if len(train) < 30 or len(test) < 3:
            continue
        model = factory()
        model.fit(train[cols].fillna(0), (train["y_next_ret"] > 0).astype(int))
        test["score"] = model.predict_proba(test[cols].fillna(0))[:, 1]
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cls_metrics(pred, cost=25):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score((pred["y_next_ret"] > 0).astype(int), pred["score"])
    acc = ((pred["score"] > 0.5) == (pred["y_next_ret"] > 0)).mean() * 100
    rows = []
    for _, g in pred.groupby("date"):
        if len(g) < 4:
            continue
        g = g.sort_values("score")
        rows.append(g.tail(2)["y_next_ret"].mean() - g.head(2)["y_next_ret"].mean())
    net = np.array(rows) - 2 * cost / 10_000
    return {"auc": auc, "acc_pct": acc, "days": len(net),
            "ls_net_pct": net.mean() * 100 if len(net) else np.nan,
            "ls_sharpe": net.mean() / net.std() * np.sqrt(252) if len(net) and net.std() > 0 else np.nan}


def main():
    df, posts = build()
    print("posts", len(posts), "panel rows", len(df), "sessions", df.date.nunique(),
          "assets", df.ticker.nunique(), "range", df.date.min().date(), "->", df.date.max().date())
    print("assets/session:", df.groupby("date").size().describe().round(1).to_dict())

    print("\nRaw transformer sentiment rank tests (25bps):")
    for sig in ["social_sent_wmean", "social_sent_wmean_rank", "social_pos_share", "social_n_log"]:
        for k in (2, 3):
            print(sig, "k=", k, {x: round(v, 3) for x, v in (rank_test(df, sig, k) or {}).items()})

    print("\nRegression models (25bps):")
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

    print("\nClassification models (25bps):")
    for name, factory in {
        "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "rf_cls": lambda: RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, random_state=1),
        "hgb_cls": lambda: HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=0.05,
                                                         l2_regularization=1.0, random_state=1),
    }.items():
        for set_name, cols in (("social", SOCIAL), ("all", SETS["all"])):
            p = forward_classify(df, cols, factory)
            print(name, set_name, {k: round(v, 3) for k, v in cls_metrics(p).items()})

    print("\nEnsemble: average Ridge regressor rank + classifier probability rank")
    # Build one regressor and one classifier prediction set, then rank-average per date.
    reg = forward_predict(df, SETS["all"], lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)), "y_rel")
    cls = forward_classify(df, SETS["all"], lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)))
    reg = reg[["date", "ticker", "y_next_ret", "pred"]].rename(columns={"pred": "reg"})
    cls = cls[["date", "ticker", "score"]].rename(columns={"score": "cls"})
    ens = reg.merge(cls, on=["date", "ticker"])
    ens["reg_rank"] = ens.groupby("date")["reg"].rank(pct=True)
    ens["cls_rank"] = ens.groupby("date")["cls"].rank(pct=True)
    ens["pred"] = 0.5 * ens["reg_rank"] + 0.5 * ens["cls_rank"]
    # metrics expects return-scale pred; use rank-based LS directly
    rows = []
    for _, g in ens.groupby("date"):
        if len(g) < 4:
            continue
        g = g.sort_values("pred")
        rows.append(g.tail(2)["y_next_ret"].mean() - g.head(2)["y_next_ret"].mean())
    net = np.array(rows) - 2 * 25 / 10_000
    rng = np.random.default_rng(1)
    boot = [rng.choice(net, size=len(net), replace=True).mean() for _ in range(2000)]
    print({"days": len(net), "net_pct": round(net.mean() * 100, 3),
           "sharpe": round(net.mean() / net.std() * np.sqrt(252), 2),
           "hit_pct": round((net > 0).mean() * 100, 1),
           "ci": [round(np.percentile(boot, 2.5) * 100, 3), round(np.percentile(boot, 97.5) * 100, 3)]})


if __name__ == "__main__":
    main()
