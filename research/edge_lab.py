"""Edge research lab.

Forward-chaining evaluation of absolute and cross-sectional next-day targets,
with rank IC, long/short baskets and transaction-cost sensitivity.

Run: .venv/bin/python research/edge_lab.py
Outputs: reports/research/edge_results.csv, reports/research/backtests.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentpred import align, features
from sentpred.config import ROOT, load_config
from sentpred.features import PRICE_FEATURES

REPORT = ROOT / "reports" / "research"
REPORT.mkdir(parents=True, exist_ok=True)

TECH = PRICE_FEATURES + ["mkt_ret_1", "mkt_ret_5", "us_ret_1", "sox_ret_1", "vix_level", "vix_chg", "fx_ret_1"]
SOCIAL = [
    "social_sent_wmean", "social_pos_share", "social_neg_share", "social_n_log", "social_eng_log",
    "social_hit_share", "sent_chg", "sent_ma3", "sent_z", "pos_neg", "mention_chg",
    "social_sent_wmean_rank", "social_n_log_rank", "sent_x_vol",
]
NEWS = ["news_sent_wmean", "news_n_log", "news_hit_share", "news_chg", "news_social_div", "news_sent_wmean_rank"]
SETS = {"tech": TECH, "social": SOCIAL, "news": NEWS, "all": TECH + SOCIAL + NEWS}
MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "enet": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=5000)),
    "rf": lambda: RandomForestRegressor(n_estimators=250, max_depth=3, min_samples_leaf=10, random_state=1, n_jobs=-1),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=120, max_depth=3, learning_rate=0.05,
                                                l2_regularization=1.0, random_state=1),
}


def load_data() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    prices = pd.read_csv("data/processed/prices.csv", parse_dates=["date"])
    market = pd.read_csv("data/processed/market.csv", parse_dates=["date"])
    posts = pd.read_csv("data/processed/social_posts.csv")
    posts["published_utc"] = pd.to_datetime(posts["published_utc"], utc=True, format="mixed")
    news = pd.read_csv("data/processed/news_long.csv")
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, format="mixed")

    calendars = {}
    for exchange in cfg.exchanges:
        ex = cfg.exchange(exchange)
        ex_prices = prices[prices["exchange"] == exchange]
        calendars[exchange] = (ex["tz"], ex["close"], align.session_calendar(ex_prices["date"]))

    social_agg = align.aggregate(posts, calendars, "social_")
    news_agg = align.aggregate(news, calendars, "news_")
    empty = pd.DataFrame(columns=["ticker", "date"])

    social_df = features.build_dataset(prices, social_agg, news_agg.copy(), market)
    news_df = features.build_dataset(prices, empty, news_agg, market)
    combined_df = features.build_dataset(prices, social_agg, news_agg, market)
    social_df = social_df[social_df["has_social"] > 0]
    news_df = news_df[news_df["has_news"] > 0]
    combined_df = combined_df[combined_df["has_news"] > 0]
    return {"social30": add_features(social_df), "news120": add_features(news_df),
            "combined": add_features(combined_df)}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["y_next_ret"]).sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)
    df["sent_chg"] = g["social_sent_wmean"].diff().fillna(0)
    df["sent_ma3"] = g["social_sent_wmean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    roll_mean = g["social_sent_wmean"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    roll_std = g["social_sent_wmean"].transform(lambda s: s.rolling(10, min_periods=3).std()).replace(0, np.nan)
    df["sent_z"] = ((df["social_sent_wmean"] - roll_mean) / roll_std).fillna(0)
    df["pos_neg"] = df["social_pos_share"] - df["social_neg_share"]
    df["mention_chg"] = g["social_n_log"].diff().fillna(0)
    df["news_chg"] = g["news_sent_wmean"].diff().fillna(0)
    df["news_social_div"] = df["news_sent_wmean"] - df["social_sent_wmean"]
    df["sent_x_vol"] = df["social_sent_wmean"] * df["vol_10"]
    for col in ("social_sent_wmean", "news_sent_wmean", "social_n_log", "ret_5", "vol_10"):
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)
    df["y_rel"] = df["y_next_ret"] - df.groupby("date")["y_next_ret"].transform("mean")
    df["y_rank"] = df.groupby("date")["y_next_ret"].rank(pct=True)
    return df


def forward_predict(df: pd.DataFrame, cols: list[str], factory, target: str, min_train: int = 8) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    out = []
    for i, date in enumerate(dates):
        if i < min_train:
            continue
        train_cut = dates[i - 2]  # purge one session so the last train label does not overlap the test
        train = df[df["date"] <= train_cut]
        test = df[df["date"] == date].copy()
        if len(train) < 30 or len(test) < 3:
            continue
        model = factory()
        model.fit(train[cols].fillna(0), train[target])
        test["pred"] = model.predict(test[cols].fillna(0))
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def metrics(pred: pd.DataFrame, cost_bps: float = 25) -> dict:
    if pred.empty:
        return {}
    err = (pred["close"] * (1 + pred["pred"]) - pred["next_close"]).abs() / pred["next_close"]
    direction = (np.sign(pred["pred"]) == np.sign(pred["y_next_ret"]))[pred["y_next_ret"] != 0].mean() * 100
    ics = []
    for _, g in pred.groupby("date"):
        if len(g) >= 4:
            ic = spearmanr(g["pred"], g["y_next_ret"]).correlation
            if np.isfinite(ic):
                ics.append(ic)
    daily = []
    for _, g in pred.groupby("date"):
        if len(g) < 4 or g["pred"].nunique() < 2:
            continue
        g = g.sort_values("pred")
        n = min(2, len(g) // 2)
        gross = g.tail(n)["y_next_ret"].mean() - g.head(n)["y_next_ret"].mean()
        net = gross - 2 * cost_bps / 10_000
        daily.append({"date": g["date"].iloc[0], "gross": gross, "net": net})
    d = pd.DataFrame(daily)
    if d.empty:
        return {"mape_pct": err.mean() * 100, "direction_acc_pct": direction, "rank_ic": np.nan,
                "pred_std": float(pred["pred"].std()), "pred_nunique": int(pred["pred"].nunique()), "n_days": 0}
    equity = (1 + d["net"]).cumprod()
    peak = equity.cummax()
    sharpe = d["net"].mean() / d["net"].std() * np.sqrt(252) if d["net"].std() > 0 else 0
    return {
        "mape_pct": err.mean() * 100,
        "direction_acc_pct": direction,
        "rank_ic": float(np.mean(ics)) if ics else np.nan,
        "rank_ic_t": float(np.mean(ics) / np.std(ics) * np.sqrt(len(ics))) if len(ics) > 1 and np.std(ics) > 0 else np.nan,
        "ls_gross_pct": float(d["gross"].mean() * 100),
        "ls_net_pct": float(d["net"].mean() * 100),
        "ls_sharpe": float(sharpe),
        "ls_hit_pct": float((d["net"] > 0).mean() * 100),
        "ls_maxdd_pct": float((equity / peak - 1).min() * 100),
        "pred_std": float(pred["pred"].std()),
        "pred_nunique": int(pred["pred"].nunique()),
        "n_days": int(len(d)),
    }


def run() -> pd.DataFrame:
    data = load_data()
    rows = []
    for dataset, df in data.items():
        for set_name, cols in SETS.items():
            if dataset == "news120" and set_name == "social":
                continue
            if dataset == "combined" and set_name == "social":
                continue
            for model_name, factory in MODELS.items():
                for target in ("y_next_ret", "y_rel"):
                    pred = forward_predict(df, cols, factory, target)
                    m = metrics(pred)
                    rows.append({"dataset": dataset, "features": set_name, "model": model_name,
                                 "target": target, "rows": len(pred), **m})
    out = pd.DataFrame(rows).sort_values(["rank_ic", "ls_sharpe"], ascending=False)
    out.to_csv(REPORT / "edge_results.csv", index=False)
    return out


if __name__ == "__main__":
    results = run()
    print(results.head(25).round(3).to_string(index=False))
    print("\nFull results:", REPORT / "edge_results.csv")
