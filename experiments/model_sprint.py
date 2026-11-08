"""Parallel model sprint: can any model beat Naive on the 30-day Apify+Exa panel?

Each --family is an independent research worker. Results are written to experiments/results/<family>.csv.
Primary validation: leave-one-session-out (20 sessions, 190 rows). Top candidates must be rechecked forward.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sentpred.features import MARKET_FEATURES, PRICE_FEATURES, SENTIMENT_FEATURES, SENTIMENT_MODEL_FEATURES

RESULTS = Path("experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

SENT_SCORE_COLS = {
    "social_sent_wmean", "social_sent_mean", "social_pos_share", "social_neg_share",
    "news_sent_wmean", "news_sent_mean", "social_sent_wmean_hit", "news_sent_wmean_hit",
}


class SentimentScaler(BaseEstimator, TransformerMixin):
    """Standardize, then multiply sentiment-score columns by `scale` (post-scaling importance)."""

    def __init__(self, scale: float = 1.0, columns: tuple[str, ...] = ()):
        self.scale = scale
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X)
        idx = [i for i, c in enumerate(self.columns) if c in SENT_SCORE_COLS]
        if idx:
            Z[:, idx] *= self.scale
        return Z


def load() -> pd.DataFrame:
    df = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"]).dropna(subset=["y_next_ret", "next_close"])
    covered = df.loc[(df["has_social"] + df["has_news"]) > 0, "date"].unique()
    df = df[df["date"].isin(covered)].sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)
    df["sent_chg"] = g["social_sent_wmean"].diff().fillna(0)
    df["sent_ma3"] = g["social_sent_wmean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    roll = g["social_sent_wmean"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    std = g["social_sent_wmean"].transform(lambda s: s.rolling(10, min_periods=3).std()).replace(0, np.nan)
    df["sent_z"] = ((df["social_sent_wmean"] - roll) / std).fillna(0)
    df["mention_chg"] = g["social_n_log"].diff().fillna(0)
    df["news_chg"] = g["news_sent_wmean"].diff().fillna(0)
    df["news_social_div"] = df["news_sent_wmean"] - df["social_sent_wmean"]
    return df


def uniq(seq):
    return list(dict.fromkeys(seq))


def sets() -> dict[str, list[str]]:
    sent = ["social_sent_wmean", "social_pos_share", "social_neg_share", "news_sent_wmean", "social_n_log", "has_social"]
    return {
        "price": PRICE_FEATURES,
        "sent": SENTIMENT_MODEL_FEATURES,
        "sent_only": sent,
        "compact": SENTIMENT_MODEL_FEATURES,
        "both": uniq(PRICE_FEATURES + sent),
        "all": uniq(PRICE_FEATURES + SENTIMENT_FEATURES + MARKET_FEATURES),
        "engineered": uniq(PRICE_FEATURES + sent + ["sent_chg", "sent_ma3", "sent_z", "mention_chg", "news_chg", "news_social_div"]),
    }


SETS = sets()


def make_model(name: str, params: dict, columns: list[str]):
    scale = params.get("scale", 1.0)
    pre = SentimentScaler(scale, tuple(columns))
    if name == "ridge":
        return make_pipeline(pre, Ridge(alpha=params.get("alpha", 10)))
    if name == "enet":
        return make_pipeline(pre, ElasticNet(alpha=params.get("alpha", 0.001), l1_ratio=params.get("l1", 0.2), max_iter=5000))
    if name == "mlp":
        return make_pipeline(pre, MLPRegressor(
            hidden_layer_sizes=params["hidden"], alpha=params.get("alpha", 1.0),
            solver=params.get("solver", "adam"),
            learning_rate_init=params.get("lr", 0.01),
            early_stopping=params.get("early_stopping", False),
            validation_fraction=params.get("validation_fraction", 0.2),
            n_iter_no_change=params.get("n_iter_no_change", 20),
            max_iter=params.get("max_iter", 3000), random_state=params.get("random_state", 1), tol=1e-4))
    if name == "rf":
        return make_pipeline(pre, RandomForestRegressor(n_estimators=300, max_depth=params.get("depth", 3),
                                                       min_samples_leaf=params.get("leaf", 10), random_state=1, n_jobs=-1))
    if name == "extra":
        return make_pipeline(pre, ExtraTreesRegressor(n_estimators=300, max_depth=params.get("depth", 3),
                                                     min_samples_leaf=params.get("leaf", 10), random_state=1, n_jobs=-1))
    if name == "hgb":
        return make_pipeline(pre, HistGradientBoostingRegressor(max_iter=120, max_depth=params.get("depth", 3),
                                                               learning_rate=0.05, l2_regularization=1.0, random_state=1))
    if name == "gbr":
        return make_pipeline(pre, GradientBoostingRegressor(n_estimators=100, max_depth=params.get("depth", 2),
                                                           learning_rate=0.05, random_state=1))
    if name == "knn":
        return make_pipeline(pre, KNeighborsRegressor(n_neighbors=params.get("k", 5), weights="distance"))
    if name == "svr":
        return make_pipeline(pre, SVR(C=params.get("c", 1.0), epsilon=0.01, gamma="scale"))
    if name == "pls":
        return make_pipeline(pre, PLSRegression(n_components=params.get("components", 3)))
    raise ValueError(name)


def run_config(df: pd.DataFrame, columns: list[str], model_name: str, params: dict, lam: float) -> dict | None:
    dates = sorted(df["date"].unique())
    rows = []
    for date in dates:
        train = df[df["date"] != date]
        test = df[df["date"] == date].copy()
        if len(train) < 30 or len(test) < 3:
            continue
        model = make_model(model_name, params, columns)
        try:
            model.fit(train[columns].fillna(0), train["y_next_ret"])
            pred_ret = model.predict(test[columns].fillna(0))
        except Exception:
            return None
        test["pred_price"] = test["close"] * (1 + lam * pred_ret)
        test["naive_price"] = test["close"]
        rows.append(test)
    if not rows:
        return None
    pred = pd.concat(rows, ignore_index=True)
    err = (pred["pred_price"] - pred["next_close"]).abs()
    nerr = (pred["naive_price"] - pred["next_close"]).abs()
    mape = (err / pred["next_close"]).mean() * 100
    nmape = (nerr / pred["next_close"]).mean() * 100
    direction = (np.sign(pred["pred_price"] - pred["close"]) == np.sign(pred["next_close"] - pred["close"]))[
        pred["next_close"] != pred["close"]].mean() * 100
    pct = pd.DataFrame({
        "date": pred["date"],
        "delta": (err / pred["next_close"] - nerr / pred["next_close"]) * 100,
    })
    by_date = pct.groupby("date")["delta"].mean()
    rng = np.random.default_rng(1)
    boot = [rng.choice(by_date.values, size=len(by_date), replace=True).mean() for _ in range(3000)]
    return {
        "model": model_name, "features": params.get("features", ""), "scale": params.get("scale", 1.0),
        "lam": lam, "rows": len(pred), "days": len(by_date), "mape": mape, "naive_mape": nmape,
        "delta_pct": by_date.mean(), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
        "direction": direction, "skill_pct": (1 - mape / nmape) * 100,
        "beats_naive": bool(mape < nmape and np.percentile(boot, 97.5) < 0),
        **{f"p_{k}": v for k, v in params.items() if k not in {"features", "scale"}},
    }


def configs(family: str):
    out = []
    if family == "linear":
        for feat in ("compact", "both", "sent_only"):
            for scale in (0.5, 1.0, 2.0, 4.0):
                for model, p in (("ridge", {"alpha": 10}), ("ridge", {"alpha": 50}), ("ridge", {"alpha": 100}),
                                 ("enet", {"alpha": 0.001}), ("enet", {"alpha": 0.01})):
                    for lam in (0.25, 0.5, 1.0):
                        out.append((feat, model, {**p, "scale": scale, "features": feat}, lam))
    elif family == "mlp_small":
        for feat in ("compact", "both"):
            for scale in (0.5, 1.0, 2.0, 4.0):
                for alpha in (0.1, 1.0):
                    for lam in (0.5, 1.0):
                        out.append((feat, "mlp", {"hidden": (8,), "alpha": alpha, "scale": scale, "features": feat}, lam))
    elif family == "mlp_deep":
        for feat in ("compact", "both"):
            for hidden in ((16, 8), (32, 16)):
                for scale in (0.5, 1.0, 2.0):
                    for alpha in (0.1, 1.0):
                        out.append((feat, "mlp", {"hidden": hidden, "alpha": alpha, "scale": scale, "features": feat}, 1.0))
    elif family == "trees":
        for feat in ("compact", "both", "all"):
            for model, p in (("rf", {"depth": 2}), ("rf", {"depth": 3}), ("extra", {"depth": 2}),
                             ("hgb", {"depth": 2}), ("hgb", {"depth": 3}), ("gbr", {"depth": 1})):
                for lam in (0.5, 1.0):
                    out.append((feat, model, {**p, "scale": 1.0, "features": feat}, lam))
    elif family == "neighbors":
        for feat in ("compact", "both"):
            for model, p in (("knn", {"k": 3}), ("knn", {"k": 5}), ("knn", {"k": 10}),
                             ("svr", {"c": 0.1}), ("svr", {"c": 1.0}), ("pls", {"components": 2}),
                             ("pls", {"components": 4})):
                for scale in (0.5, 1.0, 2.0):
                    out.append((feat, model, {**p, "scale": scale, "features": feat}, 0.5))
    elif family == "feature_sets":
        for feat in ("price", "sent", "sent_only", "compact", "both", "all", "engineered"):
            for model, p in (("ridge", {"alpha": 50}), ("mlp", {"hidden": (16, 8), "alpha": 1.0})):
                for scale in (1.0, 2.0):
                    out.append((feat, model, {**p, "scale": scale, "features": feat}, 0.5))
    elif family == "engineered":
        for feat in ("engineered", "both"):
            for model, p in (("ridge", {"alpha": 10}), ("ridge", {"alpha": 50}), ("mlp", {"hidden": (8,), "alpha": 1.0}),
                             ("rf", {"depth": 2}), ("hgb", {"depth": 2})):
                for scale in (0.5, 1.0, 2.0, 4.0):
                    out.append((feat, model, {**p, "scale": scale, "features": feat}, 0.5))
    elif family == "blend":
        for feat in ("both", "compact"):
            for model, p in (("ridge", {"alpha": 50}), ("mlp", {"hidden": (8,), "alpha": 1.0}),
                             ("rf", {"depth": 2}), ("hgb", {"depth": 2})):
                for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
                    out.append((feat, model, {**p, "scale": 1.0, "features": feat}, lam))
    elif family == "sent_scale":
        for feat in ("compact", "both"):
            for scale in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
                for model, p in (("ridge", {"alpha": 50}), ("mlp", {"hidden": (8,), "alpha": 1.0})):
                    out.append((feat, model, {**p, "scale": scale, "features": feat}, 0.5))
    elif family == "targets":
        # target transformation is not supported by run_config yet; use scale/feature variants as controls
        for feat in ("both", "compact"):
            for model, p in (("ridge", {"alpha": 10}), ("ridge", {"alpha": 100}), ("mlp", {"hidden": (16, 8), "alpha": 1.0})):
                for lam in (0.1, 0.25, 0.5, 0.75):
                    out.append((feat, model, {**p, "scale": 1.0, "features": feat}, lam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    df = load()
    rows = []
    for feat, model, params, lam in configs(args.family):
        r = run_config(df, SETS[feat], model, params, lam)
        if r:
            r["family"] = args.family
            rows.append(r)
    out = pd.DataFrame(rows).sort_values(["mape", "delta_pct"])
    path = RESULTS / f"{args.family}.csv"
    out.to_csv(path, index=False)
    print(f"{args.family}: {len(out)} configs -> {path}")
    print(out.head(8)[["model", "features", "scale", "lam", "mape", "naive_mape", "direction", "beats_naive"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
