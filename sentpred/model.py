"""Walk-forward (expanding window, out-of-sample) evaluation and the next-session forecast.

Models compared against a no-change baseline:
  * naive          - predicts zero return (price n+1 = price n)
  * ridge_price    - technical features only
  * ridge_sent     - technical + social/news sentiment features
A one-session purge gap separates training labels from the test block to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PRICE_FEATURES, SENTIMENT_FEATURES

MODELS = {"ridge_price": PRICE_FEATURES, "ridge_sent": PRICE_FEATURES + SENTIMENT_FEATURES}


def _fit(features: list[str], train: pd.DataFrame, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train[features], train["y_next_ret"])
    return model


def walk_forward(df: pd.DataFrame, min_train_dates: int, test_block: int, alpha: float) -> pd.DataFrame:
    labeled = df.dropna(subset=["y_next_ret"] + PRICE_FEATURES).copy()
    dates = sorted(labeled["date"].unique())
    if len(dates) <= min_train_dates + 1:
        raise ValueError(f"need more than {min_train_dates + 1} labelled sessions, have {len(dates)}")

    blocks = []
    for start in range(min_train_dates, len(dates), test_block):
        test_dates = dates[start:start + test_block]
        train_cut = dates[start - 2]  # purge the session just before the block
        train = labeled[labeled["date"] <= train_cut]
        test = labeled[labeled["date"].isin(test_dates)].copy()
        test["pred_naive"] = 0.0
        for name, features in MODELS.items():
            test[f"pred_{name}"] = _fit(features, train, alpha).predict(test[features])
        blocks.append(test)
    return pd.concat(blocks, ignore_index=True)


def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {"all": preds, "social_window": preds[preds["has_social"] > 0]}
    for subset_name, sub in subsets.items():
        if sub.empty:
            continue
        for model in ["naive", *MODELS.keys()]:
            pred_price = sub["close"] * (1 + sub[f"pred_{model}"])
            err = pred_price - sub["next_close"]
            direction = np.sign(sub[f"pred_{model}"]) == np.sign(sub["y_next_ret"])
            nonzero = sub["y_next_ret"] != 0
            rows.append({
                "subset": subset_name,
                "model": model,
                "n": len(sub),
                "mae_price": float(err.abs().mean()),
                "rmse_price": float(np.sqrt((err ** 2).mean())),
                "mape_pct": float((err.abs() / sub["next_close"]).mean() * 100),
                "direction_acc_pct": float(direction[nonzero].mean() * 100),
            })
    out = pd.DataFrame(rows)
    naive_mae = out[(out.model == "naive")].set_index("subset")["mae_price"]
    out["skill_vs_naive_pct"] = out.apply(
        lambda r: (1 - r["mae_price"] / naive_mae[r["subset"]]) * 100, axis=1
    )
    return out


def live_forecast(df: pd.DataFrame, metrics: pd.DataFrame, alpha: float, threshold: float) -> pd.DataFrame:
    """Fit on all labelled sessions and predict the next session close for each ticker."""
    labeled = df.dropna(subset=["y_next_ret"] + PRICE_FEATURES)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()

    overall = metrics[metrics["subset"] == "all"].set_index("model")["mae_price"]
    best = min(MODELS, key=lambda m: overall.get(m, np.inf))
    latest["pred_ret"] = _fit(MODELS[best], labeled, alpha).predict(latest[MODELS[best]])
    latest["model_used"] = best
    latest["pred_price"] = latest["close"] * (1 + latest["pred_ret"])
    latest["signal"] = np.select(
        [latest["pred_ret"] > threshold, latest["pred_ret"] < -threshold], ["LONG", "SHORT"], "FLAT"
    )
    return latest[["ticker", "exchange", "date", "close", "pred_ret", "pred_price", "signal", "model_used"]]
