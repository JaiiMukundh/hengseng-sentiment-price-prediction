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

from .features import PRICE_FEATURES, SENTIMENT_FEATURES, SENTIMENT_MODEL_FEATURES

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


def _sentiment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """All tickers on every session that has at least one social/news observation.

    Including tickers with no post that day (zero-filled sentiment) keeps the panel balanced
    instead of dropping rows and biasing the small sample.
    """
    data = df.dropna(subset=["y_next_ret"] + PRICE_FEATURES).copy()
    covered = data.loc[(data["has_social"] + data["has_news"]) > 0, "date"].unique()
    return data[data["date"].isin(covered)].reset_index(drop=True)


def leave_one_session_out(df: pd.DataFrame, alpha: float, shrink: float = 1.0) -> tuple[pd.DataFrame, int]:
    """Pooled panel validation when sentiment history is too short for an expanding window.

    Each fold holds out one trading session (all tickers) and trains on every other session.
    This gives every row an out-of-sample prediction and lets the sentiment features be fitted.
    `shrink` calibrates the forecast toward zero (a conservative prior for noisy small samples).
    """
    data = _sentiment_rows(df)
    dates = sorted(data["date"].unique())
    out = data.copy()
    out["pred_naive"] = 0.0
    for name, features in {"ridge_price": PRICE_FEATURES, "ridge_sent": SENTIMENT_MODEL_FEATURES}.items():
        out[f"pred_{name}"] = np.nan
        for session in dates:
            train = out[out["date"] != session]
            test = out["date"] == session
            out.loc[test, f"pred_{name}"] = _fit(features, train, alpha).predict(out.loc[test, features])
    out["pred_ridge_sent"] = out["pred_ridge_sent"] * shrink
    return out.dropna(subset=["pred_ridge_price", "pred_ridge_sent"]).reset_index(drop=True), len(dates)


def sentiment_coefficients(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Standardized Ridge coefficients for the sentiment model, fitted on all sentiment rows."""
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["feature", "coefficient"])
    model = _fit(SENTIMENT_MODEL_FEATURES, data, alpha)
    coefficients = model.named_steps["ridge"].coef_
    out = pd.DataFrame({"feature": SENTIMENT_MODEL_FEATURES, "coefficient": coefficients})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def sentiment_forecast(df: pd.DataFrame, alpha: float, shrink: float = 1.0) -> pd.DataFrame:
    """Fit the sentiment model on all sentiment rows and predict the next session for each ticker."""
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["ticker", "pred_sent_ret", "pred_sent_price"])
    fitted = _fit(SENTIMENT_MODEL_FEATURES, data, alpha)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest["pred_sent_ret"] = fitted.predict(latest[SENTIMENT_MODEL_FEATURES]) * shrink
    latest["pred_sent_price"] = latest["close"] * (1 + latest["pred_sent_ret"])
    return latest[["ticker", "pred_sent_ret", "pred_sent_price"]]


def nested_sentiment_predictions(df: pd.DataFrame, alphas: tuple[float, ...] = (10, 30, 50, 100, 300),
                                 shrink: float = 0.5) -> pd.DataFrame:
    """Sensitivity check: choose alpha by inner leave-one-session-out inside each outer fold."""
    data = _sentiment_rows(df)
    out = data.copy()
    out["pred_ridge_sent"] = np.nan
    for session in sorted(data["date"].unique()):
        train = data[data["date"] != session].reset_index(drop=True)
        test = data[data["date"] == session]
        best_alpha, best_mape = alphas[0], np.inf
        for alpha in alphas:
            inner = np.zeros(len(train))
            for inner_session in sorted(train["date"].unique()):
                inner_train = train[train["date"] != inner_session]
                inner_test = train["date"] == inner_session
                inner[inner_test.values] = _fit(SENTIMENT_MODEL_FEATURES, inner_train, alpha).predict(
                    train.loc[inner_test, SENTIMENT_MODEL_FEATURES])
            mape = ((train["close"] * (1 + shrink * inner) - train["next_close"]).abs()
                    / train["next_close"]).mean()
            if mape < best_mape:
                best_alpha, best_mape = alpha, mape
        out.loc[test.index, "pred_ridge_sent"] = (
            _fit(SENTIMENT_MODEL_FEATURES, train, best_alpha).predict(test[SENTIMENT_MODEL_FEATURES]) * shrink)
    return out


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
                # Naive never picks a direction, so its direction accuracy is undefined.
                "direction_acc_pct": np.nan if model == "naive" else float(direction[nonzero].mean() * 100),
            })
    out = pd.DataFrame(rows)
    naive_mape = out[(out.model == "naive")].set_index("subset")["mape_pct"]
    out["skill_vs_naive_pct"] = out.apply(
        lambda r: (1 - r["mape_pct"] / naive_mape[r["subset"]]) * 100, axis=1
    )
    return out


def live_forecast(df: pd.DataFrame, metrics: pd.DataFrame, alpha: float, threshold: float) -> pd.DataFrame:
    """Fit on all labelled sessions and predict the next session close for each ticker."""
    labeled = df.dropna(subset=["y_next_ret"] + PRICE_FEATURES)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()

    # MAPE, not absolute price error: the pooled assets trade in different currencies.
    overall = metrics[metrics["subset"] == "all"].set_index("model")["mape_pct"]
    best = min(MODELS, key=lambda m: overall.get(m, np.inf))
    latest["pred_ret"] = _fit(MODELS[best], labeled, alpha).predict(latest[MODELS[best]])
    latest["model_used"] = best
    latest["pred_price"] = latest["close"] * (1 + latest["pred_ret"])
    latest["signal"] = np.select(
        [latest["pred_ret"] > threshold, latest["pred_ret"] < -threshold], ["LONG", "SHORT"], "FLAT"
    )
    return latest[["ticker", "exchange", "date", "close", "pred_ret", "pred_price", "signal", "model_used"]]
