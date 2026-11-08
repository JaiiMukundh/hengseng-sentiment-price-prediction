"""Walk-forward (expanding window, out-of-sample) evaluation and the next-session forecast.

Models compared against a no-change baseline:
  * naive          - predicts zero return (price n+1 = price n)
  * ridge_price    - technical features only
  * ridge_sent     - technical + social/news sentiment features
A one-session purge gap separates training labels from the test block to avoid leakage.

The short-history deliverable uses a 6-component PLS model on price technicals plus sentiment
levels and dynamics. A small feedforward network, a deep MLP, Random Forest and a Ridge+MLP
ensemble were tested in ``experiments/``; PLS was the most stable across leave-one-session-out
and forward validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    ENGINEERED_MODEL_FEATURES,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    SENTIMENT_SCORE_FEATURES,
)

MODELS = {"ridge_price": PRICE_FEATURES, "ridge_sent": PRICE_FEATURES + SENTIMENT_FEATURES}


class SentimentScaler(BaseEstimator, TransformerMixin):
    """Standardize features, then multiply sentiment-score columns by ``scale``.

    A scale below 1 de-emphasizes sentiment; a scale above 1 raises its importance. The
    multiplier is applied after standardization so it is interpretable across features.
    """

    def __init__(self, scale: float = 1.0, columns: tuple[str, ...] = ()):
        self.scale = scale
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        z = self.scaler_.transform(X)
        idx = [i for i, column in enumerate(self.columns) if column in SENTIMENT_SCORE_FEATURES]
        if idx:
            z[:, idx] *= self.scale
        return z


def _fit(features: list[str], train: pd.DataFrame, alpha: float, model_name: str = "ridge",
         scale: float = 1.0, components: int = 6):
    if model_name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=300, max_depth=2, min_samples_leaf=5, random_state=1, n_jobs=-1)
    elif model_name == "pls":
        estimator = PLSRegression(n_components=min(components, len(features)), scale=True)
    else:
        estimator = Ridge(alpha=alpha)
    model = make_pipeline(SentimentScaler(scale, tuple(features)), estimator)
    model.fit(train[features], train["y_next_ret"])
    return model


def _predict(model, features: pd.DataFrame, clip: float | None = None) -> np.ndarray:
    pred = np.asarray(model.predict(features)).ravel()
    if clip is not None:
        pred = np.clip(pred, -clip, clip)
    return pred


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
    data = df.dropna(subset=["y_next_ret"] + ENGINEERED_MODEL_FEATURES).copy()
    covered = data.loc[(data["has_social"] + data["has_news"]) > 0, "date"].unique()
    return data[data["date"].isin(covered)].reset_index(drop=True)


def leave_one_session_out(df: pd.DataFrame, alpha: float, shrink: float = 1.0,
                          model_name: str = "rf", scale: float = 1.0,
                          clip: float | None = None, components: int = 6) -> tuple[pd.DataFrame, int]:
    """Pooled panel validation when sentiment history is too short for an expanding window.

    Each fold holds out one trading session (all tickers) and trains on every other session.
    This gives every row an out-of-sample prediction and lets the sentiment features be fitted.
    ``shrink`` calibrates the forecast toward zero (a conservative prior for noisy small samples).
    ``scale`` changes the post-standardization importance of the sentiment score columns.
    """
    data = _sentiment_rows(df)
    dates = sorted(data["date"].unique())
    out = data.copy()
    out["pred_naive"] = 0.0
    for name, features in {"ridge_price": PRICE_FEATURES,
                           "ridge_sent": ENGINEERED_MODEL_FEATURES}.items():
        out[f"pred_{name}"] = np.nan
        for session in dates:
            train = out[out["date"] != session]
            test = out["date"] == session
            # The control uses the same model class as the deliverable, so the ablation
            # isolates the sentiment features instead of conflating them with model choice.
            fitted = _fit(features, train, alpha, model_name,
                          scale if name == "ridge_sent" else 1.0, components)
            out.loc[test, f"pred_{name}"] = _predict(
                fitted, out.loc[test, features], clip)
    out["pred_ridge_price"] = out["pred_ridge_price"] * shrink
    out["pred_ridge_sent"] = out["pred_ridge_sent"] * shrink
    return out.dropna(subset=["pred_ridge_price", "pred_ridge_sent"]).reset_index(drop=True), len(dates)


def forward_sentiment_predictions(df: pd.DataFrame, alpha: float, shrink: float = 1.0,
                                  model_name: str = "pls", scale: float = 1.0,
                                  clip: float | None = None, components: int = 6,
                                  min_train_dates: int = 6,
                                  panel_dates=None) -> pd.DataFrame:
    """Chronological short-panel evaluation with a one-session label purge.

    Every prediction is fitted only on sessions at least two dates earlier. This is
    the primary live-like check; leave-one-session-out is only a diagnostic.
    """
    if panel_dates is None:
        data = _sentiment_rows(df)
    else:
        data = df.dropna(subset=["y_next_ret"] + ENGINEERED_MODEL_FEATURES)
        data = data[data["date"].isin(panel_dates)].reset_index(drop=True)
    dates = sorted(data["date"].unique())
    blocks = []
    for i in range(min_train_dates + 1, len(dates)):
        train = data[data["date"].isin(dates[:i - 1])]
        test = data[data["date"] == dates[i]].copy()
        test["pred_naive"] = 0.0
        for name, columns in (("ridge_price", PRICE_FEATURES),
                              ("ridge_sent", ENGINEERED_MODEL_FEATURES)):
            fitted = _fit(columns, train, alpha, model_name,
                          scale if name == "ridge_sent" else 1.0, components)
            test[f"pred_{name}"] = _predict(fitted, test[columns], clip) * shrink
        blocks.append(test)
    if not blocks:
        return pd.DataFrame(columns=list(data.columns) +
                            ["pred_naive", "pred_ridge_price", "pred_ridge_sent"])
    return pd.concat(blocks, ignore_index=True)


def sentiment_coefficients(df: pd.DataFrame, alpha: float, scale: float = 1.0,
                           model_name: str = "ridge", components: int = 6) -> pd.DataFrame:
    """Standardized model coefficients, kept as a linear reference for interpretation."""
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["feature", "coefficient"])
    model = _fit(ENGINEERED_MODEL_FEATURES, data, alpha, model_name, scale, components)
    step = "plsregression" if model_name == "pls" else "ridge"
    coefficients = np.asarray(model.named_steps[step].coef_).ravel()
    out = pd.DataFrame({"feature": ENGINEERED_MODEL_FEATURES, "coefficient": coefficients})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def sentiment_importances(df: pd.DataFrame, model_name: str = "rf",
                          alpha: float = 50, scale: float = 1.0,
                          components: int = 6) -> pd.DataFrame:
    """Feature importances for the short-history sentiment model.

    Random Forest importances are impurity based. Linear/PLS importances use the absolute
    standardized coefficients normalized to sum to one, so all models produce a comparable table.
    """
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    model = _fit(ENGINEERED_MODEL_FEATURES, data, alpha, model_name, scale, components)
    if model_name == "rf":
        importances = model.named_steps["randomforestregressor"].feature_importances_
    else:
        step = "plsregression" if model_name == "pls" else "ridge"
        coefficients = np.abs(np.asarray(model.named_steps[step].coef_).ravel())
        total = coefficients.sum()
        importances = (coefficients / total if total > 0
                       else np.full_like(coefficients, 1 / len(coefficients)))
    out = pd.DataFrame({"feature": ENGINEERED_MODEL_FEATURES, "importance": importances})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def price_only_forecast(df: pd.DataFrame, alpha: float = 10.0, shrink: float = 1.0,
                        model_name: str = "pls", clip: float | None = None,
                        components: int = 6) -> pd.DataFrame:
    """Same-class price-only control for the next session.

    Uses the same estimator, shrinkage and clipping as the sentiment model, but only the
    price technicals, so the live table is an apples-to-apples ablation.
    """
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["ticker", "pred_price_only_ret", "pred_price_only_price"])
    fitted = _fit(PRICE_FEATURES, data, alpha, model_name, 1.0, components)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest["pred_price_only_ret"] = _predict(fitted, latest[PRICE_FEATURES], clip) * shrink
    latest["pred_price_only_price"] = latest["close"] * (1 + latest["pred_price_only_ret"])
    return latest[["ticker", "pred_price_only_ret", "pred_price_only_price"]]


def sentiment_forecast(df: pd.DataFrame, alpha: float, shrink: float = 1.0,
                       model_name: str = "rf", scale: float = 1.0,
                       clip: float | None = None, components: int = 6) -> pd.DataFrame:
    """Fit the sentiment model on all sentiment rows and predict the next session for each ticker."""
    data = _sentiment_rows(df)
    if data.empty:
        return pd.DataFrame(columns=["ticker", "pred_sent_ret", "pred_sent_price"])
    fitted = _fit(ENGINEERED_MODEL_FEATURES, data, alpha, model_name, scale, components)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest["pred_sent_ret"] = _predict(
        fitted, latest[ENGINEERED_MODEL_FEATURES], clip) * shrink
    latest["pred_sent_price"] = latest["close"] * (1 + latest["pred_sent_ret"])
    return latest[["ticker", "pred_sent_ret", "pred_sent_price"]]


def nested_sentiment_predictions(df: pd.DataFrame, alphas: tuple[float, ...] = (10, 30, 50, 100, 300),
                                 shrink: float = 0.5, scale: float = 1.0,
                                 clip: float | None = None, components: int = 6) -> pd.DataFrame:
    """Sensitivity check: choose model family and alpha by inner leave-one-session-out."""
    data = _sentiment_rows(df)
    out = data.copy()
    out["pred_ridge_sent"] = np.nan
    candidates = [("pls", 0)] + [("ridge", alpha) for alpha in alphas] + [("rf", 0.0)]
    for session in sorted(data["date"].unique()):
        train = data[data["date"] != session].reset_index(drop=True)
        test = data[data["date"] == session]
        best = candidates[0]
        best_mape = np.inf
        for model_name, alpha in candidates:
            inner = np.zeros(len(train))
            for inner_session in sorted(train["date"].unique()):
                inner_train = train[train["date"] != inner_session]
                inner_test = train["date"] == inner_session
                fitted = _fit(ENGINEERED_MODEL_FEATURES, inner_train, alpha, model_name,
                              scale, components)
                inner[inner_test.values] = _predict(
                    fitted, train.loc[inner_test, ENGINEERED_MODEL_FEATURES], clip)
            mape = ((train["close"] * (1 + shrink * inner) - train["next_close"]).abs()
                    / train["next_close"]).mean()
            if mape < best_mape:
                best, best_mape = (model_name, alpha), mape
        model_name, alpha = best
        fitted = _fit(ENGINEERED_MODEL_FEATURES, train, alpha, model_name, scale, components)
        out.loc[test.index, "pred_ridge_sent"] = _predict(
            fitted, test[ENGINEERED_MODEL_FEATURES], clip) * shrink
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
