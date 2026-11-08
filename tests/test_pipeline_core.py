import numpy as np
import pandas as pd
import pytest

from sentpred import align, levels, model
from sentpred.config import Asset
from sentpred.features import (
    ENGINEERED_MODEL_FEATURES,
    PRICE_FEATURES,
    SENTIMENT_DYNAMICS,
    build_dataset,
    rsi,
)


def _prices(n_days=200, tickers=("0700.HK", "9988.HK"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rows = []
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        rows.append(pd.DataFrame({
            "ticker": t, "exchange": "HKEX", "date": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": rng.integers(1e5, 1e6, n_days),
        }))
    return pd.concat(rows, ignore_index=True)


def _sentiment_dataset(n_days=30, seed=7):
    dataset = build_dataset(_prices(n_days=n_days), pd.DataFrame(columns=["ticker", "date"]),
                            pd.DataFrame(columns=["ticker", "date"]))
    rng = np.random.default_rng(seed)
    dataset["has_social"] = 1.0
    dataset["social_n"] = 1.0
    dataset["social_n_log"] = 1.0
    dataset["social_sent_mean"] = rng.normal(0, 0.4, len(dataset))
    dataset["social_sent_wmean"] = dataset["social_sent_mean"]
    dataset["social_pos_share"] = 0.5
    dataset["social_neg_share"] = 0.5
    return dataset


def test_session_mapping_respects_close_and_weekends():
    sessions = pd.DatetimeIndex(["2026-11-05", "2026-11-06", "2026-11-09"])
    ts = pd.Series(pd.to_datetime([
        "2026-11-05 08:00:00",  # 16:00 HK -> same day before close
        "2026-11-05 09:00:00",  # 17:00 HK -> after close, rolls to 11-06
        "2026-11-07 12:00:00",  # Saturday 20:00 HK -> next session Monday
    ], utc=True))
    mapped = align.map_to_session(ts, "Asia/Hong_Kong", "16:00", sessions)
    assert list(mapped) == [pd.Timestamp("2026-11-05"), pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-09")]


def test_session_calendar_appends_next_business_day():
    cal = align.session_calendar(pd.Series(pd.to_datetime(["2026-11-05", "2026-11-06"])))
    assert cal[-1] == pd.Timestamp("2026-11-09")


def test_aggregate_weights_and_counts():
    sessions = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
    posts = pd.DataFrame({
        "source": "lihkg", "exchange": "HKEX", "ticker": "0700.HK",
        "text": ["buy strong growth", "bearish fall"],
        "published_utc": pd.to_datetime(["2026-11-05 02:00", "2026-11-05 03:00"], utc=True),
        "likes": [1000.0, 0.0], "reposts": [0.0, 0.0], "views": [0.0, 0.0], "followers": [0.0, 0.0],
    })
    agg = align.aggregate(posts, {"HKEX": ("Asia/Hong_Kong", "16:00", sessions)}, "social_")
    row = agg.iloc[0]
    assert row["social_n"] == 2
    assert row["social_sent_mean"] == pytest.approx(0.0)
    # The high-engagement bullish post should pull the weighted mean above zero.
    assert row["social_sent_wmean"] > 0


def test_rsi_bounded():
    close = pd.Series(np.linspace(1, 100, 60))
    values = rsi(close).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_build_dataset_fills_missing_sentiment_with_zero():
    prices = _prices()
    dataset = build_dataset(prices, pd.DataFrame(columns=["ticker", "date"]), pd.DataFrame(columns=["ticker", "date"]))
    assert (dataset["social_n"] == 0).all()
    assert (dataset["has_social"] == 0).all()
    assert dataset["social_sent_wmean"].isna().sum() == 0


def test_walk_forward_is_out_of_sample_and_evaluates_all_models():
    dataset = build_dataset(_prices(), pd.DataFrame(columns=["ticker", "date"]), pd.DataFrame(columns=["ticker", "date"]))
    preds = model.walk_forward(dataset, min_train_dates=60, test_block=5, alpha=1.0)
    assert preds["date"].min() > dataset["date"].sort_values().unique()[59]
    metrics = model.evaluate(preds)
    assert set(metrics["model"]) == {"naive", "ridge_price", "ridge_sent"}
    assert {"all"} <= set(metrics["subset"])


def test_live_forecast_one_row_per_ticker_with_signal():
    dataset = build_dataset(_prices(), pd.DataFrame(columns=["ticker", "date"]), pd.DataFrame(columns=["ticker", "date"]))
    preds = model.walk_forward(dataset, 60, 5, 1.0)
    metrics = model.evaluate(preds)
    fc = model.live_forecast(dataset, metrics, 1.0, 0.005)
    assert sorted(fc["ticker"]) == ["0700.HK", "9988.HK"]
    assert set(fc["signal"]) <= {"LONG", "SHORT", "FLAT"}
    last_close = dataset.sort_values("date").groupby("ticker")["close"].last()
    assert np.allclose(fc.set_index("ticker")["close"], last_close)


def test_short_history_sentiment_model_trains_real_weights():
    prices = _prices(n_days=30, tickers=("0700.HK", "9988.HK"))
    dataset = build_dataset(prices, pd.DataFrame(columns=["ticker", "date"]), pd.DataFrame(columns=["ticker", "date"]))
    rng = np.random.default_rng(5)
    dataset["has_social"] = 1.0
    dataset["social_n"] = 1.0
    dataset["social_n_log"] = 1.0
    dataset["social_sent_mean"] = rng.normal(0, 0.4, len(dataset))
    dataset["social_sent_wmean"] = dataset["social_sent_mean"]
    dataset["social_pos_share"] = 0.5
    dataset["social_neg_share"] = 0.5

    preds, n_sessions = model.leave_one_session_out(dataset, alpha=50.0, shrink=0.5)
    expected_sessions = dataset.dropna(subset=["y_next_ret"] + PRICE_FEATURES)["date"].nunique()
    assert n_sessions == expected_sessions
    assert preds["pred_ridge_sent"].notna().all()
    assert not np.allclose(preds["pred_ridge_sent"], preds["pred_ridge_price"])

    coefficients = model.sentiment_coefficients(dataset, alpha=100.0)
    assert len(coefficients) == len(model.ENGINEERED_MODEL_FEATURES)
    sentiment_coefs = coefficients[coefficients["feature"].isin(model.ENGINEERED_MODEL_FEATURES)]
    assert sentiment_coefs["coefficient"].abs().sum() > 0

    forecast = model.sentiment_forecast(dataset, alpha=50.0, shrink=0.5)
    assert sorted(forecast["ticker"]) == ["0700.HK", "9988.HK"]
    assert forecast["pred_sent_price"].notna().all()

    importances = model.sentiment_importances(dataset, "rf")
    assert len(importances) == len(model.ENGINEERED_MODEL_FEATURES)
    assert importances["importance"].sum() == pytest.approx(1.0)
    assert (importances["importance"] >= 0).all()


@pytest.mark.parametrize("model_name", ["rf", "ridge"])
def test_sentiment_importances_finite_and_normalized_for_both_models(model_name):
    dataset = _sentiment_dataset()

    importances = model.sentiment_importances(dataset, model_name)
    values = importances["importance"].to_numpy(dtype=float)

    assert list(importances.columns) == ["feature", "importance"]
    assert set(importances["feature"]) == set(model.ENGINEERED_MODEL_FEATURES)
    assert len(importances) == len(model.ENGINEERED_MODEL_FEATURES)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert values.sum() == pytest.approx(1.0)


def test_ridge_importances_are_normalized_absolute_standardized_coefficients():
    dataset = _sentiment_dataset()

    importances = model.sentiment_importances(dataset, "ridge", alpha=100.0)
    coefficients = model.sentiment_coefficients(dataset, alpha=100.0)
    expected = coefficients.set_index("feature")["coefficient"].abs()
    expected = expected / expected.sum()
    actual = importances.set_index("feature")["importance"]

    assert np.allclose(actual.sort_index(), expected.sort_index())


def test_walk_forward_requires_enough_history():
    dataset = build_dataset(_prices(n_days=40), pd.DataFrame(columns=["ticker", "date"]), pd.DataFrame(columns=["ticker", "date"]))
    with pytest.raises(ValueError):
        model.walk_forward(dataset, 60, 5, 1.0)


def test_pivot_points_formula():
    last = pd.Series({"high": 110.0, "low": 90.0, "close": 100.0})
    p = levels.pivot_points(last)
    assert p["P"] == pytest.approx(100.0)
    assert p["R1"] == pytest.approx(110.0)
    assert p["S1"] == pytest.approx(90.0)


def test_swing_levels_split_around_last_close():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 200))
    bars = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    out = levels.swing_levels(bars, lookback=120, order=5)
    last = close[-1]
    assert all(s["level"] < last for s in out["support"])
    assert all(r["level"] > last for r in out["resistance"])
    assert len(out["support"]) <= 2 and len(out["resistance"]) <= 2


def test_engineered_sentiment_dynamics_are_present_and_finite():
    dataset = build_dataset(
        _prices(n_days=40),
        pd.DataFrame(columns=["ticker", "date"]),
        pd.DataFrame(columns=["ticker", "date"]),
    )
    for column in SENTIMENT_DYNAMICS:
        assert column in dataset
        assert np.isfinite(dataset[column].to_numpy(dtype=float)).all()
    assert set(ENGINEERED_MODEL_FEATURES) <= set(dataset.columns)


def test_pls_short_history_and_same_class_price_only_forecast():
    dataset = _sentiment_dataset()
    preds, n_sessions = model.leave_one_session_out(
        dataset, alpha=10.0, shrink=0.3, model_name="pls",
        scale=1.0, clip=0.05, components=6)
    assert n_sessions > 0
    assert len(preds) > 0
    assert np.isfinite(preds["pred_ridge_sent"]).all()
    assert np.isfinite(preds["pred_ridge_price"]).all()
    assert not np.allclose(preds["pred_ridge_sent"], preds["pred_ridge_price"])

    forecast = model.price_only_forecast(
        dataset, alpha=10.0, shrink=0.3, model_name="pls", clip=0.05, components=6)
    assert sorted(forecast["ticker"]) == ["0700.HK", "9988.HK"]
    assert np.isfinite(forecast["pred_price_only_ret"]).all()
    assert np.isfinite(forecast["pred_price_only_price"]).all()
