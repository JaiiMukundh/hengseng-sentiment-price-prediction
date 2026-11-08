import numpy as np
import pandas as pd

from sentpred.evaluation import cluster_bootstrap


def _preds():
    dates = pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"])
    rows = []
    for d in dates:
        rows += [
            {"date": d, "close": 100.0, "next_close": 101.0, "y_next_ret": 0.01,
             "pred_naive": 0.0, "pred_ridge_price": 0.005, "pred_ridge_sent": 0.01},
            {"date": d, "close": 200.0, "next_close": 198.0, "y_next_ret": -0.01,
             "pred_naive": 0.0, "pred_ridge_price": -0.005, "pred_ridge_sent": -0.01},
        ]
    return pd.DataFrame(rows)


def test_bootstrap_returns_ordered_intervals():
    out = cluster_bootstrap(_preds(), n_boot=200, seed=3)
    assert set(out["model"]) == {"naive", "ridge_price", "ridge_sent"}
    assert (out["ci_low"] <= out["mean"] + 1e-9).all()
    assert (out["mean"] <= out["ci_high"] + 1e-9).all()
    assert out["mean"].notna().all()


def test_bootstrap_empty_input():
    out = cluster_bootstrap(pd.DataFrame(), n_boot=10)
    assert out.empty


def test_forward_sentiment_uses_only_past_sessions(monkeypatch):
    import numpy as np
    import pandas as pd
    from sentpred import model
    from sentpred.features import ENGINEERED_MODEL_FEATURES

    dates = pd.bdate_range("2026-01-01", periods=10)
    rows = []
    for date in dates:
        row = dict.fromkeys(ENGINEERED_MODEL_FEATURES, 0.0)
        row.update(date=date, ticker="TEST", close=100.0, next_close=101.0,
                   y_next_ret=0.01, has_social=1.0, has_news=0.0)
        rows.append(row)
    seen = []

    def fake_fit(columns, train, *args, **kwargs):
        seen.append(train["date"].max())
        class Predictor:
            def predict(self, values):
                return np.zeros(len(values))
        return Predictor()

    monkeypatch.setattr(model, "_fit", fake_fit)
    predictions = model.forward_sentiment_predictions(pd.DataFrame(rows), 1, min_train_dates=2)
    assert len(predictions) == 7
    assert all(fit_date < test_date for fit_date, test_date in zip(seen[::2], predictions["date"]))
    assert all(fit_date <= dates[list(dates).index(test_date) - 2]
               for fit_date, test_date in zip(seen[::2], predictions["date"]))


def test_cached_rows_do_not_gain_false_first_seen_time(tmp_path):
    from sentpred.cli import _append

    path = tmp_path / "posts.csv"
    row = {"source": "telegram", "ticker": "0700.HK", "text": "Tencent",
           "published_utc": "2026-11-01T10:00:00Z"}
    pd.DataFrame([row]).to_csv(path, index=False)
    _append(path, pd.DataFrame([row]), observed=True)
    assert pd.isna(pd.read_csv(path).loc[0, "first_seen_utc"])

    second = {**row, "text": "New message",
              "published_utc": pd.Timestamp("2026-11-01T10:00:00Z")}
    _append(path, pd.DataFrame([second]), observed=True)
    first_seen = pd.read_csv(path).set_index("text")["first_seen_utc"]
    assert pd.notna(first_seen["New message"])
