import numpy as np
import pandas as pd

from sentpred import features


def _prices():
    dates = pd.bdate_range("2026-09-01", periods=50)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, len(dates))))
    return pd.DataFrame({
        "ticker": "0700.HK", "exchange": "HKEX", "date": dates,
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1_000_000,
    })


def test_out_of_range_timestamps_are_dropped():
    from sentpred import align
    sessions = pd.DatetimeIndex(["2026-11-05", "2026-11-06", "2026-11-09"])
    old = pd.Series(pd.to_datetime(["2020-01-01 09:00:00"], utc=True))
    mapped = align.map_to_session(old, "Asia/Hong_Kong", "16:00", sessions)
    assert pd.isna(mapped.iloc[0])


def test_after_close_post_becomes_pending_session_row():
    prices = _prices()
    last_date = prices["date"].max()
    next_session = last_date + pd.offsets.BDay(1)
    social = pd.DataFrame({
        "ticker": ["0700.HK"], "date": [next_session],
        "social_n": [3.0], "social_sent_mean": [0.4], "social_sent_wmean": [0.5],
        "social_pos_share": [0.7], "social_neg_share": [0.1], "social_eng_log": [2.0],
        "social_hit_share": [1.0], "social_sent_mean_hit": [0.4], "social_sent_wmean_hit": [0.5],
    })
    dataset = features.build_dataset(prices, social, pd.DataFrame(columns=["ticker", "date"]))
    pending = dataset[dataset["date"] == next_session]
    assert len(pending) == 1
    row = pending.iloc[0]
    assert row["social_n"] == 3
    assert row["has_social"] == 1
    assert np.isnan(row["y_next_ret"])
    assert row["close"] == prices["close"].iloc[-1]
    inference = features.completed_session_rows(dataset, prices)
    assert inference["date"].max() == last_date
    assert next_session not in set(inference["date"])
