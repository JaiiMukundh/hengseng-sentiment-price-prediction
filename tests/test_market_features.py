import numpy as np
import pandas as pd
import pytest

from sentpred.features import MARKET_FEATURES, market_features


def _market(dates):
    def frame(series, values):
        return pd.DataFrame({"series": series, "date": dates, "close": values})

    return pd.concat([
        frame("HKEX_INDEX", [100, 101, 102, 103, 104]),
        frame("KRX_INDEX", [2500, 2525, 2500, 2550, 2575]),
        frame("US_NASDAQ", [100, 110, 99, 108.9, 120]),
        frame("US_SOX", [200, 210, 189, 207.9, 220]),
        frame("US_VIX", [20, 21, 19, 22, 18]),
        frame("USDHKD", [7.8, 7.81, 7.79, 7.82, 7.80]),
        frame("USDKRW", [1300, 1310, 1290, 1320, 1305]),
    ], ignore_index=True)


def test_market_features_cover_all_columns_without_nan():
    dates = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"ticker": "0700.HK", "exchange": "HKEX", "date": dates})
    out = market_features(prices, _market(dates))
    assert set(MARKET_FEATURES) <= set(out.columns)
    assert out[MARKET_FEATURES].isna().sum().sum() == 0


def test_us_features_use_previous_session_not_same_day():
    dates = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"ticker": "0700.HK", "exchange": "HKEX", "date": dates})
    out = market_features(prices, _market(dates))
    # US Nasdaq: 100 -> 110 on day 1 (+10%), 110 -> 99 on day 2 (-10%).
    # The HK session on day 2 must see the day-1 US return, never the same-day one.
    assert out.loc[2, "us_ret_1"] == pytest.approx(0.10)
    # The HK session on day 3 must see the previous session's return (day 2, -10%).
    assert out.loc[3, "us_ret_1"] == pytest.approx(-0.10)
    assert out.loc[1, "us_ret_1"] == pytest.approx(0.0)  # no prior US session in the fixture


def test_market_features_are_zero_when_market_is_missing():
    dates = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"ticker": "0700.HK", "exchange": "HKEX", "date": dates})
    out = market_features(prices, None)
    assert (out[MARKET_FEATURES] == 0).all().all()
