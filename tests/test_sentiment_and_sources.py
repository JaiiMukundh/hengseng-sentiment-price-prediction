import pandas as pd
import pytest

from sentpred.config import Asset
from sentpred.sentiment import mentions_asset, score_text
from sentpred.sources import _timestamp, build_run_input, normalize

TENCENT = Asset("HKEX", "0700.HK", "Tencent Holdings", ("Tencent", "腾讯", "$0700", "00700"))
SAMSUNG = Asset("KRX", "005930.KS", "Samsung Electronics", ("Samsung Electronics", "삼성전자"))


@pytest.mark.parametrize("text,expected", [
    ("strong breakout, buy", 1.0),
    ("earnings miss, bearish", -1.0),
    ("看好 但 下跌 风险", 0.0),
    ("주가 상승 매수 추천", 1.0),
    ("just watching today", 0.0),
])
def test_score_text_bilingual(text, expected):
    score, _ = score_text(text)
    assert score == pytest.approx(expected)


def test_score_text_hit_count():
    assert score_text("nothing here") == (0.0, 0)
    assert score_text("buy and sell")[1] == 2


def test_mentions_asset_uses_word_boundaries_for_ascii():
    assert mentions_asset("$0700 looks good", TENCENT)
    assert mentions_asset("Tencent results", TENCENT)
    assert mentions_asset("腾讯 今天 大涨", TENCENT)
    assert not mentions_asset("ref 100700 unrelated", TENCENT)
    assert not mentions_asset("Samsung Galaxy launch", TENCENT)


def test_mentions_asset_korean():
    assert mentions_asset("삼성전자 실적 발표", SAMSUNG)


def test_timestamp_parsing_utc_and_kst():
    assert _timestamp(1_700_000_000) == pd.Timestamp("2023-11-14T22:13:20Z")
    assert _timestamp(1_700_000_000_000) == pd.Timestamp("2023-11-14T22:13:20Z")
    kst = _timestamp("2026.11.07 16:20")
    assert kst == pd.Timestamp("2026-11-07 07:20:00", tz="UTC")
    assert _timestamp("2026-11-07T10:00:00+08:00") == pd.Timestamp("2026-11-07 02:00:00", tz="UTC")
    assert _timestamp("not a date") is None


def test_normalize_maps_fields_and_expands_to_matching_tickers():
    items = [
        {"title": "Tencent beats estimates", "content": "bullish outlook", "createdAt": "2026-11-06T03:00:00Z",
         "url": "https://example/1", "author": "u1", "likeCount": "1,200", "viewCount": 5000, "followers": 40000},
        {"text": "no mention here", "date": "2026-11-06T03:00:00Z"},
    ]
    df = normalize("lihkg", "HKEX", items, (TENCENT,))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "0700.HK"
    assert row["likes"] == 1200
    assert row["views"] == 5000
    assert row["followers"] == 40000
    assert "Tencent beats estimates" in row["text"]


def test_normalize_naver_nested_posts_uses_kst():
    items = [{"url": "x", "posts": [
        {"title": "삼성전자 매수 추천", "date": "2026.11.06 10:30", "likes": 3, "views": 100},
    ]}]
    df = normalize("naver_discussion", "KRX", items, (SAMSUNG,))
    assert len(df) == 1
    assert df.iloc[0]["published_utc"] == pd.Timestamp("2026-11-06 01:30:00", tz="UTC")


def test_build_run_input_telegram_uses_keyword_search_without_channel_list():
    cfg = {"channels": {"HKEX": []}, "discovery_language": "zh-HK", "discovery_country": "hk"}
    run = build_run_input("telegram", cfg, "HKEX", (TENCENT,), 10)
    assert run["discoverChannels"] is True
    assert run["channels"] == []
    assert run["afterDate"] == "10 days"
    assert "Tencent" in run["keywords"] and "$0700" in run["keywords"]


def test_build_run_input_lihkg_uses_search_mode():
    cfg = {"max_items": 50, "date_from": "2w"}
    run = build_run_input("lihkg", cfg, "HKEX", (TENCENT,), 10)
    assert run["mode"] == "search"
    assert "Tencent" in run["keywords"]
    assert all(not k.startswith("$") for k in run["keywords"])
