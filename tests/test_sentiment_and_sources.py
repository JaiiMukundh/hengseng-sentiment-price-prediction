import pandas as pd
import pytest

from sentpred.config import Asset, load_config
from sentpred.sentiment import mentions_asset, score_text
from sentpred.sources import _naver_time, _timestamp, build_run_input, normalize

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


def test_naver_relative_and_absolute_times():
    scraped = "2026-11-07T03:00:00Z"
    assert _naver_time("1분 전", scraped) == pd.Timestamp("2026-11-07 02:59:00", tz="UTC")
    assert _naver_time("2시간 전", scraped) == pd.Timestamp("2026-11-07 01:00:00", tz="UTC")
    assert _naver_time("1일 전", scraped) == pd.Timestamp("2026-11-06 03:00:00", tz="UTC")
    assert _naver_time("방금 전", scraped) == pd.Timestamp(scraped)
    # 04:58 KST on 2026-11-07 is 2026-11-06 19:58 UTC
    assert _naver_time("오전 4:58", scraped) == pd.Timestamp("2026-11-06 19:58:00", tz="UTC")
    assert _naver_time("11.05", scraped) == pd.Timestamp("2026-11-04 15:00:00", tz="UTC")


def test_blind_naive_timestamp_is_eastern():
    assert _timestamp("09/26/2026, 10:00 AM", "America/New_York") == pd.Timestamp("2026-09-26 14:00:00", tz="UTC")


def test_telegram_engagement_mapping():
    items = [{"text": "Tencent looks strong", "date": "2026-11-06T03:00:00+00:00",
              "totalReactions": 12, "views": 900, "channelSubscribers": 50000, "channel": "hkstock"}]
    df = normalize("telegram", "HKEX", items, (TENCENT,))
    assert df.iloc[0]["likes"] == 12
    assert df.iloc[0]["followers"] == 50000


def test_normalize_deduplicates_repeated_posts():
    item = {"title": "Tencent rally", "date": "2026-11-06T03:00:00+00:00", "like_count": 1}
    df = normalize("lihkg", "HKEX", [item, dict(item)], (TENCENT,))
    assert len(df) == 1


def test_build_run_input_telegram_uses_keyword_search_without_channel_list():
    cfg = {"channels": {"HKEX": []}, "discovery": {"HKEX": {"language": "zh-TW", "country": "hk"}}}
    run = build_run_input("telegram", cfg, "HKEX", (TENCENT,), 10)
    assert run["discoverChannels"] is True
    assert run["channels"] == []
    assert run["afterDate"] == "10 days"
    assert run["discoveryLanguage"] == "zh-TW"
    assert "Tencent" in run["keywords"] and "$0700" in run["keywords"]


def test_build_run_input_lihkg_uses_search_mode():
    cfg = {"max_items": 50, "date_from": "2w"}
    run = build_run_input("lihkg", cfg, "HKEX", (TENCENT,), 10)
    assert run["mode"] == "search"
    assert "Tencent" in run["keywords"]
    assert all(not k.startswith("$") for k in run["keywords"])


def test_config_has_five_assets_per_exchange_and_two_sources_each():
    cfg = load_config()
    for exchange in ("HKEX", "KRX"):
        assert len(cfg.assets_for(exchange)) == 5
        sources = [name for name, src in cfg.raw["sources"].items()
                   if src.get("enabled") and exchange in src["exchanges"]]
        assert len(sources) >= 2, (exchange, sources)


def test_position_label_defaults_to_flat_and_formats_holdings():
    assert Asset("HKEX", "0700.HK", "Tencent", ("Tencent",)).position_label == "flat"
    held = Asset("HKEX", "0700.HK", "Tencent", ("Tencent",), {"quantity": 200, "avg_price": 410.5})
    assert held.position_label == "200 @ 410.5"
