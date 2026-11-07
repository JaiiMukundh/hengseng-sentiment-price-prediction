"""Edge cases for source normalization: URL mentions, LIHKG fields, and timestamps."""

import pandas as pd

from sentpred.align import map_to_session
from sentpred.config import Asset
from sentpred.sources import _naver_time, _timestamp, normalize

KAKAO = Asset("KRX", "035720.KS", "Kakao", ("Kakao", "카카오"))
TENCENT = Asset("HKEX", "0700.HK", "Tencent Holdings", ("Tencent", "腾讯", "$0700", "00700"))


def test_kakao_url_does_not_create_a_ticker_match():
    items = [{
        "text": "오늘 웹툰 한 편 봤다 https://page.kakao.com/content/12345",
        "date": "2026-11-06T03:00:00Z",
    }]
    df = normalize("telegram", "KRX", items, (KAKAO,))
    assert df.empty


def test_kakao_text_mention_still_matches():
    items = [{"text": "카카오 실적 발표", "date": "2026-11-06T03:00:00Z"}]
    df = normalize("telegram", "KRX", items, (KAKAO,))
    assert df["ticker"].tolist() == ["035720.KS"]


def test_lihkg_thread_url_and_reply_count_are_mapped():
    items = [{
        "title": "Tencent beats estimates",
        "date": "2026-11-06T03:00:00Z",
        "thread_url": "https://lihkg.com/thread/12345/page/1",
        "reply_count": 27,
    }]
    df = normalize("lihkg", "HKEX", items, (TENCENT,))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["url"] == "https://lihkg.com/thread/12345/page/1"
    assert row["reposts"] == 27


def test_naver_iso_z_preserves_offset():
    assert _naver_time("2026-11-06T10:30:00Z", None) == pd.Timestamp("2026-11-06 10:30:00", tz="UTC")


def test_naver_iso_offset_preserves_offset():
    assert _naver_time("2026-11-06T10:30:00+09:00", None) == pd.Timestamp("2026-11-06 01:30:00", tz="UTC")


def test_naver_iso_space_separator_and_milliseconds():
    assert _naver_time("2026-11-06 10:30:00+09:00", None) == pd.Timestamp("2026-11-06 01:30:00", tz="UTC")
    assert _naver_time("2026-11-06T10:30:00.250Z", None) == pd.Timestamp("2026-11-06 10:30:00.250", tz="UTC")


def test_naive_naver_written_at_is_kst():
    expected = pd.Timestamp("2026-11-06 01:30:00", tz="UTC")
    assert _naver_time("2026-11-06T10:30:00", None) == expected
    assert _naver_time("2026-11-06 10:30:00", None) == expected


def test_date_only_news_moves_to_next_day_utc():
    """A date with no time is placed at the next day 00:00 UTC to avoid same-session leaks."""
    assert _timestamp("2026-11-06") == pd.Timestamp("2026-11-07 00:00:00", tz="UTC")
    assert _timestamp("2026-11-06T00:00:00") == pd.Timestamp("2026-11-06 00:00:00", tz="UTC")


def test_out_of_range_timestamp_is_not_clamped_to_first_session():
    sessions = pd.DatetimeIndex(pd.to_datetime(["2026-11-05", "2026-11-06"]))
    published = pd.Series(pd.to_datetime(["2020-01-02T00:00:00Z"]))
    mapped = map_to_session(published, "UTC", "16:00", sessions)
    assert pd.isna(mapped.iloc[0])
