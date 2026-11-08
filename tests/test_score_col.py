import pandas as pd

from sentpred import align


def test_aggregate_uses_precomputed_scores():
    sessions = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
    posts = pd.DataFrame({
        "source": "naver_direct", "exchange": "KRX", "ticker": "005930.KS",
        "text": ["a", "b"], "published_utc": pd.to_datetime(["2026-11-05 02:00", "2026-11-05 03:00"], utc=True),
        "likes": [0.0, 0.0], "reposts": [0.0, 0.0], "views": [0.0, 0.0], "followers": [0.0, 0.0],
        "sent_score": [1.0, -1.0],
    })
    out = align.aggregate(posts, {"KRX": ("Asia/Seoul", "15:30", sessions)}, "social_", score_col="sent_score")
    assert len(out) == 1
    assert out.iloc[0]["social_sent_mean"] == 0.0
    assert out.iloc[0]["social_pos_share"] == 0.5
    assert out.iloc[0]["social_neg_share"] == 0.5
