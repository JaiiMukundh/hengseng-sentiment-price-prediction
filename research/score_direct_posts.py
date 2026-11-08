"""Score the research-only direct Naver backfill with KR-FinBERT."""

from __future__ import annotations

import glob
import json

import pandas as pd

from research.transformer_sentiment import score_posts


def main():
    rows = []
    for path in sorted(glob.glob("data/raw/naver_direct_*d_*.json")):
        code = path.rsplit("_", 1)[-1].replace(".json", "")
        for p in json.load(open(path, encoding="utf-8")):
            text = " ".join([p.get("title") or "", p.get("content") or ""]).strip()
            if not text or not p.get("date"):
                continue
            rows.append({
                "source": "naver_direct",
                "exchange": "KRX",
                "ticker": f"{code}.KS",
                "text": text,
                "published_utc": pd.Timestamp(p["date"]).tz_localize("Asia/Seoul").tz_convert("UTC"),
                "url": p.get("url"),
                "author": p.get("author"),
                "likes": p.get("likes", 0),
                "reposts": 0,
                "views": p.get("views", 0),
                "followers": 0,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "text", "published_utc"])
    print("posts", len(df), "tickers", df.ticker.nunique())

    old_path = "data/processed/naver_direct_90d_scored.csv"
    try:
        old = pd.read_csv(old_path)
        old["published_utc"] = pd.to_datetime(old["published_utc"], utc=True, format="mixed")
        seen = set(zip(old["ticker"], old["text"], old["published_utc"]))
        new = df[~df.apply(lambda r: (r["ticker"], r["text"], r["published_utc"]) in seen, axis=1)].copy()
        print("already scored", len(old), "new to score", len(new))
    except FileNotFoundError:
        old = pd.DataFrame()
        new = df

    scored_new = score_posts(new) if not new.empty else new
    scored = pd.concat([old, scored_new], ignore_index=True).drop_duplicates(
        subset=["ticker", "text", "published_utc"])
    scored.to_csv("data/processed/naver_direct_all_scored.csv", index=False)
    print("saved", len(scored), scored.sent_score.describe().round(3).to_dict())


if __name__ == "__main__":
    main()
