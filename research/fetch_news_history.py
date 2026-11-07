"""Fetch 365 days of Exa news in quarterly slices (free credits, no Apify)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from sentpred.config import exa_key, load_config
from sentpred.news import EXA_URL
from sentpred.sources import _timestamp

cfg = load_config()
now = datetime.now(timezone.utc)
windows = []
for i in range(4):
    end = now - timedelta(days=90 * i)
    start = end - timedelta(days=90)
    windows.append((start, end))

rows = []
for asset in cfg.assets:
    for start, end in windows:
        body = {
            "query": f"latest news about {asset.name} ({asset.ticker}) stock",
            "objective": f"Find price-relevant news about {asset.name} ({asset.ticker}) for a next-session forecast.",
            "contents": {"highlights": True},
            "startPublishedDate": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endPublishedDate": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "numResults": 100,
        }
        resp = requests.post(EXA_URL, headers={"x-api-key": exa_key(), "Content-Type": "application/json"},
                             json=body, timeout=90)
        if resp.status_code >= 400:
            print(asset.ticker, start.date(), "HTTP", resp.status_code)
            continue
        for r in resp.json().get("results", []):
            text = " ".join([r.get("title") or "", *(r.get("highlights") or [])]).strip()
            ts = _timestamp(r.get("publishedDate"))
            if not text or ts is None:
                continue
            rows.append({
                "source": "exa_news", "exchange": asset.exchange, "ticker": asset.ticker,
                "text": text, "published_utc": ts, "url": r.get("url"), "author": r.get("author"),
                "likes": 0.0, "reposts": 0.0, "views": 0.0, "followers": 0.0,
            })
        print(asset.ticker, start.date(), "total", len(rows))

df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "text", "published_utc"])
df.to_csv("data/processed/news_365.csv", index=False)
print("saved", len(df), "rows", df.ticker.nunique(), "assets")
