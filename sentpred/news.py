"""News sentiment input via Exa (https://exa.ai) neural search, restricted to the look-back window.

Request shape follows Exa's build-with-exa guidance: query + highlights, no highlight budgets,
no numResults/category unless needed. The 10-day window is a hard task requirement, so
startPublishedDate is set. `objective` frames the search as one step of the forecasting pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import Asset
from .sources import POST_COLUMNS, _timestamp

EXA_URL = "https://api.exa.ai/search"
log = logging.getLogger(__name__)


def fetch_news(assets: tuple[Asset, ...], api_key: str, days: int, num_results: int = 10) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for asset in assets:
        body = {
            "query": f"latest news about {asset.name} ({asset.ticker}) stock",
            "objective": (
                f"Find recent, price-relevant news about {asset.name} ({asset.ticker}) for a next-session "
                f"stock forecast. Rank reputable financial outlets and company announcements first; "
                f"exclude other companies with similar names and unrelated market roundups."
            ),
            "contents": {"highlights": True},
            "startPublishedDate": start,
        }
        if num_results != 10:
            body["numResults"] = num_results
        resp = requests.post(EXA_URL, headers={"x-api-key": api_key, "Content-Type": "application/json"},
                             json=body, timeout=60)
        if resp.status_code >= 400:
            log.error("Exa error for %s: HTTP %s %s", asset.ticker, resp.status_code, resp.text[:200])
            continue
        for result in resp.json().get("results", []):
            highlights = result.get("highlights") or []
            text = " ".join([result.get("title") or "", *highlights]).strip()
            ts = _timestamp(result.get("publishedDate"))
            if not text or ts is None:
                continue
            rows.append({
                "source": "exa_news",
                "exchange": asset.exchange,
                "ticker": asset.ticker,
                "text": text,
                "published_utc": ts,
                "url": result.get("url"),
                "author": result.get("author"),
                "likes": 0.0,
                "reposts": 0.0,
                "views": 0.0,
                "followers": 0.0,
            })
    return pd.DataFrame(rows, columns=POST_COLUMNS)
