"""News sentiment input via Exa (https://exa.ai) neural search, restricted to the look-back window."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import Asset
from .sources import POST_COLUMNS, _timestamp

EXA_URL = "https://api.exa.ai/search"
log = logging.getLogger(__name__)


def fetch_news(assets: tuple[Asset, ...], api_key: str, days: int, num_results: int) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for asset in assets:
        body = {
            "query": f"{asset.name} ({asset.ticker}) stock news",
            "type": "auto",
            "category": "news",
            "numResults": num_results,
            "startPublishedDate": start,
            "contents": {"highlights": {"maxCharacters": 1000}},
        }
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
