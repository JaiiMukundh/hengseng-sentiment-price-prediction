"""News sentiment input via Exa (https://exa.ai) neural search, restricted to the look-back window.

Request shape follows Exa's build-with-exa guidance: query + highlights, no highlight budgets,
no numResults/category unless needed. The 10-day window is a hard task requirement, so
startPublishedDate is set. `objective` frames the search as one step of the forecasting pipeline.

Network calls share one ``requests.Session`` with urllib3 retries and a small explicit
retry loop for 429/5xx. A single asset's failure is logged and skipped so the news stage
still returns whatever other assets produced.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Asset
from .sources import POST_COLUMNS, _timestamp

EXA_URL = "https://api.exa.ai/search"
log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5


class ExaError(RuntimeError):
    pass


def _build_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def _backoff(attempt: int) -> None:
    time.sleep(BACKOFF_BASE_S * (2**attempt))


def _post(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST one search, retrying 429/5xx and transient network errors."""
    last_exc: requests.RequestException | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = SESSION.post(
                EXA_URL,
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 >= MAX_ATTEMPTS:
                break
            log.warning("Exa request failed (%s); retrying", exc)
            _backoff(attempt)
            continue
        if resp.status_code in RETRY_STATUS and attempt + 1 < MAX_ATTEMPTS:
            log.warning("Exa returned HTTP %s; retrying", resp.status_code)
            _backoff(attempt)
            continue
        if resp.status_code >= 400:
            raise ExaError(f"Exa HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ExaError(f"Exa returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ExaError(f"Exa returned an unexpected payload: {type(payload).__name__}")
        return payload
    raise ExaError(f"Exa request failed after {MAX_ATTEMPTS} attempts: {last_exc}") from last_exc


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
        try:
            payload = _post(body, api_key)
        except ExaError as exc:
            log.error("Exa error for %s: %s", asset.ticker, exc)
            continue
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
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
