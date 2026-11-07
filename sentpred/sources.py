"""Actor input builders and output normalizers for each social source.

Each normalizer maps raw Apify items onto one common post schema:
source, exchange, ticker, text, published_utc, url, author, likes, reposts, views, followers.
Field names differ between Actors and versions, so lookups try several candidate keys.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import pandas as pd

from .config import Asset
from .sentiment import mentions_asset

log = logging.getLogger(__name__)

POST_COLUMNS = [
    "source", "exchange", "ticker", "text", "published_utc", "url",
    "author", "likes", "reposts", "views", "followers",
]

# Runs inside apify/playwright-scraper. Naver's board is rendered client-side, so a browser is needed.
# Selectors reflect the board's classic table layout and must be re-checked on a live run.
NAVER_PAGE_FUNCTION = r"""
async function pageFunction({ page, request }) {
    await page.waitForSelector('table.type2', { timeout: 20000 }).catch(() => {});
    const posts = await page.$$eval('table.type2 tr', (rows) => rows.map((tr) => {
        const a = tr.querySelector('td.title a');
        if (!a) return null;
        const tds = tr.querySelectorAll('td');
        const num = (i) => parseInt(((tds[i] && tds[i].textContent) || '').replace(/[^0-9]/g, ''), 10) || 0;
        const dateCell = tr.querySelector('td.tah');
        return {
            title: a.textContent.trim(),
            url: a.href,
            date: dateCell ? dateCell.textContent.trim() : null,
            author: tds[2] ? tds[2].textContent.trim() : null,
            views: num(3),
            likes: num(4),
        };
    }).filter(Boolean));
    return { url: request.url, posts };
}
"""


def build_run_input(source: str, cfg: dict[str, Any], exchange: str, assets: tuple[Asset, ...], social_days: int) -> dict[str, Any] | None:
    """Return the Actor input for one source/exchange pair, or None if it cannot run yet."""
    if source == "lihkg":
        keywords = sorted({k for a in assets for k in a.keywords if not k.startswith("$")})
        return {
            "mode": "search",
            "keywords": keywords,
            "maxItems": cfg["max_items"],
            "order": "desc_reply_time",
            "dateFrom": cfg.get("date_from", f"{social_days}d"),
            "aiLabels": False,
        }
    if source == "telegram":
        channels = cfg.get("channels", {}).get(exchange, [])
        return {
            "keywords": sorted({k for a in assets for k in a.keywords}),
            "discoverChannels": True,
            "channels": channels,
            "maxChannelsPerKeyword": 20,
            "maxItems": cfg.get("max_items", 200),
            "searchDepth": "fast",
            "afterDate": cfg.get("after_date", f"{social_days} days"),
            "discoveryLanguage": cfg.get("discovery_language", "en"),
            "discoveryCountry": cfg.get("discovery_country", "us"),
            "strictMatch": True,
        }
    if source == "naver_discussion":
        urls = [
            {"url": cfg["board_url"].format(code=a.code, page=p)}
            for a in assets
            for p in range(1, cfg["pages"] + 1)
        ]
        return {
            "startUrls": urls,
            "pageFunction": NAVER_PAGE_FUNCTION,
            "proxyConfiguration": {"useApifyProxy": True},
        }
    if source == "blind":
        urls = []
        for a in assets:
            query = next((k for k in a.keywords if not k.isascii()), a.name)
            urls.append({"url": cfg["search_url"].format(query=quote(query))})
        return {"startUrls": urls, "maxItems": cfg["max_items"], "maxPages": 3, "scrapeComments": False}
    raise ValueError(f"unknown source: {source}")


def normalize(source: str, exchange: str, items: list[dict[str, Any]], assets: tuple[Asset, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        children = item["posts"] if isinstance(item.get("posts"), list) else [item]
        for child in children:
            text = _text(child)
            ts = _timestamp(_pick(child, "published_utc", "date", "createdAt", "created_at", "timestamp",
                                  "time", "postedAt", "publishedAt", "published", "createdTime"))
            if not text or ts is None:
                continue
            base = {
                "source": source,
                "exchange": exchange,
                "text": text,
                "published_utc": ts,
                "url": _pick(child, "url", "link", "postUrl", "permalink"),
                "author": _pick(child, "author", "username", "writer", "authorName", "user", "sender"),
                "likes": _num(_pick(child, "likes", "likeCount", "like_count", "upvotes", "up",
                                    "reactions", "recommend", "good", "sympathy")),
                "reposts": _num(_pick(child, "shares", "reposts", "forwards", "forward_count",
                                      "retweets", "repostCount")),
                "views": _num(_pick(child, "views", "viewCount", "view_count", "readCount", "read_count", "hits")),
                "followers": _num(_pick(child, "followers", "authorFollowers", "followersCount",
                                        "subscribers", "memberCount")),
            }
            for asset in assets:
                if mentions_asset(text, asset):
                    rows.append({**base, "ticker": asset.ticker})
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def _text(child: dict[str, Any]) -> str:
    parts = [_pick(child, "title"), _pick(child, "content", "text", "message", "body", "comment", "description")]
    return " ".join(str(p).strip() for p in parts if p)


def _pick(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", []):
            return value
    return None


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


_NAVER_DATE = re.compile(r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}(:\d{2})?$")


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        number = float(value)
        return pd.Timestamp(number, unit="ms" if number > 1e11 else "s", tz="UTC")
    if isinstance(value, str) and _NAVER_DATE.match(value.strip()):
        fmt = "%Y.%m.%d %H:%M:%S" if value.count(":") == 2 else "%Y.%m.%d %H:%M"
        local = datetime.strptime(value.strip(), fmt)
        return pd.Timestamp(local).tz_localize("Asia/Seoul").tz_convert("UTC")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
