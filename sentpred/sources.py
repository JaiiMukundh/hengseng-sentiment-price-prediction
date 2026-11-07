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

# Runs inside apify/playwright-scraper. The board is a client-rendered Next.js app, so a browser
# is required. Selectors were verified against the live page (Nov 2026); Naver hashes the class
# suffixes, so match on the stable prefix. Like counts are Lottie-animated and not readable, so
# Naver engagement uses comments and views.
NAVER_PAGE_FUNCTION = r"""
async function pageFunction({ page, request }) {
    await page.waitForSelector('[class*="DiscussionPostWrapper_article"]', { timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const posts = await page.$$eval('[class*="DiscussionPostWrapper_article"]', (lis, url) => lis.map((li) => {
        const txt = (sel) => { const e = li.querySelector(sel); return e ? e.innerText.trim() : null; };
        const num = (s) => { const m = (s || '').replace(/,/g, '').match(/(\d+)/); return m ? parseInt(m[1], 10) : 0; };
        const rtext = ((li.querySelector('[class*="ReactionBox_article"]') || {}).innerText || '').replace(/\s+/g, ' ');
        const comment = rtext.match(/댓글\s*([\d,]+)/);
        const view = rtext.match(/조회\s*([\d,]+)/);
        return {
            id: li.id || null,
            title: txt('[class*="UserPostContents_title"]'),
            content: txt('[class*="UserPostContents_description"]'),
            author: (txt('[class*="Profile_nickname"]') || '').replace('작성자', '').trim(),
            timeText: (txt('[class*="Profile_time"]') || '').replace('작성일', '').replace(/\s+/g, ' ').trim(),
            likes: 0,
            comments: comment ? num(comment[1]) : 0,
            views: view ? num(view[1]) : 0,
            url: url,
            scrapedAt: new Date().toISOString(),
        };
    }), request.url);
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
        discovery = cfg.get("discovery", {}).get(exchange, {})
        return {
            "keywords": sorted({k for a in assets for k in a.keywords}),
            "discoverChannels": True,
            "channels": channels,
            "maxChannelsPerKeyword": 20,
            "maxItems": cfg.get("max_items", 200),
            "searchDepth": "fast",
            "afterDate": cfg.get("after_date", f"{social_days} days"),
            "discoveryLanguage": discovery.get("language", "en"),
            "discoveryCountry": discovery.get("country", "us"),
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
            "maxConcurrency": 1,
            "maxRequestRetries": 2,
            "pageLoadTimeoutSecs": 90,
            "waitUntil": "domcontentloaded",
        }
    if source == "blind":
        urls = [{"url": cfg["search_url"].format(query=quote(a.name))} for a in assets]
        return {"startUrls": urls, "maxItems": cfg["max_items"], "maxPages": 3, "scrapeComments": False}
    raise ValueError(f"unknown source: {source}")


def normalize(source: str, exchange: str, items: list[dict[str, Any]], assets: tuple[Asset, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        children = item["posts"] if isinstance(item.get("posts"), list) else [item]
        ticker_hint = _ticker_from_url(item.get("url"), assets, source)
        for child in children:
            text = _text(child)
            if source == "naver_discussion":
                ts = _naver_time(child.get("timeText") or child.get("date"), child.get("scrapedAt"))
            else:
                naive_tz = "America/New_York" if source == "blind" else "UTC"
                ts = _timestamp(_pick(child, "published_utc", "date", "createdAt", "created_at", "timestamp",
                                      "time", "postedAt", "publishedAt", "published", "createdTime"), naive_tz)
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
                                    "reactions", "totalReactions", "recommend", "good", "sympathy")),
                "reposts": _num(_pick(child, "shares", "reposts", "forwards", "forward_count",
                                      "retweets", "repostCount")),
                "views": _num(_pick(child, "views", "viewCount", "view_count", "readCount", "read_count", "hits")),
                "followers": _num(_pick(child, "followers", "authorFollowers", "followersCount",
                                        "channelSubscribers", "subscribers", "memberCount")),
            }
            for asset in assets:
                if ticker_hint and asset.ticker == ticker_hint:
                    rows.append({**base, "ticker": asset.ticker})
                elif not ticker_hint and mentions_asset(text, asset):
                    rows.append({**base, "ticker": asset.ticker})
    df = pd.DataFrame(rows, columns=POST_COLUMNS)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["ticker", "text", "published_utc"])


def _text(child: dict[str, Any]) -> str:
    parts = [_pick(child, "title"), _pick(child, "content", "text", "message", "body", "comment", "description")]
    return " ".join(str(p).strip() for p in parts if p)


def _ticker_from_url(url: Any, assets: tuple[Asset, ...], source: str) -> str | None:
    """Naver boards are per-ticker, so the URL code is authoritative (titles rarely name the stock)."""
    if source != "naver_discussion" or not url:
        return None
    match = re.search(r"/stock/(\d{6})", str(url))
    if not match:
        return None
    return next((a.ticker for a in assets if a.code == match.group(1)), None)


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


def _naver_time(text: str | None, scraped_at: str | None) -> pd.Timestamp | None:
    """Parse Naver's Korean relative/absolute time labels into UTC.

    Labels look like "1분 전", "3시간 전", "2일 전", "방금 전", "오전 4:58", "11.05".
    """
    if not text:
        return None
    now = pd.Timestamp(scraped_at) if scraped_at else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    t = re.sub(r"\s+", " ", str(text).replace("작성일", "")).strip()
    if _NAVER_DATE.match(t):
        return _timestamp(t)

    rel = re.match(r"(\d+)\s*(분|시간|일)\s*전", t)
    if rel:
        n, unit = int(rel.group(1)), rel.group(2)
        return now - pd.Timedelta(**{{"분": "minutes", "시간": "hours", "일": "days"}[unit]: n})
    if "방금" in t:
        return now
    clock = re.match(r"(오전|오후)\s*(\d{1,2}):(\d{2})", t)
    if clock:
        kst_today = now.tz_convert("Asia/Seoul").normalize()
        hour = int(clock.group(2)) % 12 + (12 if clock.group(1) == "오후" else 0)
        local = kst_today + pd.Timedelta(hours=hour, minutes=int(clock.group(3)))
        return local.tz_convert("UTC")
    full = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", t)
    if full:
        return pd.Timestamp(f"{full.group(1)}-{full.group(2)}-{full.group(3)}").tz_localize("Asia/Seoul").tz_convert("UTC")
    short = re.match(r"(\d{2})\.(\d{2})", t)
    if short:
        year = now.tz_convert("Asia/Seoul").year
        return pd.Timestamp(f"{year}-{short.group(1)}-{short.group(2)}").tz_localize("Asia/Seoul").tz_convert("UTC")
    return _timestamp(t)


def _timestamp(value: Any, naive_tz: str = "UTC") -> pd.Timestamp | None:
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
    if ts.tzinfo is None:
        try:
            return ts.tz_localize(naive_tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")
        except (ValueError, TypeError):
            return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
