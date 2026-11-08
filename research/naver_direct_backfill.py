"""Research-only direct Naver backfill (no Apify cost). Not part of the submitted pipeline.

Fetches 90 days of the public Naver discussion API for 10 KRX assets, keeping up to
`per_day` posts per ticker per day. The production pipeline remains Apify-based.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://m.stock.naver.com/front-api/discussion/list"
ASSETS = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "035420": "NAVER",
    "035720": "Kakao",
    "005380": "Hyundai Motor",
    "373220": "LG Energy Solution",
    "006400": "Samsung SDI",
    "005490": "POSCO Holdings",
    "105560": "KB Financial",
    "068270": "Celltrion",
}
DAYS = 365
PER_DAY = 15
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(code: str, days: int = DAYS, per_day: int = PER_DAY) -> tuple[str, int, str]:
    cutoff = (datetime.now(timezone.utc) + timedelta(hours=9) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"})
    buckets: dict[str, list[dict]] = {}
    seen: set[str] = set()
    offset = None
    pages = 0
    while pages < 3000:
        params = {
            "discussionType": "domesticStock", "itemCode": code, "pageSize": 100,
            "isHolderOnly": "false", "excludesItemNews": "false", "isItemNewsOnly": "false",
        }
        if offset:
            params["offset"] = offset
        for attempt in range(3):
            try:
                r = session.get(BASE, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5 * (attempt + 1))
        else:
            break
        batch = data.get("result", {}).get("posts", [])
        if not batch:
            break
        for p in batch:
            day = (p.get("writtenAt") or "")[:10]
            if not day or p.get("id") in seen:
                continue
            seen.add(p.get("id"))
            buckets.setdefault(day, [])
            if len(buckets[day]) >= per_day:
                continue
            buckets[day].append({
                "id": p.get("id"), "title": p.get("title"),
                "content": p.get("contentSwReplacedButImg") or p.get("contentSwReplaced") or "",
                "author": (p.get("writer") or {}).get("nickname"),
                "date": p.get("writtenAt"), "likes": p.get("recommendCount") or 0,
                "views": p.get("viewCount") or 0,
                "url": f"https://m.stock.naver.com/domestic/stock/{code}/discussion",
            })
        oldest = batch[-1].get("writtenAt") or ""
        if oldest < cutoff:
            break
        nxt = data.get("result", {}).get("lastOffset")
        if not nxt or nxt == offset:
            break
        offset = nxt
        pages += 1
        time.sleep(0.12)
    posts = [p for day in sorted(buckets) for p in buckets[day]]
    path = OUT / f"naver_direct_{days}d_{code}.json"
    path.write_text(json.dumps(posts, ensure_ascii=False), encoding="utf-8")
    return code, len(posts), str(path)


def main():
    print(f"backfill {len(ASSETS)} assets x {DAYS} days, max {PER_DAY}/day")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, code): code for code in ASSETS}
        for fut in as_completed(futures):
            code, n, path = fut.result()
            print(f"{code} {ASSETS[code]}: {n} posts -> {path}")


if __name__ == "__main__":
    main()
