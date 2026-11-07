"""Lexicon-based sentiment for English, Traditional/Simplified Chinese and Korean.

Score is (pos - neg) / (pos + neg) in [-1, 1]; 0 when no sentiment term is found.
Deliberately transparent and dependency-free. Swap in a multilingual model for production.
"""

from __future__ import annotations

import re

from .config import Asset

POSITIVE = [
    "beat", "bullish", "buy", "upgrade", "surge", "rally", "record", "growth", "strong", "profit",
    "breakout", "outperform", "undervalued", "dividend", "gain", "rise", "soar", "win", "moon",
    "看好", "上涨", "大涨", "抄底", "加仓", "买入", "利好", "突破", "强势", "強勁", "反彈", "賺", "新高", "牛",
    "상승", "급등", "매수", "호재", "상한가", "신고가", "실적개선", "강세", "반등", "추천", "수익",
]
NEGATIVE = [
    "miss", "bearish", "sell", "downgrade", "plunge", "crash", "drop", "loss", "weak", "dump",
    "lawsuit", "fraud", "fine", "slump", "fall", "decline", "cut", "tank", "overvalued",
    "看淡", "下跌", "大跌", "暴跌", "割肉", "沽", "利淡", "套牢", "止蝕", "減持", "亏", "虧", "熊",
    "하락", "급락", "매도", "악재", "하한가", "신저가", "약세", "손실", "적자", "폭락", "공매도", "물려",
]

_ASCII_TERM = re.compile(r"^[a-z ]+$")


def _count(text: str, terms: list[str]) -> int:
    total = 0
    for term in terms:
        if _ASCII_TERM.match(term):
            total += len(re.findall(rf"\b{re.escape(term)}\b", text))
        else:
            total += text.count(term)
    return total


def score_text(text: str) -> tuple[float, int]:
    """Return (sentiment score in [-1, 1], number of sentiment hits)."""
    lowered = (text or "").lower()
    pos = _count(lowered, POSITIVE)
    neg = _count(lowered, NEGATIVE)
    hits = pos + neg
    if hits == 0:
        return 0.0, 0
    return (pos - neg) / hits, hits


def mentions_asset(text: str, asset: Asset) -> bool:
    """True if the text contains any of the asset's keywords or cashtags."""
    lowered = (text or "").lower()
    for keyword in asset.keywords:
        k = keyword.lower()
        if _ASCII_TERM.match(k) or re.fullmatch(r"[a-z0-9 $._-]+", k):
            if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", lowered):
                return True
        elif k in lowered:
            return True
    return False
