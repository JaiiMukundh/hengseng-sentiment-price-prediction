"""Lexicon-based sentiment for English, Traditional/Simplified Chinese and Korean.

Score is (pos - neg) / (pos + neg) in [-1, 1]; 0 when no sentiment term is found.
Longest match wins (so "下跌" is not double-counted as "跌"), and a short negation window
flips nearby terms ("not good", "唔好", "안 좋다"). Deliberately transparent and dependency-free.
"""

from __future__ import annotations

import re

from .config import Asset

POSITIVE = [
    "beat", "bullish", "buy", "good", "upgrade", "surge", "rally", "record", "growth", "strong",
    "profit", "breakout", "outperform", "undervalued", "dividend", "gain", "rise", "soar", "win", "moon",
    "看好", "上涨", "大涨", "抄底", "加仓", "买入", "利好", "突破", "强势", "強勁", "反彈", "賺", "新高", "牛",
    "上升", "上漲", "買入", "強勢", "回升", "做好", "向好", "樂觀", "乐观", "賺錢", "赚钱", "好消息",
    "상승", "급등", "매수", "호재", "상한가", "신고가", "실적개선", "강세", "반등", "추천", "수익",
]
NEGATIVE = [
    "miss", "bearish", "sell", "downgrade", "plunge", "crash", "drop", "loss", "weak", "bad", "dump",
    "lawsuit", "fraud", "fine", "slump", "fall", "decline", "cut", "tank", "overvalued",
    "看淡", "下跌", "大跌", "暴跌", "割肉", "沽", "利淡", "套牢", "止蝕", "減持", "亏", "虧", "熊",
    "하락", "급락", "매도", "악재", "하한가", "신저가", "약세", "손실", "적자", "폭락", "공매도", "물려",
]

# English negations must be whole words ("not" must not match "nothing" or "now").
_EN_NEGATIONS = (
    "not", "no", "never", "without", "isn't", "aren't", "wasn't", "won't", "can't", "don't", "doesn't",
)
# Korean negations only when standalone; "안" must not match inside "안정적" (stable).
_KO_STANDALONE_NEGATIONS = ("안", "못")
# CJK negations are safe as substrings (no word delimiters in written Chinese/Korean).
_ZH_NEGATIONS = ("不", "沒", "没", "別", "别", "無", "无", "未", "唔", "唔好", "唔係")
_KO_SUBSTRING_NEGATIONS = ("없", "아니", "말고", "아닌")
_CJK_NEGATIONS = (*_ZH_NEGATIONS, *_KO_SUBSTRING_NEGATIONS)
NEGATIONS = [*_EN_NEGATIONS, *_ZH_NEGATIONS, *_KO_STANDALONE_NEGATIONS, *_KO_SUBSTRING_NEGATIONS]

_ASCII_TERM = re.compile(r"^[a-z ]+$")
_EN_NEGATION_RE = re.compile(
    r"(?<![a-z0-9_])(?:" + "|".join(re.escape(n) for n in _EN_NEGATIONS) + r")(?![a-z0-9_])"
)
_KO_STANDALONE_NEGATION_RE = re.compile(
    r"(?<![가-힣])(?:" + "|".join(re.escape(n) for n in _KO_STANDALONE_NEGATIONS) + r")(?![가-힣])"
)
_CLAUSE_PUNCTUATION = re.compile(r"[,.!?;:，。！？；：、…\n]")


def _is_negated(text: str, start: int, window: int = 14) -> bool:
    context = _CLAUSE_PUNCTUATION.split(text[max(0, start - window):start])[-1]
    if _EN_NEGATION_RE.search(context) or _KO_STANDALONE_NEGATION_RE.search(context):
        return True
    return any(neg in context for neg in _CJK_NEGATIONS)


def _score(text: str) -> tuple[int, int]:
    matches: list[tuple[int, int, int]] = []
    for term, polarity in [(t, 1) for t in POSITIVE] + [(t, -1) for t in NEGATIVE]:
        if _ASCII_TERM.match(term):
            spans = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(term)}\b", text)]
        else:
            spans, start = [], 0
            while True:
                idx = text.find(term, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(term)))
                start = idx + len(term)
        for start, end in spans:
            sign = -polarity if _is_negated(text, start) else polarity
            matches.append((start, end, sign))

    # Longest match wins, so "下跌" is not also counted as "跌".
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    pos = neg = 0
    covered: list[tuple[int, int]] = []
    for start, end, sign in matches:
        if any(start < c_end and end > c_start for c_start, c_end in covered):
            continue
        covered.append((start, end))
        if sign > 0:
            pos += 1
        else:
            neg += 1
    return pos, neg


def score_text(text: str) -> tuple[float, int]:
    """Return (sentiment score in [-1, 1], number of sentiment hits)."""
    lowered = (text or "").lower()
    pos, neg = _score(lowered)
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
