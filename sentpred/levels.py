"""Support / resistance levels: classic daily pivots plus clustered swing highs and lows."""

from __future__ import annotations

import pandas as pd


def pivot_points(last: pd.Series) -> dict[str, float]:
    h, l, c = float(last["high"]), float(last["low"]), float(last["close"])
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l)}


def swing_levels(bars: pd.DataFrame, lookback: int = 120, order: int = 5, tol: float = 0.015) -> dict[str, list[dict]]:
    """Find swing pivots in the last `lookback` bars, cluster nearby ones and split around the last close."""
    window = bars.tail(lookback).reset_index(drop=True)
    highs, lows = window["high"].to_numpy(), window["low"].to_numpy()
    last_close = float(window["close"].iloc[-1])

    pivots: list[float] = []
    for i in range(order, len(window) - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            pivots.append(float(highs[i]))
        if lows[i] == lows[i - order:i + order + 1].min():
            pivots.append(float(lows[i]))

    clusters = _cluster(sorted(pivots), tol)
    supports = sorted((c for c in clusters if c["level"] < last_close), key=lambda c: -c["level"])[:2]
    resistances = sorted((c for c in clusters if c["level"] > last_close), key=lambda c: c["level"])[:2]
    return {"support": supports, "resistance": resistances}


def _cluster(levels: list[float], tol: float) -> list[dict]:
    clusters: list[list[float]] = []
    for level in levels:
        if clusters and level <= clusters[-1][0] * (1 + tol):
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [{"level": round(sum(c) / len(c), 4), "touches": len(c)} for c in clusters]
