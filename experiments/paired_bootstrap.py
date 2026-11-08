"""Reproducible paired session-bootstrap for the current short-history model.

Run after `python -m sentpred model` (or `report`), which writes data/processed/dataset.csv.
Writes reports/live/paired_bootstrap.csv. The interval is for model percentage error minus
naive percentage error: negative values favor the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentpred import model
from sentpred.config import load_config


def score(pred: pd.DataFrame, column: str, n_boot: int = 5000) -> dict:
    err = (pred["close"] * (1 + pred[column]) - pred["next_close"]).abs()
    nerr = (pred["close"] - pred["next_close"]).abs()
    mape = (err / pred["next_close"]).mean() * 100
    nmape = (nerr / pred["next_close"]).mean() * 100
    direction = (np.sign(pred[column]) == np.sign(pred["y_next_ret"]))[
        pred["y_next_ret"] != 0].mean() * 100
    by = pd.DataFrame({"date": pred["date"], "d": (err - nerr) / pred["next_close"] * 100}).groupby("date")["d"].mean()
    rng = np.random.default_rng(1)
    boot = [rng.choice(by.values, size=len(by), replace=True).mean() for _ in range(n_boot)]
    return {
        "n": len(pred), "days": len(by), "mape_pct": mape, "naive_mape_pct": nmape,
        "skill_vs_naive_pct": (1 - mape / nmape) * 100,
        "direction_acc_pct": direction, "delta_pct": by.mean(),
        "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "config.yaml"))
    m = cfg.raw["model"]
    data = pd.read_csv(root / "data" / "processed" / "dataset.csv", parse_dates=["date"])
    lodo, _ = model.leave_one_session_out(
        data, m["sentiment_alpha"], m["sentiment_shrinkage"], m["sentiment_model"],
        m["sentiment_scale"], m.get("sentiment_clip"), m["sentiment_components"])
    forward = model.forward_sentiment_predictions(
        data, m["sentiment_alpha"], m["sentiment_shrinkage"], m["sentiment_model"],
        m["sentiment_scale"], m.get("sentiment_clip"), m["sentiment_components"])
    rows = []
    for split, pred in (("lodo", lodo), ("chronological", forward)):
        for name, column in (("price_only", "pred_ridge_price"), ("sentiment", "pred_ridge_sent")):
            rows.append({"split": split, "model": name, **score(pred, column)})
    out = pd.DataFrame(rows)
    output = root / "reports" / "live" / "paired_bootstrap.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.round(4).to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
