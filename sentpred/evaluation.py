"""Evaluation helpers: session-clustered bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap(preds: pd.DataFrame, n_boot: int = 1000, seed: int = 1) -> pd.DataFrame:
    """Resample whole trading sessions (not individual rows) to respect the panel structure."""
    if preds.empty:
        return pd.DataFrame(columns=["model", "metric", "mean", "ci_low", "ci_high"])
    dates = np.array(sorted(preds["date"].unique()))
    rng = np.random.default_rng(seed)
    model_cols = {"naive": "pred_naive", "ridge_price": "pred_ridge_price", "ridge_sent": "pred_ridge_sent"}
    rows: list[dict] = []
    for model, col in model_cols.items():
        if col not in preds:
            continue
        maples, directions = [], []
        for _ in range(n_boot):
            sampled = rng.choice(dates, size=len(dates), replace=True)
            sample = pd.concat([preds[preds["date"] == d] for d in sampled], ignore_index=True)
            predicted = sample["close"] * (1 + sample[col])
            maples.append(float(((predicted - sample["next_close"]).abs() / sample["next_close"]).mean() * 100))
            if model != "naive":
                hit = np.sign(sample[col]) == np.sign(sample["y_next_ret"])
                directions.append(float(hit[sample["y_next_ret"] != 0].mean() * 100))
        for metric, values in (("mape_pct", maples), ("direction_acc_pct", directions)):
            if not values:
                continue
            rows.append({
                "model": model,
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
            })
    return pd.DataFrame(rows)
