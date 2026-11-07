import numpy as np
import pandas as pd

from sentpred.evaluation import cluster_bootstrap


def _preds():
    dates = pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"])
    rows = []
    for d in dates:
        rows += [
            {"date": d, "close": 100.0, "next_close": 101.0, "y_next_ret": 0.01,
             "pred_naive": 0.0, "pred_ridge_price": 0.005, "pred_ridge_sent": 0.01},
            {"date": d, "close": 200.0, "next_close": 198.0, "y_next_ret": -0.01,
             "pred_naive": 0.0, "pred_ridge_price": -0.005, "pred_ridge_sent": -0.01},
        ]
    return pd.DataFrame(rows)


def test_bootstrap_returns_ordered_intervals():
    out = cluster_bootstrap(_preds(), n_boot=200, seed=3)
    assert set(out["model"]) == {"naive", "ridge_price", "ridge_sent"}
    assert (out["ci_low"] <= out["mean"] + 1e-9).all()
    assert (out["mean"] <= out["ci_high"] + 1e-9).all()
    assert out["mean"].notna().all()


def test_bootstrap_empty_input():
    out = cluster_bootstrap(pd.DataFrame(), n_boot=10)
    assert out.empty
