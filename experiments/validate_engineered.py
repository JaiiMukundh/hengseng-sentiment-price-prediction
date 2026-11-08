"""Validate the final engineered-feature model against naive, price-only and ablations.

Primary check: pooled leave-one-session-out on all 20 sentiment sessions (190 rows).
Secondary check: forward chaining on the last 12 sessions (120 rows).
The paired bootstrap resamples whole sessions, because rows inside a session share shocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.model_sprint import SETS, load, make_model
from sentpred.features import ENGINEERED_MODEL_FEATURES, PRICE_FEATURES


def predict(df, cols, model_name, params, lam, mode, clip=None):
    dates = sorted(df["date"].unique())
    out = []
    for i, date in enumerate(dates):
        if mode == "forward":
            if i < 8:
                continue
            train = df[df["date"] <= dates[i - 2]]
        else:
            train = df[df["date"] != date]
        test = df[df["date"] == date].copy()
        if len(train) < 30 or len(test) < 3:
            continue
        model = make_model(model_name, params, cols)
        model.fit(train[cols].fillna(0), train["y_next_ret"])
        pred = model.predict(test[cols].fillna(0))
        if clip is not None:
            pred = np.clip(pred, -clip, clip)
        test["pred_price"] = test["close"] * (1 + lam * pred)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def score(pred):
    err = (pred["pred_price"] - pred["next_close"]).abs()
    nerr = (pred["close"] - pred["next_close"]).abs()
    mape = (err / pred["next_close"]).mean() * 100
    nmape = (nerr / pred["next_close"]).mean() * 100
    direction = (np.sign(pred["pred_price"] - pred["close"]) == np.sign(pred["next_close"] - pred["close"]))[
        pred["next_close"] != pred["close"]].mean() * 100
    by = pd.DataFrame({"date": pred["date"], "d": (err - nerr) / pred["next_close"] * 100}).groupby("date")["d"].mean()
    rng = np.random.default_rng(1)
    boot = [rng.choice(by.values, size=len(by), replace=True).mean() for _ in range(5000)]
    return {
        "n": len(pred), "days": len(by), "mape": mape, "naive": nmape,
        "skill": (1 - mape / nmape) * 100, "direction": direction,
        "delta_pct": by.mean(), "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
    }


def main():
    df = load()
    candidates = [
        # name, feature set, model, params, lambda, clip
        ("final_pls6", ENGINEERED_MODEL_FEATURES, "pls",
         {"components": 6, "scale": 1.0}, 0.30, 0.05),
        ("pls6_price_only", PRICE_FEATURES, "pls",
         {"components": 6, "scale": 1.0}, 0.30, 0.05),
        ("final_engineered_ridge", ENGINEERED_MODEL_FEATURES, "ridge",
         {"alpha": 10, "scale": 8.0}, 0.40, 0.05),
        ("engineered_scale4", ENGINEERED_MODEL_FEATURES, "ridge", {"alpha": 10, "scale": 4.0}, 0.40, 0.05),
        ("engineered_scale1", ENGINEERED_MODEL_FEATURES, "ridge", {"alpha": 10, "scale": 1.0}, 0.40, 0.05),
        ("engineered_no_clip", ENGINEERED_MODEL_FEATURES, "ridge", {"alpha": 10, "scale": 8.0}, 0.40, None),
        ("engineered_lam030", ENGINEERED_MODEL_FEATURES, "ridge", {"alpha": 10, "scale": 8.0}, 0.30, 0.05),
        ("compact_ridge", SETS["compact"], "ridge", {"alpha": 10, "scale": 8.0}, 0.40, 0.05),
        ("price_only_ridge", SETS["price"], "ridge", {"alpha": 10, "scale": 1.0}, 0.40, None),
        ("small_mlp_lbfgs", ENGINEERED_MODEL_FEATURES, "mlp",
         {"hidden": (16,), "alpha": 1.0, "scale": 8.0, "solver": "lbfgs", "max_iter": 3000}, 0.40, 0.05),
        ("small_mlp_adam", ENGINEERED_MODEL_FEATURES, "mlp",
         {"hidden": (16, 8), "alpha": 1.0, "scale": 8.0, "solver": "adam", "lr": 0.001,
          "early_stopping": True, "max_iter": 3000}, 0.40, 0.05),
    ]
    rows = []
    for name, cols, model_name, params, lam, clip in candidates:
        for mode in ("lodo", "forward"):
            pred = predict(df, cols, model_name, params, lam, mode, clip)
            result = score(pred)
            result.update({"candidate": name, "mode": mode, "model": model_name,
                           "scale": params.get("scale"), "lam": lam, "clip": clip})
            rows.append(result)
            print(f"{mode:7s} {name:20s} MAPE={result['mape']:.3f}% naive={result['naive']:.3f}% "
                  f"skill={result['skill']:+.2f}% dir={result['direction']:.1f}% "
                  f"delta={result['delta_pct']:+.3f}pp CI[{result['ci_low']:+.3f},{result['ci_high']:+.3f}]")
    pd.DataFrame(rows).to_csv("experiments/results/engineered_validation.csv", index=False)


if __name__ == "__main__":
    main()
