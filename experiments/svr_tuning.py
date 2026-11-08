"""Tune an RBF SVR on the engineered feature set.

The earlier neighbourhood sprint only tried SVR on the compact feature set. This grid uses the
same 19 engineered features as the production PLS model, LODO + forward validation, and the
same-class price-only control in ``validate_engineered.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.model_sprint import load, make_model
from sentpred.features import ENGINEERED_MODEL_FEATURES

RESULTS = "experiments/results/svr_tuning.csv"


def predict(df, mode, params, lam, clip=0.05):
    dates = sorted(df["date"].unique())
    rows = []
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
        model = make_model("svr", params, ENGINEERED_MODEL_FEATURES)
        model.fit(train[ENGINEERED_MODEL_FEATURES].fillna(0), train["y_next_ret"])
        pred = np.clip(model.predict(test[ENGINEERED_MODEL_FEATURES].fillna(0)), -clip, clip)
        test["pred_price"] = test["close"] * (1 + lam * pred)
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def score(pred):
    err = (pred["pred_price"] - pred["next_close"]).abs()
    nerr = (pred["close"] - pred["next_close"]).abs()
    mape = (err / pred["next_close"]).mean() * 100
    nmape = (nerr / pred["next_close"]).mean() * 100
    direction = (np.sign(pred["pred_price"] - pred["close"]) == np.sign(pred["next_close"] - pred["close"]))[
        pred["next_close"] != pred["close"]].mean() * 100
    by = pd.DataFrame({"date": pred["date"], "d": (err - nerr) / pred["next_close"] * 100}).groupby("date")["d"].mean()
    rng = np.random.default_rng(1)
    boot = [rng.choice(by.values, size=len(by), replace=True).mean() for _ in range(3000)]
    return {
        "mape": mape, "skill": (1 - mape / nmape) * 100, "direction": direction,
        "delta_pct": by.mean(), "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
    }


def main():
    df = load()
    rows = []
    for scale in (2.0, 4.0, 6.0):
        for c in (1.0, 3.0, 5.0, 10.0):
            for epsilon in (0.002, 0.005, 0.01):
                for lam in (0.20, 0.25, 0.30, 0.35, 0.40):
                    params = {"c": c, "epsilon": epsilon, "gamma": "scale", "scale": scale}
                    for mode in ("lodo", "forward"):
                        result = score(predict(df, mode, params, lam))
                        result.update({"scale": scale, "c": c, "epsilon": epsilon,
                                       "gamma": "scale", "lam": lam, "mode": mode})
                        rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS, index=False)
    pivot = out.pivot(index=["scale", "c", "epsilon", "lam"], columns="mode",
                      values=["mape", "skill", "direction", "ci_high"]).reset_index()
    pivot.columns = ["_".join([str(x) for x in col if x]) for col in pivot.columns]
    pivot["worst_skill"] = pivot[["skill_lodo", "skill_forward"]].min(axis=1)
    print(f"{len(pivot)} configs -> {RESULTS}; both modes positive: {(pivot.worst_skill > 0).sum()}")
    print(pivot.sort_values("worst_skill", ascending=False).head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
