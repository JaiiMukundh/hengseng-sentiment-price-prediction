"""Tune the engineered-feature Ridge around the promising forward result."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.model_sprint import PRICE_FEATURES, SentimentScaler, load


def add_extra(df):
    g = df.groupby("ticker", sort=False)
    df = df.copy()
    df["social_pos_chg"] = g["social_pos_share"].diff().fillna(0)
    df["news_sent_ma3"] = g["news_sent_wmean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["news_div_chg"] = g["news_social_div"].diff().fillna(0)
    df["sent_x_vol"] = df["social_sent_wmean"] * df["vol_10"]
    roll = g["social_n_log"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    std = g["social_n_log"].transform(lambda s: s.rolling(10, min_periods=3).std()).replace(0, np.nan)
    df["mention_z"] = ((df["social_n_log"] - roll) / std).fillna(0)
    df["sent_rank"] = df.groupby("date")["social_sent_wmean"].rank(pct=True)
    return df


def predict(df, cols, alpha, scale, lam, mode, clip=None):
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
        pre = SentimentScaler(scale, tuple(cols))
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(pre, Ridge(alpha=alpha))
        model.fit(train[cols].fillna(0), train["y_next_ret"])
        pred = model.predict(test[cols].fillna(0))
        if clip is not None:
            pred = np.clip(pred, -clip, clip)
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
    boot = [rng.choice(by.values, size=len(by), replace=True).mean() for _ in range(2000)]
    return mape, nmape, (1 - mape / nmape) * 100, direction, by.mean(), np.percentile(boot, 97.5)


def main():
    df = add_extra(load())
    base = PRICE_FEATURES + [
        "social_sent_wmean", "social_pos_share", "social_neg_share", "news_sent_wmean",
        "social_n_log", "has_social", "sent_chg", "sent_ma3", "sent_z", "mention_chg",
        "news_chg", "news_social_div",
    ]
    extra = base + ["social_pos_chg", "news_sent_ma3", "news_div_chg", "sent_x_vol", "mention_z", "sent_rank"]
    rows = []
    for name, cols in (("eng", base), ("eng_plus", extra)):
        for alpha in (1, 10, 50, 100):
            for scale in (1, 2, 4, 6, 8):
                for lam in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
                    for clip in (None, 0.05):
                        for mode in ("lodo", "forward"):
                            p = predict(df, cols, alpha, scale, lam, mode, clip)
                            mape, nmape, skill, direction, delta, ci_high = score(p)
                            rows.append({"features": name, "alpha": alpha, "scale": scale, "lam": lam,
                                         "clip": clip, "mode": mode, "mape": mape, "naive": nmape,
                                         "skill": skill, "direction": direction, "delta": delta,
                                         "ci_high": ci_high})
    out = pd.DataFrame(rows)
    out.to_csv("experiments/results/engineered_tuning.csv", index=False)
    # require both modes positive; then sort by worst-mode skill
    pivot = out.pivot_table(index=["features", "alpha", "scale", "lam", "clip"], columns="mode",
                            values=["skill", "direction", "ci_high"]).reset_index()
    pivot.columns = ["_".join([str(x) for x in c if x]) for c in pivot.columns]
    both = pivot[(pivot.skill_lodo > 0) & (pivot.skill_forward > 0)].copy()
    both["worst_skill"] = both[["skill_lodo", "skill_forward"]].min(axis=1)
    print("configs", len(pivot), "positive in both", len(both))
    print(both.sort_values("worst_skill", ascending=False).head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
