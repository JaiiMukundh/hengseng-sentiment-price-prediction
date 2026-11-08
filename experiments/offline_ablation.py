"""Cached-data source ablation on identical chronological folds. No network calls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentpred import align, features, model
from sentpred.config import load_config


def run(base: Path, output: Path) -> pd.DataFrame:
    cfg = load_config(str(base / "config.yaml"))
    processed = base / "data" / "processed"
    full = pd.read_csv(processed / "dataset.csv", parse_dates=["date"])
    posts = pd.read_csv(processed / "social_posts.csv", parse_dates=["published_utc"])
    news = pd.read_csv(processed / "news.csv", parse_dates=["published_utc"])
    prices = pd.read_csv(processed / "prices.csv", parse_dates=["date"])
    covered = full.loc[(full["has_social"] + full["has_news"] > 0)
                       & full["y_next_ret"].notna(), "date"].unique()
    audit_rows = []
    for feed, frame in (("social", posts), ("news", news)):
        times = pd.to_datetime(frame["published_utc"], utc=True)
        audit_rows.append({
            "feed": feed, "rows": len(frame),
            "verified_first_seen_rows": int(frame["first_seen_utc"].notna().sum())
            if "first_seen_utc" in frame else 0,
            "midnight_utc_timestamps": int(((times.dt.hour == 0) & (times.dt.minute == 0)
                                              & (times.dt.second == 0)).sum()),
            "future_publication_timestamps": int((times > pd.Timestamp.now(tz="UTC")).sum()),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(output.with_name("availability_audit.csv"), index=False)
    calendars = {}
    for exchange in cfg.exchanges:
        ex = cfg.exchange(exchange)
        dates = prices.loc[prices["exchange"] == exchange, "date"]
        calendars[exchange] = (ex["tz"], ex["close"], align.session_calendar(dates))
    coverage_rows = []
    panel_dates = pd.DatetimeIndex(covered)
    for source, group in posts.groupby("source"):
        agg = align.aggregate(group, calendars, "social_")
        for asset in cfg.assets:
            matched = group[group["ticker"] == asset.ticker]
            sessions = agg.loc[(agg["ticker"] == asset.ticker)
                               & agg["date"].isin(panel_dates), "date"].nunique()
            coverage_rows.append({"exchange": asset.exchange, "ticker": asset.ticker,
                                  "source": source, "posts": len(matched),
                                  "covered_sessions": int(sessions),
                                  "panel_sessions": len(panel_dates),
                                  "session_coverage_pct": round(100 * sessions / len(panel_dates), 1)})
    pd.DataFrame(coverage_rows).to_csv(output.with_name("source_coverage.csv"), index=False)

    social_cols = [c for c in full if c.startswith("social_")]
    news_cols = [c for c in full if c.startswith("news_")]
    cfg_model = cfg.raw["model"]
    rows = []
    for name in ("all", "exa_only", "social_only", "naver_only", "telegram_only", "lihkg_only"):
        data = full.copy()
        if name in ("exa_only", "naver_only", "telegram_only", "lihkg_only"):
            data[social_cols] = 0.0
            data["has_social"] = 0.0
        if name in ("social_only", "naver_only", "telegram_only", "lihkg_only"):
            data[news_cols] = 0.0
            data["has_news"] = 0.0
        if name.endswith("_only") and name.split("_")[0] in ("naver", "telegram", "lihkg"):
            source = {"naver": "naver_discussion", "telegram": "telegram", "lihkg": "lihkg"}[name.split("_")[0]]
            agg = align.aggregate(posts[posts["source"] == source], calendars, "social_")
            if not agg.empty:
                data = data.merge(agg, on=["ticker", "date"], how="left", suffixes=("", "_src"))
                for col in social_cols:
                    src_col = f"{col}_src"
                    if src_col in data:
                        data[col] = data.pop(src_col).fillna(0.0)
                data["has_social"] = (data["social_n"] > 0).astype(float)
                data["social_n_log"] = np.log1p(data["social_n"])
        if name != "all":
            data = features.add_sentiment_dynamics(data, covered_dates=covered)
        preds = model.forward_sentiment_predictions(
            data, cfg_model["sentiment_alpha"], cfg_model["sentiment_shrinkage"],
            cfg_model["sentiment_model"], cfg_model["sentiment_scale"],
            cfg_model["sentiment_clip"], cfg_model["sentiment_components"], panel_dates=covered)
        for exchange, frame in (("both", preds), *preds.groupby("exchange")):
            if frame.empty:
                continue
            metric = model.evaluate(frame)
            for _, m in metric[metric["subset"] == "all"].iterrows():
                if name != "all" and m["model"] != "ridge_sent":
                    continue
                rows.append({"features": name, "exchange": exchange, "model": m["model"],
                             "n": int(m["n"]), "mape_pct": m["mape_pct"],
                             "direction_acc_pct": m["direction_acc_pct"],
                             "skill_vs_naive_pct": m["skill_vs_naive_pct"]})
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/live/source_ablation.csv"))
    args = parser.parse_args()
    print(run(args.base, args.output).to_string(index=False))
