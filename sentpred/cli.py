"""Command line entry point.

Stages: scrape (Apify) -> news (Exa) -> prices (yfinance) -> model (align, features, walk-forward) -> report.
Run everything with `python -m sentpred run`, or try the whole pipeline offline with `python -m sentpred demo`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import align, evaluation, features, levels, model
from .config import ROOT, Config, apify_token, exa_key, load_config
from .sources import POST_COLUMNS

log = logging.getLogger("sentpred")


class Paths:
    def __init__(self, base: Path, report_dir: Path):
        self.raw = base / "raw"
        self.processed = base / "processed"
        self.report = report_dir
        for p in (self.raw, self.processed, self.report):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def posts(self) -> Path:
        return self.processed / "social_posts.csv"

    @property
    def news(self) -> Path:
        return self.processed / "news.csv"

    @property
    def prices(self) -> Path:
        return self.processed / "prices.csv"

    @property
    def market(self) -> Path:
        return self.processed / "market.csv"


def _read(path: Path, parse_dates: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    df = pd.read_csv(path)
    for col in parse_dates:
        if col not in df.columns:
            continue
        if col == "date":  # price bars stay naive exchange-local dates
            df[col] = pd.to_datetime(df[col], format="mixed")
        else:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    return df


def stage_scrape(cfg: Config, paths: Paths, reuse_raw: bool = False) -> None:
    from .apify import ApifyError, run_actor
    from .sources import build_run_input, normalize

    token = apify_token()
    if not token:
        raise SystemExit("APIFY_API_TOKEN is not set (see .env.example)")
    window = cfg.raw["window"].get("history_days", cfg.raw["window"]["social_days"])
    frames = []
    for name, src in cfg.raw["sources"].items():
        if not src.get("enabled"):
            continue
        for exchange in src["exchanges"]:
            assets = cfg.assets_for(exchange)
            raw_path = paths.raw / f"{name}_{exchange}.json"
            if reuse_raw and raw_path.exists():
                items = json.loads(raw_path.read_text(encoding="utf-8"))
                log.info("%s/%s: reusing %d cached items from %s", name, exchange, len(items), raw_path.name)
            else:
                limit = src.get("max_assets_per_run")
                batches = [assets] if not limit else [assets[i:i + limit] for i in range(0, len(assets), limit)]
                items = []
                for batch in batches:
                    run_input = build_run_input(name, src, exchange, batch, window)
                    if run_input is None:
                        continue
                    try:
                        items.extend(run_actor(src["actor"], run_input, token,
                                               max_charge_usd=src.get("max_charge_usd"),
                                               timeout_s=src.get("timeout_s", 900)))
                    except ApifyError as exc:
                        log.error("%s/%s batch failed: %s", name, exchange, exc)
                if not items:
                    continue
                raw_path.write_text(json.dumps(items, ensure_ascii=False, default=str), encoding="utf-8")
            df = normalize(name, exchange, items, assets)
            log.info("%s/%s: %d raw items -> %d asset-matched posts", name, exchange, len(items), len(df))
            frames.append(df)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=POST_COLUMNS)
    if not new.empty:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window)
        before = len(new)
        new = new[new["published_utc"] >= cutoff]
        log.info("window filter: kept %d/%d posts published in the last %d days", len(new), before, window)
    _append(paths.posts, new, observed=not reuse_raw)


def stage_news(cfg: Config, paths: Paths) -> None:
    from .news import fetch_news

    key = exa_key()
    if not key:
        raise SystemExit("EXA_API_KEY is not set (see .env.example)")
    df = fetch_news(cfg.assets, key,
                    cfg.raw["window"].get("history_days", cfg.raw["window"]["social_days"]),
                    cfg.raw["news"]["num_results"])
    _append(paths.news, df, observed=True)
    log.info("news: %d new items", len(df))


def _append(path: Path, new: pd.DataFrame, observed: bool = False) -> None:
    """Accumulate rows while retaining the earliest verified collection time.

    Old cached rows have no first-seen time and remain unknown. Reused raw files
    also remain unknown: their original collection time cannot be reconstructed.
    """
    old = _read(path, [], POST_COLUMNS)
    key = ["source", "ticker", "text", "published_utc"]
    if "published_utc" in old and not old.empty:
        old["published_utc"] = pd.to_datetime(old["published_utc"], utc=True, format="mixed")
    if "first_seen_utc" not in old:
        old["first_seen_utc"] = pd.NA
    old["first_seen_utc"] = old["first_seen_utc"].astype("string")
    new = new.copy()
    if "published_utc" in new and not new.empty:
        new["published_utc"] = pd.to_datetime(new["published_utc"], utc=True, format="mixed")
    new["first_seen_utc"] = pd.Timestamp.now(tz="UTC").isoformat() if observed else pd.NA
    new["first_seen_utc"] = new["first_seen_utc"].astype("string")
    prior = old[key + ["first_seen_utc"]].drop_duplicates(key)
    if not new.empty:
        new = new.merge(prior, on=key, how="left", suffixes=("", "_prior"), indicator=True)
        existing = new["_merge"] == "both"
        new.loc[existing, "first_seen_utc"] = new.loc[existing, "first_seen_utc_prior"]
        new = new.drop(columns=["first_seen_utc_prior", "_merge"])
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep="last")
    merged.to_csv(path, index=False)


def stage_prices(cfg: Config, paths: Paths) -> None:
    from .prices import fetch_market, fetch_prices

    period = cfg.raw["window"]["price_period"]
    df = fetch_prices(cfg.assets, period)
    df.to_csv(paths.prices, index=False)
    market = fetch_market(period)
    market.to_csv(paths.market, index=False)
    log.info("prices: %d bars for %d tickers; market context: %d series",
             len(df), df["ticker"].nunique(), market["series"].nunique())


def stage_model(cfg: Config, paths: Paths) -> dict:
    prices = _read(paths.prices, ["date"])
    market = _read(paths.market, ["date"])
    posts = _read(paths.posts, ["published_utc"], POST_COLUMNS)
    news = _read(paths.news, ["published_utc"], POST_COLUMNS)
    if prices.empty:
        raise SystemExit("no price data; run the prices stage first")

    calendars = {}
    for exchange in cfg.exchanges:
        ex = cfg.exchange(exchange)
        ex_prices = prices[prices["exchange"] == exchange]
        if ex_prices.empty:
            continue
        calendars[exchange] = (ex["tz"], ex["close"], align.session_calendar(ex_prices["date"]))

    social_agg = align.aggregate(_utc(posts), calendars, "social_")
    news_agg = align.aggregate(_utc(news), calendars, "news_")
    dataset = features.build_dataset(prices, social_agg, news_agg, market)
    # The model was trained on completed sessions with an observed close. Pending
    # after-close rows have copied price features and a different information set.
    forecast_dataset = features.completed_session_rows(dataset, prices)

    mcfg = cfg.raw["model"]
    preds = model.walk_forward(dataset, mcfg["min_train_dates"], mcfg["test_block"], mcfg["ridge_alpha"])
    metrics = model.evaluate(preds)
    forecasts = model.live_forecast(forecast_dataset, metrics, mcfg["ridge_alpha"], mcfg["signal_threshold"])
    forecasts["target_session_estimate"] = forecasts["date"].map(
        lambda day: pd.bdate_range(pd.Timestamp(day) + pd.Timedelta(days=1), periods=1)[0].date().isoformat())
    sent_sessions = int(dataset.loc[(dataset["has_social"] + dataset["has_news"] > 0)
                                    & dataset["y_next_ret"].notna(), "date"].nunique())

    short_metrics = None
    forward_metrics = None
    forward_preds = None
    nested_metrics = None
    coefficients = None
    importances = None
    bootstrap = None
    if sent_sessions < mcfg["min_train_dates"]:
        sent_alpha = mcfg.get("sentiment_alpha", 50)
        sent_shrink = mcfg.get("sentiment_shrinkage", 0.3)
        sent_model = mcfg.get("sentiment_model", "pls")
        sent_scale = mcfg.get("sentiment_scale", 1.0)
        sent_clip = mcfg.get("sentiment_clip")
        sent_components = mcfg.get("sentiment_components", 6)
        short_preds, short_dates = model.leave_one_session_out(
            dataset, sent_alpha, sent_shrink, sent_model, sent_scale, sent_clip, sent_components)
        short_metrics = model.evaluate(short_preds)
        forward_preds = model.forward_sentiment_predictions(
            dataset, sent_alpha, sent_shrink, sent_model, sent_scale, sent_clip, sent_components)
        if not forward_preds.empty:
            forward_metrics = model.evaluate(forward_preds)
            forward_preds.to_csv(paths.processed / "sentiment_forward_predictions.csv", index=False)
            forward_metrics.to_csv(paths.processed / "sentiment_forward_metrics.csv", index=False)
        bootstrap = evaluation.cluster_bootstrap(short_preds)
        nested = model.nested_sentiment_predictions(
            dataset, shrink=sent_shrink, scale=sent_scale, clip=sent_clip, components=sent_components)
        nested = nested.merge(short_preds[["date", "ticker", "pred_ridge_price"]], on=["date", "ticker"], how="left")
        nested["pred_naive"] = 0.0
        nested_metrics = model.evaluate(nested.dropna(subset=["pred_ridge_price", "pred_ridge_sent"]))
        coefficients = model.sentiment_coefficients(dataset, sent_alpha, sent_scale)
        importances = model.sentiment_importances(
            dataset, sent_model, sent_alpha, sent_scale, sent_components)
        forecasts = forecasts.merge(
            model.sentiment_forecast(forecast_dataset, sent_alpha, sent_shrink, sent_model,
                                     sent_scale, sent_clip, sent_components),
            on="ticker", how="left")
        price_only = model.price_only_forecast(
            forecast_dataset, sent_alpha, sent_shrink, sent_model, sent_clip, sent_components)
        forecasts = forecasts.merge(price_only, on="ticker", how="left")
        forecasts["pred_ret"] = forecasts["pred_sent_ret"]
        forecasts["pred_price"] = forecasts["pred_sent_price"]
        forecasts["signal"] = np.select(
            [forecasts["pred_ret"] > mcfg["signal_threshold"],
             forecasts["pred_ret"] < -mcfg["signal_threshold"]],
            ["LONG", "SHORT"], "FLAT")
        forecasts["model_used"] = f"{sent_model}_sentiment"
        if not forward_preds.empty:
            checked = forward_preds.copy()
            checked["abs_residual_ret"] = (
                checked["y_next_ret"] - checked["pred_ridge_sent"]).abs()
            band = checked.groupby("exchange")["abs_residual_ret"].quantile(0.8)
            forecasts["error_band_pct"] = forecasts["exchange"].map(band) * 100
            width = forecasts["error_band_pct"] / 100
            forecasts["forecast_low_80"] = forecasts["close"] * (1 + forecasts["pred_ret"] - width)
            forecasts["forecast_high_80"] = forecasts["close"] * (1 + forecasts["pred_ret"] + width)
            forecasts["direction_confidence"] = np.select(
                [forecasts["pred_ret"] > width, forecasts["pred_ret"] < -width],
                ["above_empirical_error", "below_empirical_error"], default="uncertain")
        short_metrics.to_csv(paths.processed / "short_history_metrics.csv", index=False)
        bootstrap.to_csv(paths.processed / "short_history_bootstrap.csv", index=False)
        nested_metrics.to_csv(paths.processed / "sentiment_nested_metrics.csv", index=False)
        coefficients.to_csv(paths.processed / "sentiment_coefficients.csv", index=False)
        importances.to_csv(paths.processed / "sentiment_importances.csv", index=False)
        log.info("short-history sentiment model: %d sessions, %d rows; coefficients written",
                 short_dates, len(short_preds))
        log.warning("only %d sessions with social/news history (need %d for a full walk-forward); "
                    "using pooled leave-one-session-out and reporting it as short-history",
                    sent_sessions, mcfg["min_train_dates"])
    else:
        log.info("sentiment history: %d sessions (full walk-forward)", sent_sessions)

    dataset.to_csv(paths.processed / "dataset.csv", index=False)
    lvl = {}
    for asset in cfg.assets:
        px = prices[prices["ticker"] == asset.ticker].sort_values("date")
        if px.empty:
            continue
        sw = levels.swing_levels(px)
        piv = levels.pivot_points(px.iloc[-1])
        # Keep the report complete when a 120-session swing window has no clustered level
        # on one side of the last close: fall back to the classic daily pivot range.
        if not sw["support"]:
            sw["support"] = [{"level": round(piv["S1"], 4), "touches": 0},
                             {"level": round(piv["S2"], 4), "touches": 0}]
        if not sw["resistance"]:
            sw["resistance"] = [{"level": round(piv["R1"], 4), "touches": 0},
                                {"level": round(piv["R2"], 4), "touches": 0}]
        lvl[asset.ticker] = {**sw, "pivots": piv}

    preds.to_csv(paths.processed / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(paths.processed / "model_metrics.csv", index=False)
    forecasts.to_csv(paths.processed / "forecasts.csv", index=False)
    log.info("model: %d out-of-sample rows; next-session headline model: %s",
             len(preds), forecasts["model_used"].iloc[0] if len(forecasts) else "n/a")
    return {"prices": prices, "posts": posts, "news": news, "forecasts": forecasts,
            "levels": lvl, "metrics": metrics, "preds": preds, "sent_sessions": sent_sessions,
            "short_metrics": short_metrics, "forward_metrics": forward_metrics,
            "forward_preds": forward_preds,
            "nested_metrics": nested_metrics,
            "coefficients": coefficients, "importances": importances, "bootstrap": bootstrap}


def stage_report(cfg: Config, paths: Paths, state: dict, banner: str | None) -> Path:
    from .report import write_report

    needed = cfg.raw["model"]["min_train_dates"]
    report_days = cfg.raw["window"].get("social_days", 10)
    history_days = cfg.raw["window"].get("history_days", report_days)
    coverage = (f"Current-sentiment tables use the last {report_days} days; model training uses up to "
                f"{history_days} days ({state['sent_sessions']} sessions). "
                f"Below {needed} sessions the pipeline reports a pooled leave-one-session-out "
                f"model as short-history evidence, not a full walk-forward.")
    banner = " ".join(filter(None, [banner, coverage]))
    path = write_report(paths.report, cfg, state["prices"], state["posts"], state["news"],
                        state["forecasts"], state["levels"], state["metrics"], state["preds"],
                        state["short_metrics"], state["coefficients"], banner, state["nested_metrics"],
                        state["importances"], state["bootstrap"], state.get("forward_metrics"),
                        state.get("forward_preds"))
    log.info("report written to %s", path)
    return path


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True)
    return out


def run_demo(cfg: Config, paths: Paths) -> Path:
    """Synthetic data: exercises every stage offline. Numbers are NOT real market data."""
    rng = np.random.default_rng(7)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    price_rows = []
    for asset in cfg.assets:
        dates = pd.bdate_range(end=today, periods=260)
        start = rng.uniform(20, 400)
        rets = rng.normal(0.0004, 0.015, len(dates))
        close = start * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
        low = close * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
        open_ = np.r_[start, close[:-1]]
        vol = rng.integers(1e6, 5e6, len(dates))
        price_rows.append(pd.DataFrame({
            "ticker": asset.ticker, "exchange": asset.exchange, "date": dates,
            "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        }))
    prices = pd.concat(price_rows, ignore_index=True)

    posting = [
        "{name} breakout looks strong, buy the dip", "{name} earnings miss, bearish outlook",
        "{name} 看好 后市 上涨", "{name} 下跌 割肉 利淡", "{name} 주가 상승 매수 추천",
        "{name} 하락 악재 매도", "just watching {name} today", "{name} rally continues, record high",
    ]
    post_rows = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for asset in cfg.assets:
        for _ in range(int(rng.integers(40, 120))):
            ts = now - pd.Timedelta(minutes=int(rng.integers(0, 10 * 24 * 60)))
            post_rows.append({
                "source": "demo_social", "exchange": asset.exchange, "ticker": asset.ticker,
                "text": rng.choice(posting).format(name=asset.name), "published_utc": ts.tz_localize("UTC"),
                "url": None, "author": "demo", "likes": float(rng.integers(0, 300)),
                "reposts": float(rng.integers(0, 50)), "views": float(rng.integers(0, 5000)),
                "followers": float(rng.integers(0, 50000)),
            })
    posts = pd.DataFrame(post_rows, columns=POST_COLUMNS)
    news = posts[posts["ticker"].isin([a.ticker for a in cfg.assets])].sample(frac=0.2, random_state=1).copy()
    news["source"] = "demo_news"
    news["likes"] = 0.0
    news["reposts"] = 0.0
    news["views"] = 0.0
    news["followers"] = 0.0

    prices.to_csv(paths.prices, index=False)
    posts.to_csv(paths.posts, index=False)
    news.to_csv(paths.news, index=False)
    log.info("demo data generated: %d bars, %d posts, %d news", len(prices), len(posts), len(news))
    state = stage_model(cfg, paths)
    return stage_report(cfg, paths, state, "DEMO: synthetic data for pipeline testing. Not real market or social data.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentpred", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--demo", action="store_true", help="write outputs under data/demo and reports/demo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    commands = {name: sub.add_parser(name) for name in
                ("scrape", "news", "prices", "model", "report", "run", "demo")}
    commands["scrape"].add_argument("--reuse-raw", action="store_true",
                                    help="reuse cached data/raw/<source>_<exchange>.json instead of paying to re-scrape")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.demo or args.cmd == "demo":
        paths = Paths(ROOT / "data" / "demo", ROOT / "reports" / "demo")
    else:
        paths = Paths(ROOT / "data", ROOT / "reports")

    if args.cmd == "demo":
        print(run_demo(cfg, paths))
        return
    if args.cmd == "scrape":
        stage_scrape(cfg, paths, reuse_raw=args.reuse_raw)
    elif args.cmd == "news":
        stage_news(cfg, paths)
    elif args.cmd == "prices":
        stage_prices(cfg, paths)
    elif args.cmd == "model":
        stage_model(cfg, paths)
    elif args.cmd == "report":
        state = stage_model(cfg, paths)
        print(stage_report(cfg, paths, state, None))
    elif args.cmd == "run":
        for stage in (stage_scrape, stage_news, stage_prices):
            try:
                stage(cfg, paths)
            except SystemExit as exc:
                log.error("%s skipped: %s", stage.__name__, exc)
        state = stage_model(cfg, paths)
        print(stage_report(cfg, paths, state, None))
