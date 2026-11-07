"""Visual report: per-exchange tables ranked by social mentions and by sentiment, plus charts."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .sentiment import score_text  # noqa: E402

DIRECTION_COLORS = {"BULLISH": "#2e7d32", "BEARISH": "#c62828", "NEUTRAL": "#757575"}


def direction(score: float, threshold: float) -> str:
    if score >= threshold:
        return "BULLISH"
    if score <= -threshold:
        return "BEARISH"
    return "NEUTRAL"


def summary_table(cfg: Config, prices: pd.DataFrame, posts: pd.DataFrame, news: pd.DataFrame,
                  forecasts: pd.DataFrame, levels: dict[str, dict]) -> pd.DataFrame:
    threshold = cfg.raw["model"]["sentiment_direction"]
    rows = []
    for asset in cfg.assets:
        px = prices[prices["ticker"] == asset.ticker].sort_values("date")
        if px.empty:
            continue
        last = px.iloc[-1]
        social = posts[(posts["ticker"] == asset.ticker)]
        news_t = news[news["ticker"] == asset.ticker] if not news.empty else news
        scores = social["text"].map(lambda t: score_text(t)[0]) if not social.empty else pd.Series(dtype=float)
        sent = float(scores.mean()) if len(scores) else 0.0
        fc = forecasts[forecasts["ticker"] == asset.ticker]
        lv = levels.get(asset.ticker, {"support": [], "resistance": []})
        rows.append({
            "exchange": asset.exchange,
            "ticker": asset.ticker,
            "name": asset.name,
            "current_price": round(float(last["close"]), 4),
            "as_of": pd.Timestamp(last["date"]).date().isoformat(),
            "social_mentions": len(social),
            "news_items": len(news_t),
            "sentiment": round(sent, 3),
            "direction": direction(sent, threshold),
            "pred_close_n1": round(float(fc["pred_price"].iloc[0]), 4) if len(fc) else None,
            "pred_move_pct": round(float(fc["pred_ret"].iloc[0]) * 100, 2) if len(fc) else None,
            "signal": fc["signal"].iloc[0] if len(fc) else None,
            "position": asset.position_label,
            "support": ", ".join(f"{s['level']:.2f}" for s in lv["support"]) or "-",
            "resistance": ", ".join(f"{r['level']:.2f}" for r in lv["resistance"]) or "-",
        })
    return pd.DataFrame(rows)


def write_report(out_dir: Path, cfg: Config, prices: pd.DataFrame, posts: pd.DataFrame, news: pd.DataFrame,
                 forecasts: pd.DataFrame, levels: dict[str, dict], metrics: pd.DataFrame,
                 preds: pd.DataFrame, short_metrics: pd.DataFrame | None = None,
                 coefficients: pd.DataFrame | None = None, banner: str | None = None) -> Path:
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    summary = summary_table(cfg, prices, posts, news, forecasts, levels)
    sections = []

    coverage = _coverage(posts, news)
    check = _prediction_check(preds)
    sections.append(
        "<h2>Data coverage</h2>"
        "<p class='note'>Posts are filtered to the 10-day window. Naver is paginated through its JSON API and "
        "sampled at up to 80 posts per ticker per day; LIHKG/Telegram coverage depends on keyword discovery. "
        "The table shows the actual first/last timestamp per feed.</p>"
        f"{_table(coverage)}"
    )

    for exchange in cfg.exchanges:
        ex = summary[summary["exchange"] == exchange]
        if ex.empty:
            continue
        by_mentions = ex.sort_values(["social_mentions", "sentiment"], ascending=False)
        by_sentiment = ex.sort_values("sentiment", ascending=False)
        mention_png = _bar(by_mentions, "social_mentions", f"{exchange}: social mentions (last window)",
                           charts / f"mentions_{exchange}.png")
        sent_png = _bar(by_sentiment, "sentiment", f"{exchange}: sentiment score", charts / f"sentiment_{exchange}.png")
        levels_png = _levels_chart(cfg, exchange, prices, forecasts, levels, charts / f"levels_{exchange}.png")
        sections.append(
            f"<h2>{html.escape(exchange)}</h2>"
            f"<h3>Ranked by social mentions</h3>{_table(by_mentions)}"
            f"<h3>Ranked by sentiment score</h3>{_table(by_sentiment)}"
            f"<div class='grid'>{_img(mention_png)}{_img(sent_png)}</div>"
            f"<h3>Price, support / resistance and next-session forecast</h3>{_img(levels_png)}"
        )

    mae_png = _mae_chart(metrics, charts / "model_mae.png")
    metric_view = metrics.round(3)
    short_html = ""
    if short_metrics is not None and not short_metrics.empty:
        short_html = (
            "<h2>Sentiment model - short-history panel</h2>"
            "<p class='note'>Pooled leave-one-session-out over the sessions that have social/news data. "
            "Every row is predicted by a model trained on the other sessions. With fewer than 60 sessions this "
            "is evidence that the sentiment features carry fitted weights, not a statistically robust backtest.</p>"
            f"{_table(short_metrics.round(3))}"
            "<h3>Standardized coefficients</h3>"
            "<p class='note'>Standardized Ridge coefficients; larger absolute values mean the feature moves the "
            "predicted next-session return more.</p>"
            f"{_table(coefficients.round(4))}"
        )
    banner_html = f"<div class='banner'>{html.escape(banner)}</div>" if banner else ""
    doc = _HTML.format(
        banner=banner_html,
        sections="".join(sections),
        metrics=_table(metric_view),
        short=short_html,
        check=_table(check),
        mae=_img(mae_png),
        forecast=_table(forecasts.round({"close": 4, "pred_ret": 4, "pred_price": 4,
                                         "pred_sent_ret": 4, "pred_sent_price": 4})),
        generated=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv", index=False)
    return path


def _coverage(posts: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not posts.empty:
        for (exchange, source), g in posts.groupby(["exchange", "source"]):
            rows.append({
                "exchange": exchange, "feed": source, "items": len(g),
                "first_utc": g["published_utc"].min().strftime("%Y-%m-%d %H:%M"),
                "last_utc": g["published_utc"].max().strftime("%Y-%m-%d %H:%M"),
            })
    if not news.empty:
        for exchange, g in news.groupby("exchange"):
            rows.append({
                "exchange": exchange, "feed": "exa_news", "items": len(g),
                "first_utc": g["published_utc"].min().strftime("%Y-%m-%d %H:%M"),
                "last_utc": g["published_utc"].max().strftime("%Y-%m-%d %H:%M"),
            })
    return pd.DataFrame(rows)


def _prediction_check(preds: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Show the most recent out-of-sample forecasts next to what actually happened."""
    if preds.empty:
        return preds
    df = preds.sort_values("date").groupby("ticker").tail(n).copy()
    df["pred_price"] = df["close"] * (1 + df["pred_ridge_price"])
    df["actual_price"] = df["next_close"]
    df["error_pct"] = (df["pred_price"] / df["actual_price"] - 1) * 100
    df["direction_hit"] = (df["pred_ridge_price"].gt(0) == df["y_next_ret"].gt(0))
    return df[["exchange", "ticker", "date", "pred_price", "actual_price", "error_pct", "direction_hit"]].round(
        {"pred_price": 2, "actual_price": 2, "error_pct": 2})


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table", border=0, na_rep="-")


def _img(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{data}' alt='{html.escape(path.stem)}'/>"


def _bar(df: pd.DataFrame, column: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [DIRECTION_COLORS.get(d, "#757575") for d in df["direction"]]
    ax.barh(df["ticker"], df[column], color=colors)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _levels_chart(cfg: Config, exchange: str, prices: pd.DataFrame, forecasts: pd.DataFrame,
                  levels: dict[str, dict], path: Path) -> Path:
    assets = cfg.assets_for(exchange)
    fig, axes = plt.subplots(1, len(assets), figsize=(4 * len(assets), 3.6), squeeze=False)
    for ax, asset in zip(axes[0], assets):
        px = prices[prices["ticker"] == asset.ticker].sort_values("date").tail(120)
        if px.empty:
            ax.set_visible(False)
            continue
        ax.plot(px["date"], px["close"], color="#1565c0", linewidth=1.3)
        for s in levels.get(asset.ticker, {}).get("support", []):
            ax.axhline(s["level"], color="#2e7d32", linestyle="--", linewidth=0.9)
        for r in levels.get(asset.ticker, {}).get("resistance", []):
            ax.axhline(r["level"], color="#c62828", linestyle="--", linewidth=0.9)
        fc = forecasts[forecasts["ticker"] == asset.ticker]
        if len(fc):
            ax.scatter([fc["date"].iloc[0] + pd.offsets.BDay(1)], [fc["pred_price"].iloc[0]],
                       color="#ff8f00", zorder=5, label="pred n+1")
        ax.set_title(asset.ticker, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.suptitle(f"{exchange}: support (green) / resistance (red) / forecast (orange)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _mae_chart(metrics: pd.DataFrame, path: Path) -> Path | None:
    sub = metrics[metrics["subset"] == "all"]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(sub["model"], sub["mape_pct"], color=["#9e9e9e", "#1565c0", "#ef6c00"][: len(sub)])
    ax.set_ylabel("MAPE % (out-of-sample)")
    ax.set_title("Walk-forward price error by model (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>HengSeng Sentiment Price Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:24px;color:#222;max-width:1400px}}
h1{{margin-bottom:4px}} h2{{border-bottom:2px solid #1565c0;padding-bottom:4px;margin-top:36px}}
table.table{{border-collapse:collapse;font-size:13px;margin:8px 0 16px}}
table.table th,table.table td{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
table.table th{{background:#f0f4fa}} img{{max-width:100%}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.banner{{background:#fff3e0;border:1px solid #ef6c00;padding:10px;margin:12px 0;font-weight:600}}
.note{{color:#555;font-size:13px}}
</style></head><body>
<h1>HengSeng Sentiment Price Prediction</h1>
<div class="note">Generated {generated}. Not investment advice. Predictions are model outputs from a walk-forward backtest.</div>
{banner}
{sections}
<h2>Out-of-sample model comparison</h2>
{mae}
{metrics}
{short}
<h2>Prediction check: recent out-of-sample forecasts vs actual</h2>
{check}
<h2>Next-session forecasts</h2>
{forecast}
</body></html>
"""
