# HengSeng Sentiment Price Prediction

A Python pipeline that scrapes social posts and news about HKEX and KRX stocks, aligns them to exchange trading sessions, and predicts each asset's **next-session close (n+1)** with a walk-forward (out-of-sample) backtest.

> Not investment advice. The predictions come from a simple model and are shown alongside a no-change baseline so you can see whether they add value.

## What it does

| Stage | Command | What happens |
|---|---|---|
| Social scrape | `python -m sentpred scrape` | Runs Apify Actors per source and exchange, keeps posts that mention an asset (keywords and cashtags), and stores them with UTC timestamps, text, likes, reposts, views and author followers. |
| News | `python -m sentpred news` | Exa neural news search for each asset within the look-back window (highlights extracted). |
| Prices | `python -m sentpred prices` | Daily OHLCV from Yahoo Finance (`yfinance`). |
| Model | `python -m sentpred model` | Aligns posts to sessions, builds features, runs walk-forward validation and produces next-session forecasts. |
| Report | `python -m sentpred report` | Writes `reports/index.html` with tables and charts. |
| All | `python -m sentpred run` | Runs every stage. Sources that lack credentials or configuration are skipped with an error. |
| Offline demo | `python -m sentpred demo` | Runs the whole pipeline on **synthetic** data. Output goes to `reports/demo/`. |

### Assets (5 per exchange)

| Exchange | Assets |
|---|---|
| HKEX | 0700.HK Tencent, 9988.HK Alibaba, 3690.HK Meituan, 0005.HK HSBC, 1810.HK Xiaomi |
| KRX | 005930.KS Samsung Electronics, 000660.KS SK hynix, 035420.KS NAVER, 035720.KS Kakao, 005380.KS Hyundai Motor |

Assets, keywords and exchange calendars live in `config.yaml`.

### Social sources (2 per exchange)

| Exchange | Source | Apify Actor ID | Actor input |
|---|---|---|---|
| HKEX | LIHKG forum keyword search | `claude_code_reviewer~lihkg-scraper` (`pwCnD34epqg3YfeMB`) | `mode: search`, `keywords`, `maxItems`, `dateFrom` |
| HKEX + KRX | Telegram keyword search across public channels | `parseforge~telegram-keyword-search-scraper` | `keywords` (tickers, cashtags, names), `discoverChannels: true`, `afterDate`, per-exchange discovery bias (`zh-TW`/`hk`, `ko`/`kr`) |
| KRX | Naver Finance stock discussion board (종목토론실) | `apify~playwright-scraper` with the custom `pageFunction` in `sentpred/sources.py` | `startUrls` (one per asset per page), `waitUntil: domcontentloaded`, serial concurrency |
| KRX (optional, disabled) | Blind (English TeamBlind site) | `hypebridge~blind-post-scraper` (`h4IM5lZDxLlFcQ3yx`) | `startUrls` (`/search/<query>`), `maxItems` |

Each run is capped with `maxTotalChargeUsd` (`max_charge_usd` in `config.yaml`). The cap is passed as a query parameter, not in the Actor input.

**Estimated cost of one full run** (published "from" prices, before plan discounts):

| Source | Price | Volume in one run | Estimated cost |
|---|---|---|---|
| LIHKG | $2.00 / 1,000 items | 10 keywords x up to 100 items | <= $2.00 |
| Telegram search (2 exchanges) | $2.80 / 1,000 messages | up to 200 messages per exchange | <= $1.12 |
| Naver (Playwright) | pay per usage | 10 page loads | ~$0.02-0.05 |
| Blind (disabled by default) | $18.00 / 1,000 posts | up to 100 posts | <= $1.80 if enabled |
| Exa news | per Exa credits | 10 searches x 10 results | free credits / plan |

About **$3.20 per full run** with Blind disabled, and a hard worst-case cap of $9 reserved across the three enabled Actors.

### News source

Exa (`POST https://api.exa.ai/search`) following the `build-with-exa` skill: `query` + `objective` + `contents.highlights: true`, with `startPublishedDate` because the brief enforces a 10-day window. No highlight budgets or domain filters. Tavily is not used, so no Tavily key is needed.

## Features and model

**Timestamp alignment.** Each post's UTC time is converted to the exchange's local time. A post belongs to the trading session whose window is `(previous close, this close]`. Posts after the close, on weekends or on holidays roll to the next session.

**Features per session `t`:**
- Price technicals: 1, 5 and 10-session returns, 10-session volatility, RSI(14), distance from the 20-session moving average, volume z-score.
- Social sentiment: log post count, mean and engagement-weighted sentiment, share of positive and negative posts, engagement sum.
- News sentiment: log item count, mean and weighted sentiment.
- Coverage flags: `has_social`, `has_news`.

**Sentiment score.** A transparent lexicon scorer covers English, Chinese (Traditional and Simplified) and Korean. Each text scores `(pos - neg) / (pos + neg)` in [-1, 1]. Weights are `(1 + log(1 + likes + reposts + views/100)) * (1 + log(1 + followers)/10)`.

**Target.** The next session's return, `close(t+1) / close(t) - 1`. The predicted price is `close(t) * (1 + predicted return)`.

**Models.** A no-change baseline (`naive`), `ridge_price` (technicals only) and `ridge_sent` (technicals + sentiment). Ridge regression is used because the sample is small.

**Validation.** Expanding-window walk-forward. A warm-up of `min_train_dates` sessions is followed by out-of-sample blocks of `test_block` sessions. Training stops two sessions before each block, so no label overlaps the test period. Reported metrics: MAE, RMSE and MAPE of the price, direction accuracy, and skill against the naive baseline. Metrics are reported for all rows and for the social-window rows only. Absolute price errors are not comparable across assets, so model selection and the headline chart use MAPE.

**Headline forecast.** The model with the lowest out-of-sample MAPE is refit on all history and predicts each asset's next close. The signal is LONG or SHORT when the predicted move exceeds `signal_threshold`, and FLAT otherwise.

**Support and resistance.** Swing highs and lows over 120 sessions, clustered within 1.5%, split around the last close. Classic daily pivots (P, R1, R2, S1, S2) are included.

**Position.** The report shows each asset's position from the `positions` block in `config.yaml` (`flat` when empty). Add entries as `{quantity: 200, avg_price: 410.5}` to populate it.

## The report

`reports/index.html` (charts are embedded) contains:
- Per exchange, the tickers ranked by **social mentions** and separately by **sentiment score**, with direction (BULLISH / NEUTRAL / BEARISH).
- Bar charts of mentions and sentiment, coloured by direction.
- Price charts with support and resistance lines and the n+1 forecast marker.
- Out-of-sample model comparison (MAE chart and table).
- Next-session forecasts, including predicted price, predicted move, signal and model used.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in APIFY_API_TOKEN and EXA_API_KEY
.venv/bin/python -m pytest -q  # 23 unit tests, no network or keys needed
.venv/bin/python -m sentpred demo   # offline end-to-end run on synthetic data
```

Keys are read from the environment or `.env`. Never commit `.env`. `data/` (raw scrapes) is git-ignored because it can contain usernames.

### Operational notes for live runs

1. **Naver.** The board is a client-rendered Next.js app, so it is scraped with Playwright (`waitUntil: domcontentloaded`, serial page loads). The parser matches stable class-name prefixes and was verified against the live page in Nov 2026. Each board exposes the latest ~50 posts per page; `?page=` does not reliably reach older posts. **Naver therefore yields a real-time sample per run, and history grows across daily runs.** Naver's like counter is Lottie-animated and not exposed in the DOM, so Naver engagement uses views and comments; likes come from the other sources.
2. **Blind.** The only available Actor supports the English `teamblind.com` site and skips localized `/kr/` paths. Its keyword search is relevance-ranked, so most hits fall outside the 10-day window. Blind is therefore disabled by default; enable it if you want long-horizon English TeamBlind posts.
3. **Telegram.** Channel discovery is a web search restricted to `t.me` and is best-effort. Add public channel handles under `sources.telegram.channels.<exchange>` to guarantee they are searched.
4. **Raw cache.** `python -m sentpred scrape --reuse-raw` rebuilds the post table from `data/raw/*.json` without paying to re-scrape.

## Results snapshot (2026-11-07)

Live run, both exchanges, 10-day window: **402 social posts** (LIHKG 8, Telegram 38, Naver 356) and **100 Exa news items** across 10 assets, mapped to 7 trading sessions with sentiment coverage.

- Social sentiment (combined with news in the report): bearish on HK tech (Meituan -0.67, Alibaba -0.43, Xiaomi -0.36, Tencent -0.27), neutral to mildly bearish in Korea (Kakao -0.17; Samsung, SK hynix, NAVER, Hyundai near neutral).
- Model, out-of-sample over 1,685 pooled rows: naive baseline MAPE **2.656%**; `ridge_price` **2.673%** (direction 52.4%); `ridge_sent` **2.679%**. The sentiment model is not yet trainable with 7 sessions of history, so it has no meaningful weights and the report says so. No edge over the baseline yet, reported honestly.
- Snapshot: [`reports/live/index.html`](reports/live/index.html) (self-contained; no usernames or raw posts).

## Running it daily

Social sentiment is only available for the look-back window, so a single run gives at most `social_days` sessions of sentiment. History accumulates in `data/processed/` across runs (duplicates are removed), so a daily cron job builds the sample. The report states how many sessions of sentiment history exist. Until there are at least `min_train_dates` (default 60), the sentiment model cannot be trained, and the report says so.

```bash
# cron example: weekdays at 10:30 UTC (18:30 HKT, 19:30 KST), after both closes
30 10 * * 1-5  cd /path/to/repo && .venv/bin/python -m sentpred run
```

## Limitations

- **Short history.** Ten days of social data is too little to train or validate a sentiment model. This repo accumulates history, and the report shows the coverage. Treat any result before about 60 sessions as a dry run.
- **Lexicon sentiment.** It has no negation handling and no sarcasm detection. A multilingual finance model would be more accurate.
- **Model.** It is a linear model on a small sample. The expected edge is small, and the report shows whether it beats the naive baseline.
- **Calendars.** Exchange holidays are not modelled. Sessions are the price bars plus the next business day.
- **Telegram** coverage depends on its channel discovery (a web search restricted to `t.me`). You can add specific public channels in `config.yaml` to guarantee they are searched.
- **Naver** yields the board's latest ~50 posts per page per run (real-time sample), not a full 10 days in one pass. Daily runs accumulate history.
- **Blind** is optional and disabled: the available Actor supports only the English TeamBlind site, and its relevance-ranked search rarely returns posts within 10 days.
- **Follower counts** are available from Telegram (`channelSubscribers`) but not from LIHKG or Naver, so reach weighting is strongest there.

## Repository layout

```
config.yaml              assets, exchange calendars, sources, model settings
sentpred/
  config.py              config loading, .env
  apify.py               Apify REST client (run, poll, dataset download, cost cap)
  sources.py             Actor input builders and normalizers per source
  news.py                Exa news search (query + objective + highlights)
  prices.py              yfinance daily bars
  sentiment.py           Chinese/English/Korean lexicon scorer and keyword matcher
  align.py               UTC to session mapping and per-ticker aggregation
  features.py            technical features and the joined dataset
  levels.py              pivots and clustered swing support/resistance
  model.py               walk-forward validation, metrics, live forecast
  report.py              HTML and PNG report
  cli.py                 command line stages
tests/                   29 unit tests (run offline)
reports/demo/            demo report built from synthetic data
reports/live/            sanitized snapshot of the 2026-11-07 live run
```
