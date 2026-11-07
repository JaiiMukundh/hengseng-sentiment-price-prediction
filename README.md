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
| HKEX | Telegram public channels | `tri_angle~telegram-scraper` (`GEHKCq8O4orlPjLFf`) | `profiles` (channel list), `scrapeLastNDays` |
| KRX | Naver Finance stock discussion board (종목토론실) | `apify~playwright-scraper` with the custom `pageFunction` in `sentpred/sources.py` | `startUrls` (one per asset per page) |
| KRX | Blind (Korean feed, keyword search) | `hypebridge~blind-post-scraper` (`h4IM5lZDxLlFcQ3yx`) | `startUrls` (search URL per asset), `maxItems` |

Each run is capped with `maxTotalChargeUsd` (`max_charge_usd` in `config.yaml`). The cap is passed as a query parameter, not in the Actor input.

### News source

Exa (`POST https://api.exa.ai/search`, `category: news`, `startPublishedDate` set to the look-back window, highlights returned). Tavily is not used, so no Tavily key is needed.

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

**Position.** The report has a `position` column set to "none (not provided)". Your own positions can be added to the summary table if you want them shown.

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

### Configuration to complete before a live run

1. **Telegram channels.** Set `sources.telegram.channels.HKEX` in `config.yaml` to the public HK channels you want scraped. The pipeline does not guess channel names.
2. **Blind search URL.** `sources.blind.search_url` is a template. Check the URL pattern on the first run, because Blind's Korean search path can change.
3. **Naver selectors.** The board renders client-side. `NAVER_PAGE_FUNCTION` in `sentpred/sources.py` targets the classic `table.type2` layout and must be checked on a live run. Set `pages: 1` in `config.yaml` for a first test, and inspect the raw output in `data/raw/naver_discussion_KRX.json`.

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
- **Telegram** only covers the channels you list. It does not search by ticker.

## Repository layout

```
config.yaml              assets, exchange calendars, sources, model settings
sentpred/
  config.py              config loading, .env
  apify.py               Apify REST client (run, poll, dataset download, cost cap)
  sources.py             Actor input builders and normalizers per source
  news.py                Exa news search
  prices.py              yfinance daily bars
  sentiment.py           bilingual lexicon scorer and asset keyword matcher
  align.py               UTC to session mapping and per-ticker aggregation
  features.py            technical features and the joined dataset
  levels.py              pivots and clustered swing support/resistance
  model.py               walk-forward validation, metrics, live forecast
  report.py              HTML and PNG report
  cli.py                 command line stages
tests/                   unit tests (run offline)
reports/demo/            demo report built from synthetic data
```
