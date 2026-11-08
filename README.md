![CI](https://github.com/JaiiMukundh/hengseng-sentiment-price-prediction/actions/workflows/ci.yml/badge.svg)

# HengSeng Sentiment Price Prediction

A Python pipeline that scrapes social posts and news about HKEX and KRX stocks, aligns them to exchange trading sessions, and predicts each asset's **next-session close (n+1)** with a walk-forward (out-of-sample) backtest.

> Not investment advice. The predictions come from a simple model and are shown alongside a no-change baseline so you can see whether they add value.

Documentation: **[Architecture](docs/architecture.md)** (data flow and validation design) · **[Model card](docs/model_card.md)** (intended use, metrics, limitations) · **[Plain-English explainer](docs/explainer.md)** (how it works, interview-friendly) · **[Interview narrative](docs/interview_narrative.md)** (role fit and resume bullets).

## What it does

| Stage | Command | What happens |
|---|---|---|
| Social scrape | `python -m sentpred scrape` | Runs Apify Actors per source and exchange, keeps posts that mention an asset (keywords and cashtags), and stores them with UTC timestamps, text, likes, reposts, views and author followers. |
| News | `python -m sentpred news` | Exa neural news search for each asset within the `history_days` window (highlights extracted). |
| Prices | `python -m sentpred prices` | Daily OHLCV from Yahoo Finance (`yfinance`). |
| Model | `python -m sentpred model` | Aligns posts to sessions, builds features, runs walk-forward validation and produces next-session forecasts. |
| Report | `python -m sentpred report` | Writes `reports/index.html` with tables and charts. |
| All | `python -m sentpred run` | Runs every stage. Sources that lack credentials or configuration are skipped with an error. |
| Offline demo | `python -m sentpred demo` | Runs the whole pipeline on **synthetic** data. Output goes to `reports/demo/`. |

Commands in the table use `python -m sentpred`. After `pip install -e .`, the equivalent `sentpred` console script is available (for example `sentpred run` or `sentpred demo`); `make test`, `make demo` and `make run` wrap the common commands.

### Windows

Two independent windows live in `config.yaml`:

| Setting | Default | Used for |
|---|---|---|
| `window.social_days` | 10 days | **Current sentiment.** The report's mention counts, sentiment scores and direction labels only use social/news items from this window. |
| `window.history_days` | 30 days | **Scrape and training.** Telegram, Naver and Exa look back this far (LIHKG is capped at 2 weeks by its own `date_from`), and history accumulates in `data/processed/` across runs, so the model can train on more than a single 30-day window once daily runs accumulate. |

A run scrapes up to `history_days` (30 days) of posts and news (LIHKG stays capped at 2 weeks), while the current-sentiment tables stay pinned to the last `social_days` (10 days). Price history is separate: `price_period: 1y` supplies technicals and the long walk-forward. That keeps the "current sentiment" definition stable while the model gets a longer training history. Metrics are regenerated from the accumulated data; see [`reports/live/`](reports/live/) for the latest numbers rather than relying on the historical snapshot below.

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
| KRX | Naver Finance stock discussion board (종목토론실) | `apify~playwright-scraper` with the custom `pageFunction` in `sentpred/sources.py` | JSON API `discussion/list` start URL per asset; `lastOffset` cursor pagination back through the `history_days` window (30 days by default) |
| KRX (optional, disabled) | Blind (English TeamBlind site) | `hypebridge~blind-post-scraper` (`h4IM5lZDxLlFcQ3yx`) | `startUrls` (`/search/<query>`), `maxItems` |

Each run is capped with `maxTotalChargeUsd` (`max_charge_usd` in `config.yaml`). The cap is passed as a query parameter, not in the Actor input.

**Estimated cost of one full run** (published "from" prices, before plan discounts; volumes and the "about $3.50" total are from the previous 10-day snapshot — a 30-day run uses more of each source's `max_charge_usd` cap, so check Apify usage for the current cost):

| Source | Price | Volume in one run | Estimated cost |
|---|---|---|---|
| LIHKG | $2.00 / 1,000 items | 10 keywords x up to 100 items | <= $2.00 |
| Telegram search (2 exchanges) | $2.80 / 1,000 messages | up to 200 messages per exchange | <= $1.12 |
| Naver (Playwright) | pay per usage | 5 API start URLs, ~600 cursor fetches inside | ~$0.32 |
| Blind (disabled by default) | $18.00 / 1,000 posts | up to 100 posts | <= $1.80 if enabled |
| Exa news | per Exa credits | 10 searches x 20 results | free credits / plan |

About **$3.50 per full run** with Blind disabled in the previous 10-day snapshot; the hard worst-case cap remains $9 reserved across the three enabled Actors.

### News source

Exa (`POST https://api.exa.ai/search`) following the `build-with-exa` skill: `query` + `objective` + `contents.highlights: true`, with `startPublishedDate` set from `history_days` (30 days by default) and 20 results per asset. The report then filters news to `social_days` for the current-sentiment tables. No highlight budgets or domain filters. Tavily is not used, so no Tavily key is needed.

## Features and model

**Timestamp alignment.** Each post's UTC time is converted to the exchange's local time. A post belongs to the trading session whose window is `(previous close, this close]`. Posts after the close, on weekends or on holidays roll to the next session.

**Features per session `t`:**
- Price technicals: 1, 5 and 10-session returns, 10-session volatility, RSI(14), distance from the 20-session moving average, volume z-score.
- Social sentiment: log post count, mean and engagement-weighted sentiment, share of positive and negative posts (`social_pos_share`, `social_neg_share`), engagement sum.
- News sentiment: log item count, mean and weighted sentiment.
- Coverage flags: `has_social`, `has_news`.

**Sentiment score.** A transparent lexicon scorer covers English, Chinese (Traditional and Simplified) and Korean. Each text scores `(pos - neg) / (pos + neg)` in [-1, 1]. Weights are `(1 + log(1 + likes + reposts + views/100)) * (1 + log(1 + followers)/10)`.

**Target.** The next session's return, `close(t+1) / close(t) - 1`. The predicted price is `close(t) * (1 + predicted return)`.

**Models.** A no-change baseline (`naive`), `ridge_price` (technicals only) and `ridge_sent` (technicals + sentiment). The long 1-year walk-forward uses Ridge because the sample is small. When sentiment history is below the 60-session warm-up, the short-history sentiment panel uses a 6-component PLS model on price technicals plus sentiment levels and dynamics, with a same-class price-only control; predictions are shrunk 0.30 toward zero and clipped at ±5%. A small MLP, deep MLP, Random Forest, gradient boosting, nearest neighbours, Ridge+MLP blends and an RBF SVR were tested in `experiments/`; PLS was the most stable across leave-one-session-out and forward validation. Nested selection is still reported as a sensitivity check. Market context (local index, US overnight, SOX, VIX, FX) is ingested but not used in the final model: an ablation showed it worsened out-of-sample MAPE (2.716% vs 2.673%). See [`reports/live/`](reports/live/) for current numbers.

**Validation.** Expanding-window walk-forward. A warm-up of `min_train_dates` sessions is followed by out-of-sample blocks of `test_block` sessions. Training stops two sessions before each block, so no label overlaps the test period. Reported metrics: MAE, RMSE and MAPE of the price, direction accuracy, and skill against the naive baseline. Metrics are reported for all rows and for the social-window rows only. Short-panel results are reported for the fixed configuration and for nested model/alpha selection; both are written under `reports/live/`. Absolute price errors are not comparable across assets, so model selection and the headline chart use MAPE.

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

The per-ticker tables apply the `social_days` filter; the model comparisons use the accumulated training history. See [`reports/live/`](reports/live/) for the current run's CSVs.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # or: pip install -e .
cp .env.example .env          # then fill in APIFY_API_TOKEN and EXA_API_KEY
.venv/bin/python -m pytest -q  # 81 unit tests, no network or keys needed
.venv/bin/sentpred demo        # offline end-to-end run on synthetic data
```

`pip install -e .` installs the package and the `sentpred` console script; the `[dev]` extra adds `pytest`. The console script is equivalent to `python -m sentpred`:

```bash
sentpred scrape    sentpred news    sentpred prices
sentpred model     sentpred report  sentpred run
sentpred demo
```

Make targets wrap the same commands: `make test`, `make demo`, `make run` (they invoke `.venv/bin/python`). `make install` creates the virtualenv from `requirements.txt`. `make demo` is fully offline; `make run` performs live scrapes and can incur Apify/Exa charges.

The `Dockerfile` builds a Python 3.12 image and runs the offline demo by default:

```bash
docker build -t sentpred .
docker run --rm sentpred
```

CI is defined in `.github/workflows/ci.yml`: on every push to `main` and every pull request it installs the requirements, runs the unit tests, runs the offline demo, and uploads the demo report as an artifact. The badge at the top links to that workflow.

Keys are read from the environment or `.env`. Never commit `.env`. `data/` (raw scrapes) is git-ignored because it can contain usernames.

### Operational notes for live runs

1. **Naver.** The board is a Next.js app backed by a public JSON API (`/front-api/discussion/list`). The scraper opens that API URL with Playwright and paginates with the `lastOffset` cursor until the `history_days` cutoff (30 days by default), keeping up to 80 posts per ticker per day. Likes come from `recommendCount`; the list endpoint does not expose comment counts. In the previous 10-day run, five tickers took roughly 18–25 minutes and ~$0.32 of Apify usage; the 30-day run is longer and uses the configured 3,600 s timeout and $3 cap.
2. **Blind.** The only available Actor supports the English `teamblind.com` site and skips localized `/kr/` paths. Its keyword search is relevance-ranked, so most hits fall outside the configured history window (30 days by default). Blind is therefore disabled by default; enable it if you want long-horizon English TeamBlind posts.
3. **Telegram.** Channel discovery is a web search restricted to `t.me` and is best-effort. Add public channel handles under `sources.telegram.channels.<exchange>` to guarantee they are searched.
4. **Raw cache.** `python -m sentpred scrape --reuse-raw` rebuilds the post table from `data/raw/*.json` without paying to re-scrape.

## Results snapshot — 30-day run (2026-11-08)

Live run: **11,403 social posts** (Naver 11,356 across all five KRX tickers, Telegram 31, LIHKG 16) and **197 Exa news items** across 10 assets, spanning 30 days. The current-sentiment tables use the last 10 days; the model trains on the full 30-day panel.

- Long walk-forward over 1,685 pooled rows: naive baseline MAPE **2.656%**; price-only Ridge **2.673%** (direction 52.4%); price+sentiment Ridge **2.806%**. This is the 1-year price test; sentiment is only available for the last 30 days, so the fair sentiment comparison is the panel below.
- **Ablation on the 30-day panel** (20 sessions x 10 tickers = 190 rows, pooled leave-one-session-out; both rows use the same PLS model class):

| Model | MAPE | Direction | Skill vs naive |
|---|---:|---:|---:|
| Naive “price stays flat” | **2.707%** | — | — |
| Price-only PLS (control) | 2.747% | 51.6% | -1.45% |
| **Price+sentiment PLS (deliverable)** | **2.664%** | **52.7%** | **+1.61%** |

  Sentiment improves MAPE versus the same-class price-only control by **0.083 percentage points** (3.0% relative), and the full model beats the naive baseline by **1.61%**. On the 111 rows with social mentions the model is essentially tied with naive (3.243% vs 3.242%) but beats the price-only control (3.310%); the overall gain is not concentrated only in social-mention rows.
- **Forward-chaining check** (last 12 sessions, 120 rows): naive **2.751%**; price-only PLS **2.800%**; price+sentiment PLS **2.702%** (**+1.80% skill**), direction **60.0%**.
- **Honest caveat:** the paired session-bootstrap 95% interval for the percentage-error improvement still includes zero (LODO -0.192 to +0.040 pp; forward -0.157 to +0.074 pp). The point estimate beats naive in both splits, but with 20 sessions this is not a statistically reliable trading edge. The model sprint tested 10 families (linear, small MLP, deep MLP, trees, neighbors, feature sets, engineered features, blends, sentiment scaling, targets). A small MLP and a Ridge+MLP ensemble looked promising for one seed but did not survive seed variation. An RBF SVR was positive but below PLS, and a 20% PLS+SVR blend moved MAPE by only ~0.003 pp, so the PLS latent-factor model remains the production choice.
- The deliverable is the **price+sentiment model**; `summary.csv` reports `model_*` for it and `price_only_*` for the same-class control.
- Current report: [`reports/live/index.html`](reports/live/index.html) is regenerated by the latest run (self-contained; no usernames or raw posts). It includes a **data-coverage table**, the ablation, bootstrap intervals, feature weights, and a **prediction-check table** comparing recent forecasts with actual next closes.

## Running it daily

Social sentiment is scraped for `history_days` (30 days by default) and filtered to `social_days` (10 days) only for the current-sentiment tables. History accumulates in `data/processed/` across runs (duplicates are removed), so a daily cron job builds the sample. The report states how many sessions of sentiment history exist. With fewer than `min_train_dates` (default 60), the pipeline automatically switches to the pooled short-history sentiment panel (leave-one-session-out) and labels it as such; once 60 sessions exist it runs the full walk-forward. A single 30-day run is still below 60 sessions, so the short panel remains the default until enough daily runs accumulate.

```bash
# cron example: weekdays at 10:30 UTC (18:30 HKT, 19:30 KST), after both closes
30 10 * * 1-5  cd /path/to/repo && .venv/bin/python -m sentpred run
```

## Limitations

- **Short history.** The current-sentiment window (`social_days`) is 10 days; a single 30-day training window (`history_days`) gives roughly 20 sessions and 190 panel rows, still below the 60-session warm-up. On that panel the price+sentiment model beats naive by **1.61% MAPE** (2.664% vs 2.707%) and the same-class price-only control by **0.083 pp**, but the paired bootstrap interval still includes zero. A 10-day snapshot previously showed 73% direction; that was small-sample noise. More daily runs are necessary but not sufficient—sentiment quality must improve too.
- **Lexicon sentiment.** It handles simple negation but not sarcasm or all Korean/Cantonese slang; roughly half of Naver posts still score neutral.
- **Model.** PLS(6) for the short sentiment panel, Ridge for the long walk-forward and as a linear reference. A small MLP, deep MLP, Random Forest, gradient boosting, nearest neighbours and Ridge+MLP blends were tested in `experiments/`; none was as stable as PLS across leave-one-session-out and forward validation. The expected edge is still small and not statistically established, and the report always shows the naive baseline next to it.
- **Market context.** Local index, US overnight, SOX, VIX and FX features are ingested but not used in the final model: they worsened out-of-sample MAPE, so the final model omits them.
- **Calendars.** Exchange holidays are not modelled. Sessions are the price bars plus the next business day.
- **Naver** is sampled at up to 80 posts per ticker per day through its cursor API, not exhaustively; the board can produce thousands of posts per day.
- **Telegram** coverage depends on its channel discovery (a web search restricted to `t.me`). You can add specific public channels in `config.yaml` to guarantee they are searched.
- **Blind** is optional and disabled: the available Actor supports only the English TeamBlind site, and its relevance-ranked search rarely returns posts inside the configured history window (30 days by default).
- **Follower counts** are available from Telegram (`channelSubscribers`) but not from LIHKG or Naver, so reach weighting is strongest there.

## Repository layout

```
config.yaml              assets, exchange calendars, sources, windows, model settings
pyproject.toml           package metadata, dependencies, sentpred console script
Makefile                 install/test/demo/run shortcuts
Dockerfile               Python 3.12 image; runs the offline demo by default
.github/workflows/ci.yml unit tests + offline demo on push/PR
docs/
  architecture.md        data flow, modeling, failure handling, cost controls
  model_card.md          intended use, data, validation, limitations
  explainer.md           plain-English explanation and interview narrative
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
experiments/             model sprint: 10 families, PLS tuning, MLP/ensemble checks
tests/                   81 unit tests (run offline)
reports/demo/            demo report built from synthetic data
reports/live/            sanitized outputs regenerated by the latest run
```
