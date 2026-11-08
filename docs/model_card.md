# Model card — sentiment price model

## Overview

This project predicts the next-session close of five HKEX and five KRX stocks using public social posts, Exa news, and daily price bars. It is a research/portfolio project, not an investment adviser or execution system.

Two windows are configured: `window.history_days` (default 30 days) sets the per-run scrape depth for Telegram/Naver/Exa (LIHKG is capped at 2 weeks by its own config) and seeds the training history, while `window.social_days` (default 10 days) filters the report's current-sentiment tables. See [architecture.md](architecture.md) for the pipeline design and the [README](../README.md) for setup and usage.

## Intended use

- Research and demonstration of an end-to-end alternative-data pipeline.
- Comparing social/news sentiment features against a no-change baseline under out-of-sample validation.
- Portfolio evidence for data engineering, feature engineering, and time-series validation.

## Not intended for

- Live trading, position sizing, or personalized investment advice.
- Any market where the user cannot independently verify prices and news.
- Claims of a proven alpha; the sentiment sample is small.

## Data

`history_days` (default 30) is the per-run scrape window and the nominal training look-back; LIHKG's source config caps it at 2 weeks. History accumulates across runs, so training can exceed a single 30-day window. `social_days` (default 10) only filters the report's current-sentiment tables; it does not limit training. The coverage below is the scrape depth.

| Source | Coverage | Notes |
|---|---|---|
| LIHKG | HKEX, keyword search, up to ~14 days | Thread titles; Chinese/Cantonese |
| Telegram | HKEX + KRX, keyword search, up to 30 days | Public channels discovered by the Actor |
| Naver 토론실 | KRX, 30 days, sampled up to 80 posts/ticker/day | JSON API cursor pagination |
| Exa | 30 days, 20 results/asset | News highlights |
| Yahoo Finance | 1 year daily OHLCV | Auto-adjusted |

Social data is public; the repo does not publish raw posts or usernames. The live report is aggregate-only.

## Features

- Price: 1/5/10-session returns, 10-session volatility, RSI(14), distance from 20-day MA, volume z-score.
- Social: log post count, mean and engagement-weighted sentiment, positive/negative shares, engagement, coverage flags.
- News: log count, mean and weighted sentiment, coverage flags.
- Market (ingested, not used in the final model): local index, US overnight, SOX, VIX, FX. Tested and omitted because it worsened out-of-sample MAPE in the previous 10-day snapshot.
- Short sentiment model: compact 10-feature set: `ret_1`, `ret_5`, `vol_10`, `rsi_14`, `social_sent_wmean`, `social_pos_share`, `social_neg_share`, `news_sent_wmean`, `social_n_log`, `has_social`.

## Target

`close(t+1) / close(t) - 1`; predicted price = `close(t) * (1 + predicted return)`.

## Models

- Long walk-forward: `naive` (no change), `ridge_price` (technicals only), and `ridge_sent` (technicals + sentiment). Ridge is used because the 1-year sample is small.
- Short-history sentiment panel (used when sentiment sessions are below `min_train_dates`): pooled leave-one-session-out panel with a compact Ridge model over the feature set above. `social_pos_share` and `social_neg_share` are the sentiment-share features. A shallow Random Forest was tested and overfit; it is not the default.
- Ridge remains a linear reference for coefficients.
- Nested selection of the model family and alpha inside each training fold is reported as a sensitivity check; predictions are shrunk 0.5 toward zero.

## Validation

- Long model: expanding-window walk-forward, 60-session warm-up, 5-session blocks, one-session purge.
- Short sentiment model: pooled leave-one-session-out over sentiment-covered sessions; nested model/alpha selection as a sensitivity check.
- Metrics: MAE, RMSE, MAPE, direction accuracy, skill vs naive. See `reports/live/model_metrics.csv`, `reports/live/short_history_metrics.csv`, and `reports/live/sentiment_nested_metrics.csv` for the latest pipeline run.

## Metrics snapshot — 30-day Apify + Exa run (2026-11-08)

The 30-day run is the current evidence. A previous 10-day snapshot showed 73% direction; the larger sample showed that was small-sample noise.

- Long walk-forward, 1,685 rows: naive **2.656%** MAPE, price-only **2.673%**, price+sentiment **2.806%**. Sentiment only exists for the last 30 days, so this is not the fair sentiment test.
- **Ablation on the 30-day panel, 190 rows / 20 sessions:**

| Model | MAPE | Direction |
|---|---:|---:|
| Naive | **2.707%** | — |
| Price-only Ridge | 2.898% | 48.9% |
| **Price+sentiment Ridge** | **2.750%** | 46.8% |

Sentiment improves MAPE versus price-only by **0.148 pp**, but the full model does not beat naive; bootstrap intervals overlap the baseline. The deliverable is the price+sentiment model; price-only is the control.

## Limitations

- The current-sentiment window (`social_days`, 10 days) is separate from the 30-day training window; a single 30-day Apify run gives ~20 sessions, still below the 60-session warm-up.
- Lexicon sentiment handles simple negation but not sarcasm or all Korean/Cantonese slang; ~50% of Naver posts score neutral.
- Naver is sampled, not exhaustive.
- Exchange holidays are not explicitly modeled.
- Market context features were tested but omitted because they worsened out-of-sample MAPE; they remain in the dataset for experiments.
- The product/business claims of the company are unrelated to this model's validity.

## Ethical / regulatory

- No personal data is published; author names are dropped from the report.
- The project is research, not investment advice.
- Respect each source's terms and rate limits; Apify Actors and Exa handle access.
