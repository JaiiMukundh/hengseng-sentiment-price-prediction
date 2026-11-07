# Interview narrative and role fit

## 30-second pitch

“I built an end-to-end alternative-data pipeline for Hong Kong and Korean equities. It scrapes LIHKG, Telegram and Naver Finance with Apify, pulls news with Exa, aligns every post to the correct trading session in UTC, engineers sentiment and price features, and validates next-day price forecasts with walk-forward and pooled panel tests. It ships with a CLI, cached raw data, cost caps, 78 tests, CI, a Dockerfile, and an HTML report that shows the model's errors instead of hiding them.”

## 2-minute walkthrough

1. **Problem**: predict the next-session close from public sentiment, without leaking the future.
2. **Ingestion**: 4 social/news feeds, 10 assets, 30-day training window, 10-day current-sentiment window.
3. **Normalization**: one schema, UTC timestamps, URL-safe ticker matching, deduplication.
4. **Alignment**: a post belongs to `(previous close, this close]`; after-close and weekend posts roll forward.
5. **Modeling**: price technicals + sentiment; long walk-forward for the price model; pooled leave-one-session-out for the sentiment panel; nested model selection; bootstrap intervals.
6. **Result**: on the 30-day panel (20 sessions, 190 rows), sentiment model 2.75% MAPE vs 2.71% naive and 46.8% direction; the earlier 10-day 73% direction did not survive the larger sample. The honest conclusion is no reliable next-day edge, and the bootstrap intervals overlap the baseline.
7. **Engineering**: cached raw data, per-source cost caps, retries, partial-run recovery, tests, CI, Docker, docs.

## Role fit

| Role | What this project proves | What to emphasize |
|---|---|---|
| **SDE intern / new grad** | Python, APIs, CLI design, error handling, tests, CI, packaging, Docker | The pipeline is modular and reproducible; 78 tests; `pip install -e .`; CI runs tests + demo |
| **Data engineer** | Ingestion, schemas, timezone alignment, idempotent cache, backfills, cost controls | Raw JSON cache, dedupe key, `(prev close, close]` sessions, partial-run recovery |
| **ML engineer** | Feature engineering, leakage prevention, walk-forward, nested CV, RF/Ridge, bootstrap | Purge gap, session-clustered bootstrap, nested model selection, model card |
| **Quant research intern** | Alpha hypothesis, panel construction, out-of-sample discipline, honest nulls | No overclaiming; baseline comparisons; IC/direction; transaction-cost caveat |
| **Product / data analyst** | Visual reporting, stakeholder communication, ranking, limitations | HTML report, per-exchange rankings, coverage table, prediction-check table |
| **AI agent developer** | LLM/agent tooling, API orchestration, prompt/structured output | Apify + Exa + agent-style pipeline; robust tool boundaries |

## Resume bullets

- Built a Python alternative-data pipeline that ingests Apify social feeds and Exa news for 10 HK/KR equities, normalizes to UTC, and aligns posts to exchange sessions without look-ahead.
- Engineered price, sentiment and market features; validated next-session forecasts with walk-forward, pooled leave-one-session-out, nested model selection and session-clustered bootstrap intervals.
- Reached 2.75% MAPE vs 2.71% naive on a 30-day sentiment panel, with 46.8% direction; documented that the earlier 10-day 73% direction was small-sample noise and that no reliable edge survived the larger sample.
- Shipped a reproducible CLI, cached raw datasets, per-source cost caps, 78 tests, GitHub Actions CI, Dockerfile and a self-contained HTML report.

## Likely interview questions

**“Why not just use FinBERT?”**
“I started with a transparent lexicon so I could debug coverage and leakage. The lexicon marks ~50% of Naver posts neutral, which is the next upgrade: a multilingual finance model. The pipeline is model-agnostic; the hard part is alignment and validation.”

**“How do you know you didn’t leak the future?”**
“Every feature at session t uses only data known at the close of t; the target is t+1. Posts after the close roll to the next session. US features use the previous US close. The walk-forward purges the session before each test block. I also fixed a bug where weekend posts were dropped at the price merge.”

**“Is 73% direction accuracy real?”**
“It was not. On the six-session panel it looked like 73%, but when I expanded to 20 sessions and 190 rows the direction accuracy fell to 46.8% and the MAPE was slightly worse than naive. The bootstrap intervals overlap the baseline. That is the most important result in the project: the validation caught the overfit instead of shipping it.”

**“What was the hardest bug?”**
“Time. Naver returns KST-local strings with several formats, Exa sometimes returns date-only news, and a 17:00 HKT post belongs to tomorrow. I centralized parsing and session mapping, then added tests for `Z`, `+09:00`, naive KST, date-only news and out-of-range timestamps.”

**“What would you do with more time?”**
“Thirty to sixty days of history, a multilingual finance transformer, a direction classifier, a paper-trading backtest with costs, and a small API/dashboard.”

## Honest gaps

- No production deployment, auth, database or monitoring. It is a batch pipeline + static report.
- The sentiment history is small (6 sessions in the 10-day snapshot; more with the 30-day training window).
- The lexicon is not a transformer and misses slang/sarcasm.
- No transaction-cost backtest yet.
- Naver is sampled, not exhaustive.

## What not to claim

- “This predicts stock prices accurately.”
- “73% win rate.”
- “This is a trading strategy.”
- “I found a reliable sentiment edge.”

Say instead: “It is a reproducible research pipeline that tests whether sentiment adds value, and reports the answer honestly.”
