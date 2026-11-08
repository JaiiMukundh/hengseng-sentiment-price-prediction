# Interview narrative and role fit

## 30-second pitch

“I built an end-to-end alternative-data pipeline for Hong Kong and Korean equities. It scrapes LIHKG, Telegram and Naver Finance with Apify, pulls news with Exa, aligns every post to the correct trading session in UTC, engineers sentiment and price features, and validates next-day price forecasts with walk-forward and pooled panel tests. It ships with a CLI, cached raw data, cost caps, 83 tests, CI, a Dockerfile, and an HTML report that shows the model's errors instead of hiding them.”

## 2-minute walkthrough

1. **Problem**: predict the next-session close from public sentiment, without leaking the future.
2. **Ingestion**: 4 social/news feeds, 10 assets, 30-day training window, 10-day current-sentiment window.
3. **Normalization**: one schema, UTC timestamps, URL-safe ticker matching, deduplication.
4. **Alignment**: a post belongs to `(previous close, this close]`; after-close and weekend posts roll forward.
5. **Modeling**: price technicals + unweighted sentiment levels and dynamics; long walk-forward for the price model; pooled leave-one-session-out diagnostic plus a chronological short-panel check; same-class price-only control; PLS(6), Ridge and MLP comparisons; bootstrap intervals.
6. **Result**: on the 30-day Apify+Exa panel, price+sentiment PLS reaches **2.663% MAPE vs 2.707% naive** (+1.63% skill, 53.7% direction) in the LODO diagnostic, and **2.654% vs 2.715%** (+2.26%, 60.8% direction) in the 130-row chronological check. The paired bootstrap interval still includes zero, so this is a point-estimate improvement, not a proven trading edge.
7. **Engineering**: cached raw data, per-source cost caps, retries, partial-run recovery, tests, CI, Docker, docs.

## Role fit

| Role | What this project proves | What to emphasize |
|---|---|---|
| **SDE intern / new grad** | Python, APIs, CLI design, error handling, tests, CI, packaging, Docker | The pipeline is modular and reproducible; 83 tests; `pip install -e .`; CI runs tests + demo |
| **Data engineer** | Ingestion, schemas, timezone alignment, idempotent cache, backfills, cost controls | Raw JSON cache, dedupe key, `(prev close, close]` sessions, partial-run recovery |
| **ML engineer** | Feature engineering, leakage prevention, walk-forward, nested CV, PLS/Ridge/MLP, bootstrap | Purge gap, same-class ablation, session-clustered bootstrap, nested model selection, model card |
| **Quant research intern** | Alpha hypothesis, panel construction, out-of-sample discipline, honest nulls | No overclaiming; baseline comparisons; direction; transaction-cost caveat |
| **Product / data analyst** | Visual reporting, stakeholder communication, ranking, limitations | HTML report, per-exchange rankings, coverage table, prediction-check table |
| **AI agent developer** | LLM/agent tooling, API orchestration, prompt/structured output | Apify + Exa + agent-style pipeline; robust tool boundaries |

## Resume bullets

- Built a Python alternative-data pipeline that ingests Apify social feeds and Exa news for 10 HK/KR equities, normalizes to UTC, and aligns posts to exchange sessions without look-ahead.
- Engineered price, sentiment-level and sentiment-dynamics features; validated next-session forecasts with walk-forward, pooled leave-one-session-out, nested model selection and session-clustered bootstrap intervals.
- Reached **2.663% MAPE vs 2.707% naive** (+1.63% skill, 53.7% direction) in LODO and **2.654% vs 2.715%** (+2.26%, 60.8% direction) in the chronological check on the 30-day Apify+Exa panel; documented that the paired CI still includes zero.
- Shipped a reproducible CLI, cached raw datasets, per-source cost caps, 83 tests, GitHub Actions CI, Dockerfile and a self-contained HTML report.

## Likely interview questions

**“Why not just use FinBERT?”**
“I started with a transparent lexicon so I could debug coverage and leakage. The lexicon marks ~50% of Naver posts neutral, which is the next upgrade: a multilingual finance model. The pipeline is model-agnostic; the hard part is alignment and validation.”

**“How do you know you didn’t leak the future?”**
“Every feature at session t uses only data known at the close of t; the target is t+1. Posts after the close roll to the next session. US features use the previous US close. The walk-forward purges the session before each test block. I also fixed a bug where weekend posts were dropped at the price merge.”

**“Is the 73% direction accuracy real?”**
“It was not. On the 10-day snapshot it looked like 73%, but when I expanded to 20 sessions and 190 rows the compact model fell to 46.8% direction and did not beat naive. I then ran a model sprint and engineered sentiment dynamics (change, moving average, z-score, mention/news changes). The final PLS model gets 53.7% direction and +1.63% MAPE versus naive in LODO, and 60.8% direction / +2.26% in the 130-row chronological check. The paired bootstrap interval still includes zero, so I present it as a point-estimate improvement, not a proven edge.”

**“What didn’t work?”**
“A small MLP and a Ridge+MLP ensemble looked good for one random seed, but across ten seeds the mean skill was negative. A deep MLP, Random Forest, gradient boosting and nearest neighbours were worse than PLS on the panel. I also tuned an RBF SVR on the earlier weighted-sentiment feature set: it was positive but below PLS, and a 20% PLS+SVR blend changed LODO by only ~0.003 pp, so I did not ship it. Scaling sentiment up helped Ridge (8× was best) but PLS learns its own latent weighting, so the extra multiplier did not help there. The most useful feature engineering was sentiment *change* and divergence, not raw sentiment level. The final model uses unweighted text sentiment so post-publication engagement cannot leak into the feature.”

**“What was the hardest bug?”**
“Time. Naver returns KST-local strings with several formats, Exa sometimes returns date-only news, and a 17:00 HKT post belongs to tomorrow. I centralized parsing and session mapping, then added tests for `Z`, `+09:00`, naive KST, date-only news and out-of-range timestamps. I also fixed a panel-definition bug where filtering to tickers with posts changed the naive baseline.”

**“What would you do with more time?”**
“Thirty to sixty days of history, a multilingual finance transformer, a direction classifier, a paper-trading backtest with costs, and a small API/dashboard.”

## Honest gaps

- No production deployment, auth, database or monitoring. It is a batch pipeline + static report.
- The sentiment history is small: 20 sessions / 190 panel rows in the 30-day run. The point estimate beats naive but the paired bootstrap CI includes zero.
- The PLS components and shrinkage were selected after a model sprint; the 130-row chronological check is encouraging but not a fully independent holdout from all selection.
- The lexicon is not a transformer and misses slang/sarcasm.
- No transaction-cost backtest yet.
- Naver is sampled, not exhaustive.

## What not to claim

- “This predicts stock prices accurately.”
- “73% win rate.”
- “This is a trading strategy.”
- “I found a statistically reliable sentiment edge.”
- “The MLP/ensemble is the reason it works.” (It was seed-sensitive; PLS is the stable model.)
- “The 4-day hold worked.” (That research layer was removed from the submission; it did not replicate.)

Say instead: “It is a reproducible research pipeline that tests whether sentiment adds value, reports a small point-estimate improvement over naive, and is explicit that the confidence interval still includes zero.”
