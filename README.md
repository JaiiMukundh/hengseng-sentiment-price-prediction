![CI](https://github.com/JaiiMukundh/hengseng-sentiment-price-prediction/actions/workflows/ci.yml/badge.svg)

# HengSeng Sentiment Price Prediction

A Python pipeline that reads public social posts and news about **10 Hong Kong and Korean stocks**, scores the sentiment, and predicts each stock's **next-session closing price**.

> Research project, not investment advice. Every prediction is shown next to a “price stays flat” baseline so you can see whether the model actually adds value.

**Docs:** [Architecture](docs/architecture.md) · [Model card](docs/model_card.md) · [Plain-English explainer](docs/explainer.md) · [Interview notes](docs/interview_narrative.md)

## What it does

1. **Scrapes social posts** with Apify: Naver Finance discussion boards, Telegram, and LIHKG.
2. **Fetches news** with Exa.
3. **Gets daily prices** from Yahoo Finance.
4. **Aligns every post** to the correct trading session in UTC, without using future information.
5. **Builds features**: price technicals + social/news sentiment + sentiment changes.
6. **Predicts** the next session's closing price with a 6-component PLS model.
7. **Writes a visual report** with sentiment rankings, direction, support/resistance, forecasts and error checks.

## Scope and data

- **Exchanges:** HKEX and KRX, exactly **5 assets each**.
- **Social/news sources:** Apify and Exa only. No direct-Naver or 365-day research data.
- **Latest run:** 11,403 social posts and 197 Exa news items over 30 days.

| Exchange | Assets |
|---|---|
| HKEX | Tencent `0700.HK`, Alibaba `9988.HK`, Meituan `3690.HK`, HSBC `0005.HK`, Xiaomi `1810.HK` |
| KRX | Samsung Electronics `005930.KS`, SK hynix `000660.KS`, NAVER `035420.KS`, Kakao `035720.KS`, Hyundai Motor `005380.KS` |

The report's current-sentiment tables use the last **10 days**; the model trains on the full **30-day** sentiment panel. Once enough daily runs accumulate, it switches to a longer walk-forward.

## Model in one paragraph

The deliverable is a **PLS regression** (a regularized linear model that compresses correlated features into a few components) trained on 19 features: seven price technicals, unweighted social/news sentiment, and six sentiment-dynamics features such as sentiment change, 3-session average, z-score, mention change and news-social divergence. Predictions are shrunk 0.30 toward zero and clipped at ±5%. The control is the same PLS model with price features only. The naive baseline always predicts “the price stays the same.”

## Results

**LODO diagnostic** — 20 sessions × 10 tickers = 190 rows:

| Model | MAPE | Direction | Skill vs naive |
|---|---:|---:|---:|
| Naive | 2.707% | — | — |
| Price-only PLS | 2.747% | 51.6% | -1.45% |
| **Price + sentiment PLS** | **2.663%** | **53.7%** | **+1.63%** |

**Chronological check** — 13 sessions × 10 tickers = 130 rows, trained only on older sessions:

| Model | MAPE | Direction | Skill vs naive |
|---|---:|---:|---:|
| Naive | 2.715% | — | — |
| Price-only PLS | 2.743% | 59.2% | -1.05% |
| **Price + sentiment PLS** | **2.654%** | **60.8%** | **+2.26%** |

The chronological check is the primary result. The paired bootstrap interval still includes zero:
LODO **-0.188 to +0.043 pp**, chronological **-0.156 to +0.053 pp**. So the model beats naive on the
point estimate, but with only 20 sessions it is **not a statistically reliable trading edge**.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                 # add APIFY_API_TOKEN and EXA_API_KEY for live runs
.venv/bin/python -m pytest -q        # 83 tests, offline
.venv/bin/sentpred demo              # offline end-to-end run on synthetic data
```

Live pipeline:

```bash
.venv/bin/sentpred run               # scrape -> news -> prices -> model -> report
```

Individual stages: `sentpred scrape`, `sentpred news`, `sentpred prices`, `sentpred model`, `sentpred report`.

## What you get

- `reports/live/index.html` — self-contained visual report.
- `reports/live/summary.csv` — 10 assets with sentiment, direction, next-session forecast, price-only control, support and resistance.
- `reports/live/short_history_metrics.csv` — LODO diagnostic.
- `reports/live/sentiment_forward_metrics.csv` — chronological check.
- `reports/live/paired_bootstrap.csv` — paired confidence intervals.
- `reports/live/source_ablation.csv` — social-only vs Exa-only comparison.

The report includes per-exchange rankings, sentiment charts, price charts with support/resistance and the n+1 forecast, model errors, and a prediction-check table.

## Limitations

- Only ~20 sentiment sessions, so the confidence interval includes zero.
- Lexicon sentiment handles simple negation but not sarcasm or all Korean/Cantonese slang.
- Naver is sampled, not exhaustive; Telegram and LIHKG coverage is sparse.
- Exchange holidays are not modelled; the target date is a business-day estimate.
- Cached rows lack verified first-seen timestamps, so the point-in-time audit is partial.
- No transaction-cost or paper-trading backtest.
- PLS and the 0.30 shrinkage were selected during research on this history; the chronological check is retrospective, not a fresh holdout.

## Repository layout

```text
config.yaml          assets, exchange calendars, sources, windows, model settings
sentpred/            pipeline code: Apify, Exa, prices, sentiment, features, model, report
experiments/         model sprint: PLS/SVR/MLP/trees/neighbours, tuning and validation
reports/live/        sanitized outputs from the latest real run
reports/demo/        synthetic demo report
docs/                architecture, model card, explainer, interview notes
tests/               83 offline unit tests
Dockerfile           Python 3.12 image; runs the offline demo
.github/workflows/   tests + offline demo on push/PR
```

The package is installable with `pip install -e .`; the `sentpred` console script and `python -m sentpred`
are equivalent. `make test`, `make demo` and `make run` wrap the common commands, and the Docker image
runs the offline demo by default.

## Disclaimer

This is a research and portfolio project. It is not an investment adviser, execution system, or proven
money-making strategy. Do not use it to trade real money without independent validation.
