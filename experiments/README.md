# Model sprint — experiments only

This folder is a separate research sprint. It does **not** add a data source: every model below
uses the same 30-day Apify social + Exa news panel as the production pipeline.

## Setup

- Panel: 20 sentiment-covered sessions x 10 tickers = **190 rows**.
- Target: next-session return, `close(t+1)/close(t) - 1`.
- Validation:
  - **LODO**: pooled leave-one-session-out (train on 19 sessions, test the held-out session).
  - **Forward**: train only on sessions at least two dates before the test session (last 12 sessions).
  - **Paired bootstrap**: resample whole sessions, 5,000 draws, and report the percentage-error
    improvement over naive. This is stricter than comparing two marginal MAPE intervals.
- Control: price technicals only, using the **same model class** as the candidate.

## Families tested

`linear`, `mlp_small`, `mlp_deep`, `trees`, `neighbors`, `feature_sets`, `engineered`, `blend`,
`sent_scale`, `targets`.

The first sprint covered ~360 configs. A second 240-config grid tuned the engineered-feature Ridge
over alpha, sentiment scale, shrinkage (lambda) and prediction clipping. A third grid tuned PLS
components/scale/lambda/clip. A fourth grid tuned an RBF SVR on the engineered features
(C, epsilon, gamma, sentiment scale, lambda, clip).

## Final comparison (all on the same 190-row panel)

| Candidate | LODO MAPE | LODO skill | LODO dir | Forward MAPE | Forward skill | Forward dir |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 2.707% | — | — | 2.751% | — | — |
| Price-only PLS(6) | 2.747% | -1.45% | 51.6% | 2.800% | -1.76% | 57.5% |
| **Price+sentiment PLS(6)** | **2.664%** | **+1.61%** | **52.7%** | **2.702%** | **+1.80%** | **60.0%** |
| Price+sentiment Ridge (engineered) | 2.680% | +1.03% | 52.1% | 2.715% | +1.30% | 61.7% |
| Price+sentiment SVR (RBF, engineered) | 2.681% | +0.97% | 54.8% | 2.725% | +0.98% | 57.5% |
| Price-only Ridge | 2.760% | -1.96% | 48.9% | 2.770% | -0.66% | 50.8% |
| Small MLP (LBFGS, 16) | 2.765% | -2.12% | 46.8% | 2.709% | +1.54% | 56.7% |
| Small MLP (Adam, early stop, 16x8) | 2.707% | +0.01% | 56.9% | 2.769% | -0.66% | 62.5% |

## What worked

- **Sentiment dynamics, not just level.** `sent_chg`, `sent_ma3`, `sent_z`, `mention_chg`,
  `news_chg`, `news_social_div` were the useful additions.
- **Same-class control.** PLS on price-only features is 2.747% / 2.800%; adding sentiment improves it
  by 0.083 pp LODO and 0.098 pp forward.
- **PLS(6)** was the most stable model across both splits. It is deterministic, low-variance on 190
  rows, and does not need the post-standardization sentiment multiplier (it learns latent weights).

## What did not survive

- **Small MLP.** LBFGS had a good forward split but negative LODO; Adam with early stopping was
  roughly flat LODO and negative forward. Results were sensitive to the random seed.
- **Ridge + MLP ensemble.** Weight 0.4 on the MLP looked strong for seed 1 (LODO 2.633%,
  forward 2.614%), but across 10 seeds the mean skill was negative. This is why PLS ships.
- **RBF SVR.** Tuned on the same engineered features (scale 4, C=10, epsilon=0.002, lambda=0.20),
  it is positive but below PLS: 2.681% LODO (+0.97%) and 2.725% forward (+0.98%), with 54.8% /
  57.5% direction. A 20% PLS+SVR blend moved LODO by only ~0.003 pp and forward by ~0.001 pp,
  which is noise, so the extra model was not shipped.
- **Trees / neighbours / deep MLP.** Worse than PLS on both splits.
- **Scaling sentiment for PLS.** Ridge preferred an 8x post-standardization multiplier; PLS
  re-standardizes internally, so the multiplier is redundant. The final config uses 1.0.

## Honest result

PLS(6) beats naive in both the LODO and forward splits, and beats the same-class price-only control.
But the paired bootstrap interval still includes zero:

- LODO: -0.192 to +0.040 pp (mean -0.066 pp)
- Forward: -0.157 to +0.074 pp (mean -0.050 pp)

That means the point estimate is positive, but with 20 sessions it is **not a statistically reliable
trading edge**. The production model ships the point improvement with that caveat in the report and
model card.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python experiments/model_sprint.py --family engineered
PYTHONPATH=. .venv/bin/python experiments/tune_engineered.py
PYTHONPATH=. .venv/bin/python experiments/svr_tuning.py
PYTHONPATH=. .venv/bin/python experiments/validate_engineered.py
```

Key files:

- `model_sprint.py` — 10-family harness.
- `tune_engineered.py` — engineered Ridge / scale / lambda / clip grid.
- `svr_tuning.py` — RBF SVR grid on the engineered features.
- `validate_engineered.py` — final candidate vs controls, LODO + forward + paired CI.
- `results/engineered_tuning.csv` — 240-config Ridge grid.
- `results/pls_tuning.csv` — PLS components/scale/lambda grid.
- `results/svr_tuning.csv` — 180-config SVR grid.
- `results/engineered_validation.csv` — final comparison with controls, SVR and MLPs.
- `results/*.csv` — per-family sprint outputs.
