# Edge research audit — 2026-11-08

## Question

Can social or news sentiment produce a reliable next-day edge for the 10 HK/KR assets, after realistic transaction costs?

## Data

| Dataset | Coverage | Rows | Sessions |
|---|---|---|---|
| `social30` | 30 days of Naver/LIHKG/Telegram + news | 190 | 20 |
| `news120` | 120 days of Exa news | 251 | 38 |
| `news365` | 365 days of Exa news | 1,194 | 242 |

Social data is the bottleneck: 30 days gives only ~20 trading sessions. News was expanded to 365 days for free via Exa.

## Methods

- Forward-chaining: train on sessions up to `t-2` (one-session purge), predict session `t`.
- Targets: absolute next-day return, cross-sectional relative return, and ranking.
- Models: Ridge, ElasticNet, Random Forest, HistGradientBoosting.
- Features: price technicals, social sentiment/levels/changes/ranks, news sentiment/counts/ranks, market context.
- Portfolios: long top 2 / short bottom 2 (or top/bottom k), equal weight, daily rebalance.
- Costs: 0, 10, 25, 50 bps per side.
- Uncertainty: session-clustered bootstrap (2,000 resamples).

## What happened

1. **Social raw sentiment looked positive.** Long high-sentiment / short low-sentiment earned 1.16%/day gross, 0.66%/day net at 25 bps, Sharpe 3.8, 19 days. But the bootstrap 95% CI was **[-0.61%, +2.03%]**, which includes zero. KR-only was negative; HK-only had 4 days. Not reliable.
2. **News models looked great — then failed falsification.** The top result was ElasticNet on news features: rank IC 0.254, 0.82%/day net, Sharpe 3.76. But in later folds ElasticNet shrank every coefficient to zero; predictions became constant, and the long/short backtest was just a fixed ticker basket (0.9%/day gross even with zero signal). Adding a constant-prediction guard killed the result.
3. **365 days of news showed no edge.** Raw news sentiment, news volume and news hit-share all had negative net returns after 25 bps; model direction was ~49–51%; rank IC ~0.02–0.05; bootstrap CIs overlapped zero.
4. **Cross-sectional models did not rescue it.** Technical-only models showed 11-day rank IC ~0.24–0.32, but that is price momentum on 11 days, not sentiment, and not stable.
5. **Late-session sentiment** (last 2 hours before close) had 14 usable days and was negative after costs.
6. **Five years of price-only cross-sectional reversal/momentum** also failed: gross long/short returns were ~0.03–0.21%/day, and 25 bps daily rebalancing costs made every variant negative (`research/price_edge.py`).
7. **Transformer sentiment + 15-asset panel** (the next escalation): added LG Energy Solution, Samsung SDI, POSCO Holdings, KB Financial and Celltrion (7,103 more Naver posts, ~$0.13), scored 18,506 posts with KR-FinBERT/FinBERT, and tested logistic/RF/HGB classifiers plus a rank ensemble. Panel: 22 sessions, 211 rows, 14 assets.
   - Raw transformer sentiment (k=2): **+0.23%/day net**, Sharpe 1.35, bootstrap 95% CI **[-0.97%, +1.50%]** — still includes zero.
   - `social_pos_share` (k=2): +0.69%/day net, Sharpe 3.47, CI **[-0.65%, +2.21%]** — includes zero.
   - Regression models: negative net and rank IC.
   - Classifiers: AUC **0.40–0.49** (worse than random); ensemble net **-0.86%/day**.
   - Conclusion: transformers and more cross-sectional assets did not create a reliable edge.

## Research-only direct Naver backfill

After the paid tier was exhausted, the user allowed a research-only direct Naver API backfill (production remains Apify-only). This is free.

- **90 days, 10 KR assets:** 29,124 posts, 64 sessions, 592 panel rows.
- Best holding variant: **H=3 days, +0.311%/day net**, Sharpe 1.77, 95% CI **[-0.406%, +1.042%]**.
- H=5: +0.282%/day, Sharpe 1.63, CI **[-0.354%, +1.001%]**.
- H=1/2 were negative; H=10 ~zero.
- Regression and classification models remained weak (AUC ~0.50; most net returns negative).
- **Still not statistically reliable.**

A 365-day backfill was then used for a **pre-registered confirmatory test** of the closest rule (long-only, H=4, positive-share top 2).

- **365 days, 10 KR assets:** 65,699 transformer-scored posts, 244 sessions, 2,065 panel rows.
- Long/short H=4: **+0.170%/day net**, Sharpe 1.04, 95% CI **[-0.142%, +0.510%]**.
- **Pre-registered long-only H=4: +0.106%/day net**, Sharpe 0.93, 95% CI **[-0.128%, +0.335%]**.
- H=3/H=5 long-short: +0.140%/+0.133%, CIs include zero.
- Regression models negative; classifiers AUC **0.49–0.52**; net long-short negative.
- **The 180-day long-only H=4 result (+0.35%/day) did not replicate.** It was sample-specific noise.

**Final conclusion: no reliable edge.** This is not a budget issue anymore: 244 sessions, 10 assets, 65k transformer-scored posts, holding periods, costs, regression, classification and ensembles all fail the reliability bar.

## Conclusion

**No reliable sentiment edge was found.** The honest state is:

- Social: positive point estimate, statistically indistinguishable from zero, 19 days.
- News: no edge after costs over 242 sessions.
- Price-only: some short-window technical signal, but not sentiment and not stable.
- The only “winner” was a constant-prediction bug, which the audit caught.

This is a valid research outcome, not a failed pipeline. The project's contribution is the reproducible ingestion, leakage-safe alignment, and falsification discipline.

## What would change the answer

1. **A pre-registered single strategy** (one signal, one holding period) tested on 365+ days, not a sweep of H/k/sides.
2. **Better sentiment labels**: domain-tuned multilingual finance model or an LLM scorer evaluated against human labels.
3. **Event features**: extreme sentiment, post velocity, author credibility, news/social divergence.
4. **Execution realism**: borrow costs, slippage, liquidity, short availability.
5. **More assets**, not just more days, to stabilize cross-sectional portfolios.

## Exact blocker

- No budget blocker remains for research: the free direct Naver backfill reached **365 days / 244 sessions / 65k transformer-scored posts**.
- The blocker is **signal quality**: no rule survives costs and bootstrap on the larger sample. More of the same data will not fix that.
- A genuine attempt would need: better labels/domain-tuned sentiment, event features, and a pre-registered single strategy — not more multiple testing.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python research/edge_lab.py
PYTHONPATH=. .venv/bin/python research/validate_candidate.py
PYTHONPATH=. .venv/bin/python research/news_edge.py
PYTHONPATH=. .venv/bin/python research/price_edge.py
PYTHONPATH=. .venv/bin/python research/transformer_edge.py

# transformer scoring (optional; install torch/transformers first):
PYTHONPATH=. .venv/bin/python research/transformer_sentiment.py

# 365-day free news fetch (Exa credits, no Apify):
PYTHONPATH=. .venv/bin/python research/fetch_news_history.py
```

Results: `reports/research/edge_results.csv`.
