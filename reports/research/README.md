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

## Conclusion

**No reliable sentiment edge was found.** The honest state is:

- Social: positive point estimate, statistically indistinguishable from zero, 19 days.
- News: no edge after costs over 242 sessions.
- Price-only: some short-window technical signal, but not sentiment and not stable.
- The only “winner” was a constant-prediction bug, which the audit caught.

This is a valid research outcome, not a failed pipeline. The project's contribution is the reproducible ingestion, leakage-safe alignment, and falsification discipline.

## What would change the answer

1. **60+ social sessions** (requires more Apify history; current free-tier budget cannot buy it).
2. **Better sentiment extraction**: FinBERT/KR-FinBERT or an LLM scorer; the lexicon marks ~50% of Naver posts neutral.
3. **More cross-sectional breadth**: 15–20 assets, not 10.
4. **Event features**: extreme sentiment, post velocity, author credibility.
5. **Execution realism**: borrow costs, slippage, liquidity, short availability.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python research/edge_lab.py
PYTHONPATH=. .venv/bin/python research/validate_candidate.py
PYTHONPATH=. .venv/bin/python research/news_edge.py

# 365-day free news fetch (Exa credits, no Apify):
PYTHONPATH=. .venv/bin/python research/fetch_news_history.py
```

Results: `reports/research/edge_results.csv`.
