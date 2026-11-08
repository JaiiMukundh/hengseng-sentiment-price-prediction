# Live run snapshot - 2026-11-08 (Apify + Exa only)

Sanitized outputs from the real pipeline: Apify social scrape + Exa news + Yahoo Finance prices.
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, model forecast, same-class price-only control, levels
- `model_metrics.csv` - long 1-year walk-forward vs naive
- `short_history_metrics.csv` - 20-session / 190-row leave-one-session-out diagnostic
- `sentiment_forward_metrics.csv` - chronological 13-session / 130-row short-panel check with a one-session label purge
- `sentiment_nested_metrics.csv` - nested model/alpha sensitivity check
- `short_history_bootstrap.csv` - marginal session-clustered bootstrap intervals
- `paired_bootstrap.csv` - paired session-bootstrap of model-minus-naive percentage error (LODO + chronological)
- `sentiment_importances.csv` - feature weights for the configured short model
- `sentiment_coefficients.csv` - Ridge coefficients (linear reference)
- `forecasts.csv` - next-session predictions (same-class price-only control + price+sentiment model)
- `source_ablation.csv` and `source_coverage.csv` - cached Apify/Exa source comparison and per-ticker coverage
- `availability_audit.csv` - publication and verified collection timestamp coverage

**Headline:** on the 30-day Apify+Exa panel, the price+sentiment PLS model reaches **2.663% MAPE**
vs **2.707%** naive (+1.63% skill, 53.7% direction) in the leave-one-session-out diagnostic, and
**2.654%** vs **2.715%** naive (+2.26%, 60.8% direction) in the chronological check. The
same-class price-only control is **2.747%** LODO and **2.743%** chronological.

The paired session-bootstrap 95% interval for model-minus-naive percentage error (`paired_bootstrap.csv`)
still includes zero: **-0.188 to +0.043 pp** LODO and **-0.156 to +0.053 pp** chronological. The point estimate
beats naive in both splits, but this is not a statistically reliable trading edge. Settings were
selected after prior experiments on this history, so the chronological result is retrospective,
not an untouched holdout.

The source ablation gives **2.658%** for social-only and **2.739%** for Exa-only on the same
chronological folds. Naver supplies almost all social observations; Telegram and LIHKG coverage
is sparse, especially for KRX Telegram. The headline forecast uses the sentiment PLS output from
the last completed price session; the price-only forecast remains a control. No Apify or Exa
request was made for this regeneration.

All 11,403 cached social rows and 197 cached Exa rows lack a verified first-seen time; 14 news
timestamps are exactly midnight UTC and may have only date-level precision. Future live ingestions
now preserve first-seen time. The target session date is a business-day estimate because exchange
holidays are not encoded.
