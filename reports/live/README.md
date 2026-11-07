# Live run snapshot - 2026-11-08 (30-day training window)

Sanitized outputs from a real pipeline run (Apify social scrape + Exa news + Yahoo Finance prices).
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, forecasts, levels
- `model_metrics.csv` - long 1-year walk-forward vs naive
- `short_history_metrics.csv` - 20-session / 190-row pooled panel
- `sentiment_nested_metrics.csv` - nested model/alpha sensitivity check
- `short_history_bootstrap.csv` - session-clustered bootstrap intervals
- `sentiment_importances.csv` - feature weights for the configured short model
- `sentiment_coefficients.csv` - Ridge coefficients
- `forecasts.csv` - next-session predictions (price model + sentiment model)

**Headline:** 30 days of data produced a negative result. Naive MAPE 2.707%; sentiment model 2.750%
(46.8% direction); nested 2.801% (43.1%); bootstrap intervals overlap the baseline. The earlier
10-day 73% direction was small-sample noise. This is documented rather than hidden.
