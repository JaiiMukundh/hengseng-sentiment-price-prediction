# Live run snapshot - 2026-11-07

Sanitized outputs from a real pipeline run (Apify social scrape + Exa news + Yahoo Finance prices).
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, forecast, levels
- `model_metrics.csv` - long walk-forward comparison against the naive baseline
- `forecasts.csv` - next-session predictions (price model + sentiment model)
- `short_history_metrics.csv` - pooled leave-one-session-out sentiment model, 6 sessions x 10 tickers
- `sentiment_coefficients.csv` - standardized Ridge coefficients for the sentiment features

History covers 6-7 trading sessions, below the 60 a full walk-forward needs. The pipeline reports
the pooled short-history panel instead, and labels it as illustrative rather than a robust edge.
