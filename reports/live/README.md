# Live run snapshot - 2026-11-07

Sanitized outputs from a real pipeline run (Apify social scrape + Exa news + Yahoo Finance prices).
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, forecast, levels
- `model_metrics.csv` - walk-forward comparison against the naive baseline
- `forecasts.csv` - next-session predictions

Sentiment history covers 7 sessions, below the 60 needed to train the sentiment model. The report
states this, and the sentiment-model results in this snapshot are not meaningful yet.
