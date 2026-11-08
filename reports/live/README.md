# Live run snapshot - 2026-11-08 (Apify + Exa only)

Sanitized outputs from the real pipeline: Apify social scrape + Exa news + Yahoo Finance prices.
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, model forecast, price-only control, levels
- `model_metrics.csv` - long 1-year walk-forward vs naive
- `short_history_metrics.csv` - 20-session / 190-row price+sentiment ablation
- `sentiment_nested_metrics.csv` - nested model/alpha sensitivity check
- `short_history_bootstrap.csv` - session-clustered bootstrap intervals
- `sentiment_importances.csv` - feature weights for the configured short model
- `sentiment_coefficients.csv` - Ridge coefficients
- `forecasts.csv` - next-session predictions (price-only control + price+sentiment model)

**Headline:** on the 30-day Apify+Exa panel, price+sentiment improves MAPE versus price-only
(**2.750% vs 2.898%**), but the full model does not beat the naive baseline (2.707%).
Direction is 46.8% vs 48.9% for price-only. Bootstrap intervals overlap the baseline.
This is a small feature gain, not a trading edge; it is documented rather than hidden.
