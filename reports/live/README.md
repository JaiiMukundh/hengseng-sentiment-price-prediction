# Live run snapshot - 2026-11-08 (Apify + Exa only)

Sanitized outputs from the real pipeline: Apify social scrape + Exa news + Yahoo Finance prices.
No usernames or raw post text are included here; raw data stays in the git-ignored `data/` folder.

- `index.html` - self-contained visual report (charts embedded)
- `summary.csv` - per-ticker price, mentions, sentiment, direction, model forecast, same-class price-only control, levels
- `model_metrics.csv` - long 1-year walk-forward vs naive
- `short_history_metrics.csv` - 20-session / 190-row price+sentiment ablation
- `sentiment_nested_metrics.csv` - nested model/alpha sensitivity check
- `short_history_bootstrap.csv` - session-clustered bootstrap intervals
- `sentiment_importances.csv` - feature weights for the configured short model
- `sentiment_coefficients.csv` - Ridge coefficients (linear reference)
- `forecasts.csv` - next-session predictions (same-class price-only control + price+sentiment model)

**Headline:** on the 30-day Apify+Exa panel, price+sentiment PLS reaches **2.664% MAPE vs 2.707% naive**
(+1.61% skill, 52.7% direction) and beats the same-class price-only control (**2.747%**).
The forward check on the last 12 sessions is **2.702% vs 2.751% naive** (+1.80%, 60.0% direction).
The paired bootstrap interval still includes zero (LODO -0.192 to +0.040 pp; forward -0.157 to +0.074 pp),
so this is a small point-estimate improvement, not a statistically reliable trading edge.
