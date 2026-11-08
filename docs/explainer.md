# How this project works — plain English

## One sentence

It reads what people are saying about ten Hong Kong and Korean stocks, scores the mood, lines each post up with the right trading day, and asks whether that mood helps predict the next day's closing price.

## The weather-forecast analogy

A weather forecast does not "know" it will rain. It combines many weak signals — pressure, humidity, wind — and checks, over many days, whether those signals beat a naive guess ("tomorrow will be like today").

This project does the same for prices:

- Social posts and news = the "humidity and pressure".
- Sentiment score = a number from -1 (very negative) to +1 (very positive).
- The model = a rule that combines sentiment with price behaviour.
- Validation = replaying history day by day, without peeking at the future, and comparing against "tomorrow will be like today".

The honest result: on the 30-day Apify+Exa panel the price+sentiment model improves MAPE versus the same-class price-only control (**2.664% vs 2.747%**) and beats the naive baseline (**2.707%**) by 1.61%. A forward check on the last 12 sessions gives **2.702% vs 2.751% naive** (+1.80%). The paired bootstrap interval still includes zero, so this is a small point-estimate improvement, not a proven money machine.

The model sprint tested ten families, including a small feedforward network, RBF-SVR and Ridge+MLP/PLS+SVR ensembles. The MLP and MLP ensemble were seed-sensitive; SVR was positive but below PLS; the PLS+SVR blend was noise. The production choice is a 6-component PLS model on price technicals plus sentiment levels and sentiment dynamics (change, moving average, z-score, mention/news changes, news-social divergence).

## The five stages

1. **Collect.** Apify Actors scrape LIHKG, Telegram and Naver Finance's discussion board. Exa fetches recent news. Yahoo Finance gives daily prices.
2. **Clean.** Every post becomes a row with a UTC timestamp, text, ticker, likes/views and source. URLs are stripped before ticker matching, duplicates are removed, and raw responses are cached.
3. **Score.** A small lexicon scores English, Chinese and Korean finance language. It handles simple negation ("not good") and weights posts by engagement and author reach.
4. **Align.** A post at 17:00 Hong Kong time belongs to the *next* trading session, not the one that just closed. Weekends roll to Monday. This is the step that prevents accidental time travel.
5. **Model and report.** Price technicals plus sentiment features feed two validations: a one-year walk-forward and a small pooled sentiment panel. The output is an HTML report with rankings, charts, forecasts and error checks.

## Why the validation is the hard part

Anyone can fit a model to history and show a good chart. The questions that matter are:

- Did the model use information that was not available at the time? (Look-ahead leakage.)
- Did it beat a trivial baseline?
- Does it still work on data it has never seen?
- Is the sample large enough to trust?

This project explicitly checks all four, and says when the answer is "not yet".

## What the numbers mean

- **MAPE**: average percentage error between predicted and actual price. Lower is better.
- **Direction accuracy**: how often the predicted up/down sign was right. 50% is a coin flip.
- **Skill vs naive**: improvement over "next price = today's price". Positive is better.
- **Nested validation**: choose the model's settings inside each training fold, so the final score is not inflated by tuning on the test data.
- **Bootstrap interval**: resample whole sessions to show how uncertain a number is. Wide interval = don't trust the point estimate.

## What makes it engineering, not just a notebook

- Reproducible config, cached raw data, and a CLI with separate stages.
- Cost caps on every paid API call and retries/timeouts.
- Unit tests for time alignment, normalization, scoring, model edge cases and evaluation.
- CI that runs tests and an offline demo on every push.
- A model card, architecture doc and a plain-English explainer.
- Honest reporting: no hidden failures, no cherry-picked metrics.

## What I would build next

1. Thirty to sixty days of sentiment history (the current bottleneck).
2. A multilingual finance transformer instead of a lexicon.
3. A direction classifier with calibrated probabilities.
4. A small web API and dashboard for live forecasts.
5. A proper paper-trading backtest with transaction costs.
