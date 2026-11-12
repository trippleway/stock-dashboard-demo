# US equity AI-sector dashboard (public demo)

This repository is a public, read-only Streamlit demo of a US-equity AI-sector snapshot. It shows the stored daily top 3, a stock deep dive, the covered universe, and the logged strategy track record. The app reads files already in the repository. It does not call market-data APIs and it does not use credentials.

**Live dashboard:** https://stock-dashboard-d-yxp4hduzpjpv5danzq8qlvmo.streamlit.app

The parent research project is https://github.com/trippleway/stock-analysis-dashboard. Its headline backtest, 209.9% over 36 months against 85.0% for SPY, does not survive the equal-weight, random-selection, and rank-IC controls. The excess return comes from AI-sector exposure, not from stock selection.

## Daily refresh

The parent repository's [`.github/workflows/daily_pipeline.yml`](https://github.com/trippleway/stock-analysis-dashboard/blob/main/.github/workflows/daily_pipeline.yml) runs the full pipeline — prices, fundamentals, FinBERT sentiment, Prophet forecasts, and Claude summaries — at 8:30 New York time on weekdays and pushes the result to `data/latest` here. `data/baseline` remains the fallback if a snapshot fails to parse.

## Run locally

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

`data/latest` is the snapshot the daily job is allowed to replace. `data/baseline` is the committed 2026-11-07 fallback and stays in place when a refresh fails. Each directory holds `manifest.json`, `daily.json`, `universe.json`, `prices.csv`, `recommendations.csv`, and `performance.json`.
