# US equity AI-sector dashboard (public demo)

This repository is a public, read-only Streamlit demo of a US-equity AI-sector snapshot. It shows the stored daily top 3, a stock deep dive, the covered universe, and the logged strategy track record. The app reads files already in the repository. It does not call market-data APIs and it does not use credentials.

**Live dashboard:** https://stock-dashboard-d-yxp4hduzpjpv5danzq8qlvmo.streamlit.app

The parent research project is https://github.com/trippleway/stock-analysis-dashboard. Its headline backtest, 209.9% over 36 months against 85.0% for SPY, does not survive the equal-weight, random-selection, and rank-IC controls. The excess return comes from AI-sector exposure, not from stock selection.

## Daily refresh

GitHub Actions runs `.github/workflows/daily_snapshot.yml` every day at 15:00 UTC, after the US cash open, and on manual dispatch. The job runs `python scripts/refresh_snapshot.py`.

The job updates `data/latest` only:

- It replaces price history from public Yahoo Finance quotes when a ticker download succeeds.
- It refreshes the latest close and moving averages from those prices.
- It recomputes a composite only when the stored component scores are already present.

The same job carries these forward and does not rebuild them:

- Daily picks, including rank, scores, news, sentiment, and written rationale
- Company profiles
- Stored Prophet forecast paths
- `recommendations.csv` and `performance.json`

It does not run FinBERT or Claude, and it does not refit forecasts. `data/baseline` is never written. If the refresh fails validation, the job does not commit, so `data/latest` and `data/baseline` stay as they were. A successful run commits only `data/latest`.

The app opens `data/latest` when that snapshot parses. If it does not, the app opens `data/baseline`.

## Run locally

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

`data/latest` is the snapshot the daily job is allowed to replace. `data/baseline` is the committed 2026-11-07 fallback and stays in place when a refresh fails. Each directory holds `manifest.json`, `daily.json`, `universe.json`, `prices.csv`, `recommendations.csv`, and `performance.json`.
