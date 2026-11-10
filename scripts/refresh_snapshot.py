# -*- coding: utf-8 -*-
"""Refresh data/latest from public Yahoo Finance prices.

No API keys. News, sentiment, profiles, Prophet paths, and the track record
are copied forward from the previous valid snapshot. data/baseline is never
written. On failure the previous latest snapshot is left in place.

Run from the demo repository root:
    python scripts/refresh_snapshot.py
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "data" / "baseline"
LATEST = ROOT / "data" / "latest"
REQUIRED = (
    "manifest.json",
    "daily.json",
    "performance.json",
    "recommendations.csv",
    "prices.csv",
    "universe.json",
)
SAFE_INFO = (
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "profitMargins",
    "grossMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "currentRatio",
    "targetMeanPrice",
    "recommendationMean",
)
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

log = logging.getLogger("refresh_snapshot")


def _assert_outside_baseline(path: Path, root: Path) -> None:
    """Refuse any write that would land in data/baseline.

    Args:
        path: Candidate write path.
        root: Demo repository root.

    Raises:
        RuntimeError: When path is the baseline directory or inside it.
    """
    baseline = (root / "data" / "baseline").resolve()
    resolved = path.resolve()
    if resolved == baseline or baseline in resolved.parents:
        raise RuntimeError(f"Refusing to modify baseline: {resolved}")


def snapshot_valid(path: Path) -> bool:
    """Return True when a snapshot directory has the files the app can open.

    Args:
        path: Directory that may contain a snapshot.

    Returns:
        True when required files parse and LRCX and INTU are present.
    """
    try:
        if not path.is_dir():
            return False
        for name in REQUIRED:
            if not (path / name).is_file():
                return False
        universe = json.loads((path / "universe.json").read_text(encoding="utf-8"))
        json.loads((path / "daily.json").read_text(encoding="utf-8"))
        json.loads((path / "performance.json").read_text(encoding="utf-8"))
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        prices = pd.read_csv(path / "prices.csv")
        recs = pd.read_csv(path / "recommendations.csv")
        tickers = {row.get("ticker") for row in universe.get("tickers", []) if isinstance(row, dict)}
        if "LRCX" not in tickers or "INTU" not in tickers:
            return False
        if prices.empty or "close" not in prices.columns or recs.empty:
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — any parse failure means the snapshot is unusable
        log.warning("Snapshot %s is not valid: %s", path, exc)
        return False


def _source_dir(root: Path) -> Path:
    """Pick the snapshot to carry forward. Latest wins when it parses."""
    latest = root / "data" / "latest"
    baseline = root / "data" / "baseline"
    backup = root / "data" / "_latest_previous"
    if snapshot_valid(latest):
        return latest
    if snapshot_valid(backup):
        return backup
    if snapshot_valid(baseline):
        return baseline
    raise RuntimeError("No valid snapshot to carry forward")


def _moving_averages(frame: pd.DataFrame) -> dict[str, float | None]:
    """Last simple moving averages of close for 5, 20, 60, and 200 sessions.

    Args:
        frame: One ticker's price rows with a close column.

    Returns:
        Mapping of ma5, ma20, ma60, and ma200. A window longer than the
        series is null.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    out: dict[str, float | None] = {}
    for window, key in ((5, "ma5"), (20, "ma20"), (60, "ma60"), (200, "ma200")):
        if len(close) < window:
            out[key] = None
            continue
        out[key] = float(close.rolling(window).mean().iloc[-1])
    return out


def _recompute_composite(row: dict[str, Any], weights: dict[str, float]) -> tuple[float | None, str | None]:
    """Weighted composite from stored component scores.

    Args:
        row: Universe row. Missing components are not invented.
        weights: Score weights stored on the snapshot.

    Returns:
        The recomputed composite, or the previous composite plus a note when
        a component score was not in the snapshot.
    """
    parts = {
        "technical": row.get("tech_score"),
        "fundamental": row.get("fund_score"),
        "valuation": row.get("valuation_score"),
        "prediction": row.get("prediction_score"),
        "volume": row.get("volume_score"),
        "news": row.get("news_score"),
    }
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in parts.values()):
        return row.get("composite_score"), "composite carried forward"
    total = 0.0
    for key, weight in weights.items():
        total += float(parts[key]) * float(weight)
    return round(total, 2), None


def _normalize_prices(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize a fetched OHLCV frame to the snapshot columns.

    Args:
        ticker: Ticker symbol.
        frame: Raw frame with a datetime index or a date column.

    Returns:
        Columns date, ticker, open, high, low, close, volume.
    """
    df = frame.copy()
    if "date" not in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        df = df.reset_index(drop=True)
        df["date"] = idx.strftime("%Y-%m-%d")
    rename = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename)
    out = pd.DataFrame({
        "date": df["date"].astype(str).str.slice(0, 10),
        "ticker": ticker,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    })
    return out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")


def _safe_info(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the public fundamental fields. None when the fetch failed."""
    if raw is None:
        return None
    kept: dict[str, Any] = {}
    for key in SAFE_INFO:
        if key in raw and raw.get(key) is not None:
            value = raw.get(key)
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                continue
            kept[key] = value
    return kept


def _fetch_yfinance(ticker: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Download one ticker from Yahoo Finance. No API key.

    Args:
        ticker: Ticker symbol.

    Returns:
        Price frame and safe info. Prices are None when the download fails.
        Info is None when only the quote profile failed.
    """
    import yfinance as yf

    instrument = yf.Ticker(ticker)
    hist = instrument.history(period="1y", auto_adjust=False)
    if hist is None or hist.empty:
        return None, None
    info: dict[str, Any] | None
    try:
        raw = instrument.info or {}
        info = _safe_info(raw if isinstance(raw, dict) else {})
    except Exception as exc:  # noqa: BLE001 — keep prices if the info call fails
        log.warning("Info failed for %s: %s", ticker, exc)
        info = None
    return _normalize_prices(ticker, hist), info


def _write_json(path: Path, payload: Any, root: Path) -> None:
    """Write JSON after checking the path is not the baseline."""
    _assert_outside_baseline(path, root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _publish(root: Path, staging: Path) -> None:
    """Replace data/latest with a validated staging directory.

    Args:
        root: Demo repository root.
        staging: Complete new snapshot directory.

    Raises:
        RuntimeError: When staging is invalid or a path points at baseline.
    """
    if not snapshot_valid(staging):
        raise RuntimeError("Refusing to publish an invalid snapshot")
    latest = root / "data" / "latest"
    backup = root / "data" / "_latest_previous"
    _assert_outside_baseline(staging, root)
    _assert_outside_baseline(latest, root)
    _assert_outside_baseline(backup, root)
    if backup.exists():
        old = root / "data" / f"_latest_previous_{int(time.time())}"
        _assert_outside_baseline(old, root)
        backup.rename(old)
    if latest.exists():
        latest.rename(backup)
    try:
        staging.rename(latest)
    except Exception:
        if backup.exists() and not latest.exists():
            backup.rename(latest)
        raise


def run_refresh(
    root: Path,
    fetch_one: Callable[[str], tuple[pd.DataFrame | None, dict[str, Any] | None]] | None = None,
) -> int:
    """Build a new latest snapshot and publish it only when it validates.

    Args:
        root: Demo repository root containing data/baseline and data/latest.
        fetch_one: Optional fetcher for tests. Defaults to Yahoo Finance.

    Returns:
        0 when a validated snapshot replaces data/latest. Non-zero when the
        previous latest and the baseline are left untouched.
    """
    fetcher = fetch_one or _fetch_yfinance
    staging = root / "data" / f"_latest_staging_{int(time.time())}"
    try:
        _assert_outside_baseline(staging, root)
        source = _source_dir(root)
        staging.mkdir(parents=True)
        universe = json.loads((source / "universe.json").read_text(encoding="utf-8"))
        daily = json.loads((source / "daily.json").read_text(encoding="utf-8"))
        weights = universe.get("weights") if isinstance(universe.get("weights"), dict) else {}
        old_prices = pd.read_csv(source / "prices.csv")
        rows = []
        updated = 0
        for row in universe.get("tickers", []):
            if not isinstance(row, dict) or not row.get("ticker"):
                continue
            ticker = str(row["ticker"])
            carried = ["news", "sentiment", "profile", "forecast", "track_record"]
            try:
                fetched_prices, fetched_info = fetcher(ticker)
            except Exception as exc:  # noqa: BLE001 — one ticker must not abort the rest
                log.warning("Refresh failed for %s: %s", ticker, exc)
                fetched_prices, fetched_info = None, None
            if fetcher is not _fetch_yfinance:
                pass
            elif fetched_prices is None:
                time.sleep(0.6)
            else:
                time.sleep(0.6)
            previous = old_prices[old_prices["ticker"] == ticker].copy()
            if fetched_prices is None or fetched_prices.empty:
                frame = previous
                row["price_note"] = "Previous row kept; Yahoo Finance refresh failed"
                carried.append("prices")
            else:
                frame = fetched_prices.sort_values("date")
                row["price_note"] = None
                updated += 1
                if not frame.empty:
                    row["current_price"] = float(pd.to_numeric(frame["close"], errors="coerce").iloc[-1])
                row["moving_averages"] = _moving_averages(frame)
            if fetched_info is None:
                carried.append("fundamentals")
            else:
                row["info"] = fetched_info
            composite, note = _recompute_composite(row, weights)
            row["composite_score"] = composite
            if note:
                carried.append(note)
            row["carried_forward"] = carried
            rows.append((row, frame))
        if updated == 0:
            log.error("No ticker received new prices; leaving latest and baseline untouched")
            return 1
        prices_out = pd.concat([frame for _, frame in rows if frame is not None and not frame.empty], ignore_index=True)
        universe["tickers"] = [row for row, _ in rows]
        picks_by_ticker = {row["ticker"]: row for row, _ in rows}
        for pick in daily.get("picks", []):
            if not isinstance(pick, dict):
                continue
            fresh = picks_by_ticker.get(pick.get("ticker"))
            if fresh and fresh.get("current_price") is not None and "prices" not in (fresh.get("carried_forward") or []):
                pick["current_price"] = fresh["current_price"]
        manifest = {
            "snapshot_id": "latest",
            "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source": "Yahoo Finance public prices; news, sentiment, profiles, Prophet paths, and track record carried forward",
            "updated_tickers": updated,
            "notes": "Failed tickers keep their previous price row. Baseline is unchanged.",
        }
        _write_json(staging / "universe.json", universe, root)
        _write_json(staging / "daily.json", daily, root)
        _write_json(staging / "manifest.json", manifest, root)
        _assert_outside_baseline(staging / "prices.csv", root)
        prices_out[PRICE_COLUMNS].to_csv(staging / "prices.csv", index=False)
        for name in ("performance.json", "recommendations.csv"):
            target = staging / name
            _assert_outside_baseline(target, root)
            shutil.copyfile(source / name, target)
        if not snapshot_valid(staging):
            log.error("New snapshot failed validation")
            return 1
        _publish(root, staging)
        log.info("Published latest snapshot with %s updated tickers", updated)
        return 0
    except Exception as exc:  # noqa: BLE001 — failure must not replace a good snapshot
        log.exception("Refresh aborted: %s", exc)
        return 1
    finally:
        if staging.exists():
            # Publish renames staging away on success. A leftover is an unpublished attempt.
            leftover = root / "data" / f"_latest_staging_old_{int(time.time())}"
            try:
                _assert_outside_baseline(leftover, root)
                staging.rename(leftover)
            except Exception:  # noqa: BLE001
                log.warning("Could not move unpublished staging directory %s", staging)


def main() -> None:
    """CLI entry point. Exits non-zero when the snapshot is not replaced."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    code = run_refresh(ROOT)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
