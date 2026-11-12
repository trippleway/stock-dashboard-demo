# -*- coding: utf-8 -*-
"""Static US equity AI-sector dashboard.

The app reads data/latest when that snapshot parses, otherwise data/baseline.
It does not call market APIs and does not use credentials.

Run from this directory:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
MISSING = "Not in the 2026-11-07 snapshot"
PAGES = ["Daily Picks", "Stock Deep Dive", "Universe Overview", "Strategy Performance"]
ACCENT = "#15385C"
TEXT = "#10141A"
TEXT_DIM = "#5B6472"
BORDER = "#E3E6EA"
POSITIVE = "#0E7C5A"
NEGATIVE = "#B3261E"
STATUS_LABELS = {
    "buy_now": "Buy now",
    "watch": "Near the entry zone",
    "wait": "Wait for a pullback",
    "avoid": "Do not chase",
}
REQUIRED = (
    "manifest.json",
    "daily.json",
    "performance.json",
    "recommendations.csv",
    "prices.csv",
    "universe.json",
)


def _snapshot_valid(path: Path) -> bool:
    """Return True when the directory is a snapshot the app can render.

    Args:
        path: Candidate snapshot directory.

    Returns:
        True when the required files parse and LRCX and INTU are present.
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
        prices = pd.read_csv(path / "prices.csv")
        recommendations = pd.read_csv(path / "recommendations.csv")
        tickers = {row.get("ticker") for row in universe.get("tickers", []) if isinstance(row, dict)}
        return "LRCX" in tickers and "INTU" in tickers and not prices.empty and not recommendations.empty
    except Exception:
        return False


def resolve_snapshot(root: Path | None = None) -> tuple[Path, str]:
    """Load latest when it parses, otherwise the committed baseline.

    Args:
        root: Repository root. Defaults to the directory that contains this file.

    Returns:
        The snapshot directory and the label ``latest`` or ``baseline``.
    """
    base = root or ROOT
    latest = base / "data" / "latest"
    baseline = base / "data" / "baseline"
    if _snapshot_valid(latest):
        return latest, "latest"
    return baseline, "baseline"


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _read_table(path: str) -> pd.DataFrame:
    """Read a CSV snapshot."""
    return pd.read_csv(path)


def _load(directory: Path) -> dict[str, Any]:
    """Read every file the four pages need.

    Args:
        directory: Valid snapshot directory.

    Returns:
        Daily picks, universe rows, prices, performance, recommendations, and manifest.
    """
    universe = _read_json(str(directory / "universe.json"))
    rows = [row for row in universe.get("tickers", []) if isinstance(row, dict)]
    by_ticker = {str(row.get("ticker")): row for row in rows}
    return {
        "manifest": _read_json(str(directory / "manifest.json")),
        "daily": _read_json(str(directory / "daily.json")),
        "performance": _read_json(str(directory / "performance.json")),
        "recommendations": _read_table(str(directory / "recommendations.csv")),
        "prices": _read_table(str(directory / "prices.csv")),
        "universe": rows,
        "by_ticker": by_ticker,
        "weights": universe.get("weights") if isinstance(universe.get("weights"), dict) else {},
    }


def _num(value: Any) -> float | None:
    """Return a finite float, or None."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _pct(value: Any, digits: int = 1) -> str:
    """Format a percent-point number such as 6.57 as +6.6%."""
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:+.{digits}f}%"


def _frac(value: Any, digits: int = 2) -> str:
    """Format a fraction such as 0.343 as +34.30%."""
    number = _num(value)
    if number is None:
        return "—"
    return f"{number * 100:+.{digits}f}%"


def _price(value: Any) -> str:
    """Format a dollar price."""
    number = _num(value)
    if number is None:
        return "—"
    return f"${number:,.2f}"


def _unavailable(value: Any) -> str:
    """Render a snapshot field. Missing values say Unavailable."""
    if value is None or value == "":
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI, and MACD from the static close series.

    Args:
        frame: One ticker's OHLCV rows.

    Returns:
        The same rows with indicator columns. No future bars are used.
    """
    out = frame.sort_values("date").copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma60"] = close.rolling(60, min_periods=60).mean()
    out["ma200"] = close.rolling(200, min_periods=200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    out["rsi"] = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-12)))
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    return out


def _style(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared light chart theme."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT),
        margin=dict(l=48, r=16, t=28, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def _prices_for(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return one ticker's OHLCV rows."""
    frame = prices[prices["ticker"].astype(str) == ticker].copy()
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["close"])


def _status_label(status: str | None) -> str:
    """Map an entry status key to the dashboard label."""
    if not status:
        return MISSING
    return STATUS_LABELS.get(str(status), str(status))


def _banner(label: str, manifest: dict[str, Any]) -> None:
    """Tell the reader which snapshot is on screen."""
    as_of = manifest.get("as_of") or "2026-11-07"
    if label == "latest":
        text = (
            f"Showing snapshot: latest · as of {as_of}. "
            "data/baseline stays available if this snapshot fails to parse."
        )
    else:
        text = (
            "Showing snapshot: baseline (fallback). "
            "data/latest did not parse, so the committed 2026-11-07 baseline is on screen."
        )
    st.markdown(
        f"<p class='banner'>{html.escape(text)}</p>",
        unsafe_allow_html=True,
    )


def _technical_figure(frame: pd.DataFrame) -> go.Figure:
    """Candles, moving averages, and RSI for one static price history."""
    view = _indicators(frame).tail(180)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(
        go.Candlestick(
            x=view["date"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Price",
            increasing_line_color=POSITIVE,
            decreasing_line_color=NEGATIVE,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma20"], name="MA20", line=dict(color=ACCENT, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma60"], name="MA60", line=dict(color=TEXT_DIM, width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["date"], y=view["rsi"], name="RSI", line=dict(color=ACCENT, width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=TEXT_DIM, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=TEXT_DIM, row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, 680)


def _forecast_figure(frame: pd.DataFrame, forecast: dict[str, Any] | None) -> go.Figure:
    """Close path plus a stored forecast band when the snapshot has one."""
    view = frame.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["date"], y=view["close"], name="Close", line=dict(color=ACCENT, width=2)))
    if forecast:
        dates = list(forecast.get("prediction_dates") or [])
        upper = list(forecast.get("prediction_upper") or [])
        lower = list(forecast.get("prediction_lower") or [])
        path = list(forecast.get("prediction_prices") or [])
        if dates and upper and lower:
            fig.add_trace(go.Scatter(x=dates, y=upper, name="Upper", line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=lower,
                    name="Stored band",
                    fill="tonexty",
                    fillcolor="rgba(21,56,92,0.15)",
                    line=dict(width=0),
                )
            )
        if dates and path:
            fig.add_trace(go.Scatter(x=dates, y=path, name="Stored 30-day path", line=dict(color=TEXT_DIM, dash="dash", width=2)))
    return _style(fig, 460)


def render_daily(data: dict[str, Any]) -> None:
    """Daily Picks: the three stored names, scores, entries, and expanders."""
    st.title("Daily picks")
    daily = data["daily"]
    market = daily.get("market_overview") if isinstance(daily.get("market_overview"), dict) else {}
    pick_date = str(data["manifest"].get("pick_date") or daily.get("date") or "—")
    st.caption(
        f"Picks for {pick_date}, computed from the previous trading day's close. "
        "The snapshot is regenerated each weekday before the US open."
    )
    st.caption(
        f"SPY {_price(market.get('spy_price'))} ({_pct(market.get('spy_change_pct'), 2)}) · "
        f"analyzed {market.get('total_analyzed', '—')}, passed filter {market.get('passed_filter', '—')}"
    )
    st.caption("Ranks the US-listed AI universe and surfaces the top 3. Scores run from 0 to 100. This page is the stored snapshot, not a live ranking.")
    picks = [pick for pick in daily.get("picks", []) if isinstance(pick, dict)][:3]
    if not picks:
        st.info(MISSING)
        return
    for pick in picks:
        ticker = str(pick.get("ticker") or "")
        news = pick.get("news") if isinstance(pick.get("news"), dict) else {}
        technical = pick.get("technical") if isinstance(pick.get("technical"), dict) else {}
        fundamental = pick.get("fundamental") if isinstance(pick.get("fundamental"), dict) else {}
        valuation = pick.get("valuation") if isinstance(pick.get("valuation"), dict) else {}
        prediction = pick.get("prediction") if isinstance(pick.get("prediction"), dict) else {}
        with st.container(border=True):
            st.subheader(f"{ticker} {pick.get('name') or ''}".strip())
            left, right = st.columns(2)
            left.metric("Price", _price(pick.get("current_price")))
            right.metric("30-day forecast", _pct(prediction.get("expected_return_pct")))
            composite = _num(pick.get("composite_score"))
            st.caption(f"Composite {composite:.2f}" if composite is not None else "Composite unavailable")
            if composite is not None:
                st.progress(min(max(composite / 100.0, 0.0), 1.0))
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Technical", technical.get("total_score"))
            c2.metric("Fundamental", fundamental.get("total_score"))
            c3.metric("Valuation", valuation.get("score"))
            c4.metric("Forecast", prediction.get("score"))
            c5.metric("News", news.get("score"))
            entry = pick.get("entry_price") if isinstance(pick.get("entry_price"), dict) else {}
            if entry:
                st.markdown(f"Entry **{_status_label(str(entry.get('current_status') or ''))}**")
                levels = entry.get("entry_prices") if isinstance(entry.get("entry_prices"), dict) else {}
                distance = entry.get("distance") if isinstance(entry.get("distance"), dict) else {}
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Level": ["Conservative", "Standard", "Aggressive"],
                            "Price": [
                                _price(levels.get("conservative")),
                                _price(levels.get("standard")),
                                _price(levels.get("aggressive")),
                            ],
                            "Distance": [
                                _pct(distance.get("to_conservative_pct")),
                                _pct(distance.get("to_standard_pct")),
                                _pct(distance.get("to_aggressive_pct")),
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption(f"Entry levels: {MISSING}")
            reasons = [str(item) for item in (pick.get("pick_reasons") or [])]
            with st.expander(f"Rationale · {ticker}"):
                if reasons:
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.caption(MISSING)
            headlines = news.get("top_headlines") if isinstance(news.get("top_headlines"), list) else []
            with st.expander(f"Related news ({news.get('news_count', len(headlines))}) · {ticker}"):
                summary = str(news.get("summary") or "").strip()
                if summary:
                    st.write(summary[:500])
                if not headlines:
                    st.caption(MISSING if not summary else "No headlines stored on this pick.")
                for item in headlines[:5]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title") or "Headline")
                    st.markdown(f"**{title}**")
                    st.caption(f"{item.get('publisher') or '—'} · {item.get('sentiment_cn') or item.get('sentiment') or '—'}")
                    link = str(item.get("link") or "")
                    if link.startswith("http"):
                        st.markdown(f"[Read original]({link})")
            if st.button("View full analysis", key=f"open_{ticker}"):
                st.session_state["nav_page"] = "Stock Deep Dive"
                st.session_state["deep_ticker"] = ticker
                st.rerun()


def _match_ticker(label: str, by_ticker: dict[str, dict[str, Any]]) -> str | None:
    """Map a competitor label onto a universe ticker when the text already matches."""
    text = str(label or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in by_ticker:
        return upper
    for token in upper.replace("(", " ").replace(")", " ").replace(",", " ").split():
        if token in by_ticker:
            return token
    lowered = text.lower()
    for ticker, row in by_ticker.items():
        name = str(row.get("name") or "").lower()
        if name and (name == lowered or name in lowered or lowered in name):
            return ticker
    return None


def render_deep(data: dict[str, Any]) -> None:
    """Stock Deep Dive for any universe ticker, including LRCX and INTU."""
    st.title("Stock deep dive")
    rows = data["universe"]
    tickers = [str(row.get("ticker")) for row in rows if row.get("ticker")]
    if "LRCX" not in tickers or "INTU" not in tickers:
        st.error("LRCX and INTU are missing from this snapshot.")
    if not tickers:
        st.info(MISSING)
        return
    current = st.session_state.get("deep_ticker", "LRCX")
    if current not in tickers:
        current = "LRCX" if "LRCX" in tickers else tickers[0]
    ticker = st.selectbox("Select a stock", tickers, index=tickers.index(current), key="deep_ticker")
    row = data["by_ticker"].get(ticker) or {}
    st.subheader(f"{ticker} {row.get('name') or ''}".strip())
    st.caption(f"{row.get('layer') or MISSING} · Peer group {row.get('peer_group') or '—'}")
    if row.get("score_note"):
        st.caption(str(row["score_note"]))
    carried = row.get("carried_forward") or []
    if carried:
        st.caption("Carried forward from the previous snapshot: " + ", ".join(str(item) for item in carried))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", _price(row.get("current_price")))
    m2.metric("Composite", row.get("composite_score") if row.get("composite_score") is not None else "—")
    m3.metric("Technical", row.get("tech_score") if row.get("tech_score") is not None else "—")
    m4.metric("Fundamental", row.get("fund_score") if row.get("fund_score") is not None else "—")
    profile_tab, tech_tab, fund_tab, forecast_tab, news_tab = st.tabs(
        ["Company profile", "Technicals", "Fundamentals", "Price forecast", "News sentiment"]
    )
    frame = _prices_for(data["prices"], ticker)
    with profile_tab:
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else None
        if not profile:
            st.info(row.get("profile_note") or f"{MISSING}. {row.get('name') or ticker} ({ticker}), {row.get('layer') or '—'}.")
        else:
            st.markdown(f"**{profile.get('name') or row.get('name')}** ({profile.get('ticker') or ticker})")
            st.caption(
                f"{profile.get('sector') or '—'} · {profile.get('industry') or '—'} · "
                f"{profile.get('country') or '—'} · Employees {profile.get('employees') or '—'}"
            )
            st.write(profile.get("description_short") or "No one-line summary in the snapshot.")
            cap = _num(profile.get("market_cap"))
            c1, c2 = st.columns(2)
            c1.metric("Market cap", f"${cap / 1e9:,.2f}B" if cap is not None else "—")
            c2.metric("Employees", profile.get("employees") if profile.get("employees") is not None else "—")
            st.markdown("**Full description**")
            st.write(profile.get("description_long") or MISSING)
            left, right = st.columns(2)
            with left:
                st.markdown("**Segments**")
                segments = profile.get("business_segments") or []
                if segments:
                    for item in segments:
                        st.markdown(f"- {item}")
                else:
                    st.caption(MISSING)
            with right:
                st.markdown("**Products**")
                products = profile.get("key_products") or []
                if products:
                    for item in products:
                        st.markdown(f"- {item}")
                else:
                    st.caption(MISSING)
            st.markdown("**Competitors**")
            st.caption("Choosing a universe name opens that stock. Other names stay as labels.")
            for index, competitor in enumerate(profile.get("competitors") or []):
                match = _match_ticker(str(competitor), data["by_ticker"])
                if match and match != ticker:
                    if st.button(str(competitor), key=f"comp_{ticker}_{index}"):
                        st.session_state["deep_ticker"] = match
                        st.rerun()
                else:
                    st.caption(str(competitor))
    with tech_tab:
        if frame.empty:
            st.info(row.get("price_note") or f"Price history: {MISSING}")
        else:
            st.plotly_chart(_technical_figure(frame), use_container_width=True)
            view = _indicators(frame)
            last = view.iloc[-1]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Indicator": "Close", "Value": _price(last.get("close"))},
                        {"Indicator": "MA20", "Value": _price(last.get("ma20"))},
                        {"Indicator": "MA60", "Value": _price(last.get("ma60"))},
                        {"Indicator": "RSI", "Value": None if _num(last.get("rsi")) is None else f"{_num(last.get('rsi')):.1f}"},
                        {"Indicator": "MACD", "Value": None if _num(last.get("macd")) is None else f"{_num(last.get('macd')):.2f}"},
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        technical = row.get("technical") if isinstance(row.get("technical"), dict) else None
        if technical and _num(technical.get("total_score")) is not None:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(technical["total_score"]), gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT}}))
            st.plotly_chart(_style(gauge, 220), use_container_width=True)
        elif row.get("tech_score") is not None:
            st.metric("Technical score from the snapshot", row.get("tech_score"))
        else:
            st.caption(f"Technical score: {MISSING}")
    with fund_tab:
        fundamental = row.get("fundamental") if isinstance(row.get("fundamental"), dict) else None
        subs = fundamental.get("sub_scores") if fundamental and isinstance(fundamental.get("sub_scores"), dict) else None
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        if subs:
            labels = ["growth", "profitability", "financial_health", "industry_outlook"]
            c1, c2, c3, c4 = st.columns(4)
            for column, key in zip((c1, c2, c3, c4), labels, strict=False):
                column.metric(key.replace("_", " ").title(), subs.get(key))
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(r=[subs.get(key) or 0 for key in labels], theta=labels, fill="toself", name=ticker))
            radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
            st.plotly_chart(_style(radar, 420), use_container_width=True)
        else:
            st.caption(f"Fundamental sub-scores: {MISSING}")
        fields = [
            ("Revenue growth", "revenueGrowth"),
            ("Earnings growth", "earningsGrowth"),
            ("Gross margin", "grossMargins"),
            ("Profit margin", "profitMargins"),
            ("ROE", "returnOnEquity"),
            ("Debt to equity", "debtToEquity"),
            ("Current ratio", "currentRatio"),
            ("Trailing P/E", "trailingPE"),
            ("Forward P/E", "forwardPE"),
            ("Analyst target", "targetMeanPrice"),
            ("Analyst rating", "recommendationMean"),
        ]
        table = pd.DataFrame([{"Field": label, "Value": _unavailable(info.get(key))} for label, key in fields])
        st.dataframe(table, hide_index=True, use_container_width=True)
        numeric = [(label, _num(info.get(key))) for label, key in fields]
        numeric = [(label, value) for label, value in numeric if value is not None]
        if numeric:
            bar = go.Figure(go.Bar(x=[label for label, _ in numeric], y=[value for _, value in numeric], marker_color=ACCENT))
            st.plotly_chart(_style(bar, 360), use_container_width=True)
        elif not subs:
            st.caption("Fundamental chart: Unavailable")
    with forecast_tab:
        valuation = row.get("valuation") if isinstance(row.get("valuation"), dict) else None
        forecast = row.get("forecast") if isinstance(row.get("forecast"), dict) else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Current price", _price(row.get("current_price")))
        if valuation and _num(valuation.get("fair_value")) is not None:
            c2.metric("Fair value", _price(valuation.get("fair_value")), _pct(valuation.get("upside_pct"), 2))
        else:
            c2.metric("Fair value", "—")
            st.caption(f"Valuation detail: {MISSING}")
        if forecast and _num(forecast.get("predicted_30d")) is not None:
            c3.metric("30-day forecast", _price(forecast.get("predicted_30d")), _pct(forecast.get("expected_return_pct")))
        else:
            c3.metric("30-day forecast", "—")
        if row.get("forecast_note"):
            st.info(str(row["forecast_note"]))
        elif forecast:
            st.caption("The band is the Prophet path stored on 2026-11-07. The daily refresh carries this path forward and does not refit it.")
        if not frame.empty:
            st.plotly_chart(_forecast_figure(frame, forecast), use_container_width=True)
        else:
            st.info(row.get("price_note") or f"Price history: {MISSING}")
        if valuation and isinstance(valuation.get("methods"), dict):
            methods = valuation["methods"]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Method": "P/E", "Valuation": (methods.get("pe_method") or {}).get("fair_price"), "Weight": (methods.get("pe_method") or {}).get("weight")},
                        {"Method": "PEG", "Valuation": (methods.get("peg_method") or {}).get("fair_price"), "Weight": (methods.get("peg_method") or {}).get("weight")},
                        {"Method": "Analyst target", "Valuation": (methods.get("analyst_target") or {}).get("fair_price"), "Weight": (methods.get("analyst_target") or {}).get("weight")},
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        if valuation and valuation.get("valuation_reliable") is False:
            st.warning(valuation.get("valuation_reason") or "Trailing P/E implies an implausible fair value.")
    with news_tab:
        news = row.get("news") if isinstance(row.get("news"), dict) else None
        if not news:
            st.info(row.get("news_note") or MISSING)
        else:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Sentiment", news.get("sentiment_label") or "—")
            n2.metric("Sentiment score", news.get("sentiment_score"))
            n3.metric("News count", news.get("news_count"))
            n4.metric(
                "Breakdown",
                f"{news.get('positive_count', 0)} / {news.get('neutral_count', 0)} / {news.get('negative_count', 0)}",
            )
            score = _num(news.get("sentiment_score"))
            if score is not None:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={"valueformat": ".2f"},
                        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": ACCENT}},
                    )
                )
                st.plotly_chart(_style(gauge, 220), use_container_width=True)
            summary = str(news.get("summary") or "").strip()
            st.markdown("**News summary**")
            st.write(summary or MISSING)
            carried_forward = row.get("carried_forward")
            if isinstance(carried_forward, list) and carried_forward and any(
                "carried forward" in str(item) for item in carried_forward
            ):
                st.caption("Sentiment and the summary were carried forward from the previous snapshot.")
            articles = news.get("analyzed_news") if isinstance(news.get("analyzed_news"), list) else []
            if not articles:
                st.caption(MISSING)
            for item in articles[:8]:
                if not isinstance(item, dict):
                    continue
                st.markdown(f"**{item.get('title') or 'Headline'}**")
                st.caption(f"{item.get('publisher') or '—'} · {item.get('sentiment_cn') or item.get('sentiment') or '—'}")


def render_overview(data: dict[str, Any]) -> None:
    """Universe table with search, peer-group filter, and an open-in-deep-dive control."""
    st.title("Universe overview")
    st.caption("Click a column header to sort. Filters apply before the sort.")
    records = []
    for row in data["universe"]:
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
        status = str(entry.get("current_status") or "")
        records.append(
            {
                "Ticker": row.get("ticker"),
                "Layer": row.get("layer"),
                "Peer group": row.get("peer_group"),
                "Company": row.get("name"),
                "Price": _num(row.get("current_price")),
                "Composite": _num(row.get("composite_score")),
                "Technical": _num(row.get("tech_score")),
                "Fundamental": _num(row.get("fund_score")),
                "Valuation": _num(row.get("valuation_score")),
                "Forecast": _num(row.get("prediction_return_pct")),
                "Entry": _status_label(status) if status else (row.get("entry_note") or MISSING),
                "_status": status,
            }
        )
    table = pd.DataFrame(records)
    keyword = st.text_input("Search ticker or company")
    groups = sorted(value for value in table["Peer group"].dropna().unique().tolist())
    chosen = st.multiselect("Filter by peer group", groups)
    ready = st.checkbox("Only show names ready to enter (buy now)", value=False)
    filtered = table.copy()
    if keyword:
        mask = filtered["Ticker"].astype(str).str.contains(keyword, case=False) | filtered["Company"].astype(str).str.contains(keyword, case=False)
        filtered = filtered[mask]
    if chosen:
        filtered = filtered[filtered["Peer group"].isin(chosen)]
    if ready:
        filtered = filtered[filtered["_status"] == "buy_now"]
    display = filtered.drop(columns=["_status"]).sort_values("Composite", ascending=False, na_position="last")
    if display.empty:
        st.caption("No stocks match the filter")
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Forecast": st.column_config.NumberColumn("Forecast %", format="%.2f"),
        },
    )
    options = display["Ticker"].astype(str).tolist()
    if options:
        choice = st.selectbox(
            "Select a stock for the full analysis",
            options,
            key="overview_pick",
        )
        if st.button("Open stock analysis", key="overview_open"):
            st.session_state["nav_page"] = "Stock Deep Dive"
            st.session_state["deep_ticker"] = choice
            st.rerun()
    else:
        st.caption("No stock is available to open from this filter.")


def render_performance(data: dict[str, Any]) -> None:
    """Track-record metrics, equity curve, leave-one-out attribution, and history."""
    st.title("Strategy performance")
    perf = data["performance"]
    summary = perf.get("summary") if isinstance(perf.get("summary"), dict) else {}
    cumulative = perf.get("cumulative") if isinstance(perf.get("cumulative"), dict) else {}
    concentration = perf.get("concentration") if isinstance(perf.get("concentration"), dict) else {}
    date_range = summary.get("date_range") if isinstance(summary.get("date_range"), dict) else {}
    st.caption(
        f"Recommendation window: {date_range.get('first', '—')} ~ {date_range.get('last', '—')} · "
        f"snapshot {perf.get('snapshot_time') or '—'}"
    )
    strategy = _num(cumulative.get("strategy_cumulative_return"))
    spy = _num(cumulative.get("spy_cumulative_return"))
    excess = _num(cumulative.get("outperformance"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommendations", summary.get("total_recommendations"))
    c2.metric("1M tracking complete", summary.get("completed_1m"))
    c3.metric("Strategy cumulative return", _frac(strategy))
    c4.metric("Excess return vs SPY", _frac(excess))
    if spy is not None:
        st.caption(f"SPY buy-and-hold over the same window: {_frac(spy)}")
    st.subheader("Cumulative return")
    dates = cumulative.get("dates") or []
    strategy_curve = cumulative.get("strategy_equity_curve") or []
    spy_curve = cumulative.get("spy_equity_curve") or []
    if dates and strategy_curve:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=strategy_curve, name="Strategy", line=dict(color=ACCENT, width=2.4)))
        if spy_curve:
            fig.add_trace(go.Scatter(x=dates, y=spy_curve, name="SPY buy-and-hold", line=dict(color=TEXT_DIM, width=2)))
        fig.add_hline(y=0, line_dash="dot", line_color=BORDER)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(_style(fig, 460), use_container_width=True)
    else:
        st.info("No equity curve in this snapshot.")
    st.subheader("Attribution (leave-one-out)")
    by_ticker = concentration.get("by_ticker") if isinstance(concentration.get("by_ticker"), dict) else {}
    if by_ticker:
        attrib = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Times selected": stats.get("picks"),
                    "Avg 1M return": stats.get("avg_return_1m"),
                    "Contribution": stats.get("contribution"),
                }
                for ticker, stats in by_ticker.items()
                if isinstance(stats, dict)
            ]
        ).sort_values("Contribution", ascending=False)
        colors = [POSITIVE if (_num(value) or 0) >= 0 else NEGATIVE for value in attrib["Contribution"]]
        bar = go.Figure(
            go.Bar(
                x=attrib["Ticker"],
                y=attrib["Contribution"],
                marker_color=colors,
                name="Leave-one-out contribution",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )
        bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(_style(bar, 380), use_container_width=True)
        show = attrib.copy()
        show["Avg 1M return"] = show["Avg 1M return"].map(_frac)
        show["Contribution"] = show["Contribution"].map(_frac)
        st.dataframe(show, hide_index=True, use_container_width=True)
    top = concentration.get("top_contributor") if isinstance(concentration.get("top_contributor"), dict) else {}
    share = _num(top.get("share_of_outperformance"))
    share_txt = f"{share * 100:.1f}" if share is not None else "—"
    name = top.get("ticker") or "—"
    if concentration.get("concentration_warning"):
        st.warning(
            f"Performance is concentrated in one name ({name} contributes {share_txt}%). "
            "The sample is too narrow to treat the stock-picking result as consistent."
        )
    st.caption(
        "Contribution = full equity-curve return minus the return recomputed without that name. "
        "Equal weight means dropping one name raises the weight of the rest, so contributions do not sum to the total return."
    )
    st.subheader("Recommendation history")
    history = data["recommendations"].copy()
    done = pd.to_numeric(history.get("return_1m"), errors="coerce").notna()
    choice = st.radio("Tracking status", ["All", "Completed", "Tracking"], horizontal=True, key="history_filter")
    if choice == "Completed":
        view = history[done]
    elif choice == "Tracking":
        view = history[~done]
    else:
        view = history
    if view.empty:
        st.caption("No records match the filter")
        return
    view = view.copy()
    view["_d"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.sort_values(["_d", "top_rank"], ascending=[False, True])
    out = pd.DataFrame(
        {
            "Date": view["date"].astype(str),
            "Ticker": view["ticker"].astype(str),
            "Peer group": view.get("sector"),
            "Rank": pd.to_numeric(view.get("top_rank"), errors="coerce"),
            "Entry price": pd.to_numeric(view.get("entry_price"), errors="coerce"),
            "Composite": pd.to_numeric(view.get("composite_score"), errors="coerce"),
            "1D": pd.to_numeric(view.get("return_1d"), errors="coerce"),
            "1W": pd.to_numeric(view.get("return_1w"), errors="coerce"),
            "2W": pd.to_numeric(view.get("return_2w"), errors="coerce"),
            "1M": pd.to_numeric(view.get("return_1m"), errors="coerce"),
            "SPY 1M": pd.to_numeric(view.get("spy_return_1m"), errors="coerce"),
            "Alpha": pd.to_numeric(view.get("alpha_1m"), errors="coerce"),
            "Status": view.get("status"),
        }
    )
    st.dataframe(
        out,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Entry price": st.column_config.NumberColumn(format="$%.2f"),
            "1D": st.column_config.NumberColumn(format="+.2%"),
            "1W": st.column_config.NumberColumn(format="+.2%"),
            "2W": st.column_config.NumberColumn(format="+.2%"),
            "1M": st.column_config.NumberColumn(format="+.2%"),
            "SPY 1M": st.column_config.NumberColumn(format="+.2%"),
            "Alpha": st.column_config.NumberColumn(format="+.2%"),
        },
    )
    st.caption(f"{len(out)} rows · 1M is 30 calendar days from the recommendation date")


def _css() -> None:
    """Light dashboard theme. No remote font download."""
    st.markdown(
        """
<style>
.stApp {background:#fff; color:#10141A;}
.banner {background:#F7F8FA; border:1px solid #E3E6EA; padding:8px 12px; border-radius:6px;}
h1 {font-size:24px !important;}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_active_page(page: str, data: dict[str, Any], snapshot_label: str) -> None:
    """Render exactly one page for this script run.

    The sidebar radio is the only page selector. The other page functions are
    not called, and this function does not keep the previous page in a
    container.

    Args:
        page: Selected page name from the sidebar radio.
        data: Parsed snapshot used by every page.
        snapshot_label: ``latest`` or ``baseline``.
    """
    _banner(snapshot_label, data["manifest"])
    if page == "Daily Picks":
        render_daily(data)
    elif page == "Stock Deep Dive":
        render_deep(data)
    elif page == "Universe Overview":
        render_overview(data)
    elif page == "Strategy Performance":
        render_performance(data)
    st.divider()
    st.caption("Research snapshot only. Not investment advice. A failed daily refresh leaves this baseline in place.")


def main() -> None:
    """Dashboard entry point."""
    st.set_page_config(page_title="US Stock Analysis Dashboard", page_icon="📈", layout="wide")
    _css()
    directory, label = resolve_snapshot()
    data = _load(directory)
    st.sidebar.markdown("**Pages**")
    st.sidebar.caption("US equity AI-sector ranking. Read-only snapshot, no live market calls.")
    page = st.sidebar.radio("Page", PAGES, key="nav_page")
    with st.sidebar.expander("How to read this"):
        weights = data["weights"]
        if weights:
            text = ", ".join(f"{key} {float(value):.0%}" for key, value in weights.items())
            st.caption(f"Composite weights: {text}.")
        st.caption("A higher score is a stronger reading on that dimension, not a predicted return.")
    render_active_page(page, data, label)


if __name__ == "__main__":
    main()
