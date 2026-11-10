# -*- coding: utf-8 -*-
"""Regression: each sidebar page replaces the previous page.

Run from the demo repository root:
    python tests/test_page_navigation.py
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _titles(at: AppTest) -> list[str]:
    """Return the visible page titles."""
    return [str(item.value) for item in at.title]


def _select_labels(at: AppTest) -> list[str]:
    """Return selectbox labels on the current page."""
    return [str(item.label) for item in at.selectbox]


def _tab_labels(at: AppTest) -> list[str]:
    """Return tab labels. The deep dive is the only page that creates tabs."""
    labels: list[str] = []
    for tab in at.tabs:
        label = getattr(tab, "label", None)
        if label:
            labels.append(str(label))
    return labels


def _assert_no_deep_dive(at: AppTest) -> None:
    """The deep-dive selectbox and its tabs must not survive a page change."""
    titles = _titles(at)
    assert "Stock deep dive" not in titles, titles
    assert "Select a stock" not in _select_labels(at), _select_labels(at)
    assert _tab_labels(at) == [], _tab_labels(at)
    assert at.get("tab") == [] or len(list(at.tabs)) == 0


def test_page_switch_clears_deep_dive() -> None:
    """Deep dive content is gone after opening Universe Overview and the other pages."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    at.sidebar.radio[0].set_value("Stock Deep Dive").run()
    assert not at.exception, at.exception
    at.selectbox[0].set_value("LRCX").run()
    assert not at.exception, at.exception
    deep_text = " ".join(_titles(at) + [str(item.value) for item in at.subheader] + [str(item.value) for item in at.markdown])
    assert "Stock deep dive" in _titles(at)
    assert "LRCX" in deep_text or "Lam" in deep_text
    assert "Select a stock" in _select_labels(at)
    assert len(list(at.tabs)) > 0 or len(at.get("tab")) > 0

    at.sidebar.radio[0].set_value("Universe Overview").run()
    assert not at.exception, at.exception
    assert "Universe overview" in _titles(at)
    assert at.text_input[0].label == "Search ticker or company"
    assert at.multiselect[0].label == "Filter by peer group"
    _assert_no_deep_dive(at)

    at.sidebar.radio[0].set_value("Daily Picks").run()
    assert not at.exception, at.exception
    assert "Daily picks" in _titles(at)
    assert "Universe overview" not in _titles(at)
    assert "Strategy performance" not in _titles(at)
    _assert_no_deep_dive(at)

    at.sidebar.radio[0].set_value("Strategy Performance").run()
    assert not at.exception, at.exception
    assert "Strategy performance" in _titles(at)
    warning = " ".join(str(item.value) for item in at.warning)
    assert "MU" in warning and "108.9" in warning
    assert "Daily picks" not in _titles(at)
    assert "Universe overview" not in _titles(at)
    _assert_no_deep_dive(at)

    at.sidebar.radio[0].set_value("Universe Overview").run()
    assert "Universe overview" in _titles(at)
    assert "Strategy performance" not in _titles(at)
    _assert_no_deep_dive(at)
    print("PAGE_NAVIGATION_OK")


if __name__ == "__main__":
    test_page_switch_clears_deep_dive()
