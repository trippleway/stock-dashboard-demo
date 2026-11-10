# -*- coding: utf-8 -*-
"""Regression: one script run renders exactly one page.

Run from the demo repository root:
    python tests/test_page_navigation.py
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"

PERFORMANCE_ONLY = (
    "Attribution (leave-one-out)",
    "Recommendation history",
)
# Strings produced by the deep-dive company-profile tab, not by the overview table.
PROFILE_ONLY = (
    "**Full description**",
    "Choosing a universe name opens that stock. Other names stay as labels.",
    "Company profile",
)


def _rendered_text(at: AppTest) -> str:
    """Return headings, markdown, captions, and widget labels from this run.

    Args:
        at: The AppTest tree produced by the latest ``run()``.

    Returns:
        The visible strings from that run, including text inside tabs.
    """
    chunks: list[str] = []
    for collection in (
        at.title,
        at.header,
        at.subheader,
        at.markdown,
        at.caption,
        at.text,
        at.info,
        at.warning,
        at.error,
        at.success,
    ):
        for item in collection:
            chunks.append(str(item.value))
    for collection in (at.tabs, at.selectbox, at.text_input, at.radio, at.button, at.checkbox, at.multiselect):
        for item in collection:
            label = getattr(item, "label", None)
            if isinstance(label, str):
                chunks.append(label)
    return "\n".join(chunks)


def test_one_run_renders_one_page() -> None:
    """Strategy Performance must not survive into Stock Deep Dive, or the profile into Overview."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    at.sidebar.radio[0].set_value("Strategy Performance").run()
    assert not at.exception, at.exception
    performance = _rendered_text(at)
    assert "Strategy performance" in performance
    assert "Attribution (leave-one-out)" in performance
    assert "Recommendation history" in performance

    at.sidebar.radio[0].set_value("Stock Deep Dive").run()
    assert not at.exception, at.exception
    deep = _rendered_text(at)
    assert "Stock deep dive" in deep, deep
    assert "Full description" in deep
    for leftover in PERFORMANCE_ONLY:
        assert leftover not in deep, leftover

    at.sidebar.radio[0].set_value("Universe Overview").run()
    assert not at.exception, at.exception
    overview = _rendered_text(at)
    assert "Universe overview" in overview
    assert "Search ticker or company" in overview
    for leftover in PROFILE_ONLY:
        assert leftover not in overview, leftover
    assert "Stock deep dive" not in overview
    print("PAGE_NAVIGATION_OK")


if __name__ == "__main__":
    test_one_run_renders_one_page()
