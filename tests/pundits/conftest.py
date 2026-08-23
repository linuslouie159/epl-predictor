"""Shared builders for the Pundit tests.

A frozen call and a parsed listing are two different shapes — one is what the page said, the other
is what survived being reconciled with the corpus — so there are two builders, and each names only
the fields a test is actually about.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

#: One row of the frozen dataset, in :data:`epl.pundits.dataset.CALL_COLUMNS`.
CALL_DEFAULTS: dict[str, object] = {
    "pundit": "lawrenson",
    "season": 2017,
    "division": "E0",
    "date": pd.Timestamp("2017-08-12").date(),
    "home_club": "arsenal",
    "away_club": "chelsea",
    "pred_home_goals": 2,
    "pred_away_goals": 0,
}

#: One row as :func:`epl.pundits.myfootballfacts.parse_page` hands it over — the Club names as the
#: source spelled them, and the result the page published beside the call.
LISTING_DEFAULTS: dict[str, object] = {
    "season": 2017,
    "pundit": "lawrenson",
    "home_name": "Arsenal",
    "away_name": "Chelsea",
    "pred_home_goals": 2,
    "pred_away_goals": 0,
    "played": True,
    "published_home_goals": 1,
    "published_away_goals": 0,
}


@pytest.fixture
def make_calls() -> Callable[..., pd.DataFrame]:
    """Rows of the frozen dataset, naming only the fields the test is about."""

    def _make(*rows: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame([{**CALL_DEFAULTS, **row} for row in rows])

    return _make


@pytest.fixture
def make_listings() -> Callable[..., pd.DataFrame]:
    """Rows as the parser hands them over, naming only the fields the test is about."""

    def _make(*rows: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame([{**LISTING_DEFAULTS, **row} for row in rows])

    return _make
