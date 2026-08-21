"""Smoke tests against the live upstream, run only with ``--run-network``.

These exist because of open risk 2 in docs/DECISIONS.md: as of 20 Aug 2026 the live path was
unverified. They check that the shapes this module depends on are still the shapes upstream serves,
which is the assumption the whole live loop rests on.
"""

from __future__ import annotations

import pytest

from epl.clubs import ClubResolver
from epl.ingest import fixtures as fx
from epl.ingest import football_data as fd
from epl.ingest.football_data import SOURCE

pytestmark = pytest.mark.network


def test_a_season_file_still_has_the_columns_we_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EPL_PROJECT_ROOT", str(tmp_path))
    path = fd.fetch_season(fd.LAST_SEASON, "E0")
    raw = fd.read_raw_csv(path)
    assert {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"} <= set(raw.columns)
    assert {"AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA"} <= set(raw.columns)


def test_the_current_season_parses_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EPL_PROJECT_ROOT", str(tmp_path))
    path = fd.fetch_season(fd.LAST_SEASON, "E0")
    frame = fd.parse_season_csv(path, fd.LAST_SEASON, "E0")
    assert tuple(frame.columns) == fd.MATCH_COLUMNS
    assert len(frame) > 0


def test_the_rolling_fixtures_file_is_reachable_and_parses(tmp_path, monkeypatch) -> None:
    """Whether it currently holds English rows depends on the calendar, so that is not asserted."""
    monkeypatch.setenv("EPL_PROJECT_ROOT", str(tmp_path))
    path = fx.fetch_fixtures()
    frame = fx.parse_fixtures(path)
    assert tuple(frame.columns) == fx.FIXTURE_COLUMNS


def test_no_new_club_spelling_has_appeared_upstream(tmp_path, monkeypatch) -> None:
    """The check that would have to pass before a live Prediction Round could be sealed."""
    monkeypatch.setenv("EPL_PROJECT_ROOT", str(tmp_path))
    resolver = ClubResolver.load()

    unknown: set[str] = set()
    for division in fd.DIVISIONS:
        raw = fd.read_raw_csv(fd.fetch_season(fd.LAST_SEASON, division))
        for column in ("HomeTeam", "AwayTeam"):
            unknown |= {
                name
                for name in raw[column].dropna().unique()
                if not resolver.knows(name, SOURCE)
            }

    fixtures_raw = fx.read_raw_csv(fx.fetch_fixtures())
    english = fixtures_raw[fixtures_raw["Div"].isin(fd.DIVISIONS)]
    for column in ("HomeTeam", "AwayTeam"):
        unknown |= {
            name for name in english[column].dropna().unique() if not resolver.knows(name, SOURCE)
        }

    assert sorted(unknown) == []
