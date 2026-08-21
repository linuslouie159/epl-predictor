"""Forward Fixtures from Football-Data's rolling ``fixtures.csv``.

This is the live path's source of upcoming Fixtures and of their Market Line, which is why the
project needs no API-Football client in v1 (decision 12).

Two properties of the file shape this module:

* It carries **every league Football-Data covers**, not just the English ones, so rows are filtered
  to E0-E3.
* It is **rolling** - roughly a week wide, replaced in place upstream. Caching it under a fixed name
  would silently destroy the previous week's evidence, so each fetch is written under the instant it
  was fetched. Nothing in ``data/raw/`` is ever overwritten.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from epl.clubs import ClubResolver
from epl.ingest.football_data import (
    DIVISIONS,
    SOURCE,
    download,
    parse_dates,
    read_raw_csv,
    require_columns,
)
from epl.paths import raw_dir

__all__ = [
    "FIXTURES_URL",
    "FIXTURE_COLUMNS",
    "fetch_fixtures",
    "fixtures_dir",
    "latest_fixtures_path",
    "parse_fixtures",
    "raw_fixtures_path",
]

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

#: Canonical column order for a table of Fixtures. A Fixture carries no result (CONTEXT.md), so
#: this is deliberately not the match schema with nulls in it.
FIXTURE_COLUMNS: tuple[str, ...] = (
    "division",
    "date",
    "time",
    "home_club",
    "away_club",
    "prematch_odds_home",
    "prematch_odds_draw",
    "prematch_odds_away",
)

_ODDS_COLUMNS = {
    "prematch_odds_home": "AvgH",
    "prematch_odds_draw": "AvgD",
    "prematch_odds_away": "AvgA",
}


def fixtures_dir() -> Path:
    """Where fetched copies of the rolling file accumulate."""
    return raw_dir() / SOURCE / "fixtures"


def raw_fixtures_path(fetched_at: dt.datetime) -> Path:
    """The cache path for a fetch made at ``fetched_at`` (UTC).

    The instant is in the filename because the upstream file is replaced in place: two fetches a
    week apart are different evidence, and one must not overwrite the other.
    """
    stamp = fetched_at.astimezone(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return fixtures_dir() / f"fixtures_{stamp}.csv"


def fetch_fixtures(
    *,
    fetched_at: dt.datetime | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download the rolling fixtures file into the raw cache, and return the cached path."""
    fetched_at = fetched_at or dt.datetime.now(dt.UTC)
    return download(FIXTURES_URL, raw_fixtures_path(fetched_at), timeout=timeout)


def latest_fixtures_path() -> Path | None:
    """The most recently fetched copy, or ``None`` if the cache holds none."""
    directory = fixtures_dir()
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("fixtures_*.csv"))
    return candidates[-1] if candidates else None


def parse_fixtures(
    path: Path | str,
    *,
    divisions: tuple[str, ...] = DIVISIONS,
    resolver: ClubResolver | None = None,
) -> pd.DataFrame:
    """Turn one cached fixtures file into canonical Fixture rows for the English tiers.

    Rows for other countries' leagues are dropped, not resolved: their Clubs are not in the Club
    table and never will be.
    """
    raw = read_raw_csv(path)
    require_columns(raw, {"Div", "Date", "HomeTeam", "AwayTeam"}, path)

    raw = raw[raw["Div"].isin(divisions)].reset_index(drop=True)

    frame = pd.DataFrame(index=raw.index)
    frame["division"] = raw["Div"].astype("string")
    frame["date"] = _parse_fixture_dates(raw["Date"], path)
    frame["time"] = (raw["Time"] if "Time" in raw.columns else pd.NA)
    frame["time"] = frame["time"].astype("string")
    frame["home_club"] = raw["HomeTeam"]
    frame["away_club"] = raw["AwayTeam"]

    for target, source_column in _ODDS_COLUMNS.items():
        values = raw[source_column] if source_column in raw.columns else pd.NA
        frame[target] = pd.to_numeric(values, errors="coerce").astype("Float64")

    resolver = resolver or ClubResolver.load()
    frame["home_club"] = resolver.resolve_series(frame["home_club"], source=SOURCE)
    frame["away_club"] = resolver.resolve_series(frame["away_club"], source=SOURCE)

    return frame[list(FIXTURE_COLUMNS)].sort_values(
        ["date", "time", "division", "home_club"], kind="stable"
    ).reset_index(drop=True)


def _parse_fixture_dates(values: pd.Series, path: Path | str) -> pd.Series:
    """Same two formats as the Season files, without a Season window to check against.

    A forward Fixture has no Season column upstream, and inferring one from the date would be a
    guess in July. The rest of the pipeline derives Season where it needs it.
    """
    return parse_dates(values, path).dt.date
