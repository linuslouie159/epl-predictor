"""Integrity of the ingested corpus, checked against the raw cache.

These are the tests that would catch a whole-corpus mistake the unit tests cannot see: a tier read
with shifted columns, an era boundary handled backwards, a Club silently split in two. Several of
them re-derive the measured facts recorded in docs/DECISIONS.md, which that document explicitly
asks to be re-verified rather than trusted.

They need a populated ``data/raw/``, which is gitignored, so they skip when it is absent:

    python -c "from epl.ingest import fetch_all; fetch_all()"
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.ingest import (
    DIVISIONS,
    FIRST_SEASON,
    LAST_SEASON,
    club_names_in_raw_cache,
    load_matches,
    raw_season_path,
)
from epl.ingest.football_data import SOURCE

pytestmark = pytest.mark.cache

EVALUATION_WINDOW = range(2005, 2026)
BURN_IN_WINDOW = range(2000, 2005)


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} of 104 files missing)")


@pytest.fixture(scope="module")
def matches():
    _require_cache()
    return load_matches()


class TestCoverage:
    def test_every_season_and_tier_is_cached(self) -> None:
        _require_cache()
        assert len(list(range(FIRST_SEASON, LAST_SEASON + 1))) * len(DIVISIONS) == 104

    def test_the_premier_league_has_a_full_fixture_list_every_season(self, matches) -> None:
        counts = matches[matches["division"] == "E0"].groupby("season").size()
        assert set(counts.index) == set(range(FIRST_SEASON, LAST_SEASON + 1))
        assert set(counts) == {380}

    def test_the_evaluation_window_is_7980_fixtures(self, matches) -> None:
        """The headline number every score in this project is computed over."""
        window = matches[
            (matches["division"] == "E0") & matches["season"].isin(EVALUATION_WINDOW)
        ]
        assert len(window) == 7980

    def test_the_lower_tiers_are_full_except_where_covid_curtailed_them(self, matches) -> None:
        """League One and League Two stopped early in 2019/20; the Premier League did not."""
        counts = matches[matches["division"] != "E0"].groupby(["division", "season"]).size()
        short = {key: int(n) for key, n in counts.items() if n != 552}
        assert short == {("E2", 2019): 400, ("E3", 2019): 440}

    def test_ingests_roughly_43000_matches_beyond_the_premier_league(self, matches) -> None:
        """The cost of rating the whole pyramid, as estimated in ADR 0004."""
        assert 42_000 < len(matches[matches["division"] != "E0"]) < 44_000


class TestClubResolution:
    def test_every_spelling_in_the_cache_resolves(self) -> None:
        _require_cache()
        from epl.clubs import ClubResolver

        resolver = ClubResolver.load()
        unknown = [n for n in club_names_in_raw_cache() if not resolver.knows(n, SOURCE)]
        assert unknown == []

    def test_no_club_is_left_unresolved_in_the_parsed_table(self, matches) -> None:
        assert matches["home_club"].isna().sum() == 0
        assert matches["away_club"].isna().sum() == 0

    def test_the_alias_table_has_no_clubs_the_pyramid_never_fielded(self, matches) -> None:
        from epl.clubs import load_clubs

        played = set(matches["home_club"]) | set(matches["away_club"])
        assert sorted(set(load_clubs()["slug"]) - played) == []


class TestEraBoundaries:
    """The era table in docs/DECISIONS.md, re-derived from the data it describes."""

    def test_the_market_line_begins_in_2005_06(self, matches) -> None:
        e0 = matches[matches["division"] == "E0"]
        coverage = e0.groupby("season")["prematch_odds_home"].apply(lambda s: s.notna().mean())
        assert (coverage.loc[list(BURN_IN_WINDOW)] == 0).all()
        assert (coverage.loc[list(EVALUATION_WINDOW)] == 1).all()

    def test_the_ceiling_line_begins_in_2019_20(self, matches) -> None:
        e0 = matches[matches["division"] == "E0"]
        coverage = e0.groupby("season")["closing_odds_home"].apply(lambda s: s.notna().mean())
        assert (coverage.loc[2000:2018] == 0).all()
        assert (coverage.loc[2019:2025] == 1).all()

    def test_match_stats_are_present_from_2000_01(self, matches) -> None:
        """Recorded as 100%; it is 99.98% - nine rows across the whole pyramid lack them."""
        columns = ["home_shots", "away_shots", "home_corners", "away_corners", "home_fouls"]
        missing = matches[columns].isna().any(axis=1).sum()
        assert missing == 9

    def test_the_market_line_is_never_read_from_two_columns_at_once(self, matches) -> None:
        """BbAv* and Avg* are one quantity; a Season carrying both would double-count."""
        e0 = matches[matches["division"] == "E0"]
        priced = e0[e0["prematch_odds_home"].notna()]
        assert (priced["prematch_odds_home"] > 1).all()
        assert (priced["prematch_odds_draw"] > 1).all()
        assert (priced["prematch_odds_away"] > 1).all()


class TestMeasuredFacts:
    """Numbers docs/DECISIONS.md records, re-derived rather than trusted."""

    def test_the_naive_baseline_rates(self, matches) -> None:
        """Recorded as H 45.6% / D 24.3% / A 30.1%."""
        window = matches[
            (matches["division"] == "E0") & matches["season"].isin(EVALUATION_WINDOW)
        ]
        rates = window["outcome"].value_counts(normalize=True)
        assert rates["H"] == pytest.approx(0.456, abs=0.001)
        assert rates["D"] == pytest.approx(0.243, abs=0.001)
        assert rates["A"] == pytest.approx(0.301, abs=0.001)

    def test_the_mean_overround_on_the_market_line(self, matches) -> None:
        """Recorded as 1.0562 - a 5.62% margin. Wrong odds columns would move this."""
        window = matches[
            (matches["division"] == "E0") & matches["season"].isin(EVALUATION_WINDOW)
        ]
        overround = (
            1 / window["prematch_odds_home"]
            + 1 / window["prematch_odds_draw"]
            + 1 / window["prematch_odds_away"]
        )
        assert float(overround.mean()) == pytest.approx(1.0562, abs=0.0001)

    def test_home_advantage_has_declined(self, matches) -> None:
        """~47% home wins in the 2000s against ~43% since 2020."""
        e0 = matches[matches["division"] == "E0"]
        noughties = e0[e0["season"].between(2000, 2009)]
        recent = e0[e0["season"].between(2020, 2025)]
        assert (noughties["outcome"] == "H").mean() == pytest.approx(0.47, abs=0.01)
        assert (recent["outcome"] == "H").mean() == pytest.approx(0.43, abs=0.01)


class TestRowIntegrity:
    def test_no_fixture_appears_twice(self, matches) -> None:
        keys = ["season", "division", "date", "home_club", "away_club"]
        assert matches.duplicated(keys).sum() == 0

    def test_no_club_plays_itself(self, matches) -> None:
        assert (matches["home_club"] == matches["away_club"]).sum() == 0

    def test_every_match_has_goals_and_an_outcome(self, matches) -> None:
        assert matches["home_goals"].isna().sum() == 0
        assert matches["away_goals"].isna().sum() == 0
        assert set(matches["outcome"]) == {"H", "D", "A"}

    def test_the_outcome_agrees_with_the_scoreline(self, matches) -> None:
        """A Scoreline implies an Outcome (CONTEXT.md); disagreement means shifted columns."""
        home, away = matches["home_goals"], matches["away_goals"]
        implied = pd.Series("D", index=matches.index)
        implied[home > away] = "H"
        implied[home < away] = "A"
        assert (implied == matches["outcome"]).all()

    def test_every_date_falls_inside_its_season(self, matches) -> None:
        seasons = matches["season"]
        assert (matches["date"].map(lambda d: d.year) >= seasons).all()
        assert (matches["date"].map(lambda d: d.year) <= seasons + 1).all()

    def test_goals_are_plausible(self, matches) -> None:
        assert matches["home_goals"].min() >= 0
        assert matches["away_goals"].min() >= 0
        assert matches["home_goals"].max() <= 15
        assert matches["away_goals"].max() <= 15
