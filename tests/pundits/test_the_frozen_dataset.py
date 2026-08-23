"""What the committed `predictions.csv` must hold, checked without the raw cache.

The dataset is committed precisely so that the accountability feature works on a fresh clone
(issue #11), which means its own integrity has to be checkable on a fresh clone too. Everything
here reads the file and nothing else. The claims that need the nine pages, or the corpus, are in
``test_over_the_corpus.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.clubs import ClubResolver
from epl.pundits import ORIGIN, dataset
from epl.windows import EVALUATION_WINDOW

#: What the nine pages yielded: 3,420 Fixtures across nine Seasons, twelve of which the archive
#: never listed. Pinned exactly, because a refresh that quietly dropped a matchday would otherwise
#: show up only as a Pundit covering slightly less.
TOTAL_CALLS = 3408

#: Per Season, in Season order. 2022/23 is the thin one — eight Fixtures the archive skipped
#: entirely, in the Season of the Queen's death and the winter World Cup.
CALLS_PER_SEASON = {
    2017: 378,
    2018: 380,
    2019: 379,
    2020: 380,
    2021: 379,
    2022: 372,
    2023: 380,
    2024: 380,
    2025: 380,
}

WHO_WORKED_WHEN = {"lawrenson": range(2017, 2022), "sutton": range(2022, 2026)}


@pytest.fixture(scope="module")
def calls() -> pd.DataFrame:
    return dataset.load()


class TestWhatIsInIt:
    def test_nine_seasons_of_calls(self, calls: pd.DataFrame) -> None:
        assert len(calls) == TOTAL_CALLS
        assert dict(calls.groupby("season").size()) == CALLS_PER_SEASON

    def test_each_pundit_appears_in_the_seasons_they_worked_and_no_others(
        self, calls: pd.DataFrame
    ) -> None:
        for pundit, seasons in WHO_WORKED_WHEN.items():
            worked = calls.loc[calls["pundit"] == pundit, "season"]
            assert set(worked) == set(seasons)

    def test_every_call_sits_inside_the_evaluation_window(self, calls: pd.DataFrame) -> None:
        """A call in the Burn-In Window could never be scored, so it would be dead weight."""
        assert set(calls["season"]) <= set(EVALUATION_WINDOW)

    def test_every_call_is_a_premier_league_one(self, calls: pd.DataFrame) -> None:
        assert set(calls["division"]) == {"E0"}


class TestOnlyFactsAreStored:
    def test_the_columns_are_the_four_things_the_ticket_permits(
        self, calls: pd.DataFrame
    ) -> None:
        """Fixture, predicted Scoreline, Predictor, date. No prose, no matchday heading, no
        commentary, and no result — a stored Prediction that knew its own Outcome is what
        ADR 0005 exists to prevent."""
        assert list(calls.columns) == list(dataset.CALL_COLUMNS)

    def test_no_column_holds_what_happened(self, calls: pd.DataFrame) -> None:
        forbidden = {"home_goals", "away_goals", "outcome", "result", "exact_score"}
        assert forbidden.isdisjoint(calls.columns)

    def test_the_origin_is_recorded_as_the_bbc(self) -> None:
        """The Pundits published for the BBC; MyFootballFacts archived them. Attribution names
        the origin, not only the archive."""
        assert ORIGIN == "BBC"


class TestItIsUsable:
    def test_every_club_is_a_club(self, calls: pd.DataFrame) -> None:
        known = set(ClubResolver.load().clubs)
        assert set(calls["home_club"]) <= known
        assert set(calls["away_club"]) <= known

    def test_no_fixture_carries_two_calls(self, calls: pd.DataFrame) -> None:
        """One Fixture, one Prediction. Two would be scored twice and averaged."""
        assert not calls.duplicated(subset=list(dataset.FIXTURE_KEY)).any()

    def test_no_club_plays_itself(self, calls: pd.DataFrame) -> None:
        """What a home and away pair resolved to one slug would look like."""
        assert (calls["home_club"] != calls["away_club"]).all()

    def test_every_scoreline_is_a_scoreline(self, calls: pd.DataFrame) -> None:
        goals = calls[["pred_home_goals", "pred_away_goals"]]
        assert (goals >= 0).all().all()
        assert (goals <= 9).all().all()

    def test_every_date_falls_inside_its_own_season(self, calls: pd.DataFrame) -> None:
        dates = pd.to_datetime(calls["date"])
        assert (dates >= pd.to_datetime(calls["season"].astype(str) + "-07-01")).all()
        assert (dates <= pd.to_datetime((calls["season"] + 1).astype(str) + "-08-31")).all()


class TestItIsFrozen:
    def test_writing_what_was_read_changes_nothing(self, tmp_path) -> None:
        """Frozen means a rebuild is a no-op in git. A dataset that reordered itself on every
        write would make every rebuild look like a change and stop meaning anything."""
        rewritten = dataset.write(dataset.read(), tmp_path / "predictions.csv")

        assert rewritten.read_bytes() == dataset.path().read_bytes()
