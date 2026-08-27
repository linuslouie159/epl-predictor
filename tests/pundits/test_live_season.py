"""The Season in progress: a partial page, and the question of whether it is early enough.

Issue #16's spike. Two things separate a live page from the nine frozen ones, and both are facts
rather than caution:

* it is **partial** — ten calls in late August, not 380 — so the backfill's size floor would refuse
  every live page there has ever been;
* it may run **behind the football**, which decides whether a Pundit can appear in a Sealed
  Prediction at all. That is what :class:`~epl.pundits.live.Coverage` measures.
"""

from __future__ import annotations

import datetime as dt

import pytest

from epl.pundits import dataset, live
from epl.pundits import myfootballfacts as mff

PAGE = mff.Page(2026, "sutton", "chris-sutton-predictions-premier-league-2026-27")


def day(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


class TestChoosingThePage:
    def test_the_page_for_a_season_is_found_among_the_discovered_ones(self) -> None:
        assert live.page_for(2026, (mff.PAGES[0], PAGE)) == PAGE

    def test_a_season_the_index_does_not_carry_is_an_error_naming_what_it_does(self) -> None:
        with pytest.raises(live.LiveError, match="2030/31"):
            live.page_for(2030, (mff.PAGES[0], PAGE))


class TestParsingAPartialPage:
    """The live page in August 2026 held ten calls. ``MIN_CALLS`` is 360."""

    def test_a_ten_call_page_parses_where_the_backfill_floor_would_refuse_it(self) -> None:
        html = _page_html(("Arsenal 3-0 Coventry City", "3 - 0"))

        assert len(live.parse(html, PAGE)) == 1
        with pytest.raises(mff.PunditSourceError, match="1 calls"):
            mff.parse_page(html, PAGE)

    def test_a_page_with_no_calls_at_all_is_allowed_because_a_season_starts_empty(self) -> None:
        """Before the opening round the page is real and says nothing. That is not a parse
        failure, and it is why the live path cannot lean on a size floor the way the backfill
        does."""
        assert live.parse("<html><body><table></table></body></html>", PAGE).empty

    def test_it_still_refuses_a_page_that_parses_to_more_than_a_season(self) -> None:
        """The upper guard survives: a page scanned twice would double, and that is as wrong
        live as it is frozen. Exercised by parsing an oversized page rather than by comparing
        two constants, which would pass whatever :func:`live.parse` did with them."""
        doubled = _page_html(
            *[(f"Arsenal {n % 5}-0 Coventry City", "1 - 0") for n in range(mff.MAX_CALLS + 1)]
        )

        with pytest.raises(mff.PunditSourceError, match="calls"):
            live.parse(doubled, PAGE)


class TestBuildingCallsForASeasonInProgress:
    def test_it_produces_the_frozen_dataset_s_own_columns(self, make_matches) -> None:
        """"Whichever source wins produces rows in the same schema as the committed backfill.\""""
        matches = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry", "home_goals": 3, "away_goals": 0},
        )
        listings = live.parse(_page_html(("Arsenal 3-0 Coventry City", "3 - 0")), PAGE)

        built = live.build(matches, listings)

        assert tuple(built.calls.columns) == dataset.CALL_COLUMNS
        assert built.calls["pundit"].tolist() == ["sutton"]
        assert built.calls["season"].tolist() == [2026]
        assert built.calls["division"].tolist() == ["E0"]

    def test_the_result_the_page_publishes_is_read_and_not_stored(self, make_matches) -> None:
        """ADR 0005's rule does not relax because a Season is in progress."""
        matches = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry", "home_goals": 3, "away_goals": 0},
        )
        listings = live.parse(_page_html(("Arsenal 3-0 Coventry City", "3 - 0")), PAGE)

        built = live.build(matches, listings)

        assert "home_goals" not in built.calls.columns
        assert "published_home_goals" not in built.calls.columns

    def test_a_page_with_nothing_on_it_still_produces_the_right_empty_frame(
        self, make_matches
    ) -> None:
        """The state every Season starts in. A build that raised here would make the live path
        unusable for the fortnight before a Season begins, which is when it is first run."""
        matches = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        listings = live.parse("<html><body><table></table></body></html>", PAGE)

        built = live.build(matches, listings)

        assert tuple(built.calls.columns) == dataset.CALL_COLUMNS
        assert built.calls.empty
        assert built.unplaced.empty

    def test_a_call_on_a_fixture_not_yet_played_is_held_back_rather_than_refused(
        self, make_matches
    ) -> None:
        """The seam :func:`epl.pundits.dataset._locate` names in its own docstring. The corpus
        holds played matches, so a call published *ahead* of kickoff has nothing to join to —
        which is a gap in the data here and a resolution error there."""
        matches = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry", "home_goals": 3, "away_goals": 0},
        )
        listings = live.parse(
            _page_html(
                ("Arsenal 3-0 Coventry City", "3 - 0"),
                ("Everton 2-1 Crystal Palace", "PP"),
            ),
            PAGE,
        )

        built = live.build(matches, listings)

        assert len(built.calls) == 1
        assert built.unplaced["home_club"].tolist() == ["everton"]

    def test_holding_one_back_does_not_lose_the_rest(self, make_matches) -> None:
        matches = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry", "home_goals": 3, "away_goals": 0},
            {"season": 2026, "date": "2026-08-22", "home_club": "everton",
             "away_club": "crystal_palace", "home_goals": 2, "away_goals": 0},
        )
        listings = live.parse(
            _page_html(
                ("Arsenal 3-0 Coventry City", "3 - 0"),
                ("Everton 2-1 Crystal Palace", "2 - 0"),
                ("Fulham 0-2 Chelsea", "PP"),
            ),
            PAGE,
        )

        built = live.build(matches, listings)

        assert sorted(built.calls["home_club"]) == ["arsenal", "everton"]
        assert built.unplaced["home_club"].tolist() == ["fulham"]


class TestCoverageOfTheNextRound:
    """The measurement that decides whether a Pundit can be sealed, rather than scored later."""

    def test_a_round_whose_calls_are_all_published_is_sealable(self, make_matches, make_calls):
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-29", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls(
            {"pundit": "sutton", "season": 2026, "date": day("2026-08-29"),
             "home_club": "arsenal", "away_club": "coventry"},
        )

        found = live.coverage(calls, fixtures, as_of=day("2026-08-27"))

        assert (found.round_fixtures, found.round_called) == (1, 1)
        assert found.sealable

    def test_a_round_the_archive_has_not_reached_is_not(self, make_matches, make_calls) -> None:
        """Measured on the real archive: two days before kickoff, nought of ten."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-29", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls(
            {"pundit": "sutton", "season": 2026, "date": day("2026-08-21"),
             "home_club": "everton", "away_club": "chelsea"},
        )

        found = live.coverage(calls, fixtures, as_of=day("2026-08-27"))

        assert (found.round_fixtures, found.round_called) == (1, 0)
        assert not found.sealable

    def test_a_partly_published_round_is_not_sealable_either(self, make_matches, make_calls):
        """Half a round is not a round. Sealing it would put a Pundit on some Fixtures of a
        Prediction Round and not others, which is not a track record of anything."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-29", "home_club": "arsenal",
             "away_club": "coventry"},
            {"season": 2026, "date": "2026-08-29", "home_club": "everton",
             "away_club": "chelsea"},
        )
        calls = make_calls(
            {"pundit": "sutton", "season": 2026, "date": day("2026-08-29"),
             "home_club": "arsenal", "away_club": "coventry"},
        )

        found = live.coverage(calls, fixtures, as_of=day("2026-08-27"))

        assert (found.round_fixtures, found.round_called) == (2, 1)
        assert not found.sealable

    def test_a_frame_with_no_upcoming_fixture_cannot_answer_and_says_so(
        self, make_matches, make_calls
    ) -> None:
        """**This is the live case, not an edge case.** The corpus holds played matches, so for a
        Season in progress there is never an upcoming Fixture in it — the next round's come from
        ``fixtures.csv`` at issue #17. Refusing beats returning ``sealable=False``, which would
        report "cannot tell yet" as "no"."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-21")})

        with pytest.raises(live.LiveError, match=r"fixtures\.csv at issue #17"):
            live.coverage(calls, fixtures, as_of=day("2026-09-01"))

    def test_the_next_kickoff_is_a_date_even_when_the_caller_held_strings(
        self, make_matches, make_calls
    ) -> None:
        """``epl.ingest.match_table`` reads the corpus without parsing dates, and it is what the
        command line hands in — so every date this returns goes through ``_dates`` or none does."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-29", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        fixtures["date"] = fixtures["date"].astype(str)
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-29")})

        found = live.coverage(calls, fixtures, as_of=day("2026-08-27"))

        assert found.next_kickoff == day("2026-08-29")
        assert found.next_anchor == day("2026-08-28")  # the Friday before a Saturday kickoff

    def test_it_prints_its_verdict(self, make_matches, make_calls) -> None:
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-29", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-21")})

        printed = live.coverage(calls, fixtures, as_of=day("2026-08-27")).describe()

        assert "2026/27" in printed
        assert "CANNOT be sealed" in printed


class TestTheLagBehindTheFootball:
    """The half that *can* be answered from the corpus, and therefore the half issue #16's
    "latency stated" criterion actually rests on."""

    def test_it_is_measured_against_football_played_not_against_today(
        self, make_matches, make_calls
    ) -> None:
        """Comparing the archive with the calendar makes an international break look like lag.
        The honest number is how far behind the *last Fixture played* the archive is."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
            {"season": 2026, "date": "2026-08-24", "home_club": "everton",
             "away_club": "chelsea"},
            {"season": 2026, "date": "2026-09-12", "home_club": "fulham",
             "away_club": "leeds"},
        )
        calls = make_calls(
            {"pundit": "sutton", "season": 2026, "date": day("2026-08-21"),
             "home_club": "arsenal", "away_club": "coventry"},
        )

        found = live.lag(calls, fixtures, as_of=day("2026-09-01"))

        assert found.last_played == day("2026-08-24")
        assert found.latest_call == day("2026-08-21")
        assert found.behind_days == 3

    def test_an_archive_that_has_caught_up_is_nought_days_behind(self, make_matches, make_calls):
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
            {"season": 2026, "date": "2026-09-12", "home_club": "fulham",
             "away_club": "leeds"},
        )
        calls = make_calls(
            {"pundit": "sutton", "season": 2026, "date": day("2026-08-21"),
             "home_club": "arsenal", "away_club": "coventry"},
        )

        assert live.lag(calls, fixtures, as_of=day("2026-09-01")).behind_days == 0

    def test_it_answers_where_coverage_cannot(self, make_matches, make_calls) -> None:
        """Nothing upcoming in the frame — the live Season's ordinary state. ``coverage`` refuses
        and the latency still comes back, which is the whole reason these are two functions."""
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-21")})

        assert live.lag(calls, fixtures, as_of=day("2026-09-01")).behind_days == 0
        with pytest.raises(live.LiveError):
            live.coverage(calls, fixtures, as_of=day("2026-09-01"))

    def test_it_prints_the_latency(self, make_matches, make_calls) -> None:
        fixtures = make_matches(
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-21")})

        printed = live.lag(calls, fixtures, as_of=day("2026-09-01")).describe()

        assert "2026/27" in printed
        assert "archive is behind by" in printed

    def test_fixtures_from_two_seasons_are_refused_rather_than_silently_halved(
        self, make_matches, make_calls
    ) -> None:
        """Both measurements are one Season's page against one Season's football. A frame carrying
        two would have reported whichever happened to be first."""
        fixtures = make_matches(
            {"season": 2025, "date": "2026-05-24", "home_club": "arsenal",
             "away_club": "chelsea"},
            {"season": 2026, "date": "2026-08-21", "home_club": "arsenal",
             "away_club": "coventry"},
        )
        calls = make_calls({"pundit": "sutton", "season": 2026, "date": day("2026-08-21")})

        with pytest.raises(live.LiveError, match="exactly one Season"):
            live.lag(calls, fixtures, as_of=day("2026-09-01"))


def _page_html(*pairs: tuple[str, str]) -> str:
    """A page shaped like the real ones, with as few matchdays as the test needs."""
    cells = "".join(f"<tr><td>{listing}</td><td>{result}</td></tr>" for listing, result in pairs)
    return (
        "<html><body><table>"
        "<thead><tr><th>Matchday 1 (21/08/26)</th><th>Result</th></tr></thead>"
        f"<tbody>{cells}</tbody></table></body></html>"
    )
