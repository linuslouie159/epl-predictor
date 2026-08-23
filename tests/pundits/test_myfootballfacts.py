"""Parsing the nine MyFootballFacts season pages.

Every fact this module has to survive was found in the real pages and is named in the test that
covers it, so a page whose HTML moves fails on the thing that moved rather than on a row count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from epl.ingest import cache, fetcher
from epl.pundits import myfootballfacts as mff


def page_html(*rows: str, header: str = "Matchday 1 (12/08/17)") -> str:
    """A page shaped like the real ones: a matchday header, then fixture/result cell pairs."""
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return (
        "<html><body><table>"
        f"<thead><tr><th>{header}</th><th>Result</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></body></html>"
    )


def pair(listing: str, result: str) -> str:
    return f"<td>{listing}</td><td>{result}</td>"


def calls(*listings: tuple[str, str], season: int = 2017, floor: int = 0):
    """Parse a page built from these (listing, result) pairs, with the size floor relaxed."""
    page = mff.Page(season=season, pundit="lawrenson", path="whatever")
    return mff.parse_page(page_html(*(pair(*one) for one in listings)), page, minimum=floor)


class TestThePages:
    def test_nine_pages_are_named_one_per_season(self) -> None:
        assert len(mff.PAGES) == 9
        assert [page.season for page in mff.PAGES] == list(range(2017, 2026))

    def test_lawrenson_worked_the_first_five_and_sutton_the_last_four(self) -> None:
        """Whose calls these are is a fact about the Season, not a label chosen at scoring time."""
        by_pundit = {page.season: page.pundit for page in mff.PAGES}
        assert [by_pundit[season] for season in range(2017, 2022)] == ["lawrenson"] * 5
        assert [by_pundit[season] for season in range(2022, 2026)] == ["sutton"] * 4

    def test_every_page_has_its_own_url_and_its_own_cache_path(self) -> None:
        urls = {mff.page_url(page) for page in mff.PAGES}
        paths = {mff.raw_page_path(page) for page in mff.PAGES}
        assert len(urls) == len(paths) == 9
        assert all(url.startswith(mff.BASE_URL) for url in urls)


class TestParsingOneListing:
    def test_the_scoreline_inside_the_cell_is_the_call_and_the_next_cell_is_the_result(
        self,
    ) -> None:
        """The source writes the pundit's call *into* the fixture and the real score beside it."""
        (call,) = calls(("Ipswich Town 1-3 Liverpool", "0 - 2")).to_dict("records")

        assert call["home_name"] == "Ipswich Town"
        assert call["away_name"] == "Liverpool"
        assert (call["pred_home_goals"], call["pred_away_goals"]) == (1, 3)
        assert (call["published_home_goals"], call["published_away_goals"]) == (0, 2)
        assert call["played"]

    def test_the_season_and_the_pundit_come_from_the_page(self) -> None:
        (call,) = calls(("Arsenal 2-0 Chelsea", "1-1"), season=2019).to_dict("records")

        assert call["season"] == 2019
        assert call["pundit"] == "lawrenson"

    @pytest.mark.parametrize("result", ["1-0", "1 - 0", "1  -  0"])
    def test_the_result_cell_is_spelled_three_ways_across_the_nine_pages(
        self, result: str
    ) -> None:
        (call,) = calls(("Arsenal 2-0 Chelsea", result)).to_dict("records")

        assert (call["published_home_goals"], call["published_away_goals"]) == (1, 0)


class TestNamesTheSourceWroteBadly:
    @pytest.mark.parametrize(
        "listing",
        [
            "Manchester  United 1-1 Liverpool",
            "Manchester    United 1-1 Liverpool",
        ],
    )
    def test_doubled_spaces_are_collapsed(self, listing: str) -> None:
        (call,) = calls((listing, "2-4")).to_dict("records")

        assert call["home_name"] == "Manchester United"

    def test_a_trailing_asterisk_is_an_annotation_not_a_spelling(self) -> None:
        (call,) = calls(("Chelsea* 2-0 Everton*", "1-0")).to_dict("records")

        assert (call["home_name"], call["away_name"]) == ("Chelsea", "Everton")

    @pytest.mark.parametrize(
        "written",
        ["Chelsea (01.02)", "Chelsea (19th May)", "Chelsea (09/02/19)", "Chelsea* (13.03)"],
    )
    def test_a_trailing_rearranged_date_is_an_annotation_too(self, written: str) -> None:
        """Several calls carry the date the Fixture was moved to. That is a note about the
        Fixture, not a spelling of the Club, so it never reaches the Alias table."""
        (call,) = calls((f"{written} 2-0 Everton", "1-0")).to_dict("records")

        assert call["home_name"] == "Chelsea"

    def test_a_v_separator_means_the_score_landed_inside_the_away_clubs_name(self) -> None:
        """Twice in nine Seasons the source wrote `Home v Away` and put the call where the `&`
        in `Brighton & Hove Albion` should have been. Read literally that is a Club called
        `Burnley v Brighton` playing one called `Hove Albion`; read as written it is Burnley at
        home to Brighton, 1-2."""
        (call,) = calls(("Burnley v Brighton 1-2 Hove Albion", "1 - 2")).to_dict("records")

        assert call["home_name"] == "Burnley"
        assert call["away_name"] == "Brighton Hove Albion"
        assert (call["pred_home_goals"], call["pred_away_goals"]) == (1, 2)


class TestListingsThatAreNotCalls:
    def test_a_season_label_is_not_a_scoreline(self) -> None:
        """`The 2017-18 Premier League Table` reads as a fixture under a loose pattern: two Clubs
        either side of `2017-18`. Goals are one or two digits, and a four-digit year is not."""
        parsed = calls(
            ("The 2017-18 Premier League Table", ""),
            ("Arsenal 2-0 Chelsea", "1-0"),
        )

        assert list(parsed["home_name"]) == ["Arsenal"]

    def test_the_running_tally_the_page_keeps_is_prose(self) -> None:
        parsed = calls(
            ("48 Correct Scores, 141 Correct Results, 181 Wrong", ""),
            ("Arsenal 2-0 Chelsea", "1-0"),
        )

        assert len(parsed) == 1

    @pytest.mark.parametrize("marker", ["PP", "AA", ""])
    def test_a_postponed_listing_is_still_a_call_but_is_not_a_played_one(
        self, marker: str
    ) -> None:
        """`PP` marks a Fixture that was not played on the date this call was published for. The
        call is real — the pundit made it — but the listing is not the one the Fixture was
        eventually played under."""
        parsed = calls(("Fulham 1-1 Chelsea", marker))
        (call,) = parsed.to_dict("records")

        assert (call["pred_home_goals"], call["pred_away_goals"]) == (1, 1)
        assert not call["played"]
        assert parsed["published_home_goals"].isna().all()

    def test_a_page_with_and_without_postponements_holds_its_results_the_same_way(self) -> None:
        """Nullable integers, stated rather than inferred: a frame that guessed would hold plain
        integers on a page with no `PP` and floats on a page with one."""
        mixed = calls(("Arsenal 2-0 Chelsea", "1-0"), ("Fulham 1-1 Chelsea", "PP"))
        clean = calls(("Arsenal 2-0 Chelsea", "1-0"))

        assert mixed["published_home_goals"].dtype == clean["published_home_goals"].dtype
        assert mixed.loc[0, "published_home_goals"] == 1


class TestRefusingAPageThatChangedShape:
    def test_a_page_yielding_far_too_few_calls_is_refused(self) -> None:
        """The failure this guards is silent: a table restructured upstream still parses, just
        into three rows instead of three hundred and eighty."""
        page = mff.Page(season=2017, pundit="lawrenson", path="whatever")

        with pytest.raises(mff.PunditSourceError, match="3 calls"):
            mff.parse_page(
                page_html(
                    pair("Arsenal 2-0 Chelsea", "1-0"),
                    pair("Everton 1-1 Burnley", "2-2"),
                    pair("Watford 0-2 Liverpool", "0-3"),
                ),
                page,
            )

    def test_a_page_yielding_far_too_many_is_refused_as_well(self) -> None:
        """A Season is 380 Fixtures. Twice that means one table is being read twice."""
        page = mff.Page(season=2017, pundit="lawrenson", path="whatever")
        one = pair("Arsenal 2-0 Chelsea", "1-0")

        with pytest.raises(mff.PunditSourceError, match="calls"):
            mff.parse_page(page_html(*[one] * (mff.MAX_CALLS + 1)), page)


class TestFetching:
    def test_a_cached_page_is_not_downloaded_again(self, project_root: Path) -> None:
        page = mff.PAGES[0]
        served = fetcher.mapping_fetcher({mff.page_url(page): b"<html></html>"})
        mff.fetch_page(page, fetcher=served)

        mff.fetch_page(page, fetcher=served)

        assert served.requested == [mff.page_url(page)]

    def test_refreshing_supersedes_rather_than_overwrites(self, project_root: Path) -> None:
        """A season page grows as the season runs, exactly as a Football-Data file does."""
        page = mff.PAGES[0]
        mff.fetch_page(page, fetcher=fetcher.mapping_fetcher({mff.page_url(page): b"<p>at 20</p>"}))

        path = mff.fetch_page(
            page,
            refresh=True,
            fetcher=fetcher.mapping_fetcher({mff.page_url(page): b"<p>at 38</p>"}),
        )

        assert path.read_bytes() == b"<p>at 38</p>"
        assert [p.read_bytes() for p in cache.superseded_dir(path).iterdir()] == [b"<p>at 20</p>"]
