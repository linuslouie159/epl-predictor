"""Finding the Season pages instead of naming them.

The nine backfill pages are named in :data:`epl.pundits.myfootballfacts.PAGES` because they are
frozen. A Season in progress cannot be, and the reason is a fact rather than a worry: the archive
has used **four** slug conventions across the eighteen Seasons it links, and the newest arrived with
2026/27 — ``chris-sutton-predictions-premier-league-2026-27`` drops the ``for-`` that the four
Seasons before it carried. Code that built the current Season's URL from the Season would have
looked right and returned a 404.

So discovery reads the index and follows its own ``rel="next"``. Every fact tested here was found
on the real index (issue #16).
"""

from __future__ import annotations

import pytest

from epl.ingest import fetcher
from epl.pundits import myfootballfacts as mff


def index_html(*slugs: str, nxt: str | None = None) -> str:
    """An index page shaped like the real one: absolute links, and a ``rel="next"`` in the head."""
    head = f'<link rel="next" href="{nxt}"/>' if nxt else ""
    links = "".join(f'<a href="{mff.BASE_URL}/{slug}/">{slug}</a>' for slug in slugs)
    return f"<html><head>{head}</head><body>{links}</body></html>"


def discover(pages: dict[str, str], **kwargs):
    """Run discovery over canned index pages, keyed by URL."""
    canned = fetcher.mapping_fetcher({url: html.encode() for url, html in pages.items()})
    return mff.discover_pages(fetcher=canned, **kwargs), canned


class TestReadingOneIndexPage:
    def test_a_slug_becomes_a_page_with_its_season_and_its_pundit(self) -> None:
        found, _ = discover(
            {mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18")},
            minimum=1,
        )

        assert found == (mff.Page(2017, "lawrenson", "mark-lawrensons-predictions-2017-18"),)

    def test_the_season_is_the_year_the_campaign_started(self) -> None:
        """``2019-20`` is Season 2019 — the same identity the rest of the project uses."""
        found, _ = discover(
            {mff.INDEX_URL: index_html("lawros-predictions-premier-league-2019-20")},
            minimum=1,
        )

        assert found[0].season == 2019

    @pytest.mark.parametrize(
        ("slug", "pundit"),
        [
            ("mark-lawrensons-predictions-2017-18", "lawrenson"),
            ("lawros-predictions-premier-league-2019-20", "lawrenson"),
            ("chris-sutton-predictions-for-premier-league-2022-23", "sutton"),
            ("chris-sutton-predictions-premier-league-2026-27", "sutton"),
        ],
    )
    def test_all_four_slug_conventions_name_their_pundit(self, slug: str, pundit: str) -> None:
        """``lawro`` and ``lawrenson`` are the same person; the fourth convention is 2026/27's."""
        found, _ = discover({mff.INDEX_URL: index_html(slug)}, minimum=1)

        assert found[0].pundit == pundit

    def test_a_slug_carrying_no_season_is_not_a_season_page(self) -> None:
        """``mark-lawrenson-predictions`` is a real link on the real index: a landing page."""
        found, _ = discover(
            {
                mff.INDEX_URL: index_html(
                    "mark-lawrenson-predictions",
                    "mark-lawrensons-predictions-2017-18",
                )
            },
            minimum=1,
        )

        assert [page.path for page in found] == ["mark-lawrensons-predictions-2017-18"]

    def test_the_index_links_to_itself_and_that_is_not_a_season(self) -> None:
        sutton = "chris-sutton-predictions-for-premier-league-2023-24"
        found, _ = discover({mff.INDEX_URL: index_html("feed", sutton)}, minimum=1)

        assert [page.season for page in found] == [2023]


class TestFollowingThePagination:
    def test_it_follows_rel_next_until_the_index_stops_offering_one(self) -> None:
        two = f"{mff.BASE_URL}/page/2/"
        found, canned = discover(
            {
                mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18", nxt=two),
                two: index_html("mark-lawrensons-predictions-2018-19"),
            },
            minimum=1,
        )

        assert [page.season for page in found] == [2017, 2018]
        assert canned.requested == [mff.INDEX_URL, two]

    def test_it_does_not_guess_a_page_that_was_never_linked(self) -> None:
        """The real index 404s on ``page/4/``. Following links means never asking for it."""
        _, canned = discover(
            {mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18")}, minimum=1
        )

        assert canned.requested == [mff.INDEX_URL]

    def test_a_next_link_that_loops_back_does_not_fetch_forever(self) -> None:
        found, canned = discover(
            {mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18", nxt=mff.INDEX_URL)},
            minimum=1,
        )

        assert [page.season for page in found] == [2017]
        assert canned.requested == [mff.INDEX_URL]


class TestWhatDiscoveryRefuses:
    def test_an_index_that_yields_too_little_is_an_error_rather_than_a_short_list(self) -> None:
        """Eighteen Season pages are linked. A parse returning one has found a redesign."""
        with pytest.raises(mff.PunditSourceError, match="1 Season page"):
            discover({mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18")})

    def test_the_floor_is_the_nine_pages_already_frozen(self) -> None:
        """Discovery that cannot see the committed backfill has stopped working, whatever else
        it found."""
        assert mff.MIN_DISCOVERED == len(mff.PAGES)

    def test_the_same_season_twice_is_refused_rather_than_silently_deduped(self) -> None:
        """Two pages for one Season means two records of one Pundit's calls, and no rule here
        for which is authoritative."""
        with pytest.raises(mff.PunditSourceError, match="2022/23"):
            discover(
                {
                    mff.INDEX_URL: index_html(
                        "chris-sutton-predictions-for-premier-league-2022-23",
                        "chris-sutton-predictions-premier-league-2022-23",
                    )
                },
                minimum=1,
            )


class TestAgainstTheFrozenNine:
    def test_the_pages_are_returned_in_season_order(self) -> None:
        found, _ = discover(
            {
                mff.INDEX_URL: index_html(
                    "chris-sutton-predictions-premier-league-2026-27",
                    "mark-lawrensons-predictions-2017-18",
                    "lawros-predictions-premier-league-2020-21",
                )
            },
            minimum=1,
        )

        assert [page.season for page in found] == [2017, 2020, 2026]

    def test_a_discovered_page_is_the_same_shape_the_backfill_already_uses(self) -> None:
        """Discovery returns :class:`~epl.pundits.myfootballfacts.Page`, so everything downstream
        — the URL, the cache path, the parse — is the code the nine already go through."""
        found, _ = discover(
            {mff.INDEX_URL: index_html("mark-lawrensons-predictions-2017-18")}, minimum=1
        )

        assert found[0] == mff.PAGES[0]
        assert mff.page_url(found[0]) == mff.page_url(mff.PAGES[0])
        assert mff.raw_page_path(found[0]) == mff.raw_page_path(mff.PAGES[0])
