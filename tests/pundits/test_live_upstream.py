"""Smoke tests against the live archive, run only with ``--run-network``.

Issue #16's spike, kept executable. ``tests/ingest/test_upstream_live.py`` asks Football-Data
whether a new Club spelling has appeared; these ask the Pundit source the same question, and they
ask the one the spike existed for: **is the archive early enough to seal a Prediction Round?**

The answer measured on 27 August 2026 was no. The latency itself cannot be re-derived here — a page
fetched today says what the archive holds now, not when each call arrived — so it was measured once
against archived snapshots and recorded in docs/DECISIONS.md. What these keep checkable is the
*direction*: the archive publishes a matchday after it has been played.

The tests that need the corpus **skip while it stops at 2025/26**. Ingesting the Season in progress
moves :data:`epl.windows.LAST_SEASON`, which is the leakage protocol's own module, and that belongs
to issue #17's live loop rather than to this spike. Skipping says so out loud instead of asserting
something weaker.
"""

from __future__ import annotations

import pytest

from epl.clubs import ClubResolver
from epl.ingest import load_matches
from epl.pundits import live
from epl.pundits import myfootballfacts as mff

pytestmark = pytest.mark.network

#: The Season in progress when the spike was run.
LIVE_SEASON = 2026


@pytest.fixture(scope="module")
def pages() -> tuple[mff.Page, ...]:
    return mff.discover_pages()


@pytest.fixture
def corpus_covering_the_live_season():
    """The corpus, or a skip naming why it cannot answer yet."""
    matches = load_matches()
    if not (matches["season"] == LIVE_SEASON).any():
        pytest.skip(
            f"the corpus stops before {LIVE_SEASON}/{(LIVE_SEASON + 1) % 100:02d}; ingesting the "
            "Season in progress moves epl.windows.LAST_SEASON and belongs to issue #17"
        )
    return matches


class TestDiscovery:
    def test_the_index_still_links_every_page_the_backfill_is_built_from(self, pages) -> None:
        """Discovery that cannot see the committed nine has stopped working, whatever else it
        found. This is the check that would have caught the 2026/27 slug change."""
        assert set(mff.PAGES) <= set(pages)

    def test_it_finds_more_seasons_than_the_backfill_uses(self, pages) -> None:
        """Eighteen Season pages are linked — 2009/10 onward. The backfill uses the last nine;
        the eight before 2017/18 are Lawrenson calls this project has never scored."""
        assert len(pages) >= len(mff.PAGES)
        assert min(page.season for page in pages) < min(page.season for page in mff.PAGES)

    def test_the_live_season_has_a_page_and_it_is_not_named_like_its_predecessors(
        self, pages
    ) -> None:
        """``chris-sutton-predictions-premier-league-2026-27`` drops the ``for-`` that 2022/23
        through 2025/26 carried. A URL built from the Season would have 404ed."""
        page = live.page_for(LIVE_SEASON, pages)

        assert page.pundit == "sutton"
        assert page.path not in {other.path for other in mff.PAGES}


class TestTheLivePage:
    def test_it_parses_where_the_backfill_floor_would_refuse_it(self, pages) -> None:
        """A live page is partial by definition, and refusing it would be refusing the point."""
        page = live.page_for(LIVE_SEASON, pages)
        listings = live.fetch(page)

        assert 0 <= len(listings) <= live.MAX_LIVE_CALLS
        assert set(listings["pundit"]) <= {"sutton"}

    def test_no_new_club_spelling_has_appeared_on_it(self, pages) -> None:
        """The check that must pass before a live Pundit row can be built at all.

        It has already failed once, correctly: 2026/27 promoted Coventry and Hull into a Premier
        League the archive had never covered them in, and neither spelling was in a table built
        from the nine backfilled Seasons.
        """
        page = live.page_for(LIVE_SEASON, pages)
        listings = live.fetch(page)
        resolver = ClubResolver.load()

        names = set(listings["home_name"]) | set(listings["away_name"])
        unknown = sorted(name for name in names if not resolver.knows(name, mff.SOURCE))

        assert unknown == []

    def test_its_calls_reconcile_into_the_frozen_dataset_s_schema(
        self, pages, corpus_covering_the_live_season
    ) -> None:
        """Issue #16's "whichever source wins produces rows in the same schema as the committed
        backfill", asked of the real page rather than of a fixture."""
        from epl.pundits.dataset import CALL_COLUMNS

        page = live.page_for(LIVE_SEASON, pages)
        listings = live.fetch(page)

        built = live.build(corpus_covering_the_live_season, listings)

        assert tuple(built.calls.columns) == CALL_COLUMNS
        assert "home_goals" not in built.calls.columns


class TestTheFindingItself:
    """**The spike's answer, in the only form that stays checkable.**

    The latency itself is a historical measurement and cannot be re-derived from the live page:
    fetching a Season page today shows what the archive holds *now*, not when each call arrived.
    That measurement was taken once, against archived snapshots of the 2025/26 page, and is
    recorded in docs/DECISIONS.md and in :mod:`epl.pundits.live`.

    What *is* checkable now is the direction, and it is the half the live loop turns on: the
    archive publishes a matchday **after** it has been played. So no call should ever name a
    Fixture the corpus has not seen. If one does, the archive has started running ahead of the
    football and a Pundit could be sealed after all — worth a failing test to find out about.
    """

    def test_no_call_is_published_for_a_fixture_that_has_not_been_played(
        self, pages, corpus_covering_the_live_season
    ) -> None:
        matches = corpus_covering_the_live_season
        page = live.page_for(LIVE_SEASON, pages)
        listings = live.fetch(page)

        built = live.build(matches, listings)

        assert built.unplaced.empty, (
            "the archive has published a call for a Fixture the corpus has no result for. Either "
            "it now runs ahead of kickoff — which would reopen sealing a Pundit, closed by issue "
            "#16 — or two Club names resolved wrongly. Both are worth reading before this is "
            "made to pass."
        )

    def test_every_call_it_does_publish_reconciles(
        self, pages, corpus_covering_the_live_season
    ) -> None:
        """The other side of the same coin: behind the football, but completely so."""
        matches = corpus_covering_the_live_season
        page = live.page_for(LIVE_SEASON, pages)
        listings = live.fetch(page)

        built = live.build(matches, listings)

        assert len(built.calls) == len(listings)
