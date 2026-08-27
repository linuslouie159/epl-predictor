"""The MyFootballFacts backfill: nine season pages of published Pundit Scorelines.

MyFootballFacts archives the BBC's weekly pundit column season by season — Mark Lawrenson for
2017/18-2021/22, Chris Sutton from 2022/23 — and it is the only place nine Seasons of those calls
exist in one shape. **BBC is the origin**; this site is the archive, and both are named wherever
the dataset is described.

What comes back from here is what the page said, resolved no further: the two Club names *as the
source spelled them*, the Scoreline the pundit called, and the score the page publishes beside it.
Resolving those spellings to Clubs is :mod:`epl.pundits.dataset`'s job, because that is where the
Alias table and the corpus are, and a parser that resolved as it read would have nowhere to report
a spelling it did not know.

The pages are hand-maintained HTML and read like it. Everything below is a fact found in the real
nine, not a defensive guess:

* **The call is written inside the fixture.** A cell reads ``Ipswich Town 1-3 Liverpool`` — that
  ``1-3`` is Sutton's call, not the result. The result sits in the next cell along, as ``0 - 2``.
* **The layout is column blocks**, two or three matchdays side by side, and which table holds them
  moves between pages — it is table 0 on one, table 2 on another. So every table is scanned and a
  call is recognised by its shape rather than by where it sits.
* **Names carry annotations.** A trailing ``*``, or a trailing ``(14.02)`` naming the date a
  Fixture was moved to. Those are notes about the Fixture; stripping them here is what keeps them
  out of the Alias table, which holds spellings of Clubs.
* **Names are also genuinely misspelled** — ``Wolverhampton Wand``, ``Brighton & Hove Alb``. Those
  *are* spellings, so they are Alias rows rather than anything this module knows about.
* **Twice, a call landed inside a Club's name.** ``Burnley v Brighton 1-2 Hove Albion`` is the
  source writing ``Burnley v Brighton & Hove Albion`` and putting the Scoreline where the ``&``
  belongs. See :func:`_split_listing`.
* **A result cell of ``PP`` or ``AA``** means the Fixture was postponed or abandoned, so this
  listing is not the one it was eventually played under. The call is still real and is still
  returned; :mod:`epl.pundits.dataset` decides which listing of a twice-listed Fixture stands.

One guard matters more than it looks. A Season is 380 Fixtures, so a page that parses into three
calls has not failed — it has quietly succeeded at reading a table that is no longer the one the
predictions are in. :data:`MIN_CALLS` and :data:`MAX_CALLS` turn that into an exception.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from epl.ingest import cache
from epl.ingest.fetcher import Fetcher, default_fetcher
from epl.paths import raw_dir
from epl.windows import season_label

if TYPE_CHECKING:  # pragma: no cover - the parser imports bs4 lazily, the annotations do not
    from bs4 import BeautifulSoup

#: The source name under which MyFootballFacts' spellings are registered as Aliases.
SOURCE = "myfootballfacts"

#: Who published the calls this archive collects. The Pundits worked for the BBC; MyFootballFacts
#: transcribed them. Attribution names both, everywhere the dataset is described (issue #11).
ORIGIN = "BBC"

#: Where the archive lives. Every page is this plus its own path segment.
BASE_URL = "https://www.myfootballfacts.com/premier-league/all-time-premier-league/predictions"


@dataclass(frozen=True, slots=True)
class Page:
    """One Season's archive page: whose calls, which Season, and where it lives."""

    season: int
    pundit: str
    path: str


#: The nine pages, in Season order (docs/DECISIONS.md, "Source URLs"). The path spellings are the
#: site's, not ours — three different naming conventions across nine Seasons — so they are data
#: here rather than something built from the Season.
PAGES: tuple[Page, ...] = (
    Page(2017, "lawrenson", "mark-lawrensons-predictions-2017-18"),
    Page(2018, "lawrenson", "mark-lawrensons-predictions-2018-19"),
    Page(2019, "lawrenson", "lawros-predictions-premier-league-2019-20"),
    Page(2020, "lawrenson", "lawros-predictions-premier-league-2020-21"),
    Page(2021, "lawrenson", "lawros-predictions-premier-league-2021-22"),
    Page(2022, "sutton", "chris-sutton-predictions-for-premier-league-2022-23"),
    Page(2023, "sutton", "chris-sutton-predictions-for-premier-league-2023-24"),
    Page(2024, "sutton", "chris-sutton-predictions-for-premier-league-2024-25"),
    Page(2025, "sutton", "chris-sutton-predictions-for-premier-league-2025-26"),
)

#: The index every Season page is linked from, and the one thing the archive has that the BBC does
#: not (issue #16). :data:`PAGES` can be a literal because those nine Seasons are over; the Season
#: in progress cannot, because the archive has used **four** slug conventions across the eighteen
#: Seasons it links, and the newest arrived with 2026/27 —
#: ``chris-sutton-predictions-premier-league-2026-27`` drops the ``for-`` its four predecessors
#: carried. A URL built from the Season would 404.
INDEX_URL = f"{BASE_URL}/"

#: How many Season pages the index must yield before discovery is believed. Eighteen are linked as
#: of August 2026; the floor is the nine already frozen, because discovery that cannot see the
#: committed backfill has stopped working whatever else it came back with.
MIN_DISCOVERED = len(PAGES)

#: How many index pages to follow before giving up on ``rel="next"`` terminating. Three exist;
#: this is the guard against a redesign that links every page to the next one forever.
MAX_INDEX_PAGES = 25

#: The Season at the end of a slug — ``2017-18``, ``2026-27``. The four-digit half is the Season's
#: identity everywhere else in the project, so that is what is kept.
_SEASON_IN_SLUG = re.compile(r"(?:^|-)(?P<start>\d{4})-(?P<end>\d{2})$")

#: Which Pundit a slug names. ``lawro`` and ``lawrenson`` are the same person written two ways, and
#: both appear — the archive renamed the column mid-record. Matched longest-first so
#: ``mark-lawrensons`` cannot be read as anything else.
_PUNDIT_IN_SLUG: tuple[tuple[str, str], ...] = (
    ("sutton", "sutton"),
    ("lawrenson", "lawrenson"),
    ("lawro", "lawrenson"),
)

#: Canonical column order for one page's listings, before any Club is resolved. Named apart from
#: :data:`epl.pundits.dataset.CALL_COLUMNS`, which is the frozen dataset's much shorter list — a
#: listing is what the page said and a call is what survived being reconciled with the corpus.
#: ``played`` says whether this listing carried a real score rather than ``PP``; the two
#: ``published_*`` columns are what the page says happened, kept only so the parse can be checked
#: against Football-Data and never stored (:mod:`epl.pundits.dataset`).
LISTING_COLUMNS: tuple[str, ...] = (
    "season",
    "pundit",
    "home_name",
    "away_name",
    "pred_home_goals",
    "pred_away_goals",
    "played",
    "published_home_goals",
    "published_away_goals",
)

#: What each column is held as. Stated rather than inferred: the two ``published_*`` columns are
#: nullable integers because a postponed listing has no result, and a frame that inferred its own
#: dtypes would hold plain integers on a page with no postponements and floats on a page with one.
LISTING_DTYPES: dict[str, str] = {
    "season": "int64",
    "pundit": "string",
    "home_name": "string",
    "away_name": "string",
    "pred_home_goals": "int64",
    "pred_away_goals": "int64",
    "played": "bool",
    "published_home_goals": "Int64",
    "published_away_goals": "Int64",
}

#: How few calls a page may yield before the parse is treated as having failed rather than found
#: little. The thinnest of the nine gives 378; a Season is 380 Fixtures, and the page is the whole
#: Season, so anything under 360 means a table moved out from under this parser.
MIN_CALLS = 360

#: And how many is too many. Re-arranged Fixtures are listed twice, so a page can exceed 380
#: honestly — but not by much, and a page scanned twice would double.
MAX_CALLS = 420

#: A call, as the page writes it: two Club names with the pundit's Scoreline between them. Goals
#: are one or two digits, which is what keeps ``The 2017-18 Premier League Table`` — two phrases
#: either side of what looks like a score — from parsing as Watford against Chelsea.
_LISTING = re.compile(
    r"^(?P<home>.+?)\s+(?P<home_goals>\d{1,2})\s*-\s*(?P<away_goals>\d{1,2})\s+(?P<away>.+)$"
)

#: The result cell beside it. Spelled ``1-0``, ``1 - 0`` and ``1  -  0`` across the nine pages.
_RESULT = re.compile(r"^(?P<home_goals>\d{1,2})\s*-\s*(?P<away_goals>\d{1,2})$")

#: A trailing note on a Club's name — ``(14.02)``, ``(19th May)``, ``(09/02/19)``. Always the date
#: a Fixture was moved to, never part of a Club.
_ANNOTATION = re.compile(r"\s*\([^)]*\)\s*$")


class PunditSourceError(Exception):
    """A MyFootballFacts page did not look the way this module requires."""


def page_url(page: Page) -> str:
    """The upstream URL for one Season's archive page."""
    return f"{BASE_URL}/{page.path}/"


def raw_page_path(page: Page) -> Path:
    """Where that URL's bytes are cached. The layout mirrors the URL, so provenance is obvious."""
    return raw_dir() / SOURCE / f"{page.path}.html"


def discover_pages(
    *,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
    minimum: int | None = None,
) -> tuple[Page, ...]:
    """Every Season page the index links to, in Season order.

    This is what :data:`PAGES` cannot be for a Season in progress. The nine frozen pages could be
    written down because those Seasons are over; the current one is reachable only by asking the
    index, since the slug convention has changed four times and changed again for 2026/27.

    Pagination follows the index's own ``rel="next"``. The alternative — walking ``page/2/``,
    ``page/3/`` until one 404s — builds a loop whose exit condition is an error, and would ask
    upstream for a page it was never told exists.

    ``minimum`` defaults to :data:`MIN_DISCOVERED`, read here rather than bound as a default so the
    constant stays the single authority. It exists for tests that build an index of one link.
    """
    from bs4 import BeautifulSoup

    minimum = MIN_DISCOVERED if minimum is None else minimum
    fetcher = fetcher or default_fetcher(timeout)

    by_season: dict[int, Page] = {}
    url: str | None = INDEX_URL
    seen: set[str] = set()

    while url and url not in seen and len(seen) < MAX_INDEX_PAGES:
        seen.add(url)
        soup = BeautifulSoup(fetcher(url), "lxml")
        for slug in _linked_slugs(soup):
            page = _page_from_slug(slug)
            if page is None:
                continue
            if page.season in by_season and by_season[page.season].path != page.path:
                raise PunditSourceError(
                    f"the index links two Season pages for {season_label(page.season)} — "
                    f"{by_season[page.season].path!r} and {page.path!r}. One Season is one "
                    "record of one Pundit's calls, and nothing here says which of two is it"
                )
            by_season.setdefault(page.season, page)
        url = _next_index(soup)

    if len(by_season) < minimum:
        raise PunditSourceError(
            f"{INDEX_URL} yielded {len(by_season)} Season page(s); expected at least {minimum}. "
            "The index has most likely been restructured upstream"
        )
    return tuple(by_season[season] for season in sorted(by_season))


def _linked_slugs(soup: BeautifulSoup) -> list[str]:
    """The path segment of every link that points at a page *under* the predictions index."""
    prefix = f"{BASE_URL}/"
    slugs = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).split("?", 1)[0].split("#", 1)[0]
        if not href.startswith(prefix):
            continue
        slug = href[len(prefix) :].strip("/")
        if slug and "/" not in slug:
            slugs.append(slug)
    return slugs


def _page_from_slug(slug: str) -> Page | None:
    """One Season page, or ``None`` for a link that is not one.

    The index carries a landing page (``mark-lawrenson-predictions``) and a feed alongside the
    Season pages, and neither names a Season. Absence of a Season is the test, rather than a list
    of things to skip that a future addition would slip past.

    A slug that *does* name a Season but no Pundit is skipped too, and that is the one place this
    is a judgement rather than a fact: it would be a third forecaster taking over the column, and
    the alternative — raising — would instead break on any unrelated link ending in a year pair.
    The skip is visible rather than silent, because the Season range discovery found is printed by
    ``python -m epl.pundits live`` and :data:`MIN_DISCOVERED` still has to be cleared.
    """
    season = _SEASON_IN_SLUG.search(slug)
    if season is None:
        return None
    for token, pundit in _PUNDIT_IN_SLUG:
        if token in slug:
            return Page(int(season.group("start")), pundit, slug)
    return None


def _next_index(soup: BeautifulSoup) -> str | None:
    """The index's own link to its successor, if it offers one."""
    link = soup.find("link", rel="next", href=True)
    return str(link["href"]) if link else None


def fetch_page(
    page: Page,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download one Season's page into the raw cache, and return the cached path.

    A cached file is reused unless ``refresh`` is set. Refreshing matters for the Season in
    progress, whose page gains a matchday a week — and, exactly as with a Football-Data file, a
    refresh archives the old bytes rather than overwriting them (:mod:`epl.ingest.cache`).
    """
    path = raw_page_path(page)
    if path.exists() and not refresh:
        return path

    fetcher = fetcher or default_fetcher(timeout)
    return cache.store(path, fetcher(page_url(page)))


def fetch_all(
    pages: Sequence[Page] = PAGES,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
) -> list[Path]:
    """Fill the raw cache for every Season page. Returns the cached paths in Season order."""
    fetcher = fetcher or default_fetcher(timeout)
    return [fetch_page(page, refresh=refresh, fetcher=fetcher) for page in pages]


def read_page(page: Page) -> pd.DataFrame:
    """One Season's calls, parsed out of the cached bytes."""
    path = raw_page_path(page)
    if not path.exists():
        raise PunditSourceError(
            f"{path} is not cached; run `python -m epl.pundits fetch` before building"
        )
    return parse_page(path.read_bytes(), page)


def read_all(pages: Sequence[Page] = PAGES) -> pd.DataFrame:
    """Every Season's calls as one frame, in Season order."""
    return pd.concat([read_page(page) for page in pages], ignore_index=True)


def parse_page(html: bytes | str, page: Page, *, minimum: int | None = None) -> pd.DataFrame:
    """Every call on one Season's page, in the order the page lists them.

    ``minimum`` defaults to :data:`MIN_CALLS`, read here rather than bound as a default so that the
    constant stays the single authority. Two callers pass it: tests that build a page of three rows
    on purpose, and :func:`epl.pundits.live.parse`, for which the floor is *wrong* rather than
    inconvenient — a Season in progress has published between nought and 380 calls and every one of
    those is the correct answer on some day (issue #16).
    """
    from bs4 import BeautifulSoup

    minimum = MIN_CALLS if minimum is None else minimum
    soup = BeautifulSoup(html, "lxml")
    calls = [
        _call(page, listing, result)
        for table in soup.find_all("table")
        for row in table.find_all("tr")
        for listing, result in _listings(
            [_flatten(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        )
    ]
    if not minimum <= len(calls) <= MAX_CALLS:
        raise PunditSourceError(
            f"{page_url(page)} parsed into {len(calls)} calls, and a Season is 380 Fixtures. "
            f"Expected between {minimum} and {MAX_CALLS}; the page's tables have most likely "
            "been restructured upstream"
        )
    return pd.DataFrame(calls, columns=list(LISTING_COLUMNS)).astype(LISTING_DTYPES)


def _listings(cells: list[str]) -> list[tuple[re.Match[str], str]]:
    """The calls in one row, each with the cell that follows it.

    A row holds two or three matchdays side by side, so a call is found by its shape and its
    result is read from the next cell along rather than from a column index — the number of
    columns, and which of them hold anything, differ between the nine pages and within one page.
    """
    found = []
    for position, text in enumerate(cells):
        listing = _LISTING.match(text)
        if listing is not None:
            found.append((listing, cells[position + 1] if position + 1 < len(cells) else ""))
    return found


def _call(page: Page, listing: re.Match[str], result_cell: str) -> dict[str, object]:
    home, away = _split_listing(listing["home"], listing["away"])
    result = _RESULT.match(result_cell)
    return {
        "season": page.season,
        "pundit": page.pundit,
        "home_name": home,
        "away_name": away,
        "pred_home_goals": int(listing["home_goals"]),
        "pred_away_goals": int(listing["away_goals"]),
        "played": result is not None,
        "published_home_goals": int(result["home_goals"]) if result else None,
        "published_away_goals": int(result["away_goals"]) if result else None,
    }


def _split_listing(home: str, away: str) -> tuple[str, str]:
    """The two Club names, with the ``Home v Away`` mangling undone.

    ``Burnley v Brighton 1-2 Hove Albion`` is one cell, and the naive reading of it is a Club
    called ``Burnley v Brighton`` at home to one called ``Hove Albion``. What happened is that the
    source wrote the fixture as ``Burnley v Brighton & Hove Albion`` and dropped the Scoreline
    where the ``&`` was. The ``v`` is therefore the separator it looks like: everything before it
    is the home Club, and everything after it — on both sides of the Scoreline — is the away one.

    Twice in 3,408 calls. Handled here rather than as two Alias rows because ``Burnley v Brighton``
    is not a spelling of a Club, and putting it in the Alias table would say that it is.
    """
    home, away = _club_name(home), _club_name(away)
    if " v " in home:
        home, fragment = home.split(" v ", 1)
        home, away = _club_name(home), _club_name(f"{fragment} {away}")
    return home, away


def _club_name(written: str) -> str:
    """One Club's name with the page's annotations taken off, and nothing else touched.

    Deliberately as conservative as :func:`epl.clubs.normalise_alias`: only a trailing ``*`` and a
    trailing parenthesis come off, because both are notes the page puts *on* a name. A misspelling
    is left exactly as written, so it fails loudly at the Alias table instead of being guessed at
    here, where no table would record the guess.
    """
    name = _flatten(written)
    while True:
        shorter = _ANNOTATION.sub("", name).rstrip("*").strip()
        if shorter == name:
            return name
        name = shorter


def _flatten(text: str) -> str:
    """Collapse the source's non-breaking and doubled spaces, which appear inside Club names."""
    return " ".join(text.replace("\xa0", " ").split())
