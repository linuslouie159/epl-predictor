"""The Season in progress, and the finding that it cannot be sealed from.

Issue #16 asked whether current-Season Pundit Scorelines can be fetched. The answer has two halves
and this module is the second one.

**The BBC cannot be used, and the reason is permission rather than reach.** Open risk 1 recorded
`www.bbc.co.uk` as unreachable during design; re-tested on 27 August 2026 it answers in 0.1s, the
opaque article IDs resolve, and a pundit column carries a machine-readable
``window.__INITIAL_DATA__`` beside its ``application/ld+json``. The blocker found instead is
`bbc.co.uk/robots.txt`, which asks in plain English for
"no scraping, crawling, or systematic extraction of content", "no creating
datasets from BBC content" and no text and data mining — and then disallows ``ClaudeBot``,
``Claude-Web`` and ``anthropic-ai`` from the whole site. Building a fetcher for a committed dataset
of BBC calls is the named case. That answer does not change when a network does, which is why it is
recorded here as settled rather than as a thing to retry.

**MyFootballFacts wins as the source, and is too slow to seal a Prediction.** It permits what the
BBC forbids — its `robots.txt` allows ``ClaudeBot`` and ``anthropic-ai`` explicitly — it publishes
an index the BBC has no equivalent of, and a 2026/27 page exists. But the archive transcribes a
matchday *after* it has been played, with the results already filled in. Measured against archived
snapshots of the 2025/26 page, standing where a Prediction Round's As-Of Instant stands:

===========  =================  ==============  ==============
Snapshot     Next round kicks   Fixtures        Already called
===========  =================  ==============  ==============
2025-08-15   same day           10              10
2026-02-04   2 days later       10              **0**
2026-03-30   11 days later      10              **0**
===========  =================  ==============  ==============

A fourth snapshot exists, 2025-12-21, and is left out of both this table and the one in
docs/DECISIONS.md because its round had already kicked off — it can only show that the archive
records the past, which nobody doubts. Three of the four can be asked the question that matters.

and the live 2026/27 page said the same thing on 27 August 2026: one day before the second round's
kickoff it held the first round only, results filled in. Only the season opener was ever covered
before kickoff, and that is the one round the archive publishes in advance.

So a Pundit **cannot be part of a Sealed Prediction** — not because the calls do not exist when
they are needed, but because the only source this project is permitted to read has not transcribed
them yet. The consequence for issue #17 is concrete: the live loop seals the models and the Market
Line, and a Pundit's column on the three-way board is filled in retrospectively, once the archive
catches up.

**Two measurements, because only one of them can be taken today.** :func:`lag` looks backwards —
how far behind the last Fixture played the archive is — and needs only played matches, which is
what the corpus is. :func:`coverage` looks forwards — are the next round's Fixtures all called
yet? — and therefore needs Fixtures that have *not* kicked off, which the corpus by definition does
not hold; those arrive with ``fixtures.csv`` at issue #17. Rolling the two together would mean
reporting "cannot tell yet" as ``sealable=False``, which is the one answer this finding must not
give by accident.

Two things separate a live page from the nine frozen ones, and both are facts:

* it is **partial**, so :data:`epl.pundits.myfootballfacts.MIN_CALLS` cannot apply — a live page
  has held between nought and 380 calls and every value is correct on some day;
* a call may name a Fixture the corpus has never seen, because the corpus is a table of *played*
  matches. :func:`epl.pundits.dataset._locate` refuses that case and says in its own docstring that
  doing so is right for the backfill and wrong here. :func:`build` holds those calls back instead
  and hands everything else to the frozen builder, so there is exactly one implementation of what
  a listing becomes.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from epl.clubs import ClubResolver
from epl.ingest.fetcher import Fetcher
from epl.pundits import dataset
from epl.pundits import myfootballfacts as mff
from epl.pundits.dataset import DIVISION
from epl.rounds import anchor
from epl.windows import season_label

#: How few calls a live page may hold. Nought: before the opening round the page is real, correct
#: and empty. The backfill's floor of 360 exists because those Seasons are complete, and there is
#: no equivalent here — which is why :func:`build` reconciles against the corpus instead, and why
#: a live page is never the thing a frozen dataset is rebuilt from.
MIN_LIVE_CALLS = 0

#: The upper guard does survive. A page scanned twice doubles whether or not the Season is over.
MAX_LIVE_CALLS = mff.MAX_CALLS


class LiveError(Exception):
    """A Season in progress could not be read, or was asked a question it has no answer to."""


@dataclass(frozen=True, slots=True)
class LiveCalls:
    """What one live page yielded: the calls that reconciled, and the ones held back.

    ``unplaced`` is not an error list. A call the corpus cannot place is either a Fixture that has
    not kicked off — the archive running ahead of the football, which the season opener does — or
    two names that resolved to the wrong Clubs. Handing both back named, rather than dropping them,
    is what keeps the second from hiding inside the first: measured, ``unplaced`` is empty on every
    live page seen so far, so anything in it is worth reading.
    """

    calls: pd.DataFrame
    unplaced: pd.DataFrame


@dataclass(frozen=True, slots=True)
class Lag:
    """How far behind the football the archive is — the latency issue #16 asks to be stated.

    Backward-looking, and that is what makes it answerable. It needs only Fixtures that have been
    *played*, which is exactly what the corpus holds, so it can be computed for a Season in
    progress today. :class:`Coverage` cannot (see there).

    ``behind_days`` is measured against the **last Fixture played** rather than against today,
    because comparing the archive with the calendar makes an international break look like lag when
    the archive is perfectly current.
    """

    season: int
    as_of: dt.date
    published: int
    last_played: dt.date | None
    latest_call: dt.date | None
    behind_days: int | None

    def describe(self) -> str:
        """The latency, for the command line."""
        behind = "unknown" if self.behind_days is None else f"{self.behind_days} day(s)"
        return (
            f"{season_label(self.season)} as of {self.as_of}\n"
            f"  calls published        {self.published}\n"
            f"  latest call is for     {self.latest_call}\n"
            f"  last Fixture played    {self.last_played}\n"
            f"  archive is behind by   {behind}"
        )


@dataclass(frozen=True, slots=True)
class Coverage:
    """Whether the archive is early enough to seal the next Prediction Round.

    Forward-looking, and therefore **not answerable from the corpus alone**: it needs the Fixtures
    of a round that has not kicked off, and the corpus is a table of played matches. Those come
    from ``fixtures.csv`` at issue #17, the same gap ``epl.simulate.checkpoints.slate_at`` names.
    So this is the question the live loop will ask, and :class:`Lag` is the one that can be asked
    today. Keeping them apart is what stops "cannot tell yet" from being reported as "no".
    """

    season: int
    as_of: dt.date
    published: int
    next_anchor: dt.date
    next_kickoff: dt.date
    round_fixtures: int
    round_called: int

    @property
    def sealable(self) -> bool:
        """Whether every Fixture of the next round already carries a call.

        Half a round is not a round: sealing it would put a Pundit on some Fixtures of a Prediction
        Round and not others, which is a track record of nothing.
        """
        return self.round_fixtures > 0 and self.round_called == self.round_fixtures

    def describe(self) -> str:
        """The verdict, for the command line."""
        verdict = (
            "the next round can be sealed"
            if self.sealable
            else "the next round CANNOT be sealed from this archive"
        )
        return (
            f"{season_label(self.season)} as of {self.as_of}\n"
            f"  calls published        {self.published}\n"
            f"  next round anchored    {self.next_anchor} (first kickoff {self.next_kickoff})\n"
            f"  its Fixtures           {self.round_fixtures}\n"
            f"  already called         {self.round_called}\n"
            f"  => {verdict}"
        )


def page_for(season: int, pages: Sequence[mff.Page]) -> mff.Page:
    """The archive page for one Season, out of the pages discovery found.

    Named rather than built. The slug convention has changed four times in nine Seasons and changed
    again for 2026/27, so a URL derived from the Season is a 404 waiting for a fixed calendar.
    """
    for page in pages:
        if page.season == season:
            return page
    known = ", ".join(season_label(page.season) for page in pages) or "none"
    raise LiveError(
        f"the index carries no page for {season_label(season)}; it carries {known}"
    )


def parse(html: bytes | str, page: mff.Page) -> pd.DataFrame:
    """One live page's calls, with the backfill's size floor lifted and its ceiling kept."""
    return mff.parse_page(html, page, minimum=MIN_LIVE_CALLS)


def fetch(
    page: mff.Page,
    *,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Refresh one live page and parse it.

    ``refresh=True`` always, which is the point: a live page gains a matchday a week, and
    :mod:`epl.ingest.cache` archives the bytes it replaces rather than overwriting them, so what
    the source said at each fetch stays recoverable.
    """
    path = mff.fetch_page(page, refresh=True, fetcher=fetcher, timeout=timeout)
    return parse(path.read_bytes(), page)


def build(
    matches: pd.DataFrame,
    listings: pd.DataFrame,
    *,
    resolver: ClubResolver | None = None,
) -> LiveCalls:
    """Turn a live page's listings into rows of the frozen dataset's own schema.

    The calls the corpus can place go through :func:`epl.pundits.dataset.build` unchanged, so the
    twice-listed-Fixture rule, the cross-check against Football-Data and the column order are the
    backfill's and not a second copy of them. Only the *filtering* is written here, because only
    the filtering is different.

    Club resolution therefore runs twice — once here to decide what is placeable, once inside the
    builder. That is deliberate: the alternative is to reach past the builder's front door and call
    its private stages in order, which would make this module a second definition of what a listing
    becomes. Resolution is a dict lookup over at most 380 rows, and correctness is worth it.
    """
    resolver = resolver or ClubResolver.load()
    if listings.empty:
        return LiveCalls(dataset.build(matches, listings, resolver=resolver).calls, listings.copy())

    resolved = listings.assign(
        home_club=resolver.resolve_series(listings["home_name"], mff.SOURCE),
        away_club=resolver.resolve_series(listings["away_name"], mff.SOURCE),
    )
    played: pd.DataFrame = matches.loc[matches["division"] == DIVISION]
    known = set(dataset.fixture_keys(played))
    placeable = [key in known for key in dataset.fixture_keys(resolved)]
    held_back = [not one for one in placeable]

    built = dataset.build(matches, listings.loc[placeable], resolver=resolver)
    return LiveCalls(built.calls, resolved.loc[held_back].reset_index(drop=True))


def coverage(
    calls: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    as_of: dt.date,
) -> Coverage:
    """How much of the next Prediction Round the archive has already published.

    ``fixtures`` must include the Fixtures of a round that has **not** kicked off, so the corpus
    alone will not do for a Season in progress — see :class:`Coverage`. Where they come from is
    otherwise not this function's business: the corpus supplies them when the measurement is taken
    over a completed Season, and issue #17's ``fixtures.csv`` will supply them live. The same
    ignorance :mod:`epl.simulate.table` has about posteriors.
    """
    kickoffs = _dates(fixtures)
    upcoming = fixtures.loc[kickoffs >= as_of]
    if upcoming.empty:
        raise LiveError(
            f"no Fixture in the frame kicks off on or after {as_of}, so there is no next "
            "Prediction Round to ask about. A corpus of played matches always looks like this "
            "for a Season in progress; the upcoming Fixtures come from fixtures.csv at issue #17"
        )

    anchors = kickoffs.loc[upcoming.index].map(anchor)
    next_anchor = anchors.min()
    in_round = upcoming.loc[anchors == next_anchor]

    called = set(zip(calls["home_club"], calls["away_club"], strict=True))
    round_called = sum(
        (home, away) in called
        for home, away in zip(in_round["home_club"], in_round["away_club"], strict=True)
    )

    return Coverage(
        season=_one_season(fixtures),
        as_of=as_of,
        published=len(calls),
        next_anchor=next_anchor,
        next_kickoff=min(kickoffs.loc[in_round.index]),
        round_fixtures=len(in_round),
        round_called=round_called,
    )


def lag(
    calls: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    as_of: dt.date,
) -> Lag:
    """How far behind the football the archive is.

    Needs only Fixtures that have been played, so unlike :func:`coverage` this can be answered
    from the corpus for a Season in progress — which is what makes it the half of issue #16's
    latency question that a command can actually print today.
    """
    kickoffs = _dates(fixtures)
    played = kickoffs.loc[kickoffs < as_of]
    last_played = max(played) if len(played) else None
    latest_call = max(_dates(calls)) if len(calls) else None
    behind = (
        (last_played - latest_call).days
        if last_played is not None and latest_call is not None
        else None
    )

    return Lag(
        season=_one_season(fixtures),
        as_of=as_of,
        published=len(calls),
        last_played=last_played,
        latest_call=latest_call,
        behind_days=max(behind, 0) if behind is not None else None,
    )


def _one_season(fixtures: pd.DataFrame) -> int:
    """The Season these Fixtures belong to, refusing a frame that spans more than one.

    Both measurements are about one Season's archive page against one Season's football, and a
    frame carrying two would silently report the first — which is the kind of number that looks
    right and is about nothing.
    """
    seasons = sorted(set(fixtures["season"].astype(int)))
    if len(seasons) != 1:
        raise LiveError(
            f"expected Fixtures from exactly one Season, got {seasons or 'none'}"
        )
    return seasons[0]



def _dates(frame: pd.DataFrame) -> pd.Series:
    """The ``date`` column as ``datetime.date``, however the caller happens to hold it.

    :func:`epl.ingest.match_table` reads the corpus off disk without parsing dates, while
    :func:`epl.ingest.load_matches` and :mod:`epl.pundits.dataset` both hand over real ones. This
    function takes a frame of Fixtures from anywhere, so it normalises rather than requiring one.
    """
    return pd.to_datetime(frame["date"]).dt.date
