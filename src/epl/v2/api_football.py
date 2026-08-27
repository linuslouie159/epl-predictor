"""An API-Football client, deferred out of v1 — on a premise that is now in doubt (decision 12).

**What it would do.** Supply the one input the live loop cannot currently confirm: the Premier
League Fixtures of an upcoming Prediction Round, with enough lead time to seal Predictions before
the first kickoff, and ideally with pre-match odds attached. :mod:`epl.live.upcoming` already turns
a table of upcoming Fixtures into a sealable round and :mod:`epl.live.seal` already writes and
commits it, so a client would be an alternative *source* behind an interface that exists, not a new
stage. It would need an account and an API key, which is why the free file was preferred.

**Why it was deferred.** Because :mod:`epl.ingest.fixtures` already fetches
``football-data.co.uk/fixtures.csv`` — a rolling file, free, unauthenticated, carrying every league
Football-Data covers with ``Avg*`` market-average odds on each row. That is the Market Line's own
source in the same shape the corpus uses, so a paid API would have bought a second spelling of data
already in hand.

**Why that reason no longer holds up, and what the ticket therefore records.** Issue #18 asks this
stub to say what would have to change for the premise to stop being true. Stage 13 went looking, and
the honest answer is that *the premise was never confirmed in the first place*. ``fixtures.csv`` has
never been observed carrying a Premier League row.

Three fetches, recorded in ``FETCHES_MEASURED`` below and in docs/DECISIONS.md under "Measured at
stage 13". All found a file holding only Fixtures dated on or before the day it was generated: one
League One tie and two Spanish on the first, one National League tie and four Spanish on the other
two. The second file was two days stale by its own ``Last-Modified`` header, and even that batch —
generated three days before a Premier League round — carried none.

The third fetch was taken eight hours after the second, on the eve of a Premier League round, and
came back **byte-identical** to it: same md5, same ``Last-Modified``. So the file is not merely
short-horizoned, it is regenerated irregularly enough to sit unchanged for two and a half days
across a matchday — which rules out the reading that a fetch timed later in the day would have
caught a fresher batch.

Two details keep this from being a network problem or a parsing problem. The first fetch carried an
``E2`` row, so this is not a file that omits English football; it has simply never been seen
carrying the one tier this project predicts. And the results half of the same upstream is fine —
``mmz4281/2627/E0.csv`` exists and parses. What is missing is specifically the *forward* horizon,
which at the moment of generation is a couple of days, shorter than a Prediction Round's own sealing
window.

**Three fetches cannot prove a negative**, which is why this is documented rather than closed, and
why ``python -m epl.live upcoming`` asks the question on any given day and writes nothing. Run it on
the Friday of a Premier League round before doing anything with this module. If E0 rows are there,
the premise holds and this stub stays a stub.

Issue #19 changed who asks. The live loop now runs on a schedule, so this question is put to
upstream twice a week by cron rather than whenever somebody remembers — and a fire that finds no
Premier League row is a quiet success by design (:class:`epl.live.upcoming.NothingToSeal`). The
consequence for this module is worth stating plainly: **the schedule will not tell anyone when the
answer changes.** It will simply start sealing rounds. ``deploy/logs/live_loop.log`` is where that
shows up, and appending a fetch here is still a hand's job.

**What is blocked on it.** More than the seal. A live Season Projection needs every *remaining*
Fixture of the campaign — :func:`epl.simulate.checkpoints.slate_at` says so in its own docstring —
and a file with a two-day horizon cannot supply the rest of a Season. Both wait on the same source.

**And a caution if this is ever picked up.** The reason ADR 0001 benchmarks against pre-match rather
than closing odds is that the Market Line's information set must match the model's. A paid API's
odds are sampled on its own schedule, not Football-Data's Tuesday/Friday convention, so swapping the
source silently changes what the benchmark knows. The Market Line for the backtest comes from the
Season files and must keep coming from them; anything fetched here would be a *live* quote whose
comparability to 7,980 scored Fixtures needs establishing rather than assuming.
"""

from __future__ import annotations

#: Every fetch of ``fixtures.csv`` made while stage 13 was measuring the premise, newest last:
#: ``(fetched, upstream Last-Modified, total rows, Premier League rows)``.
#:
#: Kept as data rather than prose so a third fetch is an append and not a rewrite. The pair of them
#: is evidence of a horizon, not proof of a rule — see the module docstring.
FETCHES_MEASURED: tuple[tuple[str, str, int, int], ...] = (
    ("2026-08-21 06:33 UTC", "not recorded", 3, 0),
    ("2026-08-27 06:12 UTC", "Tue 25 Aug 2026 09:59 GMT", 5, 0),
    ("2026-08-27 14:21 UTC", "Tue 25 Aug 2026 09:59 GMT", 5, 0),
)

#: Premier League Fixtures ever observed in the rolling file, across every fetch above.
#:
#: The number that decides whether this module stays a stub. While it is zero, the stated reason for
#: deferring an API-Football client is an assumption rather than a measurement.
#:
#: It is the last column of ``FETCHES_MEASURED`` summed, and it is written out rather than summed
#: because a stub holds no implementation — so it must be updated whenever a fetch is appended.
#: ``tests/v2/test_stubs_are_unreachable.py`` checks the two agree, which is the whole reason it is
#: safe to state a total in a module that may not compute one.
PREMIER_LEAGUE_ROWS_SEEN: int = 0

#: What would have to become true for this to stop being a stub. Any one of these is enough.
WHAT_WOULD_REVIVE_IT: tuple[str, ...] = (
    "`python -m epl.live upcoming`, run on the Friday of a Premier League round, keeps reporting no"
    " E0 rows — establishing the fetches above as a standing condition rather than a gap. Issue"
    " #19's schedule now asks this twice a week, so the evidence accumulates in"
    " deploy/logs/live_loop.log whether or not anybody is reading it",
    "the rolling file's forward horizon stays shorter than a Prediction Round's sealing window, so"
    " a Fixture is never listed while there is still time to seal a Prediction for it",
    "a live Season Projection is wanted, which needs every remaining Fixture of the campaign and"
    " cannot be fed from a file that is a couple of days wide however reliable it becomes",
    "the free file starts omitting the market-average odds columns the Market Line is built from,"
    " leaving the Fixtures usable and the benchmark unfeedable",
)

#: What building it would actually take, beyond deciding to.
WHAT_IT_NEEDS: tuple[str, ...] = (
    "an account and an API key, plus a place to keep the key that is not this repository — the"
    " first authenticated source in a project whose every input so far is a public file",
    "Aliases for the new source's Club spellings, added to epl.clubs the generated way, with the"
    " cache check that fails loudly on a spelling the table does not cover (decision 5)",
    "a rule for caching an authenticated response under data/raw/ that keeps the byte-identical"
    " guarantee, given that a paid endpoint's payload is not a file anyone else can re-fetch",
    "a decision on whether its odds may feed the Market Line at all, or whether it supplies"
    " Fixtures only and the odds keep coming from Football-Data — see ADR 0001 above",
    "a rate-limit and cost budget that survives the live loop running weekly for a whole Season",
)

__all__ = [
    "FETCHES_MEASURED",
    "PREMIER_LEAGUE_ROWS_SEEN",
    "WHAT_IT_NEEDS",
    "WHAT_WOULD_REVIVE_IT",
]
