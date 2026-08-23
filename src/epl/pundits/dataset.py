"""The frozen Pundit dataset: nine Seasons of published Scorelines, committed rather than scraped.

`predictions.csv` sits next to this module, exactly as the Club table sits next to
:mod:`epl.clubs`, and it is the thing the Pundit Predictors read. It is built once from the raw
cache and committed, so the accountability feature is backtestable on day one and a scoreboard run
does not depend on nine pages of someone else's HTML still being up (issue #11).

**Only facts are stored** — the Fixture, the predicted Scoreline, which Pundit, and the date. No
prose, no matchday heading, and in particular *not the result*. The page publishes the result
beside every call and this module reads it, but only to check itself: a stored Prediction that
knew its own Outcome is the thing ADR 0005 exists to make impossible.

Four rules turn a page's listings into that file, and each is a decision rather than plumbing:

**Clubs resolve through the Alias table, and an unknown spelling raises.** The same rule as the
ingest, for the same reason — a spelling that silently failed to map would split one Pundit's
record in two at the point the source changed how it writes a name.

**A Fixture listed twice keeps the call published for the date it was played.** Fifteen Fixtures
across the nine Seasons were postponed and re-listed, and the pundit called them twice — 2022/23's
Leicester against Aston Villa is 1-2 on the original date and 2-2 on the rearranged one. The call
that stands is the one for the date the Fixture actually has, which is the date its As-Of Instant
is derived from. Keeping the other would score a call made about a match that did not happen.

**The date comes from Football-Data, not from the page.** The matchday headings are unreliable —
2024/25's opening matchday is headed `16/08/25`, a year out — and the Fixture is identified by its
Club pairing within a Season anyway, which is what the ledger keys on. So each call is *located*
against the corpus, and a call the corpus has no Fixture for is refused rather than invented.

**The page's own published results are checked against Football-Data.** This is what closes open
risk 3, and it is a far stronger check than the one the ticket asks for: not one row cross-checked
but all 3,406 that carry one. It confirms the thing that would otherwise be untestable — that these
two names went to the right two Clubs, the right way round. Four rows disagree, which is the
archive mistranscribing; a Season that *stopped* agreeing would be a column shift, and
:data:`MIN_AGREEMENT` is where that becomes an exception rather than a plausible-looking dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from epl.clubs import ClubResolver
from epl.metrics import OUTCOMES
from epl.pundits import myfootballfacts
from epl.pundits.myfootballfacts import PAGES, SOURCE, Page
from epl.windows import season_label

_DATA_DIR = Path(__file__).resolve().parent

#: The tier the Pundits publish in. MyFootballFacts archives the Premier League column and no
#: other, so this is a fact about the source rather than a scope we chose.
DIVISION = "E0"

#: Canonical column order for the frozen dataset. Fixture, Scoreline, Predictor, date — the four
#: things issue #11 permits, and nothing else.
CALL_COLUMNS: tuple[str, ...] = (
    "pundit",
    "season",
    "division",
    "date",
    "home_club",
    "away_club",
    "pred_home_goals",
    "pred_away_goals",
)

#: What identifies a Fixture inside one Pundit's calls: the ledger's key without the tier, which
#: this source fixes at :data:`DIVISION`. Deliberately not the date — "a postponed Fixture is still
#: the Fixture that was predicted" (:mod:`epl.ledger.schema`).
FIXTURE_KEY: tuple[str, ...] = ("season", "home_club", "away_club")

#: What each column is held as, so a dataset read back off disk compares equal to the built one.
DTYPES: dict[str, str] = {
    "pundit": "string",
    "season": "int64",
    "division": "string",
    "home_club": "string",
    "away_club": "string",
    "pred_home_goals": "int64",
    "pred_away_goals": "int64",
}

#: How much of a Season's published results must match Football-Data before the parse is believed.
#:
#: The four real disagreements are transcription slips — one goal out, in one row — and leave the
#: worst Season at 99.5%. The failure this floor is for looks nothing like that: a column read one
#: to the left, or home and away the wrong way round, agrees only where the score is symmetric,
#: which is about a quarter of Fixtures. There is an enormous gap between those two, and the floor
#: sits in it rather than close to either.
MIN_AGREEMENT = 0.95

#: How many checked listings a Season needs before that floor means anything. Below this, one slip
#: is a large fraction and the rate says more about the sample than about the parse.
MIN_CHECKED = 30


class PunditDatasetError(Exception):
    """A Pundit's calls could not be reconciled with the corpus."""


@dataclass(frozen=True, slots=True)
class Backfill:
    """What a build produced: the calls to freeze, and what the cross-check found.

    The disagreements are handed back rather than logged, because they are the evidence that the
    parse landed on the right Fixtures. A build that reported none at all would be as worth
    looking at as one that reported hundreds.

    ``checked`` is how many calls could actually be compared, which is not the same as how many
    calls there are: two Fixtures in the nine Seasons were only ever listed as postponed, so they
    carry a call and no score. Reporting the disagreements against the number of calls would count
    a check that was never made as a check that passed. See :func:`_comparable`.
    """

    calls: pd.DataFrame
    disagreements: pd.DataFrame
    checked: int


def outcomes_of(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """The Outcome each Scoreline implies, in the ordinal (Home, Draw, Away) vocabulary.

    "A Scoreline implies an Outcome; an Outcome does not imply a Scoreline" (CONTEXT.md). This is
    the arrow that exists, and it is the whole of what a Pundit scored as-stated is doing.

    It lives beside the calls rather than in :mod:`epl.pundits.grading`, because both the grading
    and the Predictor need it and a Predictor should not have to import a reporting module to know
    what its own call means.
    """
    home, draw, away = OUTCOMES
    scored, conceded = home_goals.to_numpy(), away_goals.to_numpy()
    return pd.Series(
        np.select([scored > conceded, scored == conceded], [home, draw], default=away),
        index=home_goals.index,
        dtype="string",
    )


def fixture_keys(frame: pd.DataFrame) -> list[tuple[int, str, str]]:
    """:data:`FIXTURE_KEY` read off a frame, as tuples — how a call is looked up by Fixture."""
    return list(
        zip(
            frame["season"].astype(int),
            frame["home_club"].astype(str),
            frame["away_club"].astype(str),
            strict=True,
        )
    )


def path() -> Path:
    """The frozen dataset, shipped with the package."""
    return _DATA_DIR / "predictions.csv"


def load() -> pd.DataFrame:
    """The frozen dataset. Read once per process, like the Club table.

    Cached against the path rather than against nothing, so a caller that points :func:`path`
    somewhere else — which is how the command line is tested — reads that file rather than a copy
    of whatever was loaded first.
    """
    return _cached(path()).copy()


@lru_cache(maxsize=4)
def _cached(source: Path) -> pd.DataFrame:
    if not source.exists():
        raise PunditDatasetError(
            f"{source.name} has not been built; run `python -m epl.pundits build`"
        )
    return read(source)


def read(source: Path | None = None) -> pd.DataFrame:
    """Read a frozen dataset from disk, in canonical dtypes."""
    frame = pd.read_csv(source or path(), dtype=DTYPES)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame[list(CALL_COLUMNS)]


def write(calls: pd.DataFrame, destination: Path | None = None) -> Path:
    """Write the frozen dataset deterministically, so a rebuild is a no-op in git."""
    destination = destination or path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _conform(calls).to_csv(destination, index=False, lineterminator="\n")
    _cached.cache_clear()
    return destination


def build_from_cache(
    matches: pd.DataFrame,
    pages: Sequence[Page] = PAGES,
    *,
    resolver: ClubResolver | None = None,
) -> Backfill:
    """Parse the cached pages and reconcile them with the corpus."""
    return build(matches, myfootballfacts.read_all(pages), resolver=resolver)


def build(
    matches: pd.DataFrame,
    listings: pd.DataFrame,
    *,
    resolver: ClubResolver | None = None,
    min_checked: int = MIN_CHECKED,
) -> Backfill:
    """Turn one or more pages' listings into the frozen dataset, refusing what will not reconcile.

    ``matches`` is the corpus. ``listings`` is what
    :func:`epl.pundits.myfootballfacts.parse_page` produced — the two Club names as the source
    spelled them, the call, and the result the page published beside it.

    ``min_checked`` defaults to :data:`MIN_CHECKED` and exists for tests that build a Season of two
    rows on purpose; nothing in the pipeline passes it.
    """
    resolved = _resolve(listings, resolver or ClubResolver.load())
    chosen = _choose(resolved)
    located = _locate(chosen, matches)
    disagreements = _disagreements(located)
    _refuse_a_season_that_stopped_agreeing(located, disagreements, min_checked)
    return Backfill(_conform(located), disagreements, int(_comparable(located).sum()))


def _resolve(listings: pd.DataFrame, resolver: ClubResolver) -> pd.DataFrame:
    """The two spellings as slugs. Raises :class:`~epl.clubs.UnknownAliasError` on any it cannot
    place, naming every one of them rather than the first."""
    return listings.assign(
        home_club=resolver.resolve_series(listings["home_name"], SOURCE),
        away_club=resolver.resolve_series(listings["away_name"], SOURCE),
    )


def _choose(calls: pd.DataFrame) -> pd.DataFrame:
    """One call per Fixture: the listing it was played under, or the only one there is.

    Sorting by ``played`` and taking the first is what makes the choice independent of the order
    the page happens to list the two in — 2020/21 puts the played listing first and 2022/23 puts
    it second, and a rule that depended on that would be right half the time.
    """
    played = calls.loc[calls["played"]]
    doubled = played.duplicated(subset=list(FIXTURE_KEY))
    if doubled.any():
        offending = played.loc[doubled].iloc[0]
        raise PunditDatasetError(
            f"{season_label(int(offending['season']))} lists "
            f"{offending['home_club']} v {offending['away_club']} as played more than once; "
            "one Fixture has one Outcome, so the parse has read a table twice"
        )
    ordered = calls.sort_values("played", ascending=False, kind="stable")
    return ordered.drop_duplicates(subset=list(FIXTURE_KEY), keep="first")


def _locate(calls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach each call's Fixture from the corpus, refusing any call that has none.

    Both Clubs resolved and the Season is real, so a call with no Fixture behind it means the two
    names reached the wrong slugs, or reached the right ones the wrong way round. Either way it is
    a Prediction about a match nobody played, and it must not become a row.

    That reasoning holds because the corpus is a table of *played* matches and all nine archived
    Seasons are complete. A call on a Fixture that has not kicked off yet has no row to join to
    either, and would be refused here for the wrong reason — which is a problem for the live spike
    at issue #16, not for the backfill.
    """
    fixtures = matches.loc[matches["division"] == DIVISION]
    located = calls.merge(
        fixtures[
            ["season", "division", "date", "home_club", "away_club", "home_goals", "away_goals"]
        ],
        on=["season", "home_club", "away_club"],
        how="left",
        validate="one_to_one",
    )
    missing = located["date"].isna()
    if missing.any():
        first = located.loc[missing].iloc[0]
        raise PunditDatasetError(
            f"{int(missing.sum())} calls name a Fixture the corpus has never seen — "
            f"{season_label(int(first['season']))} {first['home_club']} v {first['away_club']}. "
            "Two Clubs meet once per direction per Season, so this is a resolution error rather "
            "than a gap in the data"
        )
    return located


def _comparable(located: pd.DataFrame) -> pd.Series:
    """Which calls have a published result *and* a Football-Data one to compare it against.

    A postponed listing publishes ``PP`` rather than a score, and Football-Data leaves the goals
    blank on a match it has no result for. Neither is a disagreement — there is simply nothing to
    check — and counting either as agreement would report a check that was never made.
    """
    return (
        located["played"] & located["home_goals"].notna() & located["away_goals"].notna()
    )


def _disagreements(located: pd.DataFrame) -> pd.DataFrame:
    """The rows where the page's own published result contradicts Football-Data."""
    checked = located.loc[_comparable(located)]
    wrong = (checked["published_home_goals"] != checked["home_goals"]) | (
        checked["published_away_goals"] != checked["away_goals"]
    )
    return checked.loc[
        wrong,
        [
            "pundit",
            "season",
            "date",
            "home_club",
            "away_club",
            "published_home_goals",
            "published_away_goals",
            "home_goals",
            "away_goals",
        ],
    ].reset_index(drop=True)


def _refuse_a_season_that_stopped_agreeing(
    located: pd.DataFrame, disagreements: pd.DataFrame, min_checked: int
) -> None:
    """Fail if any Season's published results stop matching Football-Data in bulk."""
    checked = located.loc[_comparable(located), "season"].value_counts()
    wrong = disagreements["season"].value_counts()
    for season, total in checked.items():
        if total < min_checked:
            continue
        agreement = 1 - int(wrong.get(season, 0)) / int(total)
        if agreement < MIN_AGREEMENT:
            raise PunditDatasetError(
                f"{season_label(int(season))}: only {agreement:.1%} of {int(total)} published "
                f"results match Football-Data, against a floor of {MIN_AGREEMENT:.0%}. That is "
                "not a handful of transcription slips; the page's columns have most likely moved"
            )


def _conform(calls: pd.DataFrame) -> pd.DataFrame:
    """The fact columns, in canonical order, dtypes and row order."""
    ordered = calls.assign(division=DIVISION).sort_values(
        ["pundit", "season", "date", "home_club"], kind="stable"
    )
    return ordered[list(CALL_COLUMNS)].astype(DTYPES).reset_index(drop=True)
