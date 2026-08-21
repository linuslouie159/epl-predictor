"""Prediction Rounds and the As-Of Instant rule — derived once, imported everywhere.

A Prediction Round is every Fixture sharing an As-Of Instant, predicted together as one batch. The
source data has no matchweek column and no round column: 2024/25's 380 Fixtures spanned 109
distinct kickoff dates. Rounds are therefore derived from kickoff dates and from nothing else
(ADR 0002).

The rule anchors each kickoff back to the most recent Tuesday or Friday, mirroring Football-Data's
stated collection convention — pre-match odds sampled Friday afternoon for weekend Fixtures,
Tuesday afternoon for midweek ones::

    Fri / Sat / Sun -> that Friday
    Mon             -> the previous Friday
    Tue             -> that Tuesday
    Wed / Thu       -> that Tuesday

Predicting each Fixture from everything played strictly before its own kickoff would be leak-free
and more accurate, so a future reader will wonder why this deliberately uses less information. The
reason is comparability: it would let the model know Saturday's results when calling Monday night's
Fixture, while the Market Line (sampled Friday) and the Pundits (published Thursday/Friday) would
not. Every three-way comparison would then silently overstate the model (ADR 0002).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

#: ``date.weekday()`` values. Monday is named because it is the one weekday that anchors backwards
#: past an anchor day, and Tuesday and Friday because they are the anchor days themselves.
MONDAY = 0
TUESDAY = 1
FRIDAY = 4

#: Canonical column order for a table of Prediction Rounds. ``season`` appears only when the
#: Fixtures it was built from named one.
ROUND_COLUMNS: tuple[str, ...] = (
    "prediction_round",
    "as_of_instant",
    "season",
    "fixtures",
    "first_kickoff",
    "last_kickoff",
)


class RoundsError(Exception):
    """A frame of Fixtures could not be resolved into Prediction Rounds."""


def anchor(kickoff: date) -> date:
    """The Tuesday or Friday a Fixture kicking off on ``kickoff`` is predicted from.

    >>> anchor(date(2024, 8, 17))       # a Saturday
    datetime.date(2024, 8, 16)
    >>> anchor(date(2024, 8, 19))       # a Monday, back to the weekend's Friday
    datetime.date(2024, 8, 16)
    >>> anchor(date(2024, 8, 21))       # a Wednesday, back to that Tuesday
    datetime.date(2024, 8, 20)
    """
    # Kept branch-for-branch with the rule the spec states as executable code, including the
    # Tuesday case that the final line would already handle. Issue #5 asks for the rule
    # "implemented exactly as stated", and a reader auditing this against the spec should be able
    # to put the two side by side.
    weekday = kickoff.weekday()
    if weekday >= FRIDAY:
        return kickoff - timedelta(days=weekday - FRIDAY)
    if weekday == MONDAY:
        return kickoff - timedelta(days=3)  # back past the weekend, to its Friday
    if weekday == TUESDAY:
        return kickoff
    return kickoff - timedelta(days=weekday - TUESDAY)


def as_of_instant(kickoff: date | datetime) -> datetime:
    """The moment a Fixture kicking off on ``kickoff`` may be predicted from.

    Midnight at the start of the anchor day. Football-Data samples its pre-match odds on the
    *afternoon* of that day, and midnight is the conservative end of the same window: no football
    is played between the two, so the information sets are identical for every input this project
    holds, and the earlier instant can only ever withhold data, never admit it.

    Midnight is also the only choice that is checkable. Kickoff times are absent before 2019/20,
    and the earliest kickoff recorded on any Tuesday or Friday is 12:30, so an afternoon instant
    would leave the strictly-before guarantee unverifiable across two thirds of the corpus.

    >>> as_of_instant(date(2024, 8, 17))
    datetime.datetime(2024, 8, 16, 0, 0)
    """
    return datetime.combine(anchor(_as_date(kickoff)), datetime.min.time())


def round_id(kickoff: date | datetime) -> str:
    """The id of the Prediction Round a Fixture kicking off on ``kickoff`` belongs to.

    The anchor date in ISO form: sortable, self-describing, and usable directly as the ledger's
    per-round filename (ADR 0005). Anchor dates never repeat across Seasons, so the id alone
    identifies a round.

    >>> round_id(date(2024, 8, 17))
    '2024-08-16'
    """
    return anchor(_as_date(kickoff)).isoformat()


def _as_date(kickoff: date | datetime) -> date:
    """Callers hold kickoffs as timestamps; the anchor depends on the date alone."""
    return kickoff.date() if isinstance(kickoff, datetime) else kickoff


def kickoff_instants(matches: pd.DataFrame) -> pd.Series:
    """When each Fixture in ``matches`` actually kicked off.

    Football-Data carries no kickoff time before 2019/20, so a Fixture without one is placed at
    the start of its day. That is earlier than it really kicked off, which is the safe direction:
    it can only ever make the strictly-after check harder to pass, never easier.
    """
    _require_date(matches)
    dates = pd.to_datetime(matches["date"])
    if "time" not in matches.columns:
        return dates.dt.normalize()

    offsets = pd.to_timedelta(matches["time"].astype("string") + ":00", errors="coerce")
    return dates.dt.normalize() + offsets.fillna(pd.Timedelta(0))


def assign_rounds(matches: pd.DataFrame) -> pd.DataFrame:
    """``matches`` with an ``as_of_instant`` and a ``prediction_round`` on every Fixture.

    Row order, the index and every original column survive untouched; the input frame itself is
    never modified.

    Raises :class:`RoundsError` if any Fixture would kick off at or before the instant it is
    predicted from. That check lives here, at the point rounds are built, rather than in a
    downstream audit — a frame that violates the project's one rule should not be constructible in
    the first place.
    """
    _require_date(matches)
    assigned = matches.copy()
    kickoff_dates = pd.to_datetime(assigned["date"]).dt.date
    assigned["as_of_instant"] = pd.to_datetime(kickoff_dates.map(as_of_instant))
    assigned["prediction_round"] = kickoff_dates.map(round_id).astype("string")
    _check_precedes_kickoff(assigned)
    return assigned


def _check_precedes_kickoff(assigned: pd.DataFrame) -> None:
    """Every Fixture must kick off strictly after the instant it was predicted from.

    Two checks, because the corpus records a kickoff time only from 2019/20. The As-Of Instant
    must never land after the kickoff day at all — that would mean the anchor rule ran backwards.
    And where a time was recorded the comparison is exact. Where it was not, the strongest
    verifiable claim is that the instant is at or before the start of the kickoff day, which,
    since no Fixture kicks off at midnight, still gives strictness.
    """
    kickoff_day = pd.to_datetime(assigned["date"]).dt.normalize()
    anchored_late = assigned["as_of_instant"] > kickoff_day
    if anchored_late.any():
        raise RoundsError(
            f"{int(anchored_late.sum())} Fixtures are anchored after their own kickoff day"
        )

    recorded = _has_recorded_time(assigned) & (
        kickoff_instants(assigned) <= assigned["as_of_instant"]
    )
    if recorded.any():
        raise RoundsError(
            f"{int(recorded.sum())} Fixtures kick off at or before their own As-Of Instant"
        )


def _has_recorded_time(assigned: pd.DataFrame) -> pd.Series:
    if "time" not in assigned.columns:
        return pd.Series(False, index=assigned.index)
    return assigned["time"].notna()


def _require_date(matches: pd.DataFrame) -> None:
    if "date" not in matches.columns:
        raise RoundsError("a frame of Fixtures needs a 'date' column to derive rounds from")


def prediction_rounds(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per Prediction Round, in the order the rounds were predicted.

    ``first_kickoff`` is the deadline a Sealed Prediction for the round must beat: ADR 0005 makes
    a round's first kickoff the moment its file stops being editable.
    """
    assigned = assign_rounds(matches)
    columns = [name for name in ROUND_COLUMNS if name != "season" or "season" in assigned.columns]
    if assigned.empty:
        return pd.DataFrame(columns=columns)

    assigned = assigned.assign(kickoff=kickoff_instants(assigned))
    aggregations: dict[str, tuple[str, str]] = {
        "as_of_instant": ("as_of_instant", "first"),
        "fixtures": ("prediction_round", "size"),
        "first_kickoff": ("kickoff", "min"),
        "last_kickoff": ("kickoff", "max"),
    }
    if "season" in assigned.columns:
        aggregations["season"] = ("season", "first")

    summary = assigned.groupby("prediction_round", sort=True).agg(**aggregations).reset_index()
    return summary[columns].sort_values("as_of_instant", kind="stable").reset_index(drop=True)
