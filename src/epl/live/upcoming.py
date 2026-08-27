"""The next Prediction Round to seal, read off Football-Data's rolling ``fixtures.csv``.

A Sealed Prediction is evidence of what was forecast before kickoff (CONTEXT.md), so the live loop
needs Fixtures that have *not* been played — the one thing the corpus cannot supply, being a table
of played matches. :mod:`epl.ingest.fixtures` fetches them; this module decides which of them form
the round that can be sealed **right now**, and refuses to guess when none can.

Three rules, and each is a refusal rather than a fallback.

**A Fixture belongs to the Live Season, which is the last Season ingested.**
The rolling file carries no Season column, and inferring one from the month is wrong on this corpus:
2019/20 ran into July 2020 and its second-tier play-off final into August, so a July match is not
reliably a new Season. Nothing is therefore inferred — the Season is :data:`epl.windows.LIVE_SEASON`
— and two guards stop that constant from going quietly stale. The corpus must already hold Fixtures
of it, and it must hold fewer than :data:`CAMPAIGN_FIXTURES` of them. A Season with a full campaign
played is over, and a Season that is over has nothing upcoming.

**A round may be sealed only inside its own window** — at or after its As-Of Instant, and strictly
before its first kickoff. The late end is ADR 0005's and :func:`epl.ledger.live.seal` enforces it
too. The early end is this module's: a round sealed on Thursday under Friday's midnight instant
claims a moment that has not happened, and the odds it would have to read do not exist yet.

**A round nobody can seal is reported, not invented.** :func:`rounds` says what the rolling file
holds and which side of its window each round is on; :func:`next_round` picks the one sealable round
and, when there is none, raises with that table in the message. The measured reason this matters:
on 27 August 2026, one day before the second round of 2026/27, ``fixtures.csv`` held five Fixtures —
one National League and four Spanish — and no Premier League row at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from epl.ledger.backtest import SCORED_DIVISIONS

# The sealing window and its three verdicts belong to the store that enforces them, and are
# imported rather than restated: a second copy of the two comparisons here is how this module would
# come to offer a round the store then refused — or, far worse, the other way round.
from epl.ledger.live import KICKED_OFF, NOT_OPEN, SEALABLE, window
from epl.rounds import assign_rounds, prediction_rounds
from epl.windows import LIVE_SEASON, season_label

#: Fixtures in a complete Premier League campaign — 20 Clubs, home and away (CONTEXT.md). Used as a
#: fact about the calendar rather than as a setting: a Season the corpus holds this many matches of
#: has finished — which is how a stale :data:`epl.windows.LIVE_SEASON` is caught, not believed.
CAMPAIGN_FIXTURES = 380

#: Canonical column order for :func:`rounds` — :data:`epl.rounds.ROUND_COLUMNS` and the verdict.
ROUND_STATUS_COLUMNS: tuple[str, ...] = (
    "prediction_round",
    "as_of_instant",
    "season",
    "fixtures",
    "first_kickoff",
    "last_kickoff",
    "status",
)


class LiveError(Exception):
    """The upcoming Prediction Round could not be identified, so nothing may be sealed."""


@dataclass(frozen=True)
class PredictionRound:
    """One Prediction Round that can be sealed right now, and the Fixtures it holds.

    ``fixtures`` carries :data:`epl.ledger.schema.FIXTURE_COLUMNS` and the Market Line, so it is
    what :func:`epl.ledger.schema.predictions_for` takes. It is deliberately *not* a corpus row:
    there is no result on it, because there is no result yet.
    """

    prediction_round: str
    as_of: pd.Timestamp
    first_kickoff: pd.Timestamp
    fixtures: pd.DataFrame

    def describe(self) -> str:
        """One line naming the round, its window, and how much of it the rolling file held."""
        return (
            f"{self.prediction_round}: {len(self.fixtures)} Fixtures, "
            f"as-of {self.as_of.isoformat()}, first kickoff {self.first_kickoff.isoformat()}"
        )


def to_predict(
    rolling: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    season: int = LIVE_SEASON,
    divisions: Sequence[str] = SCORED_DIVISIONS,
) -> pd.DataFrame:
    """The rolling file's Fixtures this project predicts, stamped with the Live Season.

    ``rolling`` is :func:`epl.ingest.fixtures.parse_fixtures`' output — every English tier, Clubs
    already resolved — and ``matches`` is the corpus, which is here only to be asked whether
    ``season`` is genuinely under way. Stamping is the whole of the transformation: a Season column
    is what separates a row of the rolling file from a Fixture the ledger can hold.
    """
    _require_columns(rolling)
    _check_under_way(matches, season, divisions)

    wanted = rolling.loc[rolling["division"].isin(list(divisions))].copy()
    wanted.insert(0, "season", season)
    return wanted.sort_values(["date", "time", "home_club"], kind="stable").reset_index(drop=True)


def _require_columns(rolling: pd.DataFrame) -> None:
    missing = [name for name in ("division", "date", "home_club", "away_club")
               if name not in rolling.columns]
    if missing:
        raise LiveError(f"the rolling fixtures file needs {missing}; got {list(rolling.columns)}")


def _check_under_way(matches: pd.DataFrame, season: int, divisions: Sequence[str]) -> None:
    """Refuse a Season that is not being played, in either direction.

    Not yet started, and there is nothing for the corpus to have taught a Predictor about it —
    but far more importantly, the loop would be about to stamp next Season's Fixtures with this
    one's number. Already finished, and the same stamp is a year out of date. Both cases are what
    a :data:`epl.windows.LIVE_SEASON` nobody remembered to move looks like from here.
    """
    absent = {"season", "division"} - set(matches.columns)
    if absent:
        raise LiveError(
            f"a match table needs {sorted(absent)} before it can be asked whether "
            f"{season_label(season)} is under way; got {sorted(matches.columns)[:8]}"
        )

    played = matches.loc[
        matches["season"].eq(season) & matches["division"].isin(list(divisions))
    ]
    if played.empty:
        raise LiveError(
            f"the corpus holds no {list(divisions)} matches in {season_label(season)}, so that "
            "Season is not under way here. Ingest it before sealing a round of it — and if it "
            "genuinely has not started, epl.windows.LIVE_SEASON is pointing at the wrong Season"
        )

    complete = played.groupby("division").size() >= CAMPAIGN_FIXTURES
    if complete.all():
        raise LiveError(
            f"{season_label(season)} is complete in the corpus — {len(played)} matches across "
            f"{list(divisions)} — so it has no upcoming Fixtures. epl.windows.LIVE_SEASON is a "
            "Season behind"
        )


def rounds(fixtures: pd.DataFrame, *, now: pd.Timestamp) -> pd.DataFrame:
    """Every Prediction Round in ``fixtures``, and whether it can be sealed at ``now``.

    :data:`ROUND_STATUS_COLUMNS`, in the order the rounds were predicted. This is the table
    :func:`next_round` chooses from and the one its complaint quotes, so a run that seals nothing
    still says exactly what it saw — which, given how thin the rolling file has measured, is most
    of what the live command is for.
    """
    table = prediction_rounds(fixtures)
    if table.empty:
        return pd.DataFrame(columns=list(ROUND_STATUS_COLUMNS))

    moment = pd.Timestamp(now)
    status = [
        window(row["as_of_instant"], row["first_kickoff"], moment)
        for _, row in table.iterrows()
    ]
    table = table.assign(status=pd.Series(status, index=table.index, dtype="string"))
    return table[[name for name in ROUND_STATUS_COLUMNS if name in table.columns]]


def next_round(fixtures: pd.DataFrame, *, now: pd.Timestamp) -> PredictionRound:
    """The one Prediction Round in ``fixtures`` that is inside its sealing window at ``now``.

    The earliest such round, which on this data is also the only one: rounds are anchored to
    consecutive Tuesdays and Fridays, so a later round's window opens after an earlier round's
    first kickoff has closed it.

    Raises :class:`LiveError` rather than returning ``None``. Nothing to seal is a normal outcome
    of a weekly loop — most of the week is not inside anybody's window — and it is also what a
    rolling file with no Premier League rows in it looks like. Those two must be told apart by a
    reader, so the complaint carries :func:`rounds`.
    """
    table = rounds(fixtures, now=now)
    open_now = table.loc[table["status"] == SEALABLE] if not table.empty else table
    if open_now.empty:
        raise LiveError(_nothing_sealable(table, now))

    chosen = str(open_now.iloc[0]["prediction_round"])
    assigned = assign_rounds(fixtures)
    held = assigned.loc[assigned["prediction_round"] == chosen].reset_index(drop=True)
    return PredictionRound(
        prediction_round=chosen,
        as_of=pd.Timestamp(open_now.iloc[0]["as_of_instant"]),
        first_kickoff=pd.Timestamp(open_now.iloc[0]["first_kickoff"]),
        fixtures=held,
    )


def _nothing_sealable(table: pd.DataFrame, now: pd.Timestamp) -> str:
    """Say which of the two silences this is: an empty file, or a clock outside every window."""
    if table.empty:
        return (
            f"no Fixture to predict at {pd.Timestamp(now).isoformat()}. The rolling file held no "
            "row in a tier this project predicts — measured on 27 August 2026, one day before a "
            "Premier League round, it held one National League Fixture and four Spanish ones"
        )
    held = ", ".join(
        f"{row['prediction_round']} ({row['status']}, {row['fixtures']} Fixtures)"
        for _, row in table.iterrows()
    )
    return (
        f"no Prediction Round is inside its sealing window at {pd.Timestamp(now).isoformat()}; "
        f"the rolling file holds {held}. A round is sealable from its As-Of Instant until its "
        "first kickoff, and not before or after (ADR 0005)"
    )


__all__ = [
    "CAMPAIGN_FIXTURES",
    "KICKED_OFF",
    "NOT_OPEN",
    "ROUND_STATUS_COLUMNS",
    "SEALABLE",
    "LiveError",
    "PredictionRound",
    "next_round",
    "rounds",
    "to_predict",
]
