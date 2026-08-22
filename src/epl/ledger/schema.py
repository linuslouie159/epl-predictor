"""The one row schema both Prediction stores share, and the checks every row must pass.

A Backtest Prediction and a Sealed Prediction are the same record written under different rules
(ADR 0005), so they are the same row. Scoring code reads a frame of these and never asks which
store it came from — which is what stops the two track records from drifting into two shapes and
the scoreboard from quietly comparing different things.

Three facts are stamped on every row, and they are what make a stored Prediction auditable long
after the run that produced it:

* ``as_of_instant`` — when it was made
* ``kickoff`` — when the Fixture started, so ``as_of_instant < kickoff`` is checkable off the file
* ``inputs_seen`` / ``latest_input`` — which input rows the Predictor actually took, so
  ``latest_input < as_of_instant`` is checkable too

No row carries an Outcome. A Prediction is sealed before its Fixture kicks off, so it cannot know
one; scoring joins to the match table instead (:mod:`epl.ledger.scoreboard`). A ledger that stored
the Outcome beside the Prediction would make a leaked one indistinguishable from a recorded one.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl import metrics, predictors
from epl.predictors import Evidence, Predictor
from epl.rounds import as_of_instant, kickoff_instants, round_id

#: Canonical column order for a Prediction, in either store.
LEDGER_COLUMNS: tuple[str, ...] = (
    "predictor",
    "prediction_round",
    "as_of_instant",
    "season",
    "division",
    "kickoff",
    "home_club",
    "away_club",
    "prob_home",
    "prob_draw",
    "prob_away",
    "inputs_seen",
    "latest_input",
)

#: What identifies a Fixture across the ledger and the match table. Two Clubs meet once per
#: direction per Season and tier, so the pairing is the identity — deliberately *not* including
#: the date, because a postponed Fixture is still the Fixture that was predicted.
FIXTURE_KEY: tuple[str, ...] = ("season", "division", "home_club", "away_club")

#: The columns a frame of Fixtures must carry to be predicted.
FIXTURE_COLUMNS: tuple[str, ...] = ("season", "division", "date", "home_club", "away_club")

#: What a Fixture is allowed to tell a Predictor about itself.
#:
#: "A Fixture ... exists before it is played and carries no result until it is" (CONTEXT.md). But
#: the corpus is a table of *played* matches, so the row a Fixture is drawn from also carries the
#: Outcome, the goals and the match statistics — and handing that row to ``predict`` would deliver
#: the answer sheet in the same call as the question. Every stored row would still audit clean,
#: because :class:`~epl.predictors.Evidence` guards the corpus and this is the other argument.
#:
#: An allow-list, not a deny-list: a column nobody has thought about is excluded rather than
#: included, and a later stage that needs a genuinely pre-kickoff feature adds it here deliberately.
#: The pre-match odds are on it because they are sampled at the As-Of Instant itself (ADR 0001).
#: The closing odds are not, and must not be: they absorb team news from after it, which is why the
#: Ceiling Line they feed is labelled everywhere as knowing more than the model can.
VISIBLE_FIXTURE_COLUMNS: tuple[str, ...] = (
    "season",
    "division",
    "date",
    "time",
    "home_club",
    "away_club",
    "prematch_odds_home",
    "prematch_odds_draw",
    "prematch_odds_away",
)

#: What a Predictor may ask for *beyond* :data:`VISIBLE_FIXTURE_COLUMNS`, by naming it in an
#: ``also_sees`` attribute of its own.
#:
#: One Predictor needs an input the allow-list deliberately withholds. The Ceiling Line *is* the
#: closing odds (ADR 0001), and closing odds absorb team news from after the As-Of Instant — so
#: appending them to the list above would hand that team news to every Predictor in the project,
#: which is precisely the leak the list exists to prevent.
#:
#: The exception is therefore made in the open and bounded twice. A Predictor claims the columns
#: it wants in its own source, where a reader of that Predictor can see the claim; and the claim
#: is checked against this tuple, so ``also_sees`` can never become a second door onto the Outcome.
#: :func:`epl.predictors.also_sees` is where a Predictor's claim is read, unvalidated.
PRIVILEGED_FIXTURE_COLUMNS: tuple[str, ...] = (
    "closing_odds_home",
    "closing_odds_draw",
    "closing_odds_away",
)

#: How every instant is written. ISO to the second: the format upstream never uses, so a hand-
#: edited file that came from a spreadsheet stands out rather than blending in.
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Written with fixed precision so a regenerated backtest file is byte-identical to the last one.
#: Nine places is far inside :data:`epl.metrics.SUM_TOLERANCE`, so a round-tripped Prediction still
#: sums to one.
FLOAT_FORMAT = "%.9f"

#: What each column is held as, in memory and coming back off disk. Named once so a stored
#: Prediction and a freshly made one compare equal without either side having to guess.
DTYPES: dict[str, str] = {
    "predictor": "string",
    "prediction_round": "string",
    "as_of_instant": "datetime64[ns]",
    "season": "int64",
    "division": "string",
    "kickoff": "datetime64[ns]",
    "home_club": "string",
    "away_club": "string",
    "prob_home": "float64",
    "prob_draw": "float64",
    "prob_away": "float64",
    "inputs_seen": "int64",
    "latest_input": "datetime64[ns]",
}


class LedgerError(Exception):
    """A Prediction could not be recorded, or a stored one did not survive its audit."""


def predictions_for(
    predictor: Predictor, fixtures: pd.DataFrame, evidence: Evidence
) -> pd.DataFrame:
    """One Prediction Round's Predictions from one Predictor, as ledger rows.

    ``fixtures`` is the round's Fixtures in the order the rows should come back in, and
    ``evidence`` is the view of the corpus cut at that round's As-Of Instant. The two are checked
    against each other before the Predictor is called: Evidence cut at the wrong instant is how a
    leak would enter without any per-row audit failing, since every row would then be consistent
    with an instant that was simply too late.

    The Predictor is handed a *Fixture* — :data:`VISIBLE_FIXTURE_COLUMNS` of the row, and no more.
    The corpus is a table of played matches, so the rest of the row is the answer sheet.
    """
    _require_columns(fixtures)
    kickoff = kickoff_instants(fixtures)
    kickoff_dates = pd.to_datetime(fixtures["date"]).dt.date
    instants = pd.to_datetime(kickoff_dates.map(as_of_instant))
    rounds = kickoff_dates.map(round_id)

    if instants.nunique() > 1:
        raise LedgerError(
            f"{instants.nunique()} As-Of Instants in one call; Fixtures must be one Prediction "
            "Round predicted as one batch (ADR 0002)"
        )
    as_of = pd.Timestamp(instants.iloc[0]) if len(instants) else None
    if as_of is not None and pd.Timestamp(evidence.as_of) != as_of:
        raise LedgerError(
            f"Evidence was cut at {evidence.as_of}, but this round's As-Of Instant is {as_of}"
        )

    probabilities = metrics.as_predictions(
        predictor.predict(visible(predictor, fixtures), evidence)
    )
    if len(probabilities) != len(fixtures):
        raise LedgerError(
            f"{predictor.name} returned {len(probabilities)} Predictions for "
            f"{len(fixtures)} Fixtures"
        )

    return conform(
        pd.DataFrame(
            {
                "predictor": predictor.name,
                "prediction_round": rounds.to_numpy(dtype=object),
                "as_of_instant": instants.to_numpy(),
                "season": fixtures["season"].to_numpy(),
                "division": fixtures["division"].to_numpy(),
                "kickoff": kickoff.to_numpy(),
                "home_club": fixtures["home_club"].to_numpy(),
                "away_club": fixtures["away_club"].to_numpy(),
                "prob_home": probabilities[:, 0],
                "prob_draw": probabilities[:, 1],
                "prob_away": probabilities[:, 2],
                "inputs_seen": evidence.rows_seen,
                "latest_input": pd.Series(
                    [evidence.latest_seen] * len(fixtures), dtype="datetime64[ns]"
                ).to_numpy(),
            },
            index=pd.RangeIndex(len(fixtures)),
        )
    )


def _require_columns(fixtures: pd.DataFrame) -> None:
    missing = [name for name in FIXTURE_COLUMNS if name not in fixtures.columns]
    if missing:
        raise LedgerError(f"a frame of Fixtures needs {missing} to be predicted")


def _named(predictor: Predictor) -> str:
    """How a Predictor is referred to in a complaint. A Predictor too broken to have a name still
    has to be nameable, or the message about it says nothing."""
    return repr(getattr(predictor, "name", predictor))


def visible(predictor: Predictor, fixtures: pd.DataFrame) -> pd.DataFrame:
    """The columns of ``fixtures`` this Predictor is allowed to see.

    :data:`VISIBLE_FIXTURE_COLUMNS`, plus whatever the Predictor claims in ``also_sees`` and
    :data:`PRIVILEGED_FIXTURE_COLUMNS` permits. One function rather than two, so a Predictor is
    handed the same Fixture whether it is being asked to predict one or asked whether it covers
    one — otherwise a Predictor could read the Outcome while answering the cheaper question and
    keep it in its own state until the expensive one.
    """
    allowed = (*VISIBLE_FIXTURE_COLUMNS, *_claimed(predictor))
    return fixtures[[name for name in allowed if name in fixtures.columns]]


def _claimed(predictor: Predictor) -> tuple[str, ...]:
    """The extra Fixture columns this Predictor claims, refusing any it may not have."""
    claimed = predictors.also_sees(predictor)
    ungranted = [name for name in claimed if name not in PRIVILEGED_FIXTURE_COLUMNS]
    if ungranted:
        raise LedgerError(
            f"{_named(predictor)} claims Fixture columns no Predictor may "
            f"see: {ungranted}. Only {list(PRIVILEGED_FIXTURE_COLUMNS)} may be claimed, and only "
            "the Ceiling Line claims them (ADR 0001)"
        )
    return claimed


def covered(predictor: Predictor, fixtures: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Which of ``fixtures`` this Predictor has anything to say about.

    A Predictor that declares no ``covers`` method covers everything, which is the ordinary case.
    The ones that do not are the Predictors whose input does not span the whole Evaluation Window:
    the Ceiling Line, whose closing odds begin in 2019/20, and a Pundit, who published in the
    Seasons they worked and no others (issue #11). Both would otherwise have to invent a
    Prediction for Fixtures they know nothing about, and a made-up Prediction that scores is worse
    than an absent one.

    Asked here rather than inside :func:`predictions_for`, because the answer decides which
    Fixtures exist for the walk at all — a Prediction Round nobody covers should not become a
    round with nothing in it.
    """
    covers = getattr(predictor, "covers", None)
    if covers is None:
        return np.ones(len(fixtures), dtype=bool)

    answered = np.asarray(covers(visible(predictor, fixtures)), dtype=bool)
    if len(answered) != len(fixtures):
        raise LedgerError(
            f"{_named(predictor)} gave {len(answered)} answers for "
            f"{len(fixtures)} Fixtures when asked which it covers"
        )
    return answered


def audit(rows: pd.DataFrame) -> list[str]:
    """Everything wrong with a frame of stored Predictions, as sentences a human can act on.

    Returns complaints rather than raising, so one pass over a store reports every problem it has
    instead of the first. :func:`check` is the raising form.

    The audit reads only the file. That is the point: it re-derives the round from the kickoff and
    compares the recorded evidence against the recorded instant, so it holds a Prediction written
    months ago by code that has since changed to the same rule as one written this morning.
    """
    complaints: list[str] = []
    missing = [name for name in LEDGER_COLUMNS if name not in rows.columns]
    unexpected = [name for name in rows.columns if name not in LEDGER_COLUMNS]
    if missing:
        complaints.append(f"missing columns: {missing}")
    if unexpected:
        complaints.append(f"unexpected columns: {unexpected}")
    if missing or rows.empty:
        return complaints

    required = [name for name in LEDGER_COLUMNS if name != "latest_input"]
    blank = [name for name in required if rows[name].isna().any()]
    if blank:
        complaints.append(f"blank values in {blank}")

    try:
        metrics.as_predictions(rows[["prob_home", "prob_draw", "prob_away"]].to_numpy(float))
    except metrics.MetricsError as invalid:
        complaints.append(str(invalid))

    as_of = pd.to_datetime(rows["as_of_instant"])
    kickoff = pd.to_datetime(rows["kickoff"])
    latest_input = pd.to_datetime(rows["latest_input"])

    # NaT compares false, so a Prediction that read nothing cannot leak.
    leaked = latest_input >= as_of
    if leaked.any():
        complaints.append(
            f"{int(leaked.sum())} Predictions used future data — {_example(rows, leaked)} saw an "
            "input at or after its own As-Of Instant"
        )

    # Two-tier, exactly as `epl.rounds` states it. Where a kickoff time was recorded the
    # comparison is exact. Where it was not, the Fixture sits at midnight on its own day, which is
    # also where its As-Of Instant sits when it is played on the Tuesday or Friday it anchors to —
    # 313 Fixtures in the corpus. Equality there withholds no data, because the Evidence cut is
    # strict and no football is played at midnight.
    timed = kickoff != kickoff.dt.normalize()
    late = (as_of > kickoff) | (timed & (as_of == kickoff))
    if late.any():
        complaints.append(
            f"{int(late.sum())} Predictions were made at or after their own kickoff — "
            f"{_example(rows, late)}"
        )

    mislabelled = rows["prediction_round"].astype("string") != kickoff.map(round_id).astype(
        "string"
    )
    if mislabelled.any():
        complaints.append(
            f"{int(mislabelled.sum())} rows name a Prediction Round their kickoff does not anchor "
            f"to — {_example(rows, mislabelled)}"
        )

    doubled = rows.duplicated(subset=["predictor", *FIXTURE_KEY, "as_of_instant"], keep=False)
    if doubled.any():
        complaints.append(
            f"{int(doubled.sum())} rows predict one Fixture twice at one As-Of Instant — "
            f"{_example(rows, doubled)}. Superseding needs a new instant (ADR 0005)"
        )

    return complaints


def _example(rows: pd.DataFrame, offending: pd.Series) -> str:
    """Name the first offending row, so a complaint points at something rather than counting."""
    row = rows.loc[offending].iloc[0]
    return (
        f"{row['predictor']} on {row['home_club']} v {row['away_club']} "
        f"({row['prediction_round']})"
    )


def check(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` if they pass :func:`audit`, otherwise a :class:`LedgerError` listing every fault.

    Called on the way into both stores. A Prediction that cannot be audited is never written, so
    the stores hold only rows that were checkable at the moment they were made.
    """
    complaints = audit(rows)
    if complaints:
        raise LedgerError("; ".join(complaints))
    return rows


def conform(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` in the canonical column order and dtypes.

    Applied on the way in and on the way out, so a Prediction read back off disk compares equal to
    the one that was written rather than merely looking like it.
    """
    return rows[list(LEDGER_COLUMNS)].astype(DTYPES)


def empty() -> pd.DataFrame:
    """An empty ledger — what a store with nothing in it reads as.

    Typed rather than bare, so an empty store flows through scoring and reporting as zero rows
    instead of failing on a missing column.
    """
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in LEDGER_COLUMNS})


def write_csv(rows: pd.DataFrame, path: Path) -> Path:
    """Write audited rows to ``path``, deterministically.

    Both stores write through here. Fixed float and date formats and an explicit line terminator
    are what let a regenerated backtest file be byte-identical to the last one and a sealed file be
    diffed in git without platform noise.
    """
    check(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    conform(rows).to_csv(
        path,
        index=False,
        float_format=FLOAT_FORMAT,
        date_format=DATE_FORMAT,
        lineterminator="\n",
    )
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """Read one ledger file. Not audited on the way in — that is what :func:`audit` is for, and a
    file that fails it needs reading before it can be reported on."""
    frame = pd.read_csv(
        path,
        dtype={name: DTYPES[name] for name in LEDGER_COLUMNS if not _is_instant(name)},
        parse_dates=[name for name in LEDGER_COLUMNS if _is_instant(name)],
    )
    return conform(frame)


def _is_instant(column: str) -> bool:
    return DTYPES[column].startswith("datetime")


def read_all(paths: Iterable[Path]) -> pd.DataFrame:
    """Every ledger file in ``paths`` as one frame, or an empty ledger if there are none.

    Both stores read through here, so a caller cannot tell from the shape of what comes back which
    one it asked — which is the whole point of there being one row schema.
    """
    frames = [read_csv(path) for path in paths if path.exists()]
    if not frames:
        return empty()
    return conform(pd.concat(frames, ignore_index=True))
