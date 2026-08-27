"""Scoring the ledger: one line per Predictor over the Evaluation Window.

This is the module issue #7 is really about. It takes ledger rows and the match table and produces
a Scorecard per Predictor, and it is written once against the Predictor contract — there is no
branch in here for a model, for the Market Line or for a Pundit, and there is nowhere for one to be
added without it being obvious (spec, user story 16).

It also knows nothing about which store a row came from. A Backtest Prediction and a Sealed
Prediction share one schema, so both arrive here as the same frame, and a Predictor's live track
record and its backtest are scored by identical code.

Two rules the join encodes:

* **The ledger holds no Outcome**, so what happened is fetched from the match table by Fixture
  identity — the Club pairing within a Season and tier, not the date, because a postponed Fixture
  is still the Fixture that was predicted.
* **Only the latest Prediction for a Fixture counts.** Correcting a sealed Prediction means adding
  a superseding row at a new As-Of Instant (ADR 0005); scoring both would count the Fixture twice
  and average the mistake back in.

**Every metric is reported twice**, pre-calibration and post-calibration (ADR 0006, issue #10). The
shared isotonic layer lives in :mod:`epl.calibration` and is applied here rather than inside any
Predictor, so a Predictor gets calibrated by being registered and there is no calibration code
anywhere for a per-Predictor branch to be added to. The size of the correction rides onto the board
beside the two sets of numbers: a layer that moves a great deal of probability mass and buys nothing
is a warning, and publishing only the better of the two columns is how that warning gets lost.

The scoreboard sits at `outputs/scoreboard.csv` rather than inside either store: it summarises
both, so it belongs to neither, and reading a store must never pick up a report as if it were a
file of Predictions. It is gitignored, because it is derived and regenerable — the same reasoning
ADR 0005 applies to the Backtest Predictions themselves. Only `outputs/live/` is evidence. The
reliability diagrams beside it at `outputs/reliability.csv` are published on the same terms.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from epl import calibration, metrics, predictors
from epl.ledger import schema
from epl.paths import outputs_dir
from epl.windows import EVALUATION_WINDOW

#: Every metric a reader compares Predictors on, in order. RPS is primary; accuracy is for lay
#: explanation only and is never the headline (CLAUDE.md).
#:
#: This is the one list. :data:`SCOREBOARD_COLUMNS` and both printed views below are derived from
#: it, so a metric added here reaches the file *and* both tables — the alternative, three
#: hand-written lists, is how a metric ends up in the CSV and in neither table anybody reads.
#:
#: The first four are :class:`epl.metrics.Scorecard`'s fields, spread by :func:`_scores`. ``ece``
#: is not a Scorecard field: it is the reliability diagram's one-number summary, and it is on the
#: board because it is the number the calibrated half exists to move.
METRICS: tuple[str, ...] = ("rps", "brier", "log_loss", "accuracy", "ece")

#: Every metric appears twice, once bare and once under this prefix (ADR 0006). A calibration
#: layer can mask a broken model by correcting its symptoms, so the pre-calibration numbers are
#: published beside the post-calibration ones and neither may be reported alone.
CALIBRATED_PREFIX = "calibrated_"


def _calibrated(*names: str) -> tuple[str, ...]:
    """The calibrated spelling of some metric names."""
    return tuple(f"{CALIBRATED_PREFIX}{name}" for name in names)


#: Canonical column order for the scoreboard.
#:
#: ``corrected`` and ``correction`` describe what the layer did: how many Predictions a fitted map
#: reached, and how much probability mass it moved. ``note`` is last and is usually empty. It
#: carries a caveat a Predictor cannot be honestly read without — the Ceiling Line's, which knows
#: team news the model cannot have and is scored over a shorter span than everything else here
#: (ADR 0001). It is read off the registered Predictor by name, so the scoreboard still has no idea
#: which Predictor any row belongs to.
SCOREBOARD_COLUMNS: tuple[str, ...] = (
    "predictor",
    "fixtures",
    *METRICS,
    *_calibrated(*METRICS),
    "corrected",
    "correction",
    "note",
)

#: The two views a reader is shown, because fifteen columns of floats on one line is a table nobody
#: reads. Same metrics, same order, twice — which is the point ADR 0006 is making.
PRE_CALIBRATION_COLUMNS: tuple[str, ...] = ("predictor", "fixtures", *METRICS)
POST_CALIBRATION_COLUMNS: tuple[str, ...] = (
    "predictor",
    "corrected",
    *_calibrated(*METRICS),
    "correction",
)

#: The probability columns, in the ordinal (Home, Draw, Away) order the metrics expect.
PROBABILITY_COLUMNS: tuple[str, ...] = ("prob_home", "prob_draw", "prob_away")

#: The same, after the shared calibration layer. Named apart from the stored columns rather than
#: overwriting them: both halves of every comparison have to survive to be reported (ADR 0006).
CALIBRATED_PROBABILITY_COLUMNS: tuple[str, ...] = _calibrated(*PROBABILITY_COLUMNS)

#: What each form of a Prediction is called wherever both are published side by side. ``raw`` is
#: pre-calibration and ``calibrated`` is post-calibration.
FORMS: tuple[str, ...] = ("raw", "calibrated")

#: Canonical column order for the published reliability diagrams — :data:`epl.metrics.BINS` bins
#: per Predictor per form, which is issue #10's fourth acceptance criterion.
RELIABILITY_REPORT_COLUMNS: tuple[str, ...] = (
    "predictor",
    "form",
    *metrics.RELIABILITY_COLUMNS,
)


def path() -> Path:
    """Where the scoreboard is written."""
    return outputs_dir() / "scoreboard.csv"


def scored_predictions(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """The ledger rows that can be scored, each with the Outcome that happened.

    Public because it is the table behind every finer-grained question the project asks later — a
    Predictor's best and worst calls, a Pundit's calibration misses — all of which are a sort or a
    group over this.
    """
    in_window = rows.loc[rows["season"].isin(list(seasons))]
    latest = (
        in_window.sort_values("as_of_instant", kind="stable")
        .drop_duplicates(subset=["predictor", *schema.FIXTURE_KEY], keep="last")
        .reset_index(drop=True)
    )
    outcomes = matches[[*schema.FIXTURE_KEY, "outcome"]].astype(
        {name: schema.DTYPES[name] for name in schema.FIXTURE_KEY}
    )
    # many-to-one: several Predictors call the same Fixture, but the match table must hold it once
    joined = latest.merge(outcomes, on=list(schema.FIXTURE_KEY), how="left", validate="m:1")
    return joined.loc[joined["outcome"].notna()].reset_index(drop=True)


def calibrated_predictions(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """:func:`scored_predictions`, with each Prediction's calibrated form beside its raw one.

    The shared isotonic layer is applied here, one Predictor at a time and walk-forward within each
    (:mod:`epl.calibration`). Per Predictor because a map is a statement about one Predictor's own
    quotes — "when *this* Predictor says 20% Draw, how often is it a Draw?" — and pooling several
    Predictors' quotes into one map would correct each of them with the others' mistakes.

    Five columns are added: the three of :data:`CALIBRATED_PROBABILITY_COLUMNS`, plus ``corrected``,
    which says whether a fitted map reached that row at all, and ``correction``, the probability
    mass it moved. The raw columns are untouched, because every metric is reported both ways
    (ADR 0006).

    Public for the same reason :func:`scored_predictions` is: the draw curve, the reliability
    diagrams and the scoreboard are all a group over this one table, and none of them should be
    walking the calibration a second time to get it.
    """
    scored = scored_predictions(rows, matches, seasons=seasons)
    if scored.empty:
        return scored.assign(
            **{name: pd.Series(dtype="float64") for name in CALIBRATED_PROBABILITY_COLUMNS},
            corrected=pd.Series(dtype="bool"),
            correction=pd.Series(dtype="float64"),
        )

    # Written back against each Predictor's own index. The walk returns its rows in the order it
    # was handed them, so the group's index is what puts each calibrated Prediction back beside the
    # raw one it came from — rather than a position in a frame that has been grouped since.
    calibrated = pd.DataFrame(
        index=scored.index, columns=list(CALIBRATED_PROBABILITY_COLUMNS), dtype=float
    )
    reached = pd.Series(False, index=scored.index)
    moved = pd.Series(0.0, index=scored.index)
    for _, group in scored.groupby("predictor", sort=True):
        walked = calibration.walk_forward(
            group[list(PROBABILITY_COLUMNS)].to_numpy(float),
            group["outcome"].tolist(),
            group["as_of_instant"],
            group["kickoff"],
        )
        calibrated.loc[group.index, list(CALIBRATED_PROBABILITY_COLUMNS)] = walked.predictions
        reached.loc[group.index] = walked.fitted
        moved.loc[group.index] = walked.moved
    return scored.join(calibrated).assign(corrected=reached, correction=moved)


def build(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """Every Predictor in ``rows``, scored over the window twice, best raw RPS first.

    Sorted on the pre-calibration RPS: that is the Predictor's own score, and ordering the board by
    what the calibration layer did to it would let the layer decide who is winning.
    """
    return lines(calibrated_predictions(rows, matches, seasons=seasons))


def lines(scored: pd.DataFrame) -> pd.DataFrame:
    """The board over an already-calibrated frame — :func:`calibrated_predictions`' output, scored.

    Split out from :func:`build` so that a caller can cut the *slate* without also cutting what
    each Predictor's calibration map was fitted on. Issue #12's three-way comparison needs exactly
    that: it scores every Predictor over the Fixtures they all cover, but the Market Line's
    calibrated form there has to be the one the Market Line actually has — fitted on its own 7,980
    Predictions — rather than a weaker map refitted on a Pundit's 1,900. Cutting first and
    calibrating after would publish a post-calibration column that exists nowhere else and belongs
    to nobody.

    Same reasoning as ADR 0001's: narrow the comparison, never the Predictor.
    """
    rows = [_line(str(name), group) for name, group in scored.groupby("predictor", sort=True)]
    board = pd.DataFrame(rows, columns=list(SCOREBOARD_COLUMNS))
    return board.sort_values("rps", kind="stable").reset_index(drop=True)


def _line(predictor: str, group: pd.DataFrame) -> dict[str, object]:
    """One Predictor's two Scorecards as a scoreboard row.

    Both sides go through :func:`_scores`, so the pre- and post-calibration halves cannot be
    computed differently — which is the whole worth of reporting them side by side. ``fixtures`` is
    dropped from the calibrated half because it is one slate scored twice; two counts would invite a
    reader to wonder whether the halves cover the same Fixtures, which is the doubt one count
    removes.

    The note is looked up by name rather than passed in, because scoring works from stored rows
    and a stored row carries only a name. A Predictor whose ledger file outlived its code scores
    exactly as before, with a blank where its caveat would be.
    """
    outcomes = group["outcome"].tolist()
    before = _scores(group[list(PROBABILITY_COLUMNS)].to_numpy(float), outcomes)
    after = _scores(group[list(CALIBRATED_PROBABILITY_COLUMNS)].to_numpy(float), outcomes)
    return (
        {"predictor": predictor}
        | before
        | {
            f"{CALIBRATED_PREFIX}{name}": value
            for name, value in after.items()
            if name != "fixtures"
        }
        | {
            "corrected": int(group["corrected"].sum()),
            "correction": float(group["correction"].mean()),
            "note": predictors.note(predictor),
        }
    )


def _scores(predictions: object, outcomes: list[object]) -> dict[str, object]:
    """Every metric the board carries, for one set of Predictions.

    The Scorecard fields are spread rather than re-listed, so a metric added to
    :class:`epl.metrics.Scorecard` reaches the file and both printed tables by being named in
    :data:`METRICS` and nowhere else. The calibration error is added on top because it is the one
    board metric that is not a Scorecard field — see :data:`METRICS`.
    """
    return asdict(metrics.score(predictions, outcomes)) | {
        "ece": metrics.expected_calibration_error(predictions, outcomes)
    }


def reliability(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """A :data:`epl.metrics.BINS`-bin reliability diagram per Predictor, in both forms.

    Issue #10's fourth acceptance criterion, and the diagram behind the ``ece`` and
    ``calibrated_ece`` columns on the scoreboard — the one number cannot say *where* a Predictor is
    off, and a correction that fixed one band while breaking another would show up here first.

    Predictors in name order, ``raw`` before ``calibrated`` within each, bins in ascending order.
    """
    scored = calibrated_predictions(rows, matches, seasons=seasons)
    diagrams = [
        metrics.reliability(
            group[list(columns)].to_numpy(float), group["outcome"].tolist()
        ).assign(predictor=str(name), form=form)
        for name, group in scored.groupby("predictor", sort=True)
        for form, columns in zip(
            FORMS, (PROBABILITY_COLUMNS, CALIBRATED_PROBABILITY_COLUMNS), strict=True
        )
    ]
    if not diagrams:
        return pd.DataFrame(columns=list(RELIABILITY_REPORT_COLUMNS))
    return pd.concat(diagrams, ignore_index=True)[list(RELIABILITY_REPORT_COLUMNS)]


def write(board: pd.DataFrame, destination: Path | None = None) -> Path:
    """Write the scoreboard, and return where it went.

    ``destination`` is for a board over a different span rather than a different metric — the live
    Season's, which is scored apart from the Evaluation Window so that the headline numbers mean
    the same thing this week as last (:mod:`epl.windows`). Same columns, same code, another file.
    """
    return _publish(board, destination or path())


def reliability_path() -> Path:
    """Where the published reliability diagrams live."""
    return outputs_dir() / "reliability.csv"


def write_reliability(diagrams: pd.DataFrame) -> Path:
    """Write the reliability diagrams, and return where they went."""
    return _publish(diagrams, reliability_path())


def _publish(table: pd.DataFrame, destination: Path) -> Path:
    """One writer for both reports, so the two cannot drift into different float formats."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, float_format=schema.FLOAT_FORMAT, lineterminator="\n")
    return destination
