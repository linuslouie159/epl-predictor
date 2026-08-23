"""Command line entry point for the ledger.

    python -m epl.ledger backfill              walk every registered Predictor over the window
    python -m epl.ledger backfill --predictor naive_baseline
    python -m epl.ledger scoreboard            score both stores and publish the scoreboard
    python -m epl.ledger reliability           publish the 10-bin diagrams, raw and calibrated
    python -m epl.ledger audit                 re-check every stored Prediction, and the seal

``audit`` is the one to run in anger. Both stores are checked on the way in, so it only ever fails
on a file that was changed after it was written — which is exactly the failure the checks inside
the code cannot see.

``scoreboard`` prints its metrics **twice**, pre-calibration and post-calibration (ADR 0006). The
two tables are printed one after the other rather than as one wide line, and the size of the
correction sits on the second: a layer that moves a lot of probability mass and buys nothing is a
warning about the layer, and a reader shown only the better of the two columns would never see it.
``reliability`` is the same comparison at bin resolution, which is where a correction that fixed
one probability band while breaking another shows up first.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from epl.ingest import match_table
from epl.ledger import backtest, live, schema, scoreboard
from epl.predictors import by_name, registered
from epl.windows import EVALUATION_WINDOW

#: Importing these is what puts Predictors on the scoreboard. Each later stage adds its own —
#: ``epl.models`` at issue #9, ``epl.pundits`` at issue #11.
PREDICTOR_PACKAGES: tuple[str, ...] = ("epl.benchmarks", "epl.models", "epl.pundits")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.ledger", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="regenerate Backtest Predictions")
    backfill.add_argument(
        "--predictor",
        default=None,
        help="only this Predictor (default: every registered one)",
    )
    sub.add_parser("scoreboard", help="score both stores and write the scoreboard")
    sub.add_parser("reliability", help="publish the reliability diagrams, raw and calibrated")
    sub.add_parser("audit", help="re-check every stored Prediction and the seal on outputs/live/")

    args = parser.parse_args(argv)
    for package in PREDICTOR_PACKAGES:
        importlib.import_module(package)

    if args.command == "backfill":
        return _backfill(args.matches, args.predictor)
    if args.command == "scoreboard":
        return _scoreboard(args.matches)
    if args.command == "reliability":
        return _reliability(args.matches)
    if args.command == "audit":
        return _audit()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _stored() -> pd.DataFrame:
    """Every Prediction in both stores. One row schema, so scoring never asks which it is
    reading (ADR 0005)."""
    return pd.concat([backtest.read(), live.read()], ignore_index=True)


def _table(board: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """One view of the board, with the ``calibrated_`` prefix dropped from the headings.

    The two tables then carry identical column names, which is what lets a reader read one against
    the other by eye. The prefix is what keeps the two apart in the file; a table already headed
    "post-calibration" does not need to repeat it in every column, and repeating it makes the line
    too wide to print.
    """
    return (
        board[list(columns)]
        .rename(columns={name: name.removeprefix(scoreboard.CALIBRATED_PREFIX) for name in columns})
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def _backfill(matches_path: Path | None, predictor: str | None) -> int:
    matches = match_table(matches_path)
    predictors = [by_name(predictor)] if predictor else list(registered())
    window = f"{min(EVALUATION_WINDOW)}-{max(EVALUATION_WINDOW)}"

    for one in predictors:
        rows = backtest.backfill(one, matches)
        written = backtest.write(rows)
        # A Predictor that covers none of the window writes no file, which is the honest outcome
        # for the Ceiling Line before 2019/20 and for a Pundit outside the Seasons they published
        # in. Said out loud rather than passed over, because it is also what a broken input column
        # would look like.
        if not written:
            print(f"{one.name}: nothing to predict in {window} — it covers none of it")
            continue
        rounds = rows["prediction_round"].nunique()
        print(f"{one.name}: {len(rows)} Predictions over {rounds} rounds, {window} -> {written[0]}")
    return 0


def _scoreboard(matches_path: Path | None) -> int:
    matches = match_table(matches_path)
    board = scoreboard.build(_stored(), matches)
    destination = scoreboard.write(board)

    # The metrics as a table and the notes underneath it. A caveat long enough to be worth
    # printing is too long to sit in a column of floats, and one that wrapped a table into
    # illegibility would end up ignored — which for the Ceiling Line's is the whole risk.
    #
    # Printed twice, because every metric is reported pre- and post-calibration (ADR 0006). Two
    # tables rather than one wide one: fourteen columns of floats on a line is a table nobody
    # reads, and an unread comparison is the same as an unpublished one.
    print("pre-calibration")
    print(_table(board, scoreboard.PRE_CALIBRATION_COLUMNS))
    print("\npost-calibration (`corrected` Predictions reached; `correction` is the mass moved)")
    print(_table(board, scoreboard.POST_CALIBRATION_COLUMNS))
    for _, line in board.loc[board["note"].astype(str) != ""].iterrows():
        print(f"  {line['predictor']}: {line['note']}")
    print(f"-> {destination}")
    # `ece` says how far off a Predictor is; it cannot say *where*, and a correction that fixed one
    # probability band by breaking another reads as a small number here. Named so the second half
    # of ADR 0006's reporting is something a reader is pointed at rather than has to know about.
    print("   the bands behind `ece`: `python -m epl.ledger reliability`")

    # A registered Predictor with nothing stored has no metrics — epl.metrics refuses to average
    # an empty slate, and a NaN on a scoreboard reads as a real number. So it is named here rather
    # than printed blank, because silently vanishing is the one thing it must not do.
    scored = set(board["predictor"])
    unscored = [one.name for one in registered() if one.name not in scored]
    if unscored:
        print(f"registered but not scored: {', '.join(unscored)}")
        print("  no stored Predictions — run `python -m epl.ledger backfill`")
    return 0


def _reliability(matches_path: Path | None) -> int:
    """The 10-bin diagrams, published and printed — issue #10's fourth acceptance criterion.

    Its own command rather than a second file written by ``scoreboard``, because the two answer
    different questions: the board says how far off a Predictor is on average, and this says
    *where*. A reader who wants the second is looking for a band, and a band is not something a
    scoreboard line can carry.
    """
    diagrams = scoreboard.reliability(_stored(), match_table(matches_path))
    if diagrams.empty:
        print("no stored Predictions — run `python -m epl.ledger backfill`")
        return 0

    destination = scoreboard.write_reliability(diagrams)
    for (predictor, form), diagram in diagrams.groupby(["predictor", "form"], sort=False):
        print(f"\n{predictor} ({form})")
        print(
            diagram.drop(columns=["predictor", "form"]).to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
    print(f"\n-> {destination}")
    return 0


def _audit() -> int:
    complaints = [
        f"outputs/backtest/: {complaint}" for complaint in schema.audit(backtest.read())
    ]
    complaints += [f"outputs/live/: {complaint}" for complaint in schema.audit(live.read())]
    complaints += [f"outputs/live/{complaint}" for complaint in live.seal_violations()]

    for complaint in complaints:
        print(complaint)
    if complaints:
        print(f"{len(complaints)} problems")
        return 1

    print("both stores audit clean; nothing under outputs/live/ has been rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
