"""Command line entry point for the ledger.

    python -m epl.ledger backfill              walk every registered Predictor over the window
    python -m epl.ledger backfill --predictor naive_baseline
    python -m epl.ledger scoreboard            score both stores and publish the scoreboard
    python -m epl.ledger audit                 re-check every stored Prediction, and the seal

``audit`` is the one to run in anger. Both stores are checked on the way in, so it only ever fails
on a file that was changed after it was written — which is exactly the failure the checks inside
the code cannot see.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from epl.ledger import backtest, live, schema, scoreboard
from epl.paths import processed_dir
from epl.predictors import by_name, registered
from epl.windows import EVALUATION_WINDOW

#: Importing these is what puts Predictors on the scoreboard. Each later stage adds its own —
#: ``epl.models`` at issue #9, ``epl.pundits`` at issue #11.
PREDICTOR_PACKAGES: tuple[str, ...] = ("epl.benchmarks",)


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
    sub.add_parser("audit", help="re-check every stored Prediction and the seal on outputs/live/")

    args = parser.parse_args(argv)
    for package in PREDICTOR_PACKAGES:
        importlib.import_module(package)

    if args.command == "backfill":
        return _backfill(args.matches, args.predictor)
    if args.command == "scoreboard":
        return _scoreboard(args.matches)
    if args.command == "audit":
        return _audit()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _backfill(matches_path: Path | None, predictor: str | None) -> int:
    matches = _load_matches(matches_path)
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
    matches = _load_matches(matches_path)
    rows = pd.concat([backtest.read(), live.read()], ignore_index=True)
    board = scoreboard.build(rows, matches)
    destination = scoreboard.write(board)

    # The metrics as a table and the notes underneath it. A caveat long enough to be worth
    # printing is too long to sit in a column of floats, and one that wrapped a table into
    # illegibility would end up ignored — which for the Ceiling Line's is the whole risk.
    print(
        board[list(scoreboard.METRIC_COLUMNS)].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    for _, line in board.loc[board["note"].astype(str) != ""].iterrows():
        print(f"  {line['predictor']}: {line['note']}")
    print(f"-> {destination}")

    # A registered Predictor with nothing stored has no metrics — epl.metrics refuses to average
    # an empty slate, and a NaN on a scoreboard reads as a real number. So it is named here rather
    # than printed blank, because silently vanishing is the one thing it must not do.
    scored = set(board["predictor"])
    unscored = [one.name for one in registered() if one.name not in scored]
    if unscored:
        print(f"registered but not scored: {', '.join(unscored)}")
        print("  no stored Predictions — run `python -m epl.ledger backfill`")
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


def _load_matches(path: Path | None) -> pd.DataFrame:
    """The cleaned match table, or an instruction for how to make one."""
    source = path or processed_dir() / "matches.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} does not exist. Build it first:\n"
            "    python -m epl.ingest fetch\n"
            "    python -m epl.ingest build"
        )
    return pd.read_csv(source, dtype={"time": "string"})


if __name__ == "__main__":
    sys.exit(main())
