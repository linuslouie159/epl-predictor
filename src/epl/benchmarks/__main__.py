"""Command line entry point for the benchmarks.

    python -m epl.benchmarks overround           the margin in each book, per Season and tier
    python -m epl.benchmarks overround --divisions E0
    python -m epl.benchmarks methods             the three vig removals compared on one book

``overround`` is the receipt issue #8 asks for: "reported alongside every Market Line ... so the
vig removal can be sanity-checked rather than trusted". It writes `outputs/overround.csv` and
prints the Premier League summary, where the margin should be seen falling from about 9.4% in
2005/06 to about 4.1% in the early 2020s. A vig removal that had quietly stopped removing anything
would leave that decline unexplained, and a scoreboard that barely moved.

``methods`` prints one book under all three removals, which is the fastest way to see for yourself
that the choice barely matters — and exactly where it does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from epl.benchmarks import market, vig
from epl.ingest import DIVISIONS
from epl.paths import outputs_dir, processed_dir

#: A typical Premier League book, for ``methods``: a favourite, a draw and a longshot.
EXAMPLE_BOOK: tuple[float, float, float] = (1.80, 3.60, 4.50)


def path() -> Path:
    """Where the overround report is written. Derived and regenerable, like the scoreboard."""
    return outputs_dir() / "overround.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.benchmarks", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("overround", help="the margin in each book, per Season and tier")
    report.add_argument(
        "--divisions",
        nargs="+",
        default=["E0"],
        choices=list(DIVISIONS),
        help="which tiers to print; all of them are written to the file regardless",
    )
    sub.add_parser("methods", help="one book under all three vig removals")

    args = parser.parse_args(argv)
    if args.command == "overround":
        return _overround(args.matches, tuple(args.divisions))
    if args.command == "methods":
        return _methods()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _overround(matches_path: Path | None, divisions: tuple[str, ...]) -> int:
    report = market.overround_report(_load_matches(matches_path))
    destination = path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, float_format="%.5f", lineterminator="\n")

    shown = report.loc[report["division"].isin(list(divisions))]
    print(
        shown.drop(columns=["season", "division"]).to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    for name, group in shown.groupby("predictor", sort=True):
        fixtures = group["fixtures"].sum()
        weighted = float((group["mean_overround"] * group["fixtures"]).sum() / fixtures)
        print(f"{name}: {int(fixtures)} Fixtures, mean overround {weighted:.5f}")
    print(f"-> {destination}")
    return 0


def _methods() -> int:
    print(f"book {EXAMPLE_BOOK}, overround {float(vig.overround([EXAMPLE_BOOK])[0]):.5f}")
    for name in sorted(vig.METHODS):
        home, draw, away = vig.remove([EXAMPLE_BOOK], method=name)[0]
        default = " (default)" if name == vig.DEFAULT_METHOD else ""
        print(f"  {name:>9}: {home:.6f} {draw:.6f} {away:.6f}{default}")
    print("  power and Shin correct favourite-longshot bias; normalisation does not (ADR 0001)")
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
