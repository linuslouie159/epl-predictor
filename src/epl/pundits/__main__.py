"""Command line entry point for the Pundit backfill.

    python -m epl.pundits fetch     cache the nine MyFootballFacts season pages
    python -m epl.pundits fetch --refresh
    python -m epl.pundits build     parse, reconcile with the corpus, freeze predictions.csv
    python -m epl.pundits grades    the two readings of every call, per Pundit and Season

``build`` is the one that has to be believed, so it prints its own evidence rather than a count.
Every call carries the result MyFootballFacts published beside it, and every one of those is
checked against Football-Data — 3,406 of them, where issue #11 asks for one. That is what confirms
the part of the parse nothing else can see: that these two spellings became the right two Clubs,
the right way round. Four rows disagree and are printed in full; a Season that stopped agreeing
stops the build (:data:`epl.pundits.dataset.MIN_AGREEMENT`).

It also prints the Fixtures nobody called. Twelve of the nine Seasons' 3,420 are missing from the
archive, and a Pundit's ``covers`` keeps them off the ledger — but an unremarked gap is how twelve
quietly becomes a hundred.

``grades`` is the lay reading beside the RPS on the scoreboard: how often a published Scoreline was
exactly right, and how often it merely picked the right Outcome (issue #11). Neither is the
headline; RPS is (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from epl.ingest import match_table
from epl.pundits import dataset, grading, myfootballfacts
from epl.windows import season_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.pundits", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="cache the nine MyFootballFacts season pages")
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even where cached; the old bytes are archived, never overwritten",
    )
    sub.add_parser("build", help="parse, reconcile with the corpus, and freeze predictions.csv")
    sub.add_parser("grades", help="exact-score and correct-Outcome rates per Pundit and Season")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _fetch(args.refresh)
    if args.command == "build":
        return _build(args.matches)
    if args.command == "grades":
        return _grades(args.matches)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _fetch(refresh: bool) -> int:
    for page in myfootballfacts.PAGES:
        path = myfootballfacts.fetch_page(page, refresh=refresh)
        print(f"{season_label(page.season)} {page.pundit:>10} -> {path}")
    return 0


def _build(matches_path: Path | None) -> int:
    matches = match_table(matches_path)
    built = dataset.build_from_cache(matches, myfootballfacts.PAGES)

    print(
        built.calls.groupby(["pundit", "season"])
        .size()
        .rename("calls")
        .reset_index()
        .assign(season_label=lambda frame: frame["season"].map(season_label))
        .drop(columns=["season"])
        .to_string(index=False)
    )
    print(f"\n{len(built.calls)} calls, {myfootballfacts.ORIGIN} by way of MyFootballFacts")

    _report_disagreements(built)
    _report_uncalled(built.calls, matches)
    print(f"-> {dataset.write(built.calls)}")
    return 0


def _report_disagreements(built: dataset.Backfill) -> None:
    """What the cross-check against Football-Data found — the evidence for the parse itself.

    Counted against the listings that *have* a published result, not against every call: a
    Fixture the archive only ever listed as postponed carries a call and no score, and putting it
    in the denominator would report a check that was never made.
    """
    checked = built.checked
    print(
        f"\n{checked - len(built.disagreements)} of {checked} published results match "
        f"Football-Data; {len(built.disagreements)} do not:"
    )
    if not built.disagreements.empty:
        print(built.disagreements.to_string(index=False))
    print("  Football-Data is the authority; the published result is read only to check the parse")


def _report_uncalled(calls: pd.DataFrame, matches: pd.DataFrame) -> None:
    """The Fixtures inside a Pundit's Seasons that the archive never listed."""
    seasons = sorted(set(calls["season"]))
    window = matches.loc[
        matches["season"].isin(seasons) & (matches["division"] == dataset.DIVISION)
    ]
    called = set(dataset.fixture_keys(calls))
    uncalled = [key for key in dataset.fixture_keys(window) if key not in called]
    print(f"\n{len(uncalled)} of {len(window)} Fixtures have no call, and are not covered:")
    for season, home, away in sorted(uncalled):
        print(f"  {season_label(int(season))} {home} v {away}")


def _grades(matches_path: Path | None) -> int:
    graded = grading.grade(dataset.load(), match_table(matches_path))
    rate = "{:.3f}".format

    print(grading.summary(graded).to_string(index=False, float_format=rate))
    print()
    print(grading.summary(graded, by=("pundit",)).to_string(index=False, float_format=rate))
    print(
        "\nexact score is the strict reading and correct Outcome the lenient one; RPS on the "
        "scoreboard is the headline (CLAUDE.md)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
