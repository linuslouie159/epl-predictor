"""Command line entry point for the Pundit backfill and the three-way scoreboard.

    python -m epl.pundits fetch     cache the nine MyFootballFacts season pages
    python -m epl.pundits fetch --refresh
    python -m epl.pundits build     parse, reconcile with the corpus, freeze predictions.csv
    python -m epl.pundits grades    the two readings of every call, per Pundit and Season
    python -m epl.pundits three-way model, market and both readings, over shared Fixtures
    python -m epl.pundits calls     every call ranked by the miss its fair reading still had
    python -m epl.pundits map       what a call of each predicted goal margin is worth

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

The last three are issue #12, and they read the ledger rather than the archive. ``three-way`` is
the headline artifact: Elo, the Market Line, the Naive Baseline and both readings of the Pundit,
scored on identical metrics over the Fixtures all five reached, with the gap between the two
readings printed underneath as the cost of stating certainty (ADR 0003). ``calls`` is the same
comparison one call at a time, ranked so the best and worst are at either end. ``map`` prints the
model itself — what a call of each predicted goal margin has historically been worth.

All three print a table and write a file, and the file always holds more than the table: ``calls``
prints ten rows of several thousand, and choosing how many to show is the reader's business rather
than this module's (issue #12's seventh acceptance criterion).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import epl.ledger as ledger
from epl.ingest import match_table
from epl.ledger import scoreboard
from epl.pundits import dataset, grading, myfootballfacts, report
from epl.windows import season_label

#: How many of the ranked calls each end of the printed table shows. The file holds every one.
SHOWN = 5


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
    sub.add_parser("three-way", help="model, market and both readings over shared Fixtures")
    sub.add_parser("calls", help="every call ranked by the miss its fair reading still had")
    sub.add_parser("map", help="what a call of each predicted goal margin is worth")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _fetch(args.refresh)
    if args.command == "build":
        return _build(args.matches)
    if args.command == "grades":
        return _grades(args.matches)

    # The three-way reports score stored Predictions, so every Predictor on the board has to be
    # registered before one is read — the board's `note` column is looked up by name.
    ledger.register_all()
    if args.command == "three-way":
        return _three_way(args.matches)
    if args.command == "calls":
        return _calls(args.matches)
    if args.command == "map":
        return _map(args.matches)
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


def _scored(matches_path: Path | None) -> pd.DataFrame:
    """The whole ledger, calibrated.

    Calibrated over every Predictor's **whole** track record, before any slate is cut. The
    three-way tables narrow the comparison afterwards; narrowing first would give the Market Line a
    calibrated form fitted on a Pundit's Fixtures, which is a number that exists nowhere else
    (:func:`epl.ledger.scoreboard.lines`).
    """
    return scoreboard.calibrated_predictions(ledger.stored(), match_table(matches_path))


def _table(frame: pd.DataFrame, places: int = 4) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.{places}f}")


def _no_predictions() -> int:
    print("no stored Predictions — run `python -m epl.ledger backfill`")
    return 0


def _three_way(matches_path: Path | None) -> int:
    """The headline artifact: five Predictors, identical metrics, identical Fixtures."""
    scored = _scored(matches_path)
    boards = report.boards(scored)
    if boards.empty:
        return _no_predictions()

    for slate, board in boards.groupby("slate", sort=True):
        print(f"\n{slate}'s Fixtures — pre-calibration")
        print(_table(board[list(scoreboard.PRE_CALIBRATION_COLUMNS)]))
        print(f"\n{slate}'s Fixtures — post-calibration")
        print(_table(board[list(scoreboard.POST_CALIBRATION_COLUMNS)]))
        # The Fixtures are identical and the metrics are identical, but `corrected` is not: the
        # shared layer needs 380 of a Predictor's own Predictions behind it, and by this slate the
        # market and Elo are long past that while a Pundit's record has barely started. So some of
        # the pundit rows here are raw pass-through and none of the market's are. Said out loud
        # rather than left in a column, because an unread caveat is the Ceiling Line's whole
        # lesson (ADR 0001) and this table is where it would bite next.
        print(
            "  read `corrected` before this table: it is not the same fraction for every row, so "
            "\n  these are the same Fixtures but not the same amount of correction"
        )

    costs = report.certainty(boards)
    print("\nthe cost of stating certainty — the same calls, read two ways (ADR 0003)")
    print(_table(costs.drop(columns=["slate"])))
    print(
        "  cost_of_certainty is the as-stated RPS less the calibrated one: what being asked for a "
        "\n  Scoreline instead of a probability charged the forecaster. Neither reading is a trick "
        "\n  and neither may be published alone."
    )
    for name, table in (("three_way", boards), ("certainty", costs)):
        print(f"-> {report.write(table, name)}")

    # The caveats, printed under the tables rather than in a column of floats — the same reasoning
    # the main scoreboard applies, and here it is carrying ADR 0003's naming rule.
    for _, line in boards.loc[boards["note"].astype(str) != ""].drop_duplicates(
        subset=["predictor"]
    ).iterrows():
        print(f"  {line['predictor']}: {line['note']}")
    return 0


def _calls(matches_path: Path | None) -> int:
    """Each Pundit's best and worst calls by the miss their fair reading still had."""
    scored = _scored(matches_path)
    ranked = report.ranked_calls(scored, dataset.load())
    if ranked.empty:
        return _no_predictions()

    for pundit, calls in ranked.groupby("pundit", sort=True):
        ordered = calls.sort_values("miss", kind="stable")
        print(f"\n{pundit}: the {SHOWN} calls their map read best")
        print(_table(ordered.head(SHOWN).drop(columns=["pundit"]), places=3))
        print(f"\n{pundit}: the {SHOWN} it read worst")
        print(_table(ordered.tail(SHOWN).drop(columns=["pundit"]), places=3))
    print(
        "\n  miss is the RPS of the fair reading — what the call was still wrong by once the "
        "\n  Scoreline had been read as what such a call is worth (spec, user story 34)"
    )
    print(f"-> {report.write(ranked, 'pundit_calls')}")
    return 0


def _map(matches_path: Path | None) -> int:
    """The model itself: what a call of each predicted goal margin has been worth.

    The only command of the three that does not read the ledger. A map is fitted from the corpus
    and the frozen calls, so it exists whether or not anything has been backfilled — which means
    an empty result here says something different from an empty result in the other two.
    """
    matches = match_table(matches_path)
    maps = report.published_maps(matches)
    if maps.empty:
        print("no Calibrated Pundit is registered, so there is no map to publish")
        return 0

    for pundit, table in maps.groupby("pundit", sort=True):
        print(f"\n{pundit}, at the As-Of Instant of their last call")
        print(_table(table.drop(columns=["pundit"]), places=3))
    print(
        "\n  each row is a predicted goal margin and what it has historically produced. Margins "
        "\n  share a row where one of them was too thin to carry a rate of its own"
    )
    print(f"-> {report.write(maps, 'margin_map')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
