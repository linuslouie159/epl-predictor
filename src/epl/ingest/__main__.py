"""Command line entry point for the ingest.

    python -m epl.ingest fetch                 fill the raw cache (skips what is already there)
    python -m epl.ingest fetch --refresh       re-download, including the growing current Season
    python -m epl.ingest fixtures              fetch the rolling forward-Fixture file
    python -m epl.ingest build                 write matches.csv + odds_availability.csv
    python -m epl.ingest clubs                 report Club spellings the Alias table does not know
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epl.clubs import ClubResolver
from epl.ingest.fixtures import fetch_fixtures, parse_fixtures
from epl.ingest.football_data import (
    DIVISIONS,
    FIRST_SEASON,
    LAST_SEASON,
    SOURCE,
    build_tables,
    club_names_in_raw_cache,
    fetch_all,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.ingest", description=__doc__)
    parser.add_argument(
        "--seasons",
        default=f"{FIRST_SEASON}-{LAST_SEASON}",
        help="Season range by start year, e.g. 2005-2025 (default: every ingested Season)",
    )
    parser.add_argument(
        "--divisions",
        default=",".join(DIVISIONS),
        help="tiers to act on (default: all four, per ADR 0004)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="fill the raw cache")
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="re-download files already cached; needed for the current Season, whose upstream "
        "file grows weekly",
    )
    sub.add_parser("fixtures", help="fetch and summarise the rolling forward-Fixture file")
    build = sub.add_parser("build", help="write the cleaned match table")
    build.add_argument("--out", type=Path, default=None)
    sub.add_parser("clubs", help="report Club spellings the Alias table does not know")

    args = parser.parse_args(argv)
    seasons = _parse_seasons(args.seasons)
    divisions = tuple(d.strip() for d in args.divisions.split(",") if d.strip())

    if args.command == "fetch":
        paths = fetch_all(seasons, divisions, refresh=args.refresh)
        print(f"cached {len(paths)} files under {paths[0].parent.parent}")
        return 0

    if args.command == "fixtures":
        path = fetch_fixtures()
        frame = parse_fixtures(path)
        print(f"fetched {path}")
        print(f"{len(frame)} English Fixtures: {frame['division'].value_counts().to_dict()}")
        if not frame.empty:
            print(f"dates {frame['date'].min()} .. {frame['date'].max()}")
            priced = int(frame["prematch_odds_home"].notna().sum())
            print(f"{priced} of {len(frame)} carry a Market Line")
        return 0

    if args.command == "build":
        matches, availability, out, record = build_tables(seasons, divisions, args.out)
        print(f"{len(matches)} matches -> {out}")

        priced = int(availability["has_market_line"].sum()) if not availability.empty else 0
        print(f"odds availability for {len(availability)} Season-tiers -> {record}")
        print(f"  {priced} carry a Market Line, {len(availability) - priced} do not")

        _warn_if_not_canonical(seasons, divisions)
        return 0

    if args.command == "clubs":
        resolver = ClubResolver.load()
        names = club_names_in_raw_cache(seasons, divisions)
        unknown = [name for name in names if not resolver.knows(name, SOURCE)]
        print(f"{len(names)} spellings in the cache, {len(unknown)} unknown")
        for name in unknown:
            print(f"  {name}")
        return 1 if unknown else 0

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _warn_if_not_canonical(seasons: list[int], divisions: tuple[str, ...]) -> None:
    """Say so loudly when a build is a subset of the corpus the models expect.

    ADR 0004 keeps all four tiers because "adding tiers later would invalidate every stored
    backtest and force a full re-run". A narrowed build is useful for a quick look and useless as
    the corpus a Predictor is scored on, and nothing downstream can tell the two apart.
    """
    narrowed = []
    if tuple(divisions) != DIVISIONS:
        narrowed.append(f"tiers {list(divisions)} instead of all four (ADR 0004)")
    if seasons != list(range(FIRST_SEASON, LAST_SEASON + 1)):
        narrowed.append(
            f"Seasons {seasons[0]}-{seasons[-1]} instead of {FIRST_SEASON}-{LAST_SEASON}"
        )
    if narrowed:
        print("WARNING: not the canonical corpus — " + "; ".join(narrowed))
        print("         do not score a Predictor on this table.")


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        start, end = (int(part) for part in text.split("-", 1))
        return list(range(start, end + 1))
    return [int(text)]


if __name__ == "__main__":
    sys.exit(main())
