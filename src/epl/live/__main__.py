"""Command line entry point for the live loop.

    python -m epl.live upcoming    what the rolling fixtures file holds, and what could be sealed
    python -m epl.live seal        predict the upcoming round, write it, and commit it
    python -m epl.live seal --supersede   correct a round already sealed, at a new As-Of Instant
    python -m epl.live score       ingest results, then score what has been sealed

``upcoming`` is the one to run first and the one that answers the open question. It writes nothing,
so it can be run at any hour; ``seal`` does the same work and then writes, and refuses outside the
round's own window.

Both fetch the rolling file by default. ``--cached`` reads the last fetch instead, which is what to
use when asking what *was* there rather than what is — the file is replaced in place upstream, so a
cached copy is the only record of a given week.

``score`` is the retrospective half and is meant for a schedule. It refreshes the Live Season from
upstream, rebuilds the match table, re-audits the seal, and scores every Sealed
Prediction whose Fixture has now been played. It scores the live Season **on its own board**: the
Evaluation Window's numbers are over closed Seasons and must go on meaning the same thing from one
week to the next (:mod:`epl.windows`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import epl.ledger as ledger
from epl.ingest import (
    DIVISIONS,
    build_tables,
    fetch_all,
    fetch_fixtures,
    latest_fixtures_path,
    match_table,
    parse_fixtures,
)
from epl.ledger import live as store
from epl.ledger import scoreboard
from epl.live import seal, upcoming
from epl.paths import outputs_dir
from epl.predictors import Corpus
from epl.windows import LIVE_SEASON, season_label


def live_scoreboard_path() -> Path:
    """Where the Live Season is scored, apart from the Evaluation Window's board."""
    return outputs_dir() / "live_scoreboard.csv"


def clock() -> pd.Timestamp:
    """Now, in UK local time — behind a function so a test can stop it.

    Deliberately **not** a command-line option. This is the value that decides whether a round is
    inside its sealing window, so an operator who could name it could seal a round after its own
    kickoff and have the file say otherwise, which is the one thing this store exists to prevent
    (ADR 0005). The commit timestamp git records is not overridable either, and for the same reason.

    A flag is not the only way to get the wrong moment, though, and the other way needs nobody to
    choose it: :func:`epl.ledger.live.uk_now` is called rather than ``pd.Timestamp.now`` because a
    machine outside the UK reads its own zone and every kickoff it is compared against is in the
    UK's. That was a latent bug the whole time this loop was run by hand from one desk, and a
    schedule is what makes it certain — a container defaults to UTC (issue #19).
    """
    return store.uk_now()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.live", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Both commands read the same rolling file the same way; only one of them then writes.
    cached_help = "read the last fetched rolling file instead of fetching a new one"
    looking = sub.add_parser("upcoming", help="report what the rolling fixtures file holds")
    looking.add_argument("--cached", action="store_true", help=cached_help)

    sealing = sub.add_parser("seal", help="predict and seal the upcoming Prediction Round")
    sealing.add_argument("--cached", action="store_true", help=cached_help)
    sealing.add_argument(
        "--supersede",
        action="store_true",
        help="correct a round already sealed, as a new revision at a new As-Of Instant (ADR 0005)",
    )
    sealing.add_argument(
        "--no-commit",
        action="store_true",
        help="write the round without committing it — it is not evidence until it is committed",
    )
    sealing.add_argument(
        "--push",
        action="store_true",
        help="push the commit to its remote — what an unattended loop on another machine needs",
    )

    scoring = sub.add_parser("score", help="ingest results and score what has been sealed")
    scoring.add_argument(
        "--no-ingest",
        action="store_true",
        help="score what is already in the match table rather than refreshing it from upstream",
    )

    args = parser.parse_args(argv)
    # `score` rebuilds the canonical match table, so pointing it at another one and asking it to
    # ingest would refresh one table and read a different one. Refused rather than resolved: which
    # of the two the caller meant is not something to guess at.
    if args.command == "score" and args.matches is not None and not args.no_ingest:
        parser.error("--matches needs --no-ingest; an ingest rebuilds the canonical match table")
    # Pushing a branch that was deliberately not given the round is the one combination that could
    # report success over an unproven seal — `pushed to origin/main`, exit 0, and the round sitting
    # uncommitted in the working tree. Refused for the same reason as above: which half the caller
    # meant is not something to guess at.
    if args.command == "seal" and args.no_commit and args.push:
        parser.error("--push contradicts --no-commit; there would be no committed round to push")

    ledger.register_all()

    if args.command == "upcoming":
        return _upcoming(args.matches, cached=args.cached)
    if args.command == "seal":
        return _seal(
            args.matches,
            cached=args.cached,
            supersede=args.supersede,
            commit=not args.no_commit,
            push=args.push,
        )
    if args.command == "score":
        return _score(args.matches, ingest_first=not args.no_ingest)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _rolling(matches: pd.DataFrame, *, cached: bool) -> pd.DataFrame:
    """The rolling file's Premier League Fixtures, fetched or read back from the cache."""
    if cached:
        path = latest_fixtures_path()
        if path is None:
            raise upcoming.LiveError(
                "no fixtures file has ever been fetched; run without --cached, or "
                "`python -m epl.ingest fixtures`"
            )
    else:
        path = fetch_fixtures()
    print(f"rolling file: {path.name}")
    return upcoming.to_predict(parse_fixtures(path), matches)


def _upcoming(matches_path: Path | None, *, cached: bool) -> int:
    """What the rolling file holds and what could be sealed from it. Writes nothing.

    Prints the whole table rather than only the sealable row, because the interesting answer on
    this data has consistently been that there is no sealable row — and "nothing to seal" and "no
    Premier League Fixture in the file at all" are different findings.
    """
    matches = match_table(matches_path)
    now = clock()
    try:
        fixtures = _rolling(matches, cached=cached)
    except upcoming.LiveError as refused:
        print(refused)
        return 1

    print(f"{len(fixtures)} upcoming Premier League Fixtures in {season_label(LIVE_SEASON)}")
    table = upcoming.rounds(fixtures, now=now)
    if table.empty:
        print(
            "the rolling file held no Fixture in a tier this project predicts, so there is no "
            "Prediction Round to seal"
        )
        return 0

    print(table.to_string(index=False))
    for _, row in fixtures.iterrows():
        kickoff = row["time"] if pd.notna(row["time"]) else "--:--"
        print(f"  {row['date']} {kickoff} {row['home_club']} v {row['away_club']}")
    sealable = table.loc[table["status"] == upcoming.SEALABLE]
    if sealable.empty:
        print(f"nothing is inside its sealing window at {now.isoformat(timespec='seconds')}")
    else:
        print(f"sealable now: {', '.join(sealable['prediction_round'])}")
    return 0


def _seal(
    matches_path: Path | None,
    *,
    cached: bool,
    supersede: bool,
    commit: bool,
    push: bool,
) -> int:
    """Seal the round that is open now, and say plainly through the exit code what happened.

    The exit code is the whole interface to a schedule, and issue #19 asks it to draw one line in
    particular: **nothing to seal is a success**. Most of the week no round is inside its window,
    and until upcoming Premier League Fixtures have a confirmed source that is every fire there
    will ever be. A loop that exits non-zero twice a week for a season teaches its owner to ignore
    it, and the next thing it ignores is a real failure.

    So :class:`~epl.live.upcoming.NothingToSeal` is 0, and every other refusal is 1 — a rolling
    file that changed shape, a stale :data:`epl.windows.LIVE_SEASON`, a round written but not
    committed, and a round committed but not pushed. The last two are loud for the same reason: an
    unproven sealed file is indistinguishable on disk from a proven one (ADR 0005).
    """
    matches = match_table(matches_path)
    now = clock()
    try:
        fixtures = _rolling(matches, cached=cached)
        chosen = upcoming.next_round(fixtures, now=now)
    except upcoming.NothingToSeal as quiet:
        print(quiet)
        return 0
    except upcoming.LiveError as refused:
        print(refused)
        return 1

    print(chosen.describe())
    # Running the loop twice inside one round is expected of a schedule and is not a failure: the
    # round is already sealed, and the second run must leave it exactly as it was. Asked of the
    # store rather than caught from `seal`, so that "nothing to do" cannot come to mean "some other
    # LedgerError whose message happened to read the same way".
    #
    # It still pushes. A second fire has nothing to write, and pushing is then the whole of what is
    # left for it to do — which is what makes the retry worth scheduling at all, since the run that
    # sealed the round is exactly the run that may have been unable to reach the network.
    if store.is_sealed(chosen.prediction_round) and not supersede:
        print(f"{chosen.prediction_round} is already sealed — nothing to do")
        print("  a genuine correction goes in with --supersede (ADR 0005)")
    else:
        try:
            sealed = seal.run(
                chosen, Corpus(matches), now=now, supersede=supersede, commit=commit
            )
        except ledger.LedgerError as refused:
            print(refused)
            return 1

        print(sealed.describe())
        if sealed.silent:
            print(f"silent (cover none of this round): {', '.join(sealed.silent)}")
            print("  a Pundit cannot be sealed at all — see epl.pundits.live and issue #16")
        if commit and sealed.commit is None:
            print("WARNING: the round was written but not committed, so nothing yet proves when.")
            return 1

    # One decision, one place. Both arms above reach here — the round was sealed just now, or it
    # was sealed by an earlier fire — and in both cases pushing is what is left to do. Written as a
    # single tail rather than a return in each arm, because two copies of "and then push" are two
    # things that can drift apart, and the one that stopped pushing would be silent about it.
    return _push() if push else 0


def _push() -> int:
    """Push the branch, and report a failure as loudly as an uncommitted seal.

    A commit on the machine that made it proves when the round was written to anyone who can reach
    that machine. Once the loop runs somewhere else — the Pi this schedule was built for — that is
    nobody, so the round is not evidence until it has left. Pushing again when there is nothing to
    push costs a network round trip and succeeds, which is what makes it safe to do on every fire.
    """
    landed = store.push()
    if landed is None:
        print("WARNING: the round is committed here and NOT PUSHED, so it proves nothing offsite.")
        print("  check the remote and the credentials; the next fire of the loop will retry")
        return 1
    print(f"pushed to {landed}")
    return 0


def _score(matches_path: Path | None, *, ingest_first: bool) -> int:
    """Ingest results and score every Sealed Prediction whose Fixture has now been played."""
    if ingest_first:
        fetch_all([LIVE_SEASON], DIVISIONS, refresh=True)
        matches, _, destination, _ = build_tables()
        print(f"{len(matches)} matches -> {destination}")

    matches = match_table(matches_path)
    sealed = store.read()
    if sealed.empty:
        print("nothing has been sealed yet — `python -m epl.live seal`")
        return 0

    violations = store.seal_violations()
    for complaint in violations:
        print(f"outputs/live/{complaint}")

    board = scoreboard.build(ledger.stored(), matches, seasons=[LIVE_SEASON])
    if board.empty:
        print(
            f"{len(sealed)} Sealed Predictions, none of whose Fixtures has been played and "
            "ingested yet"
        )
    else:
        print(f"{season_label(LIVE_SEASON)}, sealed and scored — "
              f"{len(sealed)} Predictions on {int(board['fixtures'].max())} Fixtures")
        print(board[list(scoreboard.PRE_CALIBRATION_COLUMNS)].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ))
        print(f"-> {scoreboard.write(board, live_scoreboard_path())}")

    # The calibrated half is deliberately not printed. A map needs a track record behind it before
    # it is fitted at all (epl.calibration), and the Live Season has not got one — the column
    # would be the raw one under another name, which is the sort of number that gets quoted.
    print("  the Evaluation Window's board is unchanged: `python -m epl.ledger scoreboard`")
    print("  a Pundit's column fills in retrospectively, once the archive catches up (#16)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
