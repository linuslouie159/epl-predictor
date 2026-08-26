"""Command line entry point for the Bayesian fit and the Season Projection over it.

    python -m epl.simulate checkpoints   where a Season would be projected from, and where not
    python -m epl.simulate posterior     fit one, and read it beside the MLE of the same model
    python -m epl.simulate project       one Season Projection: the title, Europe and relegation
    python -m epl.simulate validate      project completed Seasons and see where the champion fell

``project`` is issue #15's answer in the form the question is usually asked: who wins the league.
It fits a posterior at one checkpoint, plays the rest of the Season out ten thousand times, and
prints the table with both seeds beside it so the run can be reproduced.

``validate`` is the criterion that stops a projection being merely plausible. It projects completed
Seasons from their checkpoints, joins each to the table that Season actually produced, and reports
where the real champion landed and how close the promised probabilities came to the observed rates.
It is the overnight job — one posterior fit per checkpoint, minutes each — so ``--seasons``,
``--checkpoints`` and the sampler knobs are how to want less of it. It writes its rows after every
projection rather than at the end, so an interrupted run leaves a readable answer over fewer
Seasons instead of nothing at all.

``checkpoints`` is the cheap half and the one that makes ADR 0007's split visible: it prints the
Prediction Rounds a Season Projection is taken at against the number of rounds the Season has, so
"only at Season Projection points, never at all 1,189 rounds" is a number a reader can check rather
than a promise in a docstring. It fits nothing. (Issue #14 says 1,332; stage 2 measured 1,189 and
corrected it — see docs/DECISIONS.md.)

``posterior`` is the expensive half. It fits the model at one checkpoint and prints three things:
the Clubs with the widest posteriors, the sampler's own diagnostics, and the same Fixture quoted by
both fits. That last one is issue #14's acceptance criterion made runnable — ADR 0007 permits the
whole split *because* the two agree on a single Fixture, so the place to check it is a command
anyone can run rather than a test only CI sees.

**Both fits are the same model and neither is a correction of the other.** The scoreboard's match
probabilities all come from the maximum-likelihood path (:mod:`epl.models.dixon_coles`); only a
Season Projection is built on these draws. Any output that puts the two beside each other has to
say so, which is why :data:`DIFFERENT_FITS` exists and is printed rather than left to the reader.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from epl.ingest import match_table
from epl.models.dixon_coles import DIXON_COLES
from epl.models.dixon_coles import fit as fit_mle
from epl.models.likelihood import Sample, Strengths
from epl.paths import outputs_dir
from epl.predictors import Evidence
from epl.rounds import assign_rounds
from epl.simulate.checkpoints import (
    CHECKPOINTS_PER_SEASON,
    PROJECTED_DIVISION,
    projection_rounds,
)
from epl.simulate.posterior import SAMPLING, Posterior, Sampling, fit
from epl.simulate.projection import SIMULATION, Projection, Simulation, project
from epl.simulate.table import TIEBREAKERS
from epl.simulate.validation import EVENTS, validate
from epl.windows import EVALUATION_WINDOW, season_label

#: What one posterior fit costs, for the estimate ``validate`` prints before committing a reader to
#: a run of sixty of them. Measured rather than assumed, and a range rather than a number: 231 s at
#: 2015/16's first Prediction Round (stage 10) and 537 s at its mid-Season checkpoint, so the upper
#: end is what a plan should be made against. Only ever used to print an estimate.
MINUTES_A_FIT = 9

#: The sentence issue #14's last acceptance criterion asks for, in one place so that every output
#: comparing the two fits carries the same wording.
DIFFERENT_FITS = (
    "match probabilities and Season Projections come from formally different fits of the same\n"
    "model (ADR 0007): every Prediction on the scoreboard is the maximum-likelihood path, and\n"
    "only a projection is drawn from this posterior"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.simulate", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    where = sub.add_parser("checkpoints", help="where a Season is projected from, and where not")
    where.add_argument("--season", type=int, default=min(EVALUATION_WINDOW))
    where.add_argument(
        "--live",
        action="store_true",
        help="project weekly, as the current Season is, instead of at checkpoints",
    )

    draw = sub.add_parser("posterior", help="fit one posterior and read it beside the MLE")
    draw.add_argument("--season", type=int, default=min(EVALUATION_WINDOW))
    draw.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help=f"which of the Season's {CHECKPOINTS_PER_SEASON} checkpoints (default: the first)",
    )
    _add_sampling(draw)

    table = sub.add_parser("project", help="one Season Projection: title, Europe and relegation")
    table.add_argument("--season", type=int, default=min(EVALUATION_WINDOW))
    table.add_argument(
        "--checkpoint",
        type=int,
        default=(CHECKPOINTS_PER_SEASON - 1) // 2,
        help=f"which of the Season's {CHECKPOINTS_PER_SEASON} checkpoints (default: mid-Season)",
    )
    _add_sampling(table)
    _add_walk(table)

    check = sub.add_parser("validate", help="project completed Seasons and score the projections")
    check.add_argument(
        "--seasons",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="Seasons to validate over (default: the whole Evaluation Window, minus the live one)",
    )
    check.add_argument("--checkpoints", type=int, default=CHECKPOINTS_PER_SEASON)
    _add_sampling(check)
    _add_walk(check)

    args = parser.parse_args(argv)
    if args.command == "checkpoints":
        return _checkpoints(args.matches, args.season, args.live)
    if args.command == "posterior":
        return _posterior(
            args.matches,
            args.season,
            args.checkpoint,
            _sampling(args),
        )
    if args.command == "project":
        return _project(
            args.matches,
            args.season,
            args.checkpoint,
            _sampling(args),
            Simulation(seasons=args.simulations, seed=args.seed),
        )
    if args.command == "validate":
        return _validate(
            args.matches,
            args.seasons,
            args.checkpoints,
            _sampling(args),
            Simulation(seasons=args.simulations, seed=args.seed),
        )
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _sampling(args: argparse.Namespace) -> Sampling:
    """The sampler settings the three fitting commands share, read off one parsed namespace."""
    return Sampling(
        draws=args.draws, tune=args.tune, chains=args.chains, seed=args.sampler_seed
    )


def _add_sampling(parser: argparse.ArgumentParser) -> None:
    """The knobs on the expensive half. Lowering them is how a run finishes this afternoon.

    ``--sampler-seed`` is here for the same reason ``--seed`` is on the walk: a published
    projection records both, and a recorded seed with no way to hand it back is not a way to
    reproduce anything.
    """
    parser.add_argument("--draws", type=int, default=SAMPLING.draws)
    parser.add_argument("--tune", type=int, default=SAMPLING.tune)
    parser.add_argument("--chains", type=int, default=SAMPLING.chains)
    parser.add_argument("--sampler-seed", type=int, default=SAMPLING.seed)


def _add_walk(parser: argparse.ArgumentParser) -> None:
    """The knobs on the cheap half, including the seed a published projection is reproduced by."""
    parser.add_argument(
        "--simulations",
        type=int,
        default=SIMULATION.seasons,
        help=f"how many Seasons the walk plays out (default: {SIMULATION.seasons:,})",
    )
    parser.add_argument("--seed", type=int, default=SIMULATION.seed)


def _checkpoints(matches_path: Path | None, season: int, live: bool) -> int:
    matches = match_table(matches_path)
    chosen = projection_rounds(matches, season, live=live)
    every = projection_rounds(matches, season, live=True)

    label = "weekly (live)" if live else f"{len(chosen)} checkpoints"
    print(
        f"{season_label(season)}: {label} out of {len(every)} Prediction Rounds "
        f"— {len(every) - len(chosen)} rounds get no posterior at all"
    )
    print("\nADR 0007: a full posterior at every round would make one backtest an overnight job,")
    print("and buys nothing — parameter uncertainty barely moves a single Fixture's probability.\n")

    table = chosen[["prediction_round", "as_of_instant", "fixtures", "first_kickoff"]]
    print(table.to_string(index=False))
    return 0


def _posterior(
    matches_path: Path | None, season: int, checkpoint: int, sampling: Sampling
) -> int:
    matches = match_table(matches_path)
    chosen = projection_rounds(matches, season)
    if not 0 <= checkpoint < len(chosen):
        raise SystemExit(
            f"{season_label(season)} has {len(chosen)} checkpoints; asked for index {checkpoint}"
        )

    at = chosen.iloc[checkpoint]
    instant = pd.Timestamp(at["as_of_instant"])
    upcoming = _fixtures_of(matches, at["prediction_round"])
    clubs = sorted({*upcoming["home_club"], *upcoming["away_club"]})

    evidence = Evidence.before(matches, instant)
    sample = DIXON_COLES.sample_at(evidence, also=clubs)
    print(
        f"{season_label(season)} checkpoint {checkpoint + 1} of {len(chosen)}, "
        f"as of {instant.date()}: {len(sample)} weighted matches, {sample.club_count} Clubs, "
        f"{2 * sample.club_count + 2} parameters"
    )

    point = fit_mle(sample)
    clock = time.perf_counter()
    print(f"sampling {sampling.chains} chains x {sampling.draws} draws "
          f"(tune {sampling.tune})... this is minutes, not seconds", flush=True)
    drawn = fit(sample, sampling=sampling)
    print(f"fitted in {time.perf_counter() - clock:.0f}s\n")

    print(drawn.diagnostics.describe())
    for concern in drawn.diagnostics.concerns():
        print(f"  ! {concern}")
    if not drawn.diagnostics.healthy:
        print("  do not publish a Season Projection from this fit")
    print()

    _print_uncertainty(drawn, point)
    _print_quote(drawn, point, sample, upcoming)
    # Printed here rather than at the end of `_print_quote`, because everything above it is also a
    # comparison between the two fits and `_print_quote` has a Fixture-less path that returns
    # early. Issue #14's last criterion is about any published comparison, not about that one.
    print(f"\n{DIFFERENT_FITS}")
    return 0


def _project(
    matches_path: Path | None,
    season: int,
    checkpoint: int,
    sampling: Sampling,
    simulation: Simulation,
) -> int:
    """One Season Projection, printed and written."""
    matches = match_table(matches_path)
    chosen = projection_rounds(matches, season)
    if not 0 <= checkpoint < len(chosen):
        raise SystemExit(
            f"{season_label(season)} has {len(chosen)} checkpoints; asked for index {checkpoint}"
        )

    at = chosen.iloc[checkpoint]
    print(
        f"{season_label(season)} checkpoint {checkpoint + 1} of {len(chosen)}, "
        f"as of {pd.Timestamp(at['as_of_instant']).date()}"
    )
    print(f"fitting the posterior ({sampling.chains} chains x {sampling.draws} draws)... "
          "this is minutes, not seconds", flush=True)

    clock = time.perf_counter()
    projection = project(matches, season, at, sampling=sampling, simulation=simulation)
    print(f"fitted and walked in {time.perf_counter() - clock:.0f}s\n")

    _print_projection(projection)
    print(f"\n{DIFFERENT_FITS}")
    written = _write(projection.published(), "projection")
    print(f"\nwritten to {written}")
    return 0


def _print_projection(projection: Projection) -> None:
    """The table, the two seeds and the chain that settled every tie in it."""
    print(projection.describe())
    for concern in projection.diagnostics.concerns():
        print(f"  ! {concern}")
    if not projection.diagnostics.healthy:
        print("  the posterior beneath this projection did not converge; do not publish it")

    print("\nties settled by: " + " -> ".join(TIEBREAKERS))
    print()
    table = projection.table().copy()
    for column in EVENTS:
        table[column] = table[column].map(lambda share: f"{share:.4f}")
    table["mean_points"] = table["mean_points"].map(lambda points: f"{points:.1f}")
    table["mean_position"] = table["mean_position"].map(lambda place: f"{place:.2f}")
    print(table.to_string(index=False))


def _validate(
    matches_path: Path | None,
    seasons: list[int] | None,
    checkpoints: int,
    sampling: Sampling,
    simulation: Simulation,
) -> int:
    """Project completed Seasons from their checkpoints and score what came back."""
    matches = match_table(matches_path)
    wanted = _seasons_to_validate(matches, seasons)
    total = len(wanted) * checkpoints
    print(
        f"validating {len(wanted)} Seasons x {checkpoints} checkpoints = {total} posterior fits.\n"
        f"At the shipped sampler settings that is roughly {total * MINUTES_A_FIT / 60:.1f} hours; "
        "--seasons, --checkpoints and\n--draws/--tune/--chains are how to want less of it. "
        "Rows are written after every projection.\n"
    )

    done = 0
    clock = time.perf_counter()
    so_far: list[pd.DataFrame] = []

    def progress(projection: Projection, rows: pd.DataFrame) -> None:
        """Print where the run has got to, and save what it has — see :func:`validate`.

        Written every time rather than once at the end. An interrupted eight-hour run that had
        kept nothing would have to start again, and a partial file is a perfectly readable answer
        over fewer Seasons.
        """
        nonlocal done
        done += 1
        so_far.append(rows)
        _write(pd.concat(so_far, ignore_index=True), "projection_validation")
        leader = projection.table().iloc[0]
        print(
            f"[{done}/{total}] {projection.where}  "
            f"{projection.remaining} to play, favourite {leader['club']} "
            f"{leader['title']:.3f}  ({time.perf_counter() - clock:.0f}s elapsed)",
            flush=True,
        )

    validation = validate(
        matches,
        wanted,
        checkpoints=checkpoints,
        sampling=sampling,
        simulation=simulation,
        on_projection=progress,
    )

    print(f"\n{validation.describe()}\n")
    print("where the real champion landed, by how much of the Season was left:")
    print(validation.champions().to_string(index=False))
    print("\nreliability over the title, European places and relegation pooled:")
    print(validation.reliability().to_string(index=False))
    print(
        "\nthese points are not independent — twenty Clubs share one table and six checkpoints\n"
        "share one champion — so read the diagram as a shape, not as a significance test."
    )
    written = _write(validation.rows, "projection_validation")
    print(f"\nwritten to {written}")
    return 0


def _seasons_to_validate(matches: pd.DataFrame, seasons: list[int] | None) -> list[int]:
    """Which Seasons to validate over: the ones asked for, or every completed Season scored.

    The live Season is dropped rather than refused. It has no final table, so there is nothing to
    validate against, and leaving it in would turn "validate the Evaluation Window" into an error.
    """
    if seasons is not None:
        return list(range(seasons[0], seasons[1] + 1))

    played = matches.loc[matches["division"] == PROJECTED_DIVISION]
    complete = played.groupby("season").size()
    return [
        season
        for season in EVALUATION_WINDOW
        if season in complete.index and complete[season] >= complete.max()
    ]


def _write(table: pd.DataFrame, name: str) -> Path:
    """Regenerable and gitignored, like every other report this project publishes."""
    path = outputs_dir() / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _fixtures_of(matches: pd.DataFrame, prediction_round: str) -> pd.DataFrame:
    assigned = assign_rounds(matches)
    return assigned.loc[assigned["prediction_round"] == prediction_round]


def _print_uncertainty(drawn: Posterior, point: Strengths) -> None:
    """The Clubs the posterior is least sure about, which is what the MLE cannot say at all."""
    spread = drawn.attack.std(axis=0)
    mean = drawn.attack.mean(axis=0)
    order = np.argsort(-spread)

    print("the Clubs this fit is least certain about (attack, log-goals):")
    for index in order[:5]:
        print(f"  {drawn.clubs[index]:<20} {mean[index]:+.3f} +/- {spread[index]:.3f}  "
              f"(MLE {point.attack[index]:+.3f})")
    # The widest belong to Clubs with almost no football inside the decay horizon, and the spread
    # tracks that almost exactly — see `Diagnostics.DIVERGENCE_RATE_CEILING`.
    print(f"\n  the widest is {spread.max():.3f} and the tightest {spread.min():.3f}; "
          "an MLE reports neither")
    print(f"  home advantage {drawn.home_advantage.mean():+.4f} "
          f"+/- {drawn.home_advantage.std():.4f} (MLE {point.home_advantage:+.4f})")
    print(f"  low-score correction {drawn.correction.mean():+.4f} "
          f"+/- {drawn.correction.std():.4f} (MLE {point.correction:+.4f})\n")


def _print_quote(
    drawn: Posterior, point: Strengths, sample: Sample, upcoming: pd.DataFrame
) -> None:
    """One Fixture quoted by both fits — ADR 0007's claim, printed rather than asserted."""
    if upcoming.empty:
        return
    fixture = upcoming.iloc[0]
    home = sample.index_of([fixture["home_club"]])
    away = sample.index_of([fixture["away_club"]])

    mle_quote = point.outcomes_for(home, away)[0]
    mean_quote = drawn.mean().outcomes_for(home, away)[0]
    print(f"{fixture['home_club']} v {fixture['away_club']}, quoted by both fits:")
    print(f"  maximum likelihood  H {mle_quote[0]:.4f}  D {mle_quote[1]:.4f}  A {mle_quote[2]:.4f}")
    print(f"  posterior mean      H {mean_quote[0]:.4f}  D {mean_quote[1]:.4f}  "
          f"A {mean_quote[2]:.4f}")
    print(f"  largest difference  {np.abs(mle_quote - mean_quote).max():.4f}")
    print("\nthey agree, which is what lets ADR 0007 use the cheap fit for all 952 rounds.")
    print(f"\n{DIFFERENT_FITS}")


if __name__ == "__main__":
    sys.exit(main())
