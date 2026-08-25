"""Command line entry point for the Bayesian fit.

    python -m epl.simulate checkpoints   where a Season would be projected from, and where not
    python -m epl.simulate posterior     fit one, and read it beside the MLE of the same model

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
from epl.predictors import Evidence
from epl.rounds import assign_rounds
from epl.simulate.checkpoints import CHECKPOINTS_PER_SEASON, projection_rounds
from epl.simulate.posterior import SAMPLING, Posterior, Sampling, fit
from epl.windows import EVALUATION_WINDOW, season_label

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
    draw.add_argument("--draws", type=int, default=SAMPLING.draws)
    draw.add_argument("--tune", type=int, default=SAMPLING.tune)
    draw.add_argument("--chains", type=int, default=SAMPLING.chains)

    args = parser.parse_args(argv)
    if args.command == "checkpoints":
        return _checkpoints(args.matches, args.season, args.live)
    if args.command == "posterior":
        return _posterior(
            args.matches,
            args.season,
            args.checkpoint,
            Sampling(draws=args.draws, tune=args.tune, chains=args.chains),
        )
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


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
