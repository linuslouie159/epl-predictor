"""Command line entry point for the models.

    python -m epl.models fit        re-derive the frozen hyperparameters on the Burn-In Window
    python -m epl.models draws      the draw rate against Supremacy, predicted and observed
    python -m epl.models ratings    the pool at a Season's first Prediction Round

``fit`` is the one that keeps the frozen numbers honest. ADR 0008 puts every hyperparameter inside
2000/01-2004/05 and then freezes it as a literal in `epl.models.elo`, which means the literals and
the fit that produced them can drift apart without anything failing. This runs the fit again and
prints both, so the two can be compared; ``tests/models/test_elo_over_the_corpus.py`` asserts it.

``draws`` is ADR 0006's receipt. The claim is that the draw band narrows on its own as Supremacy
grows, with no hand-coded taper anywhere — so the curve is printed for every Predictor with stored
Predictions, predicted beside observed, and a taper that had stopped tapering would show up here as
a flat column rather than as a slightly worse RPS.

``ratings`` answers the two questions ADR 0004 exists for: whether promoted Clubs arrive with
ratings that differ, and how much football is behind a rating by the time it is first scored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from epl.ingest import match_table
from epl.ledger import backtest, scoreboard
from epl.models import burn_in, draw_curve
from epl.models.elo import ELO, FROZEN_LOGIT, FROZEN_SETTINGS, newcomers
from epl.predictors import Evidence
from epl.rounds import as_of_instant
from epl.windows import EVALUATION_WINDOW, season_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.models", description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="the cleaned match table (default: data/processed/matches.csv)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fit", help="re-derive the frozen hyperparameters on the Burn-In Window")
    curve = sub.add_parser("draws", help="the draw rate against Supremacy")
    curve.add_argument(
        "--predictor",
        default=None,
        help="only this Predictor (default: every one with stored Predictions)",
    )
    pool = sub.add_parser("ratings", help="the pool at a Season's first Prediction Round")
    pool.add_argument(
        "--season",
        type=int,
        default=min(EVALUATION_WINDOW),
        help="the Season to stand at the start of (default: the first scored one)",
    )

    args = parser.parse_args(argv)
    if args.command == "fit":
        return _fit(args.matches)
    if args.command == "draws":
        return _draws(args.matches, args.predictor)
    if args.command == "ratings":
        return _ratings(args.matches, args.season)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _fit(matches_path: Path | None) -> int:
    matches = match_table(matches_path)
    fitted = burn_in.fit(matches)
    window = (
        f"{season_label(min(burn_in.FITTING_SEASONS))}-"
        f"{season_label(max(burn_in.FITTING_SEASONS))}"
    )
    print(f"fitted on {window}, warmed from {season_label(min(burn_in.FITTING_SEASONS) - 1)}")
    print(f"  found:  {fitted.describe()}")
    print(f"  frozen: k={FROZEN_SETTINGS.k:g} home_advantage={FROZEN_SETTINGS.home_advantage:g} "
          f"scale={FROZEN_LOGIT.scale:.4f} "
          f"cutpoints=({FROZEN_LOGIT.cutpoints[0]:.6f}, {FROZEN_LOGIT.cutpoints[1]:.6f})")
    print(f"  the floor on the same sample: {burn_in.base_rate_rps(matches):.5f} RPS")

    # Named rather than asserted: this command reports, and the test is what fails. A reader who
    # runs it after changing the ingest wants to see *what* moved, not just that something did.
    if (fitted.settings, fitted.logit) != (FROZEN_SETTINGS, FROZEN_LOGIT):
        print("  the fit and the frozen literals differ — see epl.models.elo.FROZEN_SETTINGS")
    return 0


def _draws(matches_path: Path | None, predictor: str | None) -> int:
    """The curve raw, and the curve after the shared calibration layer.

    Both, because the layer exists precisely to correct this: #9 measured Elo quoting draws too
    often in all ten buckets and the Market Line quoting them too rarely at the even end, and #10
    is what was meant to fix it. A curve printed only post-calibration would hide whether the layer
    did anything; one printed only pre-calibration would hide whether it broke something.
    """
    rows = scoreboard.calibrated_predictions(
        backtest.read(predictor), match_table(matches_path)
    )
    if rows.empty:
        print("no stored Predictions — run `python -m epl.ledger backfill`")
        return 0

    for name, group in rows.groupby("predictor", sort=True):
        outcomes = group["outcome"].to_numpy(dtype=object)
        raw = draw_curve(group[list(scoreboard.PROBABILITY_COLUMNS)].to_numpy(float), outcomes)
        calibrated = draw_curve(
            group[list(scoreboard.CALIBRATED_PROBABILITY_COLUMNS)].to_numpy(float), outcomes
        )

        print(f"\n{name}: {len(group)} Fixtures")
        print(raw.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        # The observed column is read off the raw curve on purpose: calibration moves Fixtures
        # between Supremacy buckets, so the two curves are cut differently and only one of them
        # can be the answer to "and how often was it actually a Draw".
        for label, curve, column in (
            ("predicted", raw, "predicted_draw"),
            ("calibrated", calibrated, "predicted_draw"),
            ("observed", raw, "observed_draw"),
        ):
            print(f"  {label:<11} {curve[column].iloc[0]:.3f} -> {curve[column].iloc[-1]:.3f}")
    return 0


def _ratings(matches_path: Path | None, season: int) -> int:
    matches = match_table(matches_path)
    played = matches.loc[matches["season"] == season]
    if played.empty:
        raise SystemExit(f"no matches in {season_label(season)}")

    instant = pd.Timestamp(as_of_instant(pd.to_datetime(played["date"]).min().date()))
    pool = ELO.ratings_at(Evidence.before(matches, instant))
    print(f"{season_label(season)} opens on {instant.date()}; the pool holds "
          f"{len(pool.clubs)} Clubs")

    arriving = newcomers(matches, season)
    print(f"\npromoted into the Premier League ({len(arriving)}):")
    for club in sorted(arriving, key=pool.rating, reverse=True):
        print(f"  {club:<20} {pool.rating(club):8.1f}  from {pool.played(club)} matches")

    holdovers = sorted(
        set(played.loc[played["division"] == "E0", "home_club"]) - set(arriving),
        key=pool.rating,
        reverse=True,
    )
    print(f"\nalready in it ({len(holdovers)}), best and worst rated:")
    for club in [*holdovers[:3], *holdovers[-3:]]:
        print(f"  {club:<20} {pool.rating(club):8.1f}  from {pool.played(club)} matches")

    coldest = min(
        (pool.played(club) for club in [*arriving, *holdovers]), default=0
    )
    print(f"\nthe least-warmed rating in the division rests on {coldest} matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
