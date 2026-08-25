"""Command line entry point for the models.

    python -m epl.models fit          re-derive the frozen hyperparameters on the Burn-In Window
    python -m epl.models draws        the draw rate against Supremacy, predicted and observed
    python -m epl.models ratings      the pool at a Season's first Prediction Round
    python -m epl.models strengths    Dixon-Coles' attack and defence at a Season's first round
    python -m epl.models sequential   what predicting per Fixture instead of per round would buy

``fit`` is the one that keeps the frozen numbers honest. ADR 0008 puts every hyperparameter inside
2000/01-2004/05 and then freezes it as a literal in `epl.models.elo` and `epl.models.dixon_coles`,
which means the literals and the fit that produced them can drift apart without anything failing.
This runs both fits again and prints them beside what is frozen, so the two can be compared;
``tests/models/test_elo_over_the_corpus.py`` and its Dixon-Coles counterpart assert it.

``draws`` is ADR 0006's receipt. The claim is that the draw band narrows on its own as Supremacy
grows, with no hand-coded taper anywhere — so the curve is printed for every Predictor with stored
Predictions, predicted beside observed, and a taper that had stopped tapering would show up here as
a flat column rather than as a slightly worse RPS.

``ratings`` answers the two questions ADR 0004 exists for: whether promoted Clubs arrive with
ratings that differ, and how much football is behind a rating by the time it is first scored.
``strengths`` asks the goals model the same two questions, where the answer is a pair of numbers
per Club rather than one.

``sequential`` is ADR 0002's receipt, and the one command here whose output must never be quoted as
a score. The project predicts in weekly batches on purpose, giving up Saturday's results when it
calls Monday night's game so that the model, the market and the Pundits all see the same
information. What that costs is a measurement rather than an argument, and this is where it is
taken.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from epl import metrics
from epl.ingest import match_table
from epl.ledger import backtest, scoreboard
from epl.models import ModelError, burn_in, dixon_coles, draw_curve
from epl.models.dixon_coles import DIXON_COLES, FITTED_DIVISIONS, FROZEN_DECAY
from epl.models.elo import ELO, FROZEN_LOGIT, FROZEN_SETTINGS, newcomers
from epl.paths import outputs_dir
from epl.predictors import Evidence, by_name
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
    table = sub.add_parser(
        "strengths", help="Dixon-Coles' attack and defence at a Season's first round"
    )
    table.add_argument(
        "--season",
        type=int,
        default=min(EVALUATION_WINDOW),
        help="the Season to stand at the start of (default: the first scored one)",
    )
    walk = sub.add_parser(
        "sequential", help="what predicting per Fixture instead of per round would buy (ADR 0002)"
    )
    walk.add_argument(
        "--predictor",
        default=DIXON_COLES.name,
        help=f"which Predictor to measure it on (default: {DIXON_COLES.name})",
    )
    walk.add_argument(
        "--seasons",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="narrow the span, inclusive (default: the whole Evaluation Window)",
    )

    args = parser.parse_args(argv)
    if args.command == "fit":
        return _fit(args.matches)
    if args.command == "draws":
        return _draws(args.matches, args.predictor)
    if args.command == "ratings":
        return _ratings(args.matches, args.season)
    if args.command == "strengths":
        return _strengths(args.matches, args.season)
    if args.command == "sequential":
        return _sequential(args.matches, args.predictor, args.seasons)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _fit(matches_path: Path | None) -> int:
    """Both fits, each beside the literals it was frozen into.

    Dixon-Coles' half-life is the slow half, and unavoidably so: Elo's constants can be judged by
    replaying the Burn-In pyramid once per candidate, but a half-life can only be judged by
    *predicting* with it, so every candidate walks the window's 189 Prediction Rounds and refits at
    each. That is minutes rather than seconds, and it is the price of the one hyperparameter here
    that a single replay cannot choose.
    """
    matches = match_table(matches_path)
    window = (
        f"{season_label(min(burn_in.FITTING_SEASONS))}-"
        f"{season_label(max(burn_in.FITTING_SEASONS))}"
    )
    print(f"fitted on {window}, warmed from {season_label(min(burn_in.FITTING_SEASONS) - 1)}")
    print(f"  the floor on the same sample: {burn_in.base_rate_rps(matches):.5f} RPS")

    # Each half is reported whether or not the other one could be taken, and a half that could not
    # be taken sets the exit code. One search running to the wall of its own grid is a real answer
    # about that search, and it is not a reason to withhold the other model's numbers.
    return 0 if all([_fit_elo(matches), _fit_decay_on(matches)]) else 1


def _fit_elo(matches: pd.DataFrame) -> bool:
    print("\nelo")
    try:
        fitted = burn_in.fit(matches)
    except ModelError as refused:
        print(f"  could not be taken on this corpus: {refused}")
        return False

    print(f"  found:  {fitted.describe()}")
    print(f"  frozen: k={FROZEN_SETTINGS.k:g} home_advantage={FROZEN_SETTINGS.home_advantage:g} "
          f"scale={FROZEN_LOGIT.scale:.4f} "
          f"cutpoints=({FROZEN_LOGIT.cutpoints[0]:.6f}, {FROZEN_LOGIT.cutpoints[1]:.6f})")

    # Named rather than asserted: this command reports, and the test is what fails. A reader who
    # runs it after changing the ingest wants to see *what* moved, not just that something did.
    if (fitted.settings, fitted.logit) != (FROZEN_SETTINGS, FROZEN_LOGIT):
        print("  the fit and the frozen literals differ — see epl.models.elo.FROZEN_SETTINGS")
    return True


def _fit_decay_on(matches: pd.DataFrame) -> bool:
    print("\ndixon_coles")
    try:
        decayed = burn_in.fit_decay(matches)
    except ModelError as refused:
        print(f"  could not be taken on this corpus: {refused}")
        return False

    print(f"  found:  {decayed.describe()}")
    print(f"  frozen: half_life={FROZEN_DECAY.half_life_days:g} days "
          f"(horizon {FROZEN_DECAY.horizon:.0f}) over {'+'.join(FITTED_DIVISIONS)}")
    if decayed.decay != FROZEN_DECAY or decayed.divisions != FITTED_DIVISIONS:
        print("  the fit and the frozen literal differ — see epl.models.dixon_coles.FROZEN_DECAY")
    return True


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


def _strengths(matches_path: Path | None, season: int) -> int:
    """Dixon-Coles' own view of the pyramid at the moment a Season opens.

    ADR 0004's two questions, asked of the goals model: do promoted Clubs arrive with strengths that
    differ, and how much football is behind them. The answer here is a pair of numbers per Club
    rather than one — an attack and a defence — and the promoted Clubs are the reason the fit spans
    four tiers at all.
    """
    matches = match_table(matches_path)
    played = matches.loc[matches["season"] == season]
    if played.empty:
        raise SystemExit(f"no matches in {season_label(season)}")

    instant = pd.Timestamp(as_of_instant(pd.to_datetime(played["date"]).min().date()))
    sample = DIXON_COLES.sample_at(Evidence.before(matches, instant))
    found = dixon_coles.fit(sample)
    table = found.table().set_index("club")

    print(
        f"{season_label(season)} opens on {instant.date()}; the fit holds {len(found.clubs)} Clubs "
        f"from {len(sample)} matches, weighted back {DIXON_COLES.decay.horizon:.0f} days"
    )
    print(f"home advantage {found.home_advantage:+.4f} log-goals "
          f"(x{np.exp(found.home_advantage):.3f}), low-score correction {found.correction:+.4f}")
    # Both are log-goals against an average Club, and both read upward: a higher attack scores more
    # and a higher defence concedes less, because the away rate is exp(attack - defence).
    print("attack and defence are both log-goals, and higher is better in both")

    arriving = newcomers(matches, season)
    print(f"\npromoted into the Premier League ({len(arriving)}):")
    for club in sorted(arriving, key=lambda name: -table.loc[name, "attack"]):
        print(f"  {club:<20} attack {table.loc[club, 'attack']:+.3f}  "
              f"defence {table.loc[club, 'defence']:+.3f}")

    holdovers = sorted(
        set(played.loc[played["division"] == "E0", "home_club"]) - set(arriving),
        key=lambda name: -table.loc[name, "attack"],
    )
    print(f"\nalready in it ({len(holdovers)}), best and worst attacks:")
    for club in [*holdovers[:3], *holdovers[-3:]]:
        print(f"  {club:<20} attack {table.loc[club, 'attack']:+.3f}  "
              f"defence {table.loc[club, 'defence']:+.3f}")
    return 0


def _sequential(
    matches_path: Path | None, predictor: str, seasons: list[int] | None
) -> int:
    """ADR 0002's receipt: what the weekly batch gives up, measured rather than argued.

    Printed over two spans on purpose. Football-Data records no kickoff time before 2019/20, so
    across most of the window a Fixture sits at midnight on its own day and cannot see the earlier
    kickoffs of that day either; from 2019/20 every Premier League Fixture is timed. Read the second
    row, and read it as generous — :func:`epl.ledger.backtest.sequential` explains why the timed
    reading is an upper bound rather than an estimate.

    Neither number is a score. The batch column *is* the Predictor's score and is on the scoreboard;
    the sequential column is what a model that broke the comparison would get, and it exists so that
    the size of the choice is on the record.
    """
    matches = match_table(matches_path)
    window = range(min(EVALUATION_WINDOW), max(EVALUATION_WINDOW) + 1)
    if seasons is not None:
        window = range(seasons[0], seasons[1] + 1)

    # Said before the walk rather than after it. Every Fixture is predicted twice and the second
    # reading needs one fit per distinct kickoff in a round, so this is several times a backfill —
    # about half an hour for Dixon-Coles over the whole window, and `--seasons` is how to want less.
    print(
        f"predicting {season_label(window.start)}-{season_label(window.stop - 1)} twice over: "
        "once per Prediction Round and once per kickoff. Several times the cost of a backfill"
    )
    readings = backtest.sequential(by_name(predictor), matches, seasons=window)
    if readings.empty:
        print(f"{predictor} covers nothing in {season_label(window.start)}-"
              f"{season_label(window.stop - 1)}")
        return 0

    destination = outputs_dir() / SEQUENTIAL_REPORT
    destination.parent.mkdir(parents=True, exist_ok=True)
    readings.to_csv(destination, index=False, lineterminator="\n")

    print(f"{predictor}: predicting per Fixture instead of per Prediction Round (ADR 0002)")
    print("a diagnostic, never a score — the batch column is the one on the scoreboard\n")
    print(_sequential_table(readings).to_string(index=False, float_format=_four_places))
    print(
        f"\nno kickoff time is recorded before {season_label(TIMED_FROM)}, so a Fixture before it "
        "cannot see its own day;\nafter it, a cut at kickoff can see a match still being played, "
        "so `withheld` there is an upper bound"
    )
    print(f"\nwritten to {destination}")
    return 0


#: Where the per-Fixture diagnostic is published. Beside the scoreboard rather than inside either
#: store, and gitignored, because it is derived and regenerable (ADR 0005).
SEQUENTIAL_REPORT = "sequential.csv"

#: The spans the diagnostic is reported over. "timed" is the half of the window in which
#: Football-Data records a kickoff time, and so the only half where a Fixture's own cut can see the
#: earlier kickoffs of its own day.
TIMED_FROM = 2019


def _four_places(value: float) -> str:
    return f"{value:.4f}"


def _sequential_table(readings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, span in (
        ("whole window", readings),
        (f"{season_label(TIMED_FROM)} on", readings.loc[readings["season"] >= TIMED_FROM]),
    ):
        if span.empty:
            continue
        outcomes = span["outcome"].to_numpy(dtype=object)
        batch = metrics.rps(span[["prob_home", "prob_draw", "prob_away"]].to_numpy(float), outcomes)
        later = metrics.rps(
            span[
                ["sequential_prob_home", "sequential_prob_draw", "sequential_prob_away"]
            ].to_numpy(float),
            outcomes,
        )
        rows.append(
            {
                "over": label,
                "fixtures": len(span),
                "batch_rps": batch,
                "sequential_rps": later,
                "withheld": batch - later,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
