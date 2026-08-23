"""Scoring the ledger: one line per registered Predictor over the Evaluation Window.

The scoreboard is where issue #7's first acceptance criterion is cashed in — scoring code written
once against the Predictor contract that never special-cases a Predictor. It reads ledger rows and
the match table and nothing else, so it cannot tell which store a Prediction came from or which
Predictor made it beyond the name it prints.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl import metrics, predictors
from epl.calibration import MINIMUM_SAMPLE
from epl.ledger import schema, scoreboard
from epl.predictors import Evidence
from epl.rounds import as_of_instant, round_id

CERTAIN_HOME = (1.0, 0.0, 0.0)
CERTAIN_AWAY = (0.0, 0.0, 1.0)

#: A Season of Outcomes at 45% Home, 25% Draw, 30% Away, repeated a round at a time. Exact rather
#: than sampled, so a walk-forward map fitted on whole rounds reads back those rates to the digit
#: and a correction can be worked out by hand.
SEASON_RATES = (0.45, 0.25, 0.30)
ROUND_OUTCOMES = ["H"] * 9 + ["D"] * 5 + ["A"] * 6


@pytest.fixture
def played(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """Two Premier League Fixtures of 2005/06, both Home wins."""
    return make_matches(
        {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", "outcome": "H"},
        {"season": 2005, "date": "2005-08-13", "home_club": "everton", "outcome": "H"},
    )


def _season(
    quotes: dict[str, tuple[float, float, float]], rounds: int = 40
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A Season of weekly Prediction Rounds as ledger rows, and the matches behind them.

    Long enough to get past :data:`epl.calibration.MINIMUM_SAMPLE` part-way through, which is the
    only way to see the layer both hold off and act within one scoreboard. Every Fixture is a fresh
    pair of Clubs, so nothing is superseded by accident.
    """
    kickoffs = [
        pd.Timestamp("2005-08-13") + pd.Timedelta(days=7 * number)
        for number in range(rounds)
        for _ in ROUND_OUTCOMES
    ]
    fixtures = pd.DataFrame(
        {
            "season": 2005,
            "division": "E0",
            "home_club": [f"home_{index}" for index in range(len(kickoffs))],
            "away_club": [f"away_{index}" for index in range(len(kickoffs))],
            "outcome": ROUND_OUTCOMES * rounds,
        }
    )
    rows = pd.concat(
        [
            schema.conform(
                fixtures.assign(
                    predictor=name,
                    prediction_round=[round_id(kickoff) for kickoff in kickoffs],
                    as_of_instant=[as_of_instant(kickoff) for kickoff in kickoffs],
                    kickoff=kickoffs,
                    prob_home=quote[0],
                    prob_draw=quote[1],
                    prob_away=quote[2],
                    inputs_seen=0,
                    latest_input=pd.NaT,
                ).drop(columns=["outcome"])
            )
            for name, quote in quotes.items()
        ],
        ignore_index=True,
    )
    return schema.check(rows), fixtures


def _rows(
    predictor: object, fixtures: pd.DataFrame, as_of: str = "2005-08-12"
) -> pd.DataFrame:
    return schema.predictions_for(predictor, fixtures, Evidence.before(fixtures.head(0), as_of))


class TestTheScoreboard:
    def test_it_scores_each_predictor_on_the_outcomes_that_happened(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Both Fixtures were Home wins: certainty on Home scores 0.00 RPS, certainty on Away
        scores 1.00. Worked by hand in tests/metrics/test_scores.py."""
        rows = pd.concat(
            [
                _rows(make_predictor("oracle", CERTAIN_HOME), played),
                _rows(make_predictor("fool", CERTAIN_AWAY), played),
            ],
            ignore_index=True,
        )

        board = scoreboard.build(rows, played)

        assert list(board["predictor"]) == ["oracle", "fool"]
        assert list(board["rps"]) == [0.0, 1.0]
        assert list(board["fixtures"]) == [2, 2]

    def test_the_best_predictor_is_listed_first(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """RPS is primary and lower is better, so the scoreboard is sorted by it (CLAUDE.md)."""
        rows = pd.concat(
            [
                _rows(make_predictor("fool", CERTAIN_AWAY), played),
                _rows(make_predictor("oracle", CERTAIN_HOME), played),
            ],
            ignore_index=True,
        )

        assert list(scoreboard.build(rows, played)["predictor"]) == ["oracle", "fool"]

    def test_it_reports_every_metric_the_module_produces(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        board = scoreboard.build(_rows(make_predictor("oracle", CERTAIN_HOME), played), played)

        assert list(board.columns) == list(scoreboard.SCOREBOARD_COLUMNS)
        assert board.iloc[0]["accuracy"] == 1.0
        assert board.iloc[0]["brier"] == 0.0

    def test_a_fixture_with_no_outcome_yet_is_not_scored(
        self, make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        """Sealed Predictions exist before their Fixtures are played. An unplayed Fixture is absent
        from the scoreboard, not scored as a miss."""
        fixtures = make_matches(
            {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", "outcome": "H"},
            {"season": 2005, "date": "2005-08-13", "home_club": "everton", "outcome": pd.NA},
        )

        board = scoreboard.build(_rows(make_predictor(), fixtures), fixtures)

        assert board.iloc[0]["fixtures"] == 1

    def test_a_postponed_fixture_is_still_the_fixture_that_was_predicted(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """The join is on the Club pairing, not the date. A Fixture moved after its Prediction was
        sealed would otherwise silently drop off the scoreboard."""
        rows = _rows(make_predictor("oracle", CERTAIN_HOME), played)
        rearranged = played.assign(date=pd.to_datetime("2005-09-20").date())

        assert scoreboard.build(rows, rearranged).iloc[0]["fixtures"] == 2

    def test_only_the_latest_prediction_for_a_fixture_counts(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Superseding is how a sealed mistake is corrected (ADR 0005): a new row at a new As-Of
        Instant. Scoring both would count the Fixture twice and average in the mistake."""
        wrong = _rows(make_predictor("oracle", CERTAIN_AWAY), played)
        corrected = _rows(make_predictor("oracle", CERTAIN_HOME), played)
        corrected["as_of_instant"] += pd.Timedelta(hours=6)

        board = scoreboard.build(pd.concat([wrong, corrected], ignore_index=True), played)

        assert board.iloc[0]["fixtures"] == 2
        assert board.iloc[0]["rps"] == 0.0

    def test_seasons_outside_the_evaluation_window_are_not_scored(
        self, make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        """Nothing in the Burn-In Window is ever scored — that is what keeps the Evaluation Window
        uncontaminated (ADR 0008)."""
        burn_in = make_matches(
            {"season": 2002, "date": "2002-08-17", "home_club": "arsenal", "outcome": "H"}
        )

        board = scoreboard.build(_rows(make_predictor(), burn_in, "2002-08-16"), burn_in)

        assert board.empty

    def test_an_empty_ledger_gives_an_empty_scoreboard(self, played: pd.DataFrame) -> None:
        board = scoreboard.build(schema.empty(), played)

        assert board.empty
        assert list(board.columns) == list(scoreboard.SCOREBOARD_COLUMNS)


class TestBothSidesOfTheCalibration:
    """Every metric twice, pre- and post-calibration, and the size of the correction beside them.

    ADR 0006: a calibration layer can mask a broken model by correcting its symptoms, so publishing
    only the post-calibration column would turn a warning into a silent fix. Issue #10 asks for both
    columns and for the correction itself to be a reported number.
    """

    def test_every_metric_is_reported_twice(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        board = scoreboard.build(_rows(make_predictor(), played), played)

        for metric in ("rps", "brier", "log_loss", "accuracy", "ece"):
            assert metric in board.columns
            assert f"{scoreboard.CALIBRATED_PREFIX}{metric}" in board.columns

    def test_a_short_track_record_is_left_uncalibrated(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Fitted on out-of-sample Predictions only, so the first rounds have nothing to be
        corrected against. Both columns are still reported — equal, and honestly so."""
        board = scoreboard.build(_rows(make_predictor(), played), played)

        assert board.loc[0, "corrected"] == 0
        assert board.loc[0, "correction"] == 0.0
        assert board.loc[0, "calibrated_rps"] == board.loc[0, "rps"]

    def test_an_overconfident_predictor_is_pulled_back_toward_the_base_rate(self) -> None:
        """Forty rounds of (0.9, 0.05, 0.05) over Outcomes that ran 45/25/30. Once a map has
        `MINIMUM_SAMPLE` Predictions behind it, it reads those rates straight back, so each
        corrected Prediction moves half of |0.45-0.9| + |0.25-0.05| + |0.30-0.05| = 0.45.

        Nineteen rounds of twenty Fixtures are needed before the pool reaches 380, so 380 of the
        800 pass through and 420 are corrected — which is what pins the cut being strict."""
        rows, matches = _season({"confident": (0.9, 0.05, 0.05)})

        board = scoreboard.build(rows, matches, seasons=[2005])

        assert board.loc[0, "fixtures"] == 800
        assert board.loc[0, "corrected"] == 800 - MINIMUM_SAMPLE
        assert board.loc[0, "correction"] == pytest.approx(0.45 * 420 / 800)
        assert board.loc[0, "calibrated_rps"] < board.loc[0, "rps"]

    def test_each_predictor_is_calibrated_by_its_own_map(self) -> None:
        """A map is a statement about one Predictor's own quotes. Pooling several into one would
        correct each of them with the others' mistakes — so the honest Predictor here must not move
        an inch for sharing a slate with a wildly overconfident one."""
        rows, matches = _season(
            {"honest": SEASON_RATES, "confident": (0.9, 0.05, 0.05)}
        )

        board = scoreboard.build(rows, matches, seasons=[2005]).set_index("predictor")

        assert board.loc["honest", "correction"] == pytest.approx(0.0)
        assert board.loc["confident", "correction"] > 0.2

    def test_the_board_is_ordered_by_the_score_the_predictor_earned(self) -> None:
        """Sorted on pre-calibration RPS. Ordering by what the layer did would let the layer decide
        who is winning, and the layer is not a Predictor."""
        rows, matches = _season(
            {"honest": SEASON_RATES, "confident": (0.9, 0.05, 0.05)}
        )

        board = scoreboard.build(rows, matches, seasons=[2005])

        assert list(board["rps"]) == sorted(board["rps"])
        assert list(board["predictor"]) == ["honest", "confident"]


class TestTheCalibratedPredictions:
    def test_the_raw_prediction_survives_beside_the_calibrated_one(self) -> None:
        """Both halves of every comparison have to survive to be reported, so the calibrated
        Prediction is added in new columns rather than written over the stored one."""
        rows, matches = _season({"confident": (0.9, 0.05, 0.05)})

        scored = scoreboard.calibrated_predictions(rows, matches, seasons=[2005])

        assert set(scoreboard.CALIBRATED_PROBABILITY_COLUMNS) <= set(scored.columns)
        assert list(scored["prob_home"].unique()) == [0.9]
        assert scored.loc[scored["corrected"], "calibrated_prob_home"].unique() == pytest.approx(
            [SEASON_RATES[0]]
        )

    def test_an_empty_ledger_still_carries_the_columns(self, played: pd.DataFrame) -> None:
        scored = scoreboard.calibrated_predictions(schema.empty(), played)

        assert scored.empty
        assert set(scoreboard.CALIBRATED_PROBABILITY_COLUMNS) <= set(scored.columns)


class TestThePublishedReliabilityDiagrams:
    """Issue #10's fourth acceptance criterion: ten bins, per Predictor, in both forms.

    The scoreboard's `ece` is this table as one number, and one number cannot say *where* a
    Predictor is off — nor whether a correction fixed one probability band by breaking another.
    """

    def test_there_is_one_diagram_per_predictor_per_form(self) -> None:
        rows, matches = _season(
            {"honest": SEASON_RATES, "confident": (0.9, 0.05, 0.05)}
        )

        diagrams = scoreboard.reliability(rows, matches, seasons=[2005])

        assert list(diagrams.columns) == list(scoreboard.RELIABILITY_REPORT_COLUMNS)
        assert len(diagrams) == 2 * len(scoreboard.FORMS) * metrics.BINS
        assert sorted(set(diagrams["form"])) == sorted(scoreboard.FORMS)
        assert sorted(set(diagrams["predictor"])) == ["confident", "honest"]

    def test_the_two_forms_differ_where_the_layer_acted(self) -> None:
        rows, matches = _season({"confident": (0.9, 0.05, 0.05)})

        diagrams = scoreboard.reliability(rows, matches, seasons=[2005]).set_index("form")
        occupied = diagrams.loc[diagrams["predictions"] > 0]

        assert occupied.loc["raw", "mean_predicted"].max() == pytest.approx(0.9)
        assert occupied.loc["calibrated", "mean_predicted"].max() < 0.9

    def test_an_empty_ledger_publishes_an_empty_table(self, played: pd.DataFrame) -> None:
        diagrams = scoreboard.reliability(schema.empty(), played)

        assert diagrams.empty
        assert list(diagrams.columns) == list(scoreboard.RELIABILITY_REPORT_COLUMNS)

    def test_they_are_written_beside_the_scoreboard(
        self, project_root: Path, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        diagrams = scoreboard.reliability(_rows(make_predictor(), played), played)

        written = scoreboard.write_reliability(diagrams)

        assert written.name == "reliability.csv"
        assert written.parent.name == "outputs"
        assert len(pd.read_csv(written)) == len(scoreboard.FORMS) * metrics.BINS


class TestTheScoreboardFile:
    def test_it_is_written_where_regenerable_output_belongs(
        self, project_root: Path, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """A scoreboard is derived from the stores, so it is regenerable and gitignored — the same
        reasoning ADR 0005 applies to outputs/backtest/ itself. It sits beside the two stores
        rather than inside one, so that reading a store never picks up a report."""
        board = scoreboard.build(_rows(make_predictor(), played), played)

        written = scoreboard.write(board)

        assert written.name == "scoreboard.csv"
        assert written.parent.name == "outputs"
        assert pd.read_csv(written)["predictor"].tolist() == ["fixed"]


class TestTheCaveatsTravelWithTheScores:
    """A Predictor may carry a note, and it has to reach every place its score is reported.

    The Ceiling Line is why (issue #8): a line that knows team news the model cannot have, scored
    over a shorter span than everything else on the board, is a misleading number rather than an
    incomplete one if it appears bare. The mechanism is generic — the scoreboard reads a note off
    whatever Predictor the row names, and has no idea which one that is.
    """

    def test_a_note_reaches_the_board(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        caveated = make_predictor("caveated", CERTAIN_HOME)
        caveated.note = "knows something the others do not"
        predictors.register(caveated)

        board = scoreboard.build(_rows(caveated, played), played, seasons=[2005])

        assert list(board["note"]) == ["knows something the others do not"]

    def test_a_predictor_without_one_gets_a_blank(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        plain = predictors.register(make_predictor("plain", CERTAIN_HOME))

        board = scoreboard.build(_rows(plain, played), played, seasons=[2005])

        assert list(board["note"]) == [""]

    def test_a_stored_predictor_nobody_registered_still_scores(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        """A ledger file can outlive the code that wrote it (ADR 0005). Scoring must not depend on
        the Predictor still being registered — only the note does, and its absence is a blank."""
        board = scoreboard.build(_rows(make_predictor("ghost"), played), played, seasons=[2005])

        assert board.loc[0, "fixtures"] == 2
        assert board.loc[0, "note"] == ""
