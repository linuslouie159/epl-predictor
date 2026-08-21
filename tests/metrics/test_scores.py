"""Every score the project reports, against examples worked out by hand.

Not one expected value here was produced by the code under test. A metric checked against its own
output only proves the code agrees with itself, and every comparison this project publishes —
model against Market Line against Pundit — rests on these numbers being right.

The anchors, from the spec and ADR 0006: a certain Prediction wrong in the worst direction scores
exactly 1.00 RPS, wrong by one ordinal step scores 0.50, and correct scores 0.00.
"""

from __future__ import annotations

import pytest

from epl import metrics

CERTAIN_HOME = [1.0, 0.0, 0.0]
CERTAIN_DRAW = [0.0, 1.0, 0.0]
CERTAIN_AWAY = [0.0, 0.0, 1.0]
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


class TestRankedProbabilityScore:
    """RPS by hand.

    For Prediction ``p`` over (Home, Draw, Away) and Outcome ``o``, with cumulative sums taken
    over the first two categories::

        RPS = ((P1 - O1)^2 + (P2 - O2)^2) / 2

    Certain Home, Outcome Away: P = (1, 1), O = (0, 0) -> (1 + 1) / 2 = 1.00
    Certain Home, Outcome Draw: P = (1, 1), O = (0, 1) -> (1 + 0) / 2 = 0.50
    Certain Home, Outcome Home: P = (1, 1), O = (1, 1) -> (0 + 0) / 2 = 0.00
    """

    def test_a_correct_certain_prediction_scores_zero(self) -> None:
        assert metrics.rps(CERTAIN_HOME, "H") == 0.0

    def test_a_certain_prediction_wrong_by_one_ordinal_step_scores_a_half(self) -> None:
        assert metrics.rps(CERTAIN_HOME, "D") == 0.5

    def test_a_certain_prediction_wrong_in_the_worst_direction_scores_one(self) -> None:
        assert metrics.rps(CERTAIN_HOME, "A") == 1.0

    def test_the_worst_direction_is_symmetric(self) -> None:
        assert metrics.rps(CERTAIN_AWAY, "H") == 1.0

    def test_a_certain_draw_can_never_be_wrong_by_two_steps(self) -> None:
        """Draw sits between Home and Away, so it is never further than one step from either."""
        assert metrics.rps(CERTAIN_DRAW, "H") == 0.5
        assert metrics.rps(CERTAIN_DRAW, "A") == 0.5

    def test_a_uniform_prediction_on_a_home_win(self) -> None:
        """P = (1/3, 2/3), O = (1, 1) -> ((2/3)^2 + (1/3)^2) / 2 = (4/9 + 1/9) / 2 = 5/18."""
        assert metrics.rps(UNIFORM, "H") == pytest.approx(5 / 18)

    def test_a_uniform_prediction_on_a_draw(self) -> None:
        """P = (1/3, 2/3), O = (0, 1) -> ((1/3)^2 + (1/3)^2) / 2 = (1/9 + 1/9) / 2 = 1/9."""
        assert metrics.rps(UNIFORM, "D") == pytest.approx(1 / 9)

    def test_a_uniform_prediction_on_an_away_win(self) -> None:
        """P = (1/3, 2/3), O = (0, 0) -> ((1/3)^2 + (2/3)^2) / 2 = (1/9 + 4/9) / 2 = 5/18."""
        assert metrics.rps(UNIFORM, "A") == pytest.approx(5 / 18)

    def test_a_realistic_prediction_on_a_home_win(self) -> None:
        """(0.5, 0.3, 0.2) on Home: P = (0.5, 0.8), O = (1, 1) -> (0.25 + 0.04) / 2 = 0.145."""
        assert metrics.rps([0.5, 0.3, 0.2], "H") == pytest.approx(0.145)

    def test_a_realistic_prediction_on_an_away_win(self) -> None:
        """(0.5, 0.3, 0.2) on Away: P = (0.5, 0.8), O = (0, 0) -> (0.25 + 0.64) / 2 = 0.445."""
        assert metrics.rps([0.5, 0.3, 0.2], "A") == pytest.approx(0.445)


class TestRpsRespectsTheOrdinalScale:
    """The reason RPS is primary and accuracy is not (CLAUDE.md)."""

    def test_being_wrong_by_two_steps_costs_more_than_being_wrong_by_one(self) -> None:
        assert metrics.rps(CERTAIN_HOME, "A") > metrics.rps(CERTAIN_HOME, "D")

    def test_being_wrong_by_one_step_costs_more_than_being_right(self) -> None:
        assert metrics.rps(CERTAIN_HOME, "D") > metrics.rps(CERTAIN_HOME, "H")

    def test_shifting_probability_towards_the_outcome_always_helps(self) -> None:
        assert metrics.rps([0.6, 0.3, 0.1], "H") < metrics.rps([0.4, 0.3, 0.3], "H")


class TestScoringManyPredictionsAtOnce:
    def test_returns_the_mean_over_the_fixtures(self) -> None:
        """1.00 and 0.00 by hand, so the mean is 0.50."""
        assert metrics.rps([CERTAIN_HOME, CERTAIN_HOME], ["A", "H"]) == 0.5

    def test_one_prediction_scores_the_same_alone_as_in_a_batch(self) -> None:
        assert metrics.rps([CERTAIN_HOME], ["D"]) == metrics.rps(CERTAIN_HOME, "D")

    def test_rejects_a_prediction_count_that_does_not_match_the_outcome_count(self) -> None:
        with pytest.raises(metrics.MetricsError, match="1 Predictions against 2 Outcomes"):
            metrics.rps([CERTAIN_HOME], ["H", "D"])


class TestBrierScore:
    """Multi-category Brier: the squared error summed over all three Outcomes.

    Summed, not averaged, which is Brier's original multi-category definition and puts the worst
    possible score at 2.00 rather than 1.00.

    Certain Home, Outcome Away: (1-0)^2 + (0-0)^2 + (0-1)^2 = 2.00
    Uniform, any Outcome:       (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 2/3
    """

    def test_a_correct_certain_prediction_scores_zero(self) -> None:
        assert metrics.brier(CERTAIN_HOME, "H") == 0.0

    def test_a_certain_prediction_wrong_in_the_worst_direction_scores_two(self) -> None:
        assert metrics.brier(CERTAIN_HOME, "A") == 2.0

    def test_a_uniform_prediction_scores_two_thirds_whatever_happens(self) -> None:
        for outcome in ("H", "D", "A"):
            assert metrics.brier(UNIFORM, outcome) == pytest.approx(2 / 3)

    def test_a_realistic_prediction_on_a_home_win(self) -> None:
        """(0.5, 0.3, 0.2) on Home: 0.25 + 0.09 + 0.04 = 0.38."""
        assert metrics.brier([0.5, 0.3, 0.2], "H") == pytest.approx(0.38)

    def test_a_realistic_prediction_on_an_away_win(self) -> None:
        """(0.5, 0.3, 0.2) on Away: 0.25 + 0.09 + 0.64 = 0.98."""
        assert metrics.brier([0.5, 0.3, 0.2], "A") == pytest.approx(0.98)

    def test_it_cannot_tell_one_ordinal_step_from_two(self) -> None:
        """Exactly why RPS is primary and Brier is the cross-check (CLAUDE.md)."""
        assert metrics.brier(CERTAIN_HOME, "D") == metrics.brier(CERTAIN_HOME, "A")
        assert metrics.rps(CERTAIN_HOME, "D") != metrics.rps(CERTAIN_HOME, "A")


class TestLogLoss:
    """The negative log of the probability placed on what actually happened."""

    def test_a_correct_certain_prediction_scores_zero(self) -> None:
        assert metrics.log_loss(CERTAIN_HOME, "H") == 0.0

    def test_a_uniform_prediction_scores_the_log_of_three(self) -> None:
        """-ln(1/3) = ln 3 = 1.0986122886681098."""
        assert metrics.log_loss(UNIFORM, "H") == pytest.approx(1.0986122886681098)

    def test_a_realistic_prediction_scores_the_log_of_what_it_gave_the_outcome(self) -> None:
        """-ln(0.5) = 0.6931471805599453; -ln(0.2) = 1.6094379124341003."""
        assert metrics.log_loss([0.5, 0.3, 0.2], "H") == pytest.approx(0.6931471805599453)
        assert metrics.log_loss([0.5, 0.3, 0.2], "A") == pytest.approx(1.6094379124341003)

    def test_a_certain_prediction_that_is_wrong_is_penalised_but_stays_finite(self) -> None:
        """Unclipped this is infinite, and one such row would destroy a whole scoreboard."""
        assert metrics.LOG_LOSS_FLOOR == 1e-15
        assert metrics.log_loss(CERTAIN_HOME, "A") == pytest.approx(34.538776394910684)

    def test_the_clip_leaves_every_ordinary_prediction_untouched(self) -> None:
        assert metrics.log_loss([0.5, 0.3, 0.2], "H") == pytest.approx(0.6931471805599453)

    def test_it_punishes_stated_certainty_far_harder_than_rps_does(self) -> None:
        """Why a Pundit's Scoreline is scored on RPS, not log loss (ADR 0003)."""
        assert metrics.log_loss(CERTAIN_HOME, "A") > 30 * metrics.rps(CERTAIN_HOME, "A")


class TestTopPick:
    def test_names_the_most_likely_outcome(self) -> None:
        assert list(metrics.top_pick([0.5, 0.3, 0.2])) == ["H"]
        assert list(metrics.top_pick([0.2, 0.5, 0.3])) == ["D"]
        assert list(metrics.top_pick([0.2, 0.3, 0.5])) == ["A"]

    def test_picks_for_a_whole_slate_at_once(self) -> None:
        picks = metrics.top_pick([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        assert list(picks) == ["H", "A"]

    def test_needs_no_outcome_because_it_is_not_a_score(self) -> None:
        assert list(metrics.top_pick(UNIFORM)) == ["H"]


class TestAccuracy:
    """Also called the top-pick hit rate: how often the most likely Outcome was the one that
    happened. Reported for lay explanation only, and never as the headline (CLAUDE.md)."""

    def test_a_correct_top_pick_is_a_hit(self) -> None:
        assert metrics.accuracy([0.5, 0.3, 0.2], "H") == 1.0

    def test_a_wrong_top_pick_is_a_miss(self) -> None:
        assert metrics.accuracy([0.5, 0.3, 0.2], "D") == 0.0

    def test_it_ignores_how_confident_the_pick_was(self) -> None:
        """A 0.99 Home call and a 0.34 Home call both count as one hit — RPS is what separates
        them, which is why accuracy is never the headline."""
        assert metrics.accuracy([0.99, 0.005, 0.005], "H") == metrics.accuracy(
            [0.34, 0.33, 0.33], "H"
        )

    def test_the_hit_rate_over_a_slate_is_the_fraction_that_came_in(self) -> None:
        picks = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]
        assert metrics.accuracy(picks, ["H", "H", "H", "A"]) == 0.5

    def test_a_tied_top_pick_earns_only_its_share_of_the_credit(self) -> None:
        """A tie is not a pick. Awarding the whole hit to whichever Outcome happens to be listed
        first would hand a Predictor that never picked anything a Home-win-shaped hit rate."""
        assert metrics.accuracy([0.4, 0.4, 0.2], "H") == 0.5
        assert metrics.accuracy([0.4, 0.4, 0.2], "D") == 0.5
        assert metrics.accuracy([0.4, 0.4, 0.2], "A") == 0.0

    def test_a_uniform_prediction_scores_one_third_whatever_happens(self) -> None:
        for outcome in ("H", "D", "A"):
            assert metrics.accuracy(UNIFORM, outcome) == pytest.approx(1 / 3)

    def test_hits_are_available_per_fixture(self) -> None:
        hits = metrics.hits([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], ["H", "H"])
        assert list(hits) == [1.0, 0.0]


class TestPerPrediction:
    """The per-Fixture table, for surfacing a Predictor's best and worst calls."""

    def test_returns_one_row_per_fixture(self) -> None:
        table = metrics.per_prediction([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert len(table) == 2

    def test_carries_every_score_and_the_outcome_it_was_scored_against(self) -> None:
        table = metrics.per_prediction([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert list(table.columns) == list(metrics.PER_PREDICTION_COLUMNS)

    def test_scores_each_fixture_the_way_the_scalar_metrics_do(self) -> None:
        table = metrics.per_prediction([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert list(table["rps"]) == [0.0, 1.0]
        assert list(table["brier"]) == [0.0, 2.0]
        assert list(table["hit"]) == [1.0, 0.0]

    def test_the_pick_and_the_hit_disagree_on_a_tie_and_that_is_deliberate(self) -> None:
        """``top_pick`` must name one Outcome for display; ``hit`` refuses to award a whole hit to
        a Predictor that never picked anything. The row shows both rather than hiding either."""
        table = metrics.per_prediction([[0.4, 0.4, 0.2]], ["D"])
        assert table.loc[0, "top_pick"] == "H"
        assert table.loc[0, "hit"] == 0.5

    def test_names_the_outcome_and_the_pick_in_words(self) -> None:
        table = metrics.per_prediction([CERTAIN_HOME], ["A"])
        assert table.loc[0, "outcome"] == "A"
        assert table.loc[0, "top_pick"] == "H"


class TestScorecard:
    """One call, every headline number, for the slate of Fixtures handed in."""

    def test_reports_the_hand_worked_means_of_a_two_fixture_slate(self) -> None:
        """One certain call right and one certain call wrong in the worst direction:
        RPS (0.00 + 1.00) / 2, Brier (0.00 + 2.00) / 2, accuracy (1 + 0) / 2."""
        card = metrics.score([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert card.rps == 0.5
        assert card.brier == 1.0
        assert card.accuracy == 0.5

    def test_reports_the_mean_log_loss_with_the_floor_applied(self) -> None:
        """(0 + 34.538776394910684) / 2."""
        card = metrics.score([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert card.log_loss == pytest.approx(17.269388197455342)

    def test_counts_the_fixtures_it_scored(self) -> None:
        card = metrics.score([CERTAIN_HOME, CERTAIN_HOME], ["H", "A"])
        assert card.fixtures == 2

    def test_agrees_with_the_individual_metrics(self) -> None:
        predictions = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], UNIFORM]
        outcomes = ["H", "D", "A"]
        card = metrics.score(predictions, outcomes)
        assert card.rps == metrics.rps(predictions, outcomes)
        assert card.brier == metrics.brier(predictions, outcomes)
        assert card.log_loss == metrics.log_loss(predictions, outcomes)
        assert card.accuracy == metrics.accuracy(predictions, outcomes)

    def test_refuses_to_score_an_empty_slate(self) -> None:
        """An empty scoreboard row reads as a Predictor with nothing to answer for."""
        with pytest.raises(metrics.MetricsError, match="no Fixtures"):
            metrics.score([], [])


class TestAnEmptySlate:
    """Averaging nothing is undefined, and a NaN on a scoreboard reads as a real number."""

    @pytest.mark.parametrize(
        "metric", [metrics.rps, metrics.brier, metrics.log_loss, metrics.accuracy]
    )
    def test_every_headline_metric_refuses_it(self, metric: object) -> None:
        with pytest.raises(metrics.MetricsError, match="no Fixtures"):
            metric([], [])  # type: ignore[operator]

    def test_the_per_fixture_scores_are_simply_empty(self) -> None:
        assert len(metrics.rps_per_prediction([], [])) == 0
        assert metrics.per_prediction([], []).empty
