"""The shared calibration layer: isotonic maps, and the walk that fits them without hindsight.

Issue #10 asks for one calibration step that wraps every Predictor identically, fitted walk-forward
on out-of-sample Predictions only. Two things here are therefore worth more than the rest:

* a Prediction is never corrected by a map that saw its own Outcome
* the training cut is *strict*, so a Fixture kicking off at an As-Of Instant is not in the map
  applied at that instant

Expected values are worked by hand. A one-knot isotonic map is just the observed rate, so a sample
in which every Prediction is the same gives a correction anyone can check on paper.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from epl import calibration
from epl.calibration import CalibrationError, Curve, Isotonic

CONFIDENT = (0.9, 0.05, 0.05)
EVEN = (1 / 3, 1 / 3, 1 / 3)


def _instants(*days: str) -> npt.NDArray[np.datetime64]:
    return pd.to_datetime(list(days)).to_numpy()


class TestOneIsotonicMap:
    def test_an_already_calibrated_predictor_passes_through_unchanged(self) -> None:
        """Ten Predictions of (0.5, 0.3, 0.2) on five Home wins, three Draws and two Away wins.
        Every quoted probability is exactly what happened, so there is nothing to correct."""
        quoted = [(0.5, 0.3, 0.2)] * 10
        outcomes = ["H"] * 5 + ["D"] * 3 + ["A"] * 2

        calibrated = calibration.fit(quoted, outcomes).apply([(0.5, 0.3, 0.2)])

        assert calibrated[0] == pytest.approx([0.5, 0.3, 0.2])

    def test_a_systematically_overconfident_predictor_is_pulled_back(self) -> None:
        """Ten Predictions of (0.9, 0.05, 0.05) where Home came in six times, Draw twice and Away
        twice. One quoted value per Outcome means one knot each, so the map is the observed rate:
        (0.6, 0.2, 0.2)."""
        quoted = [CONFIDENT] * 10
        outcomes = ["H"] * 6 + ["D"] * 2 + ["A"] * 2

        calibrated = calibration.fit(quoted, outcomes).apply([CONFIDENT])

        assert calibrated[0] == pytest.approx([0.6, 0.2, 0.2])

    def test_it_records_how_much_it_was_fitted_on(self) -> None:
        """The sample is the map's own receipt — a correction from twelve Predictions and one from
        seven thousand are not the same claim."""
        fitted = calibration.fit([CONFIDENT] * 10, ["H"] * 6 + ["D"] * 2 + ["A"] * 2)

        assert fitted.sample == 10

    def test_it_refuses_a_sample_it_cannot_fit(self) -> None:
        with pytest.raises(CalibrationError, match="nothing to fit"):
            calibration.fit([], [])

    def test_it_refuses_predictions_and_outcomes_of_different_lengths(self) -> None:
        with pytest.raises(CalibrationError, match="2 Predictions against 1"):
            calibration.fit([CONFIDENT, CONFIDENT], ["H"])


class TestTheMapIsMonotone:
    def test_a_ranking_the_outcomes_agree_with_is_left_alone(self) -> None:
        """Quoting 0.4 Home where Home comes in 20% of the time and 0.8 where it comes in 60% is
        badly calibrated and correctly *ordered*, so isotonic keeps both rates."""
        quoted = [(0.4, 0.3, 0.3)] * 10 + [(0.8, 0.1, 0.1)] * 10
        outcomes = ["H"] * 2 + ["A"] * 8 + ["H"] * 6 + ["A"] * 4

        fitted = calibration.fit(quoted, outcomes)

        assert fitted.curves[0]([0.4, 0.8]) == pytest.approx([0.2, 0.6])

    def test_a_ranking_the_outcomes_contradict_is_pooled_flat(self) -> None:
        """The same two bands with the rates the other way round. A monotone map cannot say 0.4 is
        worth more than 0.8, so the pool-adjacent-violators step averages the two to 0.4."""
        quoted = [(0.4, 0.3, 0.3)] * 10 + [(0.8, 0.1, 0.1)] * 10
        outcomes = ["H"] * 6 + ["A"] * 4 + ["H"] * 2 + ["A"] * 8

        fitted = calibration.fit(quoted, outcomes)

        assert fitted.curves[0]([0.4, 0.8]) == pytest.approx([0.4, 0.4])

    def test_a_quoted_value_between_two_knots_lands_between_them(self) -> None:
        curve = Curve(quoted=(0.2, 0.6), happened=(0.1, 0.5))

        assert curve([0.4]) == pytest.approx([0.3])

    def test_a_quoted_value_outside_the_knots_is_clipped_to_the_nearest(self) -> None:
        """A Predictor may quote a probability it has never quoted before. Extending a monotone map
        past its own evidence would invent a correction, so the end knot is held instead."""
        curve = Curve(quoted=(0.2, 0.6), happened=(0.1, 0.5))

        assert curve([0.0, 1.0]) == pytest.approx([0.1, 0.5])


class TestApplyingAMap:
    def test_every_calibrated_prediction_sums_to_one(self) -> None:
        """Three independent monotone maps do not sum to one on their own. A Prediction lying
        between two the Predictor has made before reads each map at a different place, and the
        three answers have to be renormalised before the ledger will take them."""
        quoted = [(0.7, 0.2, 0.1)] * 6 + [(0.3, 0.3, 0.4)] * 6
        outcomes = ["H"] * 4 + ["D", "A"] + ["H", "D", "D", "A", "A", "A"]
        fitted = calibration.fit(quoted, outcomes)

        between = [(0.4, 0.28, 0.32)]
        unmapped = sum(
            float(curve([between[0][index]])[0])
            for index, curve in enumerate(fitted.curves)
        )

        assert unmapped != pytest.approx(1.0)
        assert fitted.apply(between).sum(axis=1) == pytest.approx([1.0])

    def test_a_prediction_every_map_sends_to_zero_keeps_its_raw_shape(self) -> None:
        """Three maps can all read zero at one point, and a row of zeros has no shape left to
        renormalise. Handing back the raw Prediction is the only answer that is still one."""
        flat = Curve(quoted=(0.0, 1.0), happened=(0.0, 0.0))

        calibrated = Isotonic(curves=(flat, flat, flat), sample=10).apply([CONFIDENT])

        assert calibrated[0] == pytest.approx(CONFIDENT)


class TestTheWalkForward:
    """The half of this module the project's one rule turns on.

    A map is fitted on (Prediction, Outcome) pairs, and an Outcome is knowable only once its Fixture
    has been played. So the pool a round may fit on is the Predictions whose Fixtures kicked off
    strictly before that round's As-Of Instant — the same cut :class:`epl.predictors.Evidence`
    applies to the corpus, for the same reason.
    """

    def test_a_prediction_is_never_corrected_by_a_map_that_saw_its_own_outcome(self) -> None:
        """The last Fixture's Outcome is flipped and its calibrated Prediction must not move. Were
        the fit to include the row it is correcting, the two runs would differ."""
        quoted = [CONFIDENT] * 11
        played = _instants(*(["2005-08-13"] * 10), "2005-08-20")
        as_of = _instants(*(["2005-08-12"] * 10), "2005-08-19")

        walked = [
            calibration.walk_forward(
                quoted, [*["H"] * 6, *["A"] * 4, last], as_of, played, minimum=10
            )
            for last in ("H", "A")
        ]

        assert walked[0].predictions[10] == pytest.approx(walked[1].predictions[10])
        assert walked[0].predictions[10] == pytest.approx([0.6, 0.0, 0.4])

    def test_the_training_cut_is_strict(self) -> None:
        """A Fixture kicking off *at* an As-Of Instant has not been played when that round is
        predicted, so it is not in the map. Only the first row here may be fitted on: with it alone
        the Home curve reads 1.0, and with the second as well it would read 0.5."""
        quoted = [(0.5, 0.3, 0.2)] * 3
        played = _instants("2005-08-12", "2005-08-16", "2005-08-20")
        as_of = _instants("2005-08-09", "2005-08-16", "2005-08-16")

        walked = calibration.walk_forward(quoted, ["H", "A", "D"], as_of, played, minimum=1)

        assert walked.predictions[2] == pytest.approx([1.0, 0.0, 0.0])

    def test_predictions_are_left_alone_until_there_is_enough_to_fit_on(self) -> None:
        """Out-of-sample only has a price: the first Predictions of the Evaluation Window have no
        prior Predictions to be corrected against, and pass through rather than being invented."""
        quoted = [CONFIDENT] * 4
        played = _instants("2005-08-13", "2005-08-20", "2005-08-27", "2005-09-03")
        as_of = _instants("2005-08-12", "2005-08-19", "2005-08-26", "2005-09-02")

        walked = calibration.walk_forward(quoted, ["H"] * 4, as_of, played, minimum=3)

        assert list(walked.fitted) == [False, False, False, True]
        assert walked.predictions[0] == pytest.approx(CONFIDENT)

    def test_a_round_is_calibrated_as_one_batch(self) -> None:
        """Every Fixture sharing an As-Of Instant is predicted together (ADR 0002), so every Fixture
        sharing one is calibrated by the same map — never by each other's Outcomes."""
        quoted = [CONFIDENT] * 12
        played = _instants(*(["2005-08-13"] * 10), "2005-08-20", "2005-08-21")
        as_of = _instants(*(["2005-08-12"] * 10), "2005-08-19", "2005-08-19")

        walked = calibration.walk_forward(
            quoted, [*["H"] * 6, *["A"] * 4, "H", "A"], as_of, played, minimum=10
        )

        assert walked.predictions[10] == pytest.approx(walked.predictions[11])

    def test_nothing_to_walk_gives_nothing_back(self) -> None:
        walked = calibration.walk_forward([], [], _instants(), _instants())

        assert walked.predictions.shape == (0, 3)
        assert walked.corrected == 0
        assert walked.correction == 0.0

    def test_it_refuses_a_walk_whose_columns_do_not_line_up(self) -> None:
        with pytest.raises(CalibrationError, match="2 Predictions against"):
            calibration.walk_forward(
                [CONFIDENT, CONFIDENT], ["H", "A"], _instants("2005-08-12"), _instants("2005-08-13")
            )


class TestTheSizeOfTheCorrection:
    """Issue #10 asks that the size of the correction be a reported number, so that a large one
    reads as a warning about the underlying model rather than as a silent fix (ADR 0006)."""

    def test_it_is_the_probability_mass_the_layer_moved(self) -> None:
        """(0.9, 0.05, 0.05) corrected to (0.6, 0.2, 0.2) takes 0.3 off Home and puts 0.15 on each
        of the others: half the total absolute change, which is 0.3. Ten of the twenty rows were
        corrected, so the mean over the slate is 0.15."""
        quoted = [CONFIDENT] * 20
        outcomes = [*["H"] * 6, *["D"] * 2, *["A"] * 2] * 2
        played = _instants(*(["2005-08-13"] * 10), *(["2005-08-20"] * 10))
        as_of = _instants(*(["2005-08-12"] * 10), *(["2005-08-19"] * 10))

        walked = calibration.walk_forward(quoted, outcomes, as_of, played, minimum=10)

        assert walked.corrected == 10
        assert walked.moved[10] == pytest.approx(0.3)
        assert walked.correction == pytest.approx(0.15)

    def test_a_layer_with_little_to_correct_reports_a_small_number(self) -> None:
        quoted = [EVEN] * 20
        outcomes = ["H", "D", "A"] * 6 + ["H", "D"]
        played = _instants(*(["2005-08-13"] * 10), *(["2005-08-20"] * 10))
        as_of = _instants(*(["2005-08-12"] * 10), *(["2005-08-19"] * 10))

        walked = calibration.walk_forward(quoted, outcomes, as_of, played, minimum=10)

        assert walked.correction < 0.05
