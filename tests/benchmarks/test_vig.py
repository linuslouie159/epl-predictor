"""Vig removal: three methods behind one interface, none of them trusted.

Issue #8 asks for normalisation, the power method and Shin's method "behind one interface, with
Shin as the default", and for the overround to be "reported alongside every Market Line ... so the
vig removal can be sanity-checked rather than trusted". These tests are that sanity check applied
to the arithmetic itself.

Every expected value here is worked out from the definition rather than read off the
implementation, in the same spirit as `tests/metrics/` — a method verified against the code that
produces it only proves the code agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from epl import metrics
from epl.benchmarks import vig

#: A book with no margin in it at all: the three raw probabilities already sum to one.
FAIR = [2.0, 4.0, 4.0]

#: The same book with a 5.26% margin loaded on evenly. Every method must strip it back to FAIR's
#: probabilities *except* where it corrects favourite-longshot bias, which is the whole difference
#: between the three.
VIGGED = [1.9, 3.8, 3.8]

#: A typical Premier League book — a favourite, a draw and a longshot, 5.56% margin. Hand-worked
#: expectations below are all derived from this one.
TYPICAL = [1.80, 3.60, 4.50]


class TestOverround:
    def test_a_fair_book_has_an_overround_of_one(self) -> None:
        assert vig.overround([FAIR]) == pytest.approx([1.0])

    def test_it_is_the_sum_of_the_raw_implied_probabilities(self) -> None:
        """1/1.9 + 1/3.8 + 1/3.8 = 0.526316 + 0.263158 + 0.263158."""
        assert vig.overround([VIGGED]) == pytest.approx([1.0526315789473684])

    def test_it_reports_one_number_per_book(self) -> None:
        assert vig.overround([FAIR, VIGGED, TYPICAL]).shape == (3,)

    def test_the_margin_is_the_overround_less_one(self) -> None:
        """The number ADR 0001's readers care about: 5.56% taken out of the typical book."""
        assert float(vig.overround([TYPICAL])[0]) - 1.0 == pytest.approx(0.0555555, abs=1e-6)


class TestEveryMethod:
    """Properties all three share. A method that breaks one of these is not a vig removal."""

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_returns_a_probability_distribution(self, method: str) -> None:
        removed = vig.remove([FAIR, VIGGED, TYPICAL], method=method)

        assert removed.shape == (3, 3)
        metrics.as_predictions(removed)  # raises if any row does not sum to one

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_leaves_a_fair_book_alone(self, method: str) -> None:
        """With no margin there is nothing to remove, so all three must agree exactly here.
        A method that moved a fair book would be inventing an opinion of its own."""
        assert vig.remove([FAIR], method=method)[0] == pytest.approx([0.5, 0.25, 0.25])

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_splits_a_symmetric_book_evenly(self, method: str) -> None:
        """Three equal prices carry no favourite, so there is no bias to correct and the three
        methods have nothing to disagree about."""
        assert vig.remove([[2.85, 2.85, 2.85]], method=method)[0] == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3]
        )

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_shorter_odds_always_mean_a_higher_probability(self, method: str) -> None:
        """Vig removal reprices a book; it may never reorder it."""
        removed = vig.remove([TYPICAL], method=method)[0]

        assert removed[0] > removed[1] > removed[2]

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_removes_the_margin_rather_than_rescaling_it(self, method: str) -> None:
        """The removed line must be strictly inside the raw one: every raw implied probability is
        inflated by the margin, so every one of them has to come down."""
        removed = vig.remove([TYPICAL], method=method)[0]

        assert (removed < 1.0 / np.asarray(TYPICAL)).all()

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_one_book_and_many_books_give_the_same_answer(self, method: str) -> None:
        """Each book is priced on its own. A method that pooled them would make a Prediction
        depend on which other Fixtures happened to be in the same Prediction Round."""
        alone = vig.remove([TYPICAL], method=method)
        together = vig.remove([FAIR, TYPICAL, VIGGED], method=method)

        assert together[1] == pytest.approx(alone[0], abs=1e-12)

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_is_deterministic(self, method: str) -> None:
        """`outputs/backtest/` is only regenerable if a rebuild writes the same bytes."""
        assert (
            vig.remove([TYPICAL], method=method) == vig.remove([TYPICAL], method=method)
        ).all()

    @pytest.mark.parametrize("method", sorted(vig.METHODS))
    def test_it_accepts_an_empty_slate(self, method: str) -> None:
        assert vig.remove(np.empty((0, 3)), method=method).shape == (0, 3)


class TestNormalisation:
    """The naive method: divide out the overround and keep every price's share of the book."""

    def test_it_divides_the_book_out_proportionally(self) -> None:
        """1/1.9 / 1.052632 = 0.5 exactly, and the two 3.8s split the rest."""
        assert vig.normalise([VIGGED])[0] == pytest.approx([0.5, 0.25, 0.25])

    def test_it_preserves_the_ratio_between_any_two_prices(self) -> None:
        """Its defining property, and its weakness: the margin is assumed to sit evenly on every
        Outcome, which is exactly what favourite-longshot bias says is false."""
        removed = vig.normalise([TYPICAL])[0]

        assert removed[0] / removed[2] == pytest.approx(4.50 / 1.80)


class TestThePowerMethod:
    def test_it_raises_the_raw_probabilities_to_a_common_exponent(self) -> None:
        """The definition: find k with sum((1/o)**k) == 1. For (1.80, 3.60, 4.50) that k is
        1.05640660, which is what the exponent below is checked against."""
        raw = 1.0 / np.asarray(TYPICAL)
        removed = vig.power([TYPICAL])[0]

        exponent = np.log(removed) / np.log(raw)
        assert exponent == pytest.approx([1.0564066] * 3, abs=1e-6)

    def test_it_is_the_hand_worked_line(self) -> None:
        assert vig.power([TYPICAL])[0] == pytest.approx(
            [0.5374384, 0.2584148, 0.2041468], abs=1e-6
        )

    def test_it_takes_more_off_the_longshot_than_normalisation_does(self) -> None:
        """Favourite-longshot bias: bookmakers load margin disproportionately onto long prices, so
        removing it proportionally leaves longshots overstated. This is what power corrects, and
        why DECISIONS.md keeps it available even though it barely moves the RPS."""
        proportional = vig.normalise([TYPICAL])[0]
        powered = vig.power([TYPICAL])[0]

        assert powered[0] > proportional[0]
        assert powered[2] < proportional[2]


class TestShin:
    def test_it_solves_shins_equation(self) -> None:
        """Shin's inverse, with B the overround: p = (sqrt(z^2 + 4(1-z)(1/o)^2/B) - z) / (2(1-z)),
        z chosen so the three sum to one. For (1.80, 3.60, 4.50) that z is 0.02792287."""
        raw = 1.0 / np.asarray(TYPICAL)
        book = raw.sum()
        z = 0.02792287

        expected = (np.sqrt(z**2 + 4 * (1 - z) * raw**2 / book) - z) / (2 * (1 - z))
        assert vig.shin([TYPICAL])[0] == pytest.approx(expected, abs=1e-7)

    def test_it_is_the_hand_worked_line(self) -> None:
        assert vig.shin([TYPICAL])[0] == pytest.approx(
            [0.5342750, 0.2602381, 0.2054869], abs=1e-6
        )

    def test_it_corrects_the_same_bias_as_power_but_less_hard(self) -> None:
        """Shin attributes the margin to insider trading rather than to a free exponent, which is
        a weaker correction. The ordering matters: it is why the three RPS figures come out
        power < Shin < normalised, all within 0.0002 of each other."""
        proportional = vig.normalise([TYPICAL])[0]
        shin = vig.shin([TYPICAL])[0]
        powered = vig.power([TYPICAL])[0]

        assert proportional[0] < shin[0] < powered[0]
        assert proportional[2] > shin[2] > powered[2]

    def test_the_insider_share_is_zero_for_a_fair_book(self) -> None:
        """z is the fraction of money Shin's model attributes to insiders. With no margin there is
        nothing for it to explain."""
        assert vig.shin_z([FAIR]) == pytest.approx([0.0], abs=1e-9)

    def test_the_insider_share_grows_with_the_margin(self) -> None:
        wide = vig.shin_z([[1.7, 3.4, 3.4]])
        narrow = vig.shin_z([[1.95, 3.9, 3.9]])

        assert float(wide[0]) > float(narrow[0]) > 0.0


class TestTheInterface:
    def test_shin_is_the_default(self) -> None:
        """DECISIONS.md: 'Shin is the default'. The three differ by 0.0002 RPS, so the choice is
        near-immaterial for benchmarking — but it has to be one of them, stated in one place."""
        assert vig.DEFAULT_METHOD == "shin"
        assert vig.remove([TYPICAL]) == pytest.approx(vig.shin([TYPICAL]))

    def test_all_three_methods_are_reachable_by_name(self) -> None:
        assert sorted(vig.METHODS) == ["normalise", "power", "shin"]

    def test_every_named_method_is_the_function_it_names(self) -> None:
        assert vig.METHODS["shin"] is vig.shin
        assert vig.METHODS["power"] is vig.power
        assert vig.METHODS["normalise"] is vig.normalise

    def test_an_unknown_method_names_the_ones_that_exist(self) -> None:
        with pytest.raises(vig.VigError, match="normalise, power, shin"):
            vig.remove([TYPICAL], method="proportional")


class TestABookThatIsNotOne:
    """A malformed book is a bug in the ingest, not a Predictor with an unusual opinion."""

    def test_it_refuses_odds_that_are_not_a_three_way_book(self) -> None:
        with pytest.raises(vig.VigError, match="three decimal odds"):
            vig.remove([[1.8, 3.6]])

    def test_it_refuses_odds_of_one_or_less(self) -> None:
        """Decimal odds of 1.0 imply certainty and below 1.0 imply more than certainty. Either is
        corrupt data, and quietly returning a probability above one would put a meaningless number
        on the scoreboard."""
        with pytest.raises(vig.VigError, match="greater than 1"):
            vig.remove([[1.0, 3.6, 4.5]])

    def test_it_refuses_a_missing_price(self) -> None:
        """A Season with no odds is a Season with no market comparison (ADR 0001), which is a
        different thing from a book with a hole in it."""
        with pytest.raises(vig.VigError, match="missing"):
            vig.remove([[1.8, np.nan, 4.5]])

    def test_it_refuses_a_book_that_pays_more_than_it_takes(self) -> None:
        """An overround below one is an arbitrage, not a market average. Every method would still
        return numbers, so this has to be checked rather than assumed."""
        with pytest.raises(vig.VigError, match="overround"):
            vig.remove([[4.0, 4.0, 4.0]])


class TestAskingWhetherSomethingIsABook:
    """The same question as :func:`vig.as_book`, answered per row instead of raised.

    A Predictor asks this before claiming to cover a Fixture. The two must never disagree: a row
    called a book here and refused there would stop a whole backfill on a row nobody meant to walk
    over, which is why both are defined from one set of faults.
    """

    def test_a_book_is_a_book(self) -> None:
        assert vig.is_book([FAIR, VIGGED, TYPICAL]).tolist() == [True, True, True]

    def test_it_answers_per_row_rather_than_raising(self) -> None:
        mixed = [TYPICAL, [1.8, np.nan, 4.5], [4.0, 4.0, 4.0], [1.0, 3.6, 4.5]]

        assert vig.is_book(mixed).tolist() == [True, False, False, False]

    def test_it_agrees_with_what_as_book_refuses(self) -> None:
        for book in ([1.8, np.nan, 4.5], [4.0, 4.0, 4.0], [1.0, 3.6, 4.5]):
            assert not vig.is_book([book])[0]
            with pytest.raises(vig.VigError):
                vig.as_book([book])

    def test_it_agrees_with_what_as_book_accepts(self) -> None:
        for book in (FAIR, VIGGED, TYPICAL):
            assert vig.is_book([book])[0]
            assert vig.as_book([book]).shape == (1, 3)

    def test_a_shape_that_is_not_a_book_at_all_still_raises(self) -> None:
        """A missing price is a Fixture with no market; a two-column frame is a bug in the caller,
        and there is no per-row answer to give about it."""
        with pytest.raises(vig.VigError, match="three decimal odds"):
            vig.is_book([[1.8, 3.6]])

    def test_an_empty_slate_answers_emptily(self) -> None:
        assert vig.is_book(np.empty((0, 3))).tolist() == []
