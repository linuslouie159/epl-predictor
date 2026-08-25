"""The shared calibration layer over the real ledger, end to end.

Issue #10's acceptance is structural — one step, every Predictor, fitted out-of-sample, both sides
reported — and unit tests cover all of that. What only the corpus can show is the **answer**, and
on this corpus the answer is a surprise worth pinning:

**Calibration makes every Predictor worse**, by 0.0009 to 0.0015 RPS, moving 3% to 5% of each
Prediction's probability mass to do it. Two things are behind that, and this module separates them
rather than reporting the sum as one fact about the corpus:

* **Knot resolution.** A map gets a knot per distinct quote, and market odds and Elo edges are
  nearly continuous, so most knots rest on a single Fixture. Cutting the knots at ten probability
  bands instead recovers most of the loss — see :class:`TestMostOfTheLossIsKnotResolution`, which
  measures the alternative that was not taken so the claim is not left resting on prose.
* **The corpus.** Even coarse, both stay worse than raw. Four of the five sit at a pooled ten-bin
  error of about 0.006 before the layer touches them — Dixon-Coles at 0.008 — so there is little
  real miscalibration left to find.

That is exactly the reading ADR 0006 built the double reporting for, pointed at the layer rather
than at a model: a large correction that buys nothing is a warning. Publishing only the corrected
column would have turned this into a silent 0.001 RPS tax that nobody would ever have seen.

These re-derive it from the ingested cache rather than trusting docs/DECISIONS.md, in the same
spirit as ``tests/models/test_elo_over_the_corpus.py``. They need a populated ``data/raw/``:

    python -m epl.ingest fetch
    python -m epl.ingest build
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.isotonic import isotonic_regression

from epl import calibration, metrics
from epl.benchmarks import CEILING_LINE, MARKET_LINE, NAIVE_BASELINE
from epl.calibration import MINIMUM_SAMPLE, Curve, Isotonic
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, scoreboard
from epl.models import DIXON_COLES, ELO, draw_curve

pytestmark = pytest.mark.cache

#: What each Predictor scores before and after the layer, over the Evaluation Window. Every pair
#: goes the wrong way, which is the finding; the sizes are what a change would have to move.
SCORES: dict[str, tuple[float, float]] = {
    "market_line": (0.19362, 0.19450),
    "ceiling_line": (0.19676, 0.19800),
    "dixon_coles": (0.19752, 0.19793),
    "elo": (0.19943, 0.20037),
    "naive_baseline": (0.22938, 0.23087),
}

#: The pooled ten-bin calibration error, before and after. It rises for all five — the layer is not
#: merely failing to buy RPS, it is leaving each Predictor *less* calibrated than it found it,
#: which is what rules out "the correction is right and RPS is the wrong judge".
CALIBRATION_ERROR: dict[str, tuple[float, float]] = {
    "market_line": (0.00614, 0.01241),
    "ceiling_line": (0.00598, 0.00843),
    "dixon_coles": (0.00804, 0.01018),
    "elo": (0.00549, 0.00972),
    "naive_baseline": (0.00613, 0.01609),
}

#: What the same walk scores when each map's knots are cut at ten equal-width probability bands
#: instead of one per distinct quote — the alternative that was measured and not taken. Most of the
#: loss above is knot resolution; the rest is the corpus, and these two numbers are what separate
#: the claims. Only the Predictors with near-continuous quotes are listed: the Naive Baseline has
#: 943 distinct Home quotes over 7,980 Fixtures and the Ceiling Line has a quarter of the sample,
#: so neither separates the two effects.
COARSE_SCORES: dict[str, float] = {
    "market_line": 0.19404,
    "elo": 0.19968,
}

#: How much probability mass the layer moves per Prediction, averaged over the whole slate.
CORRECTION: dict[str, float] = {
    "market_line": 0.0342,
    "ceiling_line": 0.0328,
    "dixon_coles": 0.0378,
    "elo": 0.0307,
    "naive_baseline": 0.0455,
}

#: Elo's draw probability at the two ends of its own Supremacy range: quoted, quoted after the
#: layer, and what actually happened. The layer was built for this defect and does move the even
#: end toward the truth — 0.302 to 0.293 against 0.276 observed — while overshooting the wide end.
ELO_DRAW_RAW = (0.3016, 0.1450)
ELO_DRAW_CALIBRATED = (0.2926, 0.1330)
ELO_DRAW_OBSERVED = (0.2757, 0.1378)


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} of 104 files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    return load_matches()


@pytest.fixture(scope="module")
def stored(matches: pd.DataFrame) -> pd.DataFrame:
    """Every Predictor walked over the Evaluation Window, as ledger rows.

    Backfilled here rather than read off ``outputs/backtest/``, so these hold against the code as
    it stands rather than against whatever was last written to disk.
    """
    return pd.concat(
        [
            backtest.backfill(one, matches)
            for one in (MARKET_LINE, CEILING_LINE, ELO, DIXON_COLES, NAIVE_BASELINE)
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def scored(stored: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Those rows with each Prediction's calibrated form beside its raw one."""
    return scoreboard.calibrated_predictions(stored, matches)


@pytest.fixture(scope="module")
def board(stored: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The published scoreboard itself, not a reimplementation of it.

    Built through :func:`epl.ledger.scoreboard.build` so these numbers are the ones a reader sees;
    a hand-rolled table here would agree with itself and prove nothing about what gets published.
    """
    published = scoreboard.build(stored, matches).set_index("predictor")
    assert sorted(published.index) == sorted(SCORES), (
        "every Predictor named in this module's tables has to reach the board, or the assertions "
        "below pass by not being made"
    )
    return published


def _raw(group: pd.DataFrame) -> np.ndarray:
    return group[list(scoreboard.PROBABILITY_COLUMNS)].to_numpy(float)


def _calibrated(group: pd.DataFrame) -> np.ndarray:
    return group[list(scoreboard.CALIBRATED_PROBABILITY_COLUMNS)].to_numpy(float)


class TestTheLayerCostsEveryPredictorSomething:
    """The finding, pinned so it cannot quietly change into its opposite in either direction."""

    @pytest.mark.parametrize("predictor", sorted(SCORES))
    def test_the_scores_are_what_was_measured(
        self, board: pd.DataFrame, predictor: str
    ) -> None:
        before, after = SCORES[predictor]

        assert board.loc[predictor, "rps"] == pytest.approx(before, abs=5e-5)
        assert board.loc[predictor, "calibrated_rps"] == pytest.approx(after, abs=5e-5)

    def test_calibration_worsens_every_predictor(self, board: pd.DataFrame) -> None:
        """The headline. Not one of the five is improved, which is why both columns are published
        and why the README's target is still measured against the raw score."""
        assert (board["calibrated_rps"] > board["rps"]).all()

    @pytest.mark.parametrize("predictor", sorted(CALIBRATION_ERROR))
    def test_it_leaves_them_less_calibrated_than_it_found_them(
        self, board: pd.DataFrame, predictor: str
    ) -> None:
        """This is what rules out "the correction is right and RPS is judging it unfairly". The
        layer is being scored on its own terms here — the ten-bin error it exists to shrink — and it
        raises that too."""
        before, after = CALIBRATION_ERROR[predictor]

        assert board.loc[predictor, "ece"] == pytest.approx(before, abs=5e-5)
        assert board.loc[predictor, "calibrated_ece"] == pytest.approx(after, abs=5e-5)
        assert after > before

    @pytest.mark.parametrize("predictor", sorted(CORRECTION))
    def test_the_correction_it_makes_is_large(
        self, board: pd.DataFrame, predictor: str
    ) -> None:
        """Three to five per cent of every Prediction's mass, moved for nothing. A small correction
        that cost a little would be unremarkable; this size is the part that makes it a warning."""
        assert board.loc[predictor, "correction"] == pytest.approx(CORRECTION[predictor], abs=5e-4)


class TestMostOfTheLossIsKnotResolution:
    """The rejected alternative, measured — because "the layer costs RPS" would otherwise read as
    a fact about the corpus when most of it is a fitting choice.

    A map gets a knot per distinct quote, and market odds and Elo edges are nearly continuous, so
    most knots rest on one Fixture. Cutting the knots at ten equal-width probability bands instead
    recovers most of the loss and does not turn it positive — both halves of that matter, and both
    are asserted here. The coarse variant is deliberately *not* in `src/`: the band count would be a
    hyperparameter, and ADR 0008 wants those fitted in a Burn-In Window that holds no stored
    Prediction. It is built here from `Curve` and `Isotonic`, the module's own public API.
    """

    @pytest.mark.parametrize("predictor", sorted(COARSE_SCORES))
    def test_a_coarser_fit_recovers_most_of_the_loss(
        self, scored: pd.DataFrame, predictor: str
    ) -> None:
        one = scored.loc[scored["predictor"] == predictor]
        outcomes = one["outcome"].tolist()
        coarse = metrics.rps(_banded_walk(one), outcomes)
        _, shipped = SCORES[predictor]

        assert coarse == pytest.approx(COARSE_SCORES[predictor], abs=5e-5)
        assert coarse < shipped

    @pytest.mark.parametrize("predictor", sorted(COARSE_SCORES))
    def test_and_still_does_not_beat_leaving_the_prediction_alone(
        self, scored: pd.DataFrame, predictor: str
    ) -> None:
        """The half that stops this reading as "just use fewer knots". Even coarse, both Predictors
        score worse calibrated than raw, and that residual is the corpus rather than the fit."""
        one = scored.loc[scored["predictor"] == predictor]
        outcomes = one["outcome"].tolist()

        assert metrics.rps(_banded_walk(one), outcomes) > metrics.rps(_raw(one), outcomes)


def _banded_walk(group: pd.DataFrame, bands: int = metrics.BINS) -> npt.NDArray[np.float64]:
    """The same walk-forward rule, with each map's knots cut at ``bands`` probability bands.

    Deliberately a re-statement of :func:`epl.calibration.walk_forward`'s loop rather than a call
    into it: what is being measured is a different fit under the same rule, so the rule has to be
    visible here to be seen to be the same one.
    """
    raw = _raw(group)
    observed = metrics.as_outcomes(group["outcome"].tolist())
    instants = np.asarray(group["as_of_instant"], dtype="datetime64[ns]")
    kickoffs = np.asarray(group["kickoff"], dtype="datetime64[ns]")

    walked = raw.copy()
    for instant in np.unique(instants):
        pool = kickoffs < instant
        if int(pool.sum()) < MINIMUM_SAMPLE:
            continue
        due = instants == instant
        walked[due] = Isotonic(
            curves=tuple(
                _banded_curve(raw[pool, index], (observed[pool] == index).astype(float), bands)
                for index in range(len(metrics.OUTCOMES))
            ),
            sample=int(pool.sum()),
        ).apply(raw[due])
    return walked


def _banded_curve(
    quoted: npt.NDArray[np.float64], happened: npt.NDArray[np.float64], bands: int
) -> Curve:
    """One knot per occupied band, at that band's mean quote and its observed rate, made monotone.

    The quote itself is never rounded — only the knots are coarse — so the comparison is about how
    finely the map is fitted and not about quantising the Predictor's own output.
    """
    index = np.clip(np.floor(np.round(quoted * bands, 9)), 0, bands - 1).astype(int)
    counts = np.bincount(index, minlength=bands)
    occupied = counts > 0
    centre = np.bincount(index, weights=quoted, minlength=bands)[occupied] / counts[occupied]
    rate = np.bincount(index, weights=happened, minlength=bands)[occupied] / counts[occupied]
    monotone = isotonic_regression(
        rate,
        sample_weight=counts[occupied].astype(float),
        y_min=0.0,
        y_max=1.0,
        increasing=True,
    )
    return Curve(
        quoted=tuple(float(value) for value in centre),
        happened=tuple(float(value) for value in np.asarray(monotone, dtype=float)),
    )


class TestItIsOverfittingRatherThanMisbuilt:
    """A split half, which says the same thing as the walk without any of the walk's machinery.

    The distinction matters: "the layer is wired up wrong" and "the layer is fitting noise" would
    both show as a worse post-calibration column, and only the second is a fact about the corpus
    rather than a bug. Fit on the older half of the Market Line's Fixtures, then score both halves.
    """

    def test_the_map_improves_the_half_it_was_fitted_on(self, scored: pd.DataFrame) -> None:
        older, _ = _halves(scored, "market_line")
        fitted = calibration.fit(_raw(older), older["outcome"].tolist())

        gain = metrics.rps(_raw(older), older["outcome"].tolist()) - metrics.rps(
            fitted.apply(_raw(older)), older["outcome"].tolist()
        )

        assert gain == pytest.approx(0.0017, abs=5e-4)

    def test_and_costs_the_half_it_was_not(self, scored: pd.DataFrame) -> None:
        older, newer = _halves(scored, "market_line")
        fitted = calibration.fit(_raw(older), older["outcome"].tolist())

        cost = metrics.rps(
            fitted.apply(_raw(newer)), newer["outcome"].tolist()
        ) - metrics.rps(_raw(newer), newer["outcome"].tolist())

        assert cost == pytest.approx(0.0005, abs=5e-4)
        assert cost > 0


def _halves(scored: pd.DataFrame, predictor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Predictor's Fixtures split in two by kickoff, older first."""
    one = scored.loc[scored["predictor"] == predictor].sort_values("kickoff", kind="stable")
    middle = len(one) // 2
    return one.iloc[:middle], one.iloc[middle:]


class TestItStillDoesWhatItWasAskedTo:
    """Issue #9 handed #10 a specific defect: Elo quotes draws too often in every Supremacy bucket.

    The layer does move that in the right direction. It is worth separating from the RPS finding,
    because "the correction is wrong" and "the correction is right and the noise around it costs
    more than it buys" are different diagnoses, and only the second is true here.
    """

    def test_it_pulls_elos_draw_quote_toward_what_happened(
        self, scored: pd.DataFrame
    ) -> None:
        elo = scored.loc[scored["predictor"] == "elo"]
        outcomes = elo["outcome"].to_numpy(dtype=object)
        raw = draw_curve(_raw(elo), outcomes)
        calibrated = draw_curve(_calibrated(elo), outcomes)

        assert raw["predicted_draw"].iloc[0] == pytest.approx(ELO_DRAW_RAW[0], abs=5e-4)
        assert raw["observed_draw"].iloc[0] == pytest.approx(ELO_DRAW_OBSERVED[0], abs=5e-4)
        assert calibrated["predicted_draw"].iloc[0] == pytest.approx(
            ELO_DRAW_CALIBRATED[0], abs=5e-4
        )
        # Closer to the truth at the even end than the raw quote was, and still short of it.
        assert (
            ELO_DRAW_OBSERVED[0]
            < calibrated["predicted_draw"].iloc[0]
            < raw["predicted_draw"].iloc[0]
        )

    def test_it_overshoots_at_the_widest_supremacy(self, scored: pd.DataFrame) -> None:
        """The other end, where the raw quote was nearly right and the layer pushed it past."""
        elo = scored.loc[scored["predictor"] == "elo"]
        outcomes = elo["outcome"].to_numpy(dtype=object)
        calibrated = draw_curve(_calibrated(elo), outcomes)

        assert calibrated["predicted_draw"].iloc[-1] == pytest.approx(
            ELO_DRAW_CALIBRATED[1], abs=5e-4
        )
        assert calibrated["predicted_draw"].iloc[-1] < ELO_DRAW_OBSERVED[1]


class TestNothingWasCorrectedWithHindsight:
    """The project's one rule, applied to the layer that fits on Outcomes.

    A calibration map is the one thing in this project fitted on results, so it is the one place a
    leak could enter without any stored row looking wrong — the ledger's audit checks what a
    *Predictor* saw, and the layer runs after that.
    """

    def test_a_round_is_corrected_only_where_the_pool_before_it_was_deep_enough(
        self, scored: pd.DataFrame
    ) -> None:
        """The rule re-derived off the real As-Of Instants and kickoffs rather than asked of the
        module: at each round, count that Predictor's own Fixtures already played, and the round is
        corrected exactly when that count reaches MINIMUM_SAMPLE."""
        for _, group in scored.groupby("predictor", sort=True):
            kickoffs = group["kickoff"].to_numpy()
            for instant, due in group.groupby("as_of_instant", sort=True):
                pool = int((kickoffs < np.datetime64(instant)).sum())
                assert set(due["corrected"]) == {pool >= MINIMUM_SAMPLE}

    def test_flipping_a_later_outcome_changes_nothing_before_it(
        self, scored: pd.DataFrame
    ) -> None:
        """The strongest form of the claim, over the real ledger: rewrite the last Prediction
        Round's Outcomes and every earlier calibrated Prediction has to come back identical.

        One Predictor is enough. The walk has no branch per Predictor — that is the whole point of
        the layer sitting behind the contract — so what holds for Elo's 7,980 holds for all of them.
        """
        elo = scored.loc[scored["predictor"] == "elo"]
        outcomes = elo["outcome"].tolist()
        raw, as_of, kickoff = _raw(elo), elo["as_of_instant"], elo["kickoff"]
        last = elo["as_of_instant"] == elo["as_of_instant"].max()
        meddled = [
            ("D" if outcome != "D" else "H") if flipped else outcome
            for outcome, flipped in zip(outcomes, last, strict=True)
        ]

        before = calibration.walk_forward(raw, outcomes, as_of, kickoff)
        after = calibration.walk_forward(raw, meddled, as_of, kickoff)

        assert meddled != outcomes
        earlier = ~last.to_numpy()
        assert np.array_equal(before.predictions[earlier], after.predictions[earlier])
