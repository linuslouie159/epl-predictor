"""The Pundit backfill against the real nine pages and the real corpus.

This is where issue #11's last acceptance criterion is discharged. Open risk 3 was that
"MyFootballFacts parseability is unverified ... the HTML has not been parsed across all nine season
pages", and the ticket asks for at least one row cross-checked against Football-Data. All nine are
parsed here and every played listing is cross-checked, which is a much stronger statement: the page
publishes the result beside every call, so agreement confirms the one thing no unit test can — that
two spellings became the right two Clubs, the right way round.

It also pins the finding stage 6 asked for by name. The shared calibration layer makes every other
Predictor slightly worse, and CLAUDE.md says a Pundit scored as-stated "should be measured again
rather than assumed to behave like these four". It does not behave like them at all.

Needs a populated ``data/raw/``, which is gitignored, so these skip when it is absent:

    python -m epl.ingest fetch
    python -m epl.pundits fetch
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.benchmarks import MARKET_LINE, NAIVE_BASELINE
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, schema, scoreboard
from epl.pundits import dataset, grading, myfootballfacts
from epl.pundits.predictor import LAWRENSON, SUTTON

pytestmark = pytest.mark.cache

#: What each Pundit scores as-stated, over their own Seasons. ADR 0003's illustrative "~0.36
#: against a market at ~0.19" against the measured thing: 0.334 against 0.195.
AS_STATED_RPS = {"lawrenson": 0.3341, "sutton": 0.3343}

#: And after the shared isotonic layer. Stage 6 measured that layer costing every Predictor
#: 0.0009-0.0015 RPS; here it *buys* about 0.09, because for the first time it has a genuinely
#: miscalibrated Predictor to correct (ADR 0006, CLAUDE.md).
CALIBRATED_RPS = {"lawrenson": 0.2374, "sutton": 0.2473}

#: The four rows where MyFootballFacts' own published result contradicts Football-Data — one goal
#: out, in one row, four times in 3,406. Named rather than counted, so a fifth is a failure that
#: says which one.
KNOWN_DISAGREEMENTS = {
    (2017, "bournemouth", "southampton"),
    (2024, "ipswich", "bournemouth"),
    (2024, "ipswich", "arsenal"),
    (2025, "bournemouth", "west_ham"),
}

#: The Fixtures inside the nine Seasons that the archive never listed. Twelve of 3,420, eight of
#: them in 2022/23. A Pundit's ``covers`` keeps them off the ledger; naming them here is what stops
#: twelve quietly becoming a hundred.
UNCALLED = {
    (2017, "bournemouth", "man_united"),
    (2017, "brighton", "tottenham"),
    (2019, "west_ham", "liverpool"),
    (2021, "southampton", "brentford"),
    (2022, "brighton", "crystal_palace"),
    (2022, "brighton", "man_city"),
    (2022, "brighton", "man_united"),
    (2022, "liverpool", "fulham"),
    (2022, "man_city", "west_ham"),
    (2022, "man_united", "chelsea"),
    (2022, "man_united", "leeds"),
    (2022, "newcastle", "brighton"),
}


def _require_football_data_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} files missing)")


def _require_page_cache() -> None:
    missing = [
        page for page in myfootballfacts.PAGES if not myfootballfacts.raw_page_path(page).exists()
    ]
    if missing:
        pytest.skip(f"{len(missing)} of 9 MyFootballFacts pages not cached")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_football_data_cache()
    return load_matches()


@pytest.fixture(scope="module")
def listings() -> pd.DataFrame:
    """Every call on every page, before a single Club is resolved."""
    _require_page_cache()
    return myfootballfacts.read_all()


@pytest.fixture(scope="module")
def built(matches: pd.DataFrame, listings: pd.DataFrame) -> dataset.Backfill:
    return dataset.build(matches, listings)


@pytest.fixture(scope="module")
def stored(matches: pd.DataFrame) -> pd.DataFrame:
    """Both Pundits and both reference Predictors, walked over the Evaluation Window.

    Walked here rather than read out of ``outputs/backtest/``, so the numbers below come from the
    code under test rather than from whatever was last written to disk.
    """
    return pd.concat(
        [
            backtest.backfill(predictor, matches)
            for predictor in (LAWRENSON, SUTTON, MARKET_LINE, NAIVE_BASELINE)
        ],
        ignore_index=True,
    )


def _same_fixtures(stored: pd.DataFrame, pundit: str) -> pd.DataFrame:
    """Every Predictor's rows, cut to the Fixtures this Pundit spoke to.

    The Ceiling Line's lesson, applied here: a bare RPS beside another one measured over different
    Fixtures reads as a comparison and is not one (ADR 0001).
    """
    theirs = stored.loc[stored["predictor"] == pundit]
    key = set(zip(theirs["season"], theirs["home_club"], theirs["away_club"], strict=True))
    return stored.loc[
        [
            fixture in key
            for fixture in zip(
                stored["season"], stored["home_club"], stored["away_club"], strict=True
            )
        ]
    ]


class TestAllNinePagesParse:
    """Open risk 3, closed."""

    @pytest.mark.parametrize("page", myfootballfacts.PAGES, ids=lambda page: str(page.season))
    def test_each_page_yields_a_season_of_calls(self, page: myfootballfacts.Page) -> None:
        _require_page_cache()

        calls = myfootballfacts.read_page(page)

        assert myfootballfacts.MIN_CALLS <= len(calls) <= myfootballfacts.MAX_CALLS

    def test_every_spelling_on_every_page_resolves_to_a_club(
        self, built: dataset.Backfill
    ) -> None:
        """Reaching a built dataset at all means no name raised on the way. Stated as its own
        test because it is an acceptance criterion, not a side effect."""
        assert len(built.calls) == 3408

    def test_the_pundits_between_them_called_almost_every_fixture(
        self, built: dataset.Backfill, matches: pd.DataFrame
    ) -> None:
        seasons = sorted(set(built.calls["season"]))
        window = matches.loc[matches["season"].isin(seasons) & (matches["division"] == "E0")]
        called = set(
            zip(
                built.calls["season"],
                built.calls["home_club"],
                built.calls["away_club"],
                strict=True,
            )
        )
        uncalled = {
            fixture
            for fixture in zip(
                window["season"], window["home_club"], window["away_club"], strict=True
            )
            if fixture not in called
        }

        assert len(window) == 3420
        assert uncalled == UNCALLED


class TestCrossCheckedAgainstFootballData:
    def test_all_but_four_published_results_agree(self, built: dataset.Backfill) -> None:
        """3,402 rows of agreement is the evidence that the Clubs resolved correctly. Four rows
        of disagreement is MyFootballFacts mistranscribing a score, and Football-Data is the
        authority — the published one is read to check the parse and is never stored."""
        found = set(
            zip(
                built.disagreements["season"],
                built.disagreements["home_club"],
                built.disagreements["away_club"],
                strict=True,
            )
        )

        assert found == KNOWN_DISAGREEMENTS

    def test_two_calls_have_no_published_result_to_check(self, built: dataset.Backfill) -> None:
        """The archive listed these two Fixtures as postponed and never listed them again, so the
        pundit's only call on each stands with no score beside it. Counting them as agreements
        would report a check that was never made."""
        assert built.checked == len(built.calls) - 2

    def test_every_season_agrees_far_above_the_floor(self, built: dataset.Backfill) -> None:
        by_season = built.disagreements["season"].value_counts()
        calls = built.calls.groupby("season").size()

        worst = min(
            1 - int(by_season.get(season, 0)) / int(total) for season, total in calls.items()
        )

        assert worst > 0.99


class TestTheCommittedDatasetIsWhatTheBuildProduces:
    def test_rebuilding_from_the_cache_reproduces_it_byte_for_byte(
        self, built: dataset.Backfill, tmp_path
    ) -> None:
        """"Committed and frozen rather than re-scraped on every run" only means something if the
        committed file is provably the file the build makes."""
        rebuilt = dataset.write(built.calls, tmp_path / "predictions.csv")

        assert rebuilt.read_bytes() == dataset.path().read_bytes()


class TestScoredAsStated:
    @pytest.mark.parametrize("pundit", sorted(AS_STATED_RPS))
    def test_the_as_stated_rps_is_what_the_adr_predicted(
        self, pundit: str, stored: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """ADR 0003 put a pundit at "~0.36 RPS against a market at ~0.19" on illustrative rates.
        Measured, it is 0.334 against 0.195 — the gap is real and it is enormous."""
        board = scoreboard.build(stored.loc[stored["predictor"] == pundit], matches)

        assert board.loc[0, "rps"] == pytest.approx(AS_STATED_RPS[pundit], abs=5e-5)

    @pytest.mark.parametrize("pundit", sorted(AS_STATED_RPS))
    def test_the_gap_to_the_market_is_mostly_the_format_of_the_question(
        self, pundit: str, stored: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """The point of ADR 0003, measured over the Fixtures both spoke to. On RPS the Pundit is
        0.14 behind the market and far behind the Naive Baseline. On accuracy — which asks only
        who they picked, not how sure they claimed to be — they are four points behind the market
        and seven ahead of the floor. Nothing about the Pundit changed between those two
        sentences; only the question did."""
        board = scoreboard.build(_same_fixtures(stored, pundit), matches).set_index("predictor")

        assert board.loc[pundit, "rps"] > board.loc["naive_baseline", "rps"] + 0.09
        assert board.loc[pundit, "accuracy"] > board.loc["naive_baseline", "accuracy"] + 0.04
        assert board.loc[pundit, "accuracy"] > board.loc["market_line", "accuracy"] - 0.06

    @pytest.mark.parametrize("pundit", sorted(AS_STATED_RPS))
    def test_the_walk_audits_clean(self, pundit: str, stored: pd.DataFrame) -> None:
        assert schema.audit(stored.loc[stored["predictor"] == pundit]) == []

    @pytest.mark.parametrize("pundit", sorted(AS_STATED_RPS))
    def test_no_stored_row_saw_an_input(self, pundit: str, stored: pd.DataFrame) -> None:
        """A Pundit reads its call off the Fixture, so it has no history to leak — the same thing
        the Market Line records, and correct for the same reason (CLAUDE.md)."""
        rows = stored.loc[stored["predictor"] == pundit]

        assert (rows["inputs_seen"] == 0).all()
        assert rows["latest_input"].isna().all()


class TestTheSharedCalibrationLayerMeetsAPredictorItCanHelp:
    @pytest.mark.parametrize("pundit", sorted(CALIBRATED_RPS))
    def test_calibration_buys_a_pundit_what_it_costs_everyone_else(
        self, pundit: str, stored: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """Stage 6's whole finding was that a monotone map costs 0.0009-0.0015 RPS because all
        four Predictors were already well calibrated. Here it gains about 0.09, which is what
        confirms that diagnosis rather than contradicting it: the layer works, and it had nothing
        to find. CLAUDE.md asks for exactly this re-measurement."""
        board = scoreboard.build(stored.loc[stored["predictor"] == pundit], matches)

        assert board.loc[0, "calibrated_rps"] == pytest.approx(CALIBRATED_RPS[pundit], abs=5e-5)
        assert board.loc[0, "calibrated_rps"] < board.loc[0, "rps"] - 0.08

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED_RPS))
    def test_an_as_stated_pundit_is_the_worst_calibrated_predictor_on_the_board(
        self, pundit: str, stored: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """"A published Scoreline read as [1, 0, 0] is the most miscalibrated Prediction there is"
        (CLAUDE.md). The four Predictors before it all sit at about 0.006."""
        board = scoreboard.build(stored, matches).set_index("predictor")

        assert board.loc[pundit, "ece"] > 0.3
        assert board.loc["market_line", "ece"] < 0.01


class TestGradedBothWays:
    @pytest.fixture(scope="module")
    def graded(self, matches: pd.DataFrame) -> pd.DataFrame:
        return grading.grade(dataset.load(), matches)

    def test_every_call_on_a_played_fixture_is_graded(self, graded: pd.DataFrame) -> None:
        assert len(graded) == 3408

    def test_about_a_tenth_of_scorelines_are_exactly_right(self, graded: pd.DataFrame) -> None:
        """The strict reading. MyFootballFacts keeps its own tally for 2017/18 — "48 Correct
        Scores" — and this grading finds 48 too, which is a check on the grading that owes nothing
        to our parse of the Scorelines."""
        career = grading.summary(graded, by=("pundit",)).set_index("pundit")

        assert career.loc["lawrenson", "exact_rate"] == pytest.approx(0.110, abs=5e-4)
        assert career.loc["sutton", "exact_rate"] == pytest.approx(0.091, abs=5e-4)

    def test_about_half_of_outcomes_are_right(self, graded: pd.DataFrame) -> None:
        """The lenient reading, and the honest headline for a lay reader: a pundit picks the
        winner about half the time, which is a long way above the floor and below the market."""
        career = grading.summary(graded, by=("pundit",)).set_index("pundit")

        assert career.loc["lawrenson", "outcome_rate"] == pytest.approx(0.509, abs=5e-4)
        assert career.loc["sutton", "outcome_rate"] == pytest.approx(0.492, abs=5e-4)

    def test_calling_the_score_implies_calling_the_outcome(self, graded: pd.DataFrame) -> None:
        """The strict reading is a subset of the lenient one, always. A grading that let them
        cross would mean the implied Outcome had been computed two different ways."""
        assert not (graded["exact_score"] & ~graded["correct_outcome"]).any()

    def test_the_report_covers_every_pundit_and_season(self, graded: pd.DataFrame) -> None:
        assert len(grading.summary(graded)) == 9
        assert len(grading.summary(graded, by=("pundit",))) == 2
