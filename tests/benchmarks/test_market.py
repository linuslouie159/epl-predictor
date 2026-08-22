"""The Market Line and the Ceiling Line: the market registered as a Predictor.

Issue #8. The Market Line is the opponent this whole project is measured against, and the Ceiling
Line is the same arithmetic applied to a book that knows things the model cannot (ADR 0001).

Two properties of the Predictor contract are load-bearing here and are tested rather than assumed:

* the Market Line reads its odds off the **Fixture**, never off the corpus, so it records
  ``inputs_seen = 0`` — a Predictor that consumes no history has no history to leak
* the Ceiling Line reads a column the ledger's allow-list deliberately withholds, and gets it
  through a claim it makes in its own source rather than through a hole in the allow-list
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from epl import predictors
from epl.benchmarks import CEILING_LINE, CLOSING_COLUMNS, MARKET_LINE, market, vig
from epl.benchmarks.market import OddsLine
from epl.ledger import backtest, schema
from epl.predictors import Evidence

#: A book of (1.80, 3.60, 4.50) — a favourite, a draw and a longshot, 5.56% margin.
TYPICAL = {"prematch_odds_home": 1.80, "prematch_odds_draw": 3.60, "prematch_odds_away": 4.50}

#: The same Fixture's closing book, a shade shorter on the favourite.
TYPICAL_CLOSE = {"closing_odds_home": 1.74, "closing_odds_draw": 3.70, "closing_odds_away": 4.75}


@pytest.fixture
def priced_round(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_matches(
        {"date": "2024-08-17", "home_club": "arsenal", **TYPICAL, **TYPICAL_CLOSE},
        {
            "date": "2024-08-17",
            "home_club": "everton",
            "prematch_odds_home": 2.50,
            "prematch_odds_draw": 3.30,
            "prematch_odds_away": 2.80,
            "closing_odds_home": 2.55,
            "closing_odds_draw": 3.30,
            "closing_odds_away": 2.75,
        },
    )


def _evidence(matches: pd.DataFrame, as_of: str = "2024-08-16") -> Evidence:
    return Evidence.before(matches, pd.Timestamp(as_of))


class TestTheMarketLine:
    def test_it_is_the_vig_removed_book(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        predicted = MARKET_LINE.predict(priced_round, _evidence(make_matches()))

        assert predicted[0] == pytest.approx(vig.shin([[1.80, 3.60, 4.50]])[0])

    def test_it_prices_every_fixture_it_is_handed(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        predicted = MARKET_LINE.predict(priced_round, _evidence(make_matches()))

        assert predicted.shape == (2, 3)
        assert predicted[0].tolist() != predicted[1].tolist()

    def test_it_uses_shin_by_default(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        assert MARKET_LINE.method == vig.DEFAULT_METHOD == "shin"

    def test_the_method_can_be_changed_without_touching_the_predictor(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Issue #8 asks for the three "behind one interface, so a reader can see for themselves
        that the choice barely matters". Comparing them has to be a loop, not a rewrite."""
        lines = {
            method: OddsLine("probe", MARKET_LINE.columns, method=method).predict(
                priced_round, _evidence(make_matches())
            )
            for method in vig.METHODS
        }

        assert lines["power"][0][0] > lines["shin"][0][0] > lines["normalise"][0][0]

    def test_it_reports_the_overround_it_removed(self, priced_round: pd.DataFrame) -> None:
        """"Reported alongside every Market Line ... so the vig removal can be sanity-checked
        rather than trusted" (issue #8)."""
        assert MARKET_LINE.overround(priced_round)[0] == pytest.approx(1.0555555, abs=1e-6)

    def test_it_reads_no_history_at_all(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The market's price is a fact about the Fixture, sampled at the As-Of Instant itself. A
        Predictor that consumes no history has no history to leak, and its rows record that."""
        played = make_matches({"date": "2024-08-13"}, {"date": "2024-08-14"})
        evidence = _evidence(played)

        MARKET_LINE.predict(priced_round, evidence)

        assert evidence.rows_seen == 0
        assert evidence.latest_seen is None

    def test_its_stored_rows_say_it_saw_nothing(self, priced_round: pd.DataFrame) -> None:
        rows = backtest.backfill(MARKET_LINE, priced_round, seasons=[2024])

        assert list(rows["inputs_seen"]) == [0, 0]
        assert rows["latest_input"].isna().all()
        assert schema.audit(rows) == []


class TestTheCeilingLine:
    def test_it_is_the_vig_removed_closing_book(
        self, priced_round: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        visible = schema.visible(CEILING_LINE, priced_round)

        predicted = CEILING_LINE.predict(visible, _evidence(make_matches()))

        assert predicted[0] == pytest.approx(vig.shin([[1.74, 3.70, 4.75]])[0])

    def test_its_claim_matches_what_the_ledger_grants(self) -> None:
        """`CLOSING_COLUMNS` is spelled out in the benchmarks rather than imported from the ledger,
        so that the claim is an independent statement the ledger genuinely checks rather than the
        ledger checking its own tuple against itself. This is what keeps the two in step."""
        assert CLOSING_COLUMNS == schema.PRIVILEGED_FIXTURE_COLUMNS

    def test_it_claims_the_closing_odds_by_name(self) -> None:
        """The labelled exception. The closing odds are absent from the ledger's allow-list
        because they carry team news from after the As-Of Instant; the Ceiling Line is the one
        Predictor entitled to them, and it says so where a reader of it can see (ADR 0001)."""
        assert predictors.also_sees(CEILING_LINE) == schema.PRIVILEGED_FIXTURE_COLUMNS

    def test_the_market_line_claims_nothing(self) -> None:
        """Its odds are on the ordinary allow-list, so it needs no exception. If it ever claimed
        one, the Market Line would stop being an honest opponent."""
        assert predictors.also_sees(MARKET_LINE) == ()

    def test_the_ledger_actually_hands_it_the_closing_odds(
        self, priced_round: pd.DataFrame
    ) -> None:
        rows = backtest.backfill(CEILING_LINE, priced_round, seasons=[2024])

        assert rows.loc[0, "prob_home"] == pytest.approx(
            vig.shin([[1.74, 3.70, 4.75]])[0][0], abs=1e-9
        )

    def test_it_covers_only_the_fixtures_that_have_a_closing_book(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Closing odds begin in 2019/20 (ADR 0001). Earlier Fixtures are not Fixtures the market
        priced at nothing — they are Fixtures with no closing book, and the Ceiling Line has
        nothing to say about them."""
        mixed = make_matches(
            {"date": "2024-08-17", "home_club": "arsenal", **TYPICAL},
            {"date": "2024-08-24", "home_club": "everton", **TYPICAL, **TYPICAL_CLOSE},
        )

        rows = backtest.backfill(CEILING_LINE, mixed, seasons=[2024])

        assert list(rows["home_club"]) == ["everton"]

    def test_it_carries_a_caveat_wherever_it_is_scored(self) -> None:
        """"Labelled everywhere it appears as knowing team news the model cannot have" (issue #8).
        Its RPS is also measured over a different, smaller span than everything else on the board,
        so a bare number beside the Market Line's would mislead twice over."""
        caveat = predictors.note(CEILING_LINE)

        assert "team news" in caveat
        assert "2019/20" in caveat

    def test_the_market_lines_note_points_at_its_own_audit_instead(self) -> None:
        """It needs no caveat about what it knows — its information set is the honest one. What it
        does carry is a pointer to the margin behind the number, because a vig-removed line is a
        derived Prediction and issue #8 asks that the removal be checkable rather than trusted."""
        caveat = predictors.note(MARKET_LINE)

        assert "team news" not in caveat
        assert "overround" in caveat


class TestASeasonWithNoMarket:
    """"Seasons 2000/01-2001/02 have no odds at all and therefore no market comparison; this is
    handled explicitly rather than as missing data" (issue #8, ADR 0001).

    Two different explicit behaviours, because they answer two different questions. Asked to walk
    a Season it cannot price, the Market Line writes nothing for it. Handed one Fixture of it
    directly, it refuses — a hole in a book that is supposed to exist is a bug in the ingest, and
    silently returning a third each would hide it behind a plausible-looking floor.
    """

    def test_it_writes_no_rows_for_a_season_it_cannot_price(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        unpriced = make_matches({"season": 2001, "date": "2001-08-18", "home_club": "arsenal"})

        rows = backtest.backfill(MARKET_LINE, unpriced, seasons=[2001])

        assert rows.empty

    def test_it_refuses_a_fixture_it_is_handed_with_no_book(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        unpriced = make_matches({"season": 2001, "date": "2001-08-18"})

        with pytest.raises(market.MarketError, match="2001/02"):
            MARKET_LINE.predict(unpriced, _evidence(make_matches()))

    def test_a_priced_season_beside_an_unpriced_one_keeps_only_the_priced_rows(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        mixed = make_matches(
            {"season": 2001, "date": "2001-08-18", "home_club": "arsenal"},
            {"season": 2005, "date": "2005-08-20", "home_club": "everton", **TYPICAL},
        )

        rows = backtest.backfill(MARKET_LINE, mixed, seasons=[2001, 2005])

        assert list(rows["season"]) == [2005]


class TestTheSplice:
    """"`BbAvH/D/A` (2005/06-2018/19) is spliced to `AvgH/D/A` (2019/20 onward) with no change of
    definition visible at the join" (issue #8).

    The splice itself happens in the ingest, which refuses a file carrying both spellings. What is
    checked here is that the Market Line does not care which side of the join a Fixture came from
    — it reads one column, so there is nowhere for a per-era branch to hide.
    """

    def test_it_reads_one_column_whichever_era_the_fixture_is_from(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        eras = make_matches(
            {"season": 2018, "date": "2019-05-04", "home_club": "arsenal", **TYPICAL},
            {"season": 2019, "date": "2019-08-17", "home_club": "everton", **TYPICAL},
        )

        predicted = MARKET_LINE.predict(eras, _evidence(make_matches()))

        assert predicted[0].tolist() == predicted[1].tolist()

    def test_the_columns_it_reads_are_the_spliced_ones(self) -> None:
        assert MARKET_LINE.columns == (
            "prematch_odds_home",
            "prematch_odds_draw",
            "prematch_odds_away",
        )


class TestBothAreOnTheScoreboard:
    def test_each_is_registered_under_its_slug(self) -> None:
        assert predictors.by_name("market_line") is MARKET_LINE
        assert predictors.by_name("ceiling_line") is CEILING_LINE

    def test_each_satisfies_the_predictor_contract(self) -> None:
        assert isinstance(MARKET_LINE, predictors.Predictor)
        assert isinstance(CEILING_LINE, predictors.Predictor)

    def test_they_stand_beside_the_naive_baseline(self) -> None:
        """Registering a Predictor is what puts it on the board; there is no branch per Predictor
        anywhere downstream and there must not be one (spec, user story 16)."""
        names = {one.name for one in predictors.registered()}

        assert {"naive_baseline", "market_line", "ceiling_line"} <= names
