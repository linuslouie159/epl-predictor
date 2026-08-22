"""One row schema, shared by both Prediction stores.

Issue #7: "one row schema shared by both stores, so scoring code never needs to know which it is
reading". These tests pin the schema and the audits every row in either store must pass — above
all the project's one rule, that no Prediction saw a row timestamped at or after its own As-Of
Instant.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from epl.ledger import schema
from epl.metrics import MetricsError
from epl.predictors import Evidence

UNIFORM = (1 / 3, 1 / 3, 1 / 3)


class ReadsNothing:
    """A Predictor that never opens its Evidence — the floor before it has seen a Season."""

    name = "reads_nothing"

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        return np.tile(UNIFORM, (len(fixtures), 1))


class ShortChanged:
    """A Predictor that hands back fewer Predictions than it was given Fixtures."""

    name = "short_changed"

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        return np.tile(UNIFORM, (len(fixtures) - 1, 1))


@pytest.fixture
def round_fixtures(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """Two Fixtures of one Prediction Round: the Saturday and the Sunday after 2024-08-16."""
    return make_matches(
        {"date": "2024-08-17", "home_club": "arsenal", "away_club": "wolves"},
        {"date": "2024-08-18", "home_club": "everton", "away_club": "brighton"},
    )


@pytest.fixture
def evidence(make_matches: Callable[..., pd.DataFrame]) -> Evidence:
    """One match played the Wednesday before that round."""
    played = make_matches({"date": "2024-08-14"})
    return Evidence.before(played, pd.Timestamp("2024-08-16"))


class TestBuildingRows:
    def test_one_row_per_fixture_in_the_order_they_were_handed_in(
        self,
        round_fixtures: pd.DataFrame,
        evidence: Evidence,
        make_predictor: Callable[..., object],
    ) -> None:
        rows = schema.predictions_for(make_predictor(), round_fixtures, evidence)

        assert list(rows.columns) == list(schema.LEDGER_COLUMNS)
        assert list(rows["home_club"]) == ["arsenal", "everton"]

    def test_a_row_carries_the_predictor_the_round_and_the_probabilities(
        self,
        round_fixtures: pd.DataFrame,
        evidence: Evidence,
        make_predictor: Callable[..., object],
    ) -> None:
        rows = schema.predictions_for(
            make_predictor("coin", (0.5, 0.3, 0.2)), round_fixtures, evidence
        )
        first = rows.iloc[0]

        assert first["predictor"] == "coin"
        assert first["prediction_round"] == "2024-08-16"
        assert first["as_of_instant"] == pd.Timestamp("2024-08-16")
        assert first["kickoff"] == pd.Timestamp("2024-08-17")
        assert (first["prob_home"], first["prob_draw"], first["prob_away"]) == (0.5, 0.3, 0.2)

    def test_a_row_records_the_input_rows_the_prediction_saw(
        self,
        round_fixtures: pd.DataFrame,
        evidence: Evidence,
        make_predictor: Callable[..., object],
    ) -> None:
        """Not what was available — what the Predictor actually took, read back off the row."""
        rows = schema.predictions_for(make_predictor(), round_fixtures, evidence)

        assert set(rows["inputs_seen"]) == {1}
        assert set(rows["latest_input"]) == {pd.Timestamp("2024-08-14")}

    def test_a_predictor_that_read_nothing_records_no_input(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        rows = schema.predictions_for(ReadsNothing(), round_fixtures, evidence)

        assert set(rows["inputs_seen"]) == {0}
        assert rows["latest_input"].isna().all()

    def test_fixtures_from_two_rounds_are_refused(
        self,
        make_matches: Callable[..., pd.DataFrame],
        evidence: Evidence,
        make_predictor: Callable[..., object],
    ) -> None:
        """A Prediction Round is one batch at one As-Of Instant (ADR 0002). Two rounds in one call
        would give half the rows an instant the Evidence was never cut at."""
        spanning = make_matches({"date": "2024-08-17"}, {"date": "2024-08-21"})

        with pytest.raises(schema.LedgerError, match="one Prediction Round"):
            schema.predictions_for(make_predictor(), spanning, evidence)

    def test_evidence_cut_at_a_different_instant_is_refused(
        self,
        round_fixtures: pd.DataFrame,
        make_matches: Callable[..., pd.DataFrame],
        make_predictor: Callable[..., object],
    ) -> None:
        """The check that makes the contract worth anything: Evidence cut at the wrong instant is
        how a leak would enter without a single line of the row audit failing."""
        late = Evidence.before(make_matches({"date": "2024-08-14"}), pd.Timestamp("2024-08-20"))

        with pytest.raises(schema.LedgerError, match="As-Of Instant"):
            schema.predictions_for(make_predictor(), round_fixtures, late)

    def test_a_predictor_returning_the_wrong_number_of_rows_is_refused(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        with pytest.raises(schema.LedgerError, match="2 Fixtures"):
            schema.predictions_for(ShortChanged(), round_fixtures, evidence)

    def test_a_prediction_that_does_not_sum_to_one_is_refused(
        self,
        round_fixtures: pd.DataFrame,
        evidence: Evidence,
        make_predictor: Callable[..., object],
    ) -> None:
        """Delegated to epl.metrics, so the ledger and the scoreboard agree on what a Prediction
        is rather than each holding an opinion."""
        with pytest.raises(MetricsError, match="sum to 1"):
            schema.predictions_for(
                make_predictor(probabilities=(0.5, 0.3, 0.9)), round_fixtures, evidence
            )


@pytest.fixture
def rows(
    round_fixtures: pd.DataFrame, evidence: Evidence, make_predictor: Callable[..., object]
) -> pd.DataFrame:
    """Two clean ledger rows, for tests that then break one of them."""
    return schema.predictions_for(make_predictor(), round_fixtures, evidence)


class TestTheAudit:
    def test_clean_rows_have_nothing_to_complain_about(self, rows: pd.DataFrame) -> None:
        assert schema.audit(rows) == []

    def test_a_prediction_that_saw_a_row_at_its_own_as_of_instant_is_a_leak(
        self, rows: pd.DataFrame
    ) -> None:
        """The project's one rule, checked off the stored file rather than trusted of the code.

        At the instant, not merely after it: a Fixture kicking off at the As-Of Instant has not
        been played, so its result cannot be an input.
        """
        rows.loc[0, "latest_input"] = rows.loc[0, "as_of_instant"]

        assert any("future data" in complaint for complaint in schema.audit(rows))

    def test_a_prediction_that_saw_a_row_from_after_its_as_of_instant_is_a_leak(
        self, rows: pd.DataFrame
    ) -> None:
        rows.loc[0, "latest_input"] = rows.loc[0, "as_of_instant"] + pd.Timedelta(days=1)

        assert any("future data" in complaint for complaint in schema.audit(rows))

    def test_a_prediction_made_after_its_own_kickoff_is_refused(
        self, rows: pd.DataFrame
    ) -> None:
        rows.loc[0, "as_of_instant"] = rows.loc[0, "kickoff"] + pd.Timedelta(hours=1)

        assert any("kickoff" in complaint for complaint in schema.audit(rows))

    def test_a_fixture_with_no_recorded_kickoff_time_may_be_predicted_on_its_own_day(
        self, make_matches, evidence: Evidence, make_predictor: Callable[..., object]
    ) -> None:
        """Football-Data records no kickoff time before 2019/20, so those Fixtures sit at midnight
        on their own day — and 313 of them are played on the Tuesday or Friday they anchor to.

        The As-Of Instant is midnight at the start of that day, so it equals the recorded kickoff.
        Nothing is leaked: the Evidence cut is strict, so no Fixture from that day is visible. This
        is the same two-tier rule ``epl.rounds`` applies, and the audit has to agree with it.
        """
        friday = make_matches({"date": "2024-08-16", "time": None})
        rows = schema.predictions_for(make_predictor(), friday, evidence)

        assert rows.iloc[0]["as_of_instant"] == rows.iloc[0]["kickoff"]
        assert schema.audit(rows) == []

    def test_a_round_label_that_does_not_match_its_kickoff_is_refused(
        self, rows: pd.DataFrame
    ) -> None:
        """The round is derivable from the kickoff (ADR 0002), so a stored one that disagrees
        means the file was assembled by something that did not follow the anchor rule."""
        rows.loc[0, "prediction_round"] = "2024-08-09"

        assert any("Prediction Round" in complaint for complaint in schema.audit(rows))

    def test_one_predictor_predicting_one_fixture_twice_at_one_instant_is_refused(
        self, rows: pd.DataFrame
    ) -> None:
        """Superseding a sealed Prediction means a *new* As-Of Instant (ADR 0005). Two rows at the
        same instant are not a supersede, they are a double count."""
        doubled = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

        assert any("twice" in complaint for complaint in schema.audit(doubled))

    def test_a_superseding_row_at_a_later_instant_is_allowed(self, rows: pd.DataFrame) -> None:
        """A supersede keeps the round it belongs to and carries a later instant (ADR 0005)."""
        superseding = rows.iloc[[0]].copy()
        superseding["as_of_instant"] += pd.Timedelta(hours=12)

        assert schema.audit(pd.concat([rows, superseding], ignore_index=True)) == []

    def test_probabilities_that_do_not_sum_to_one_are_refused(self, rows: pd.DataFrame) -> None:
        rows.loc[0, "prob_draw"] = 0.9

        assert any("sum to 1" in complaint for complaint in schema.audit(rows))

    def test_a_frame_in_the_wrong_shape_is_refused(self, rows: pd.DataFrame) -> None:
        assert schema.audit(rows.drop(columns=["latest_input"])) != []

    def test_check_returns_the_rows_when_they_are_clean(self, rows: pd.DataFrame) -> None:
        assert schema.check(rows) is rows

    def test_check_raises_with_every_complaint(self, rows: pd.DataFrame) -> None:
        rows.loc[0, "latest_input"] = rows.loc[0, "as_of_instant"]
        rows.loc[1, "prediction_round"] = "2024-08-09"

        with pytest.raises(schema.LedgerError) as raised:
            schema.check(rows)

        assert "future data" in str(raised.value)
        assert "Prediction Round" in str(raised.value)


class Cheat:
    """A Predictor that reads the Outcome off the Fixture it is being asked to predict.

    It exists to prove it cannot. Everything else in this project guards the *corpus* a Predictor
    sees; this guards the other argument, which is where a leak would be much harder to notice
    because every stored row would still audit clean.
    """

    name = "cheat"
    saw: pd.DataFrame

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        self.saw = fixtures
        return np.tile(UNIFORM, (len(fixtures), 1))


class TestAFixtureCarriesNoResult:
    """CONTEXT.md: "A Fixture ... exists before it is played and carries no result until it is."

    The corpus is a table of *played* matches, so the row a Fixture is drawn from also holds the
    Outcome, the goals and the match statistics. A Predictor is handed the Fixture, not the row.
    """

    def test_a_predictor_cannot_see_the_outcome_of_what_it_is_predicting(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        cheat = Cheat()

        schema.predictions_for(cheat, round_fixtures, evidence)

        assert "outcome" not in cheat.saw.columns
        assert "home_goals" not in cheat.saw.columns
        assert "away_goals" not in cheat.saw.columns

    def test_it_can_still_see_which_clubs_are_playing_and_when(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        cheat = Cheat()

        schema.predictions_for(cheat, round_fixtures, evidence)

        assert list(cheat.saw["home_club"]) == ["arsenal", "everton"]
        assert set(cheat.saw.columns) <= set(schema.VISIBLE_FIXTURE_COLUMNS)

    def test_it_can_see_the_market_line_the_fixture_was_priced_at(
        self, make_matches: Callable[..., pd.DataFrame], evidence: Evidence
    ) -> None:
        """The pre-match odds are sampled at the As-Of Instant itself (ADR 0001), so they are a
        fact about the Fixture rather than about how it turned out. Issue #8 needs them."""
        priced = make_matches({"date": "2024-08-17", "prematch_odds_home": 1.62})
        cheat = Cheat()

        schema.predictions_for(cheat, priced, evidence)

        assert cheat.saw["prematch_odds_home"].tolist() == [1.62]

    def test_the_closing_line_is_not_a_fact_about_the_fixture(
        self, make_matches: Callable[..., pd.DataFrame], evidence: Evidence
    ) -> None:
        """Closing odds absorb team news from after the As-Of Instant. They are the Ceiling Line's
        input and nothing else's, and the Ceiling Line is labelled everywhere as knowing more than
        the model can (ADR 0001)."""
        priced = make_matches({"date": "2024-08-17", "closing_odds_home": 1.66})
        cheat = Cheat()

        schema.predictions_for(cheat, priced, evidence)

        assert "closing_odds_home" not in cheat.saw.columns


class Privileged:
    """A Predictor claiming closing odds as a labelled exception, the way the Ceiling Line does."""

    name = "privileged"
    also_sees = ("closing_odds_home", "closing_odds_draw", "closing_odds_away")
    saw: pd.DataFrame

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        self.saw = fixtures
        return np.tile(UNIFORM, (len(fixtures), 1))


class Greedy:
    """A Predictor that tries to grant itself the answer sheet through the same door."""

    name = "greedy"
    also_sees = ("outcome",)

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        return np.tile(UNIFORM, (len(fixtures), 1))


class TestTheLabelledException:
    """How the Ceiling Line gets the one input the allow-list deliberately withholds (ADR 0001).

    Appending the closing odds to :data:`schema.VISIBLE_FIXTURE_COLUMNS` would hand team news from
    after the As-Of Instant to *every* Predictor, which is the leak the allow-list exists to
    prevent. So a Predictor names the extra columns it claims, the claim is checked against a
    short list of columns that may be claimed at all, and it is visible in the Predictor's own
    source rather than buried in the ledger.
    """

    def test_a_predictor_that_claims_the_closing_odds_is_handed_them(
        self, make_matches: Callable[..., pd.DataFrame], evidence: Evidence
    ) -> None:
        priced = make_matches({"date": "2024-08-17", "closing_odds_home": 1.66})
        ceiling = Privileged()

        schema.predictions_for(ceiling, priced, evidence)

        assert ceiling.saw["closing_odds_home"].tolist() == [1.66]

    def test_it_still_cannot_see_anything_else_about_the_fixture(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        """The exception widens the allow-list by exactly what was claimed and by nothing else."""
        ceiling = Privileged()

        schema.predictions_for(ceiling, round_fixtures, evidence)

        allowed = set(schema.VISIBLE_FIXTURE_COLUMNS) | set(Privileged.also_sees)
        assert set(ceiling.saw.columns) <= allowed
        assert "outcome" not in ceiling.saw.columns

    def test_only_the_closing_odds_may_be_claimed(self) -> None:
        """The list of claimable columns is itself an allow-list. Without it, `also_sees` would be
        a way for any Predictor to grant itself the Outcome and still audit clean."""
        assert set(schema.PRIVILEGED_FIXTURE_COLUMNS) == {
            "closing_odds_home",
            "closing_odds_draw",
            "closing_odds_away",
        }

    def test_claiming_anything_else_is_refused_by_name(
        self, round_fixtures: pd.DataFrame, evidence: Evidence
    ) -> None:
        with pytest.raises(schema.LedgerError, match="outcome"):
            schema.predictions_for(Greedy(), round_fixtures, evidence)

    def test_a_predictor_that_claims_nothing_sees_the_ordinary_allow_list(
        self, make_matches: Callable[..., pd.DataFrame], evidence: Evidence
    ) -> None:
        priced = make_matches({"date": "2024-08-17", "closing_odds_home": 1.66})
        cheat = Cheat()

        schema.predictions_for(cheat, priced, evidence)

        assert "closing_odds_home" not in cheat.saw.columns
