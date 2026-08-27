"""Every registered Predictor run over the upcoming round, and the round sealed before kickoff.

The property that matters here is that there is no branch per Predictor. Five of the nine registered
Predictors cannot speak to an unplayed Fixture, and this module is supposed to find that out by
asking rather than by knowing — so the tests use Predictors that answer the question two different
ways and check that the run records the answer instead of assuming it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from epl.ledger import live as store
from epl.ledger import schema
from epl.live import seal, upcoming
from epl.predictors import Corpus, Evidence, register

#: The round every test here seals: two Fixtures on Saturday, anchored to the Friday before.
ROUND_FIXTURES: tuple[dict[str, object], ...] = (
    {"date": "2026-08-29", "time": "15:00", "home_club": "arsenal", "away_club": "wolves"},
    {"date": "2026-08-29", "time": "17:30", "home_club": "everton", "away_club": "brighton"},
)

AS_OF = pd.Timestamp("2026-08-28")
INSIDE_THE_WINDOW = pd.Timestamp("2026-08-28 14:00:30")


class Speechless:
    """A Predictor that covers no upcoming Fixture — a Pundit's permanent answer (issue #16)."""

    name = "speechless"

    def covers(self, fixtures: pd.DataFrame) -> npt.NDArray[np.bool_]:
        return np.zeros(len(fixtures), dtype=bool)

    def predict(  # pragma: no cover - never reached, which is what the test asserts
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        raise AssertionError("a Predictor that covers nothing must never be asked to predict")


@pytest.fixture
def corpus(make_matches: Callable[..., pd.DataFrame]) -> Corpus:
    """Three matches behind the round, so a Predictor that reads history has some."""
    return Corpus(
        make_matches(
            {"date": "2026-08-21"}, {"date": "2026-08-22"}, {"date": "2026-08-24"}
        )
    )


@pytest.fixture
def round_(make_matches: Callable[..., pd.DataFrame]) -> upcoming.PredictionRound:
    """The upcoming round, built the way `epl.live.upcoming` builds it."""
    fixtures = make_matches(*ROUND_FIXTURES).drop(
        columns=["home_goals", "away_goals", "outcome"]
    )
    return upcoming.next_round(fixtures, now=INSIDE_THE_WINDOW)


class TestWhoSpeaks:
    def test_a_predictor_that_covers_nothing_is_recorded_rather_than_asked(
        self, registry: dict[str, object], round_: upcoming.PredictionRound, corpus: Corpus,
        make_predictor: Callable[..., object]
    ) -> None:
        register(make_predictor("talker"))
        register(Speechless())

        rows, spoke, silent = seal.sealed_predictions(round_, corpus)

        assert spoke == ("talker",)
        assert silent == ("speechless",)
        assert set(rows["predictor"]) == {"talker"}

    def test_each_predictor_records_the_inputs_it_asked_for(
        self, registry: dict[str, object], round_: upcoming.PredictionRound, corpus: Corpus,
        make_predictor: Callable[..., object]
    ) -> None:
        """Evidence is per Predictor because `inputs_seen` is a fact about the Predictor. One
        shared Evidence would credit a Predictor that read nothing with what another one read."""
        register(make_predictor("reads_all", divisions=None))
        register(make_predictor("reads_none", divisions=()))

        rows, _, _ = seal.sealed_predictions(round_, corpus)

        seen = rows.groupby("predictor")["inputs_seen"].max().to_dict()
        assert seen == {"reads_all": 3, "reads_none": 0}

    def test_a_round_nobody_covers_is_refused_rather_than_sealed_empty(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        register(Speechless())

        with pytest.raises(schema.LedgerError, match="nothing to seal"):
            seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)


class TestSealingTheRound:
    def test_the_round_is_written_to_one_file_and_stamped_at_its_own_instant(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        register(make_predictor("elo_ish"))
        register(make_predictor("market_ish"))

        sealed = seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)

        assert sealed.path.name == "2026-08-28.csv"
        assert set(sealed.rows["as_of_instant"]) == {AS_OF}
        assert sorted(set(sealed.rows["predictor"])) == ["elo_ish", "market_ish"]

    def test_a_second_run_inside_the_same_round_refuses_rather_than_rewrites(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        """Acceptance criterion 3: running the live step twice within one round neither duplicates
        nor overwrites a Sealed Prediction."""
        register(make_predictor())
        seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)

        with pytest.raises(schema.LedgerError, match="already sealed"):
            seal.run(round_, corpus, now=pd.Timestamp("2026-08-28 18:00"), commit=False)

        assert len(store.sealed_rounds()) == 1

    def test_rows_come_back_in_a_fixed_order(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        """The same order the backtest writes in, so a sealed file and a backtest file of the same
        round diff against each other rather than against how the caller assembled them."""
        register(make_predictor("zeta"))
        register(make_predictor("alpha"))

        sealed = seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)

        assert list(sealed.rows["predictor"]) == ["alpha", "alpha", "zeta", "zeta"]
        assert list(sealed.rows["home_club"]) == ["arsenal", "everton"] * 2


class TestSuperseding:
    def test_a_correction_is_stamped_when_it_was_made_and_written_as_a_revision(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        register(make_predictor())
        seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)

        fixed = seal.run(
            round_, corpus, now=pd.Timestamp("2026-08-28 18:20:45.5"),
            supersede=True, commit=False,
        )

        assert fixed.path.name == "2026-08-28.1.csv"
        assert set(fixed.rows["as_of_instant"]) == {pd.Timestamp("2026-08-28 18:20:45")}
        assert set(fixed.rows["prediction_round"]) == {"2026-08-28"}

    def test_the_correction_sees_the_corpus_at_its_own_later_instant(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        """A superseding Prediction genuinely knows more, because it was genuinely made later.
        Recording it at the round's midnight to keep the comparison tidy would be the fiction the
        sealed store exists to prevent."""
        register(make_predictor())
        corpus = Corpus(
            make_matches({"date": "2026-08-24"}, {"date": "2026-08-28", "time": "12:30"})
        )
        seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)

        fixed = seal.run(
            round_, corpus, now=pd.Timestamp("2026-08-28 18:00"), supersede=True, commit=False
        )

        assert set(store.read().groupby("as_of_instant")["inputs_seen"].max()) == {1, 2}
        assert set(fixed.rows["inputs_seen"]) == {2}

    def test_the_store_audits_clean_with_both_readings_in_it(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        register(make_predictor())
        seal.run(round_, corpus, now=INSIDE_THE_WINDOW, commit=False)
        seal.run(round_, corpus, now=pd.Timestamp("2026-08-28 18:00"), supersede=True, commit=False)

        assert schema.audit(store.read()) == []


class TestCommitting:
    def test_a_sealed_round_is_committed_by_default(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        subprocess.run(["git", "-C", str(project_root), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(project_root), "config", "user.email", "t@example.com"], check=True
        )
        subprocess.run(["git", "-C", str(project_root), "config", "user.name", "T"], check=True)
        register(make_predictor())

        sealed = seal.run(round_, corpus, now=INSIDE_THE_WINDOW)

        assert sealed.commit is not None
        assert "committed" in sealed.describe()

    def test_without_a_repository_the_run_says_the_round_is_unproven(
        self, registry: dict[str, object], project_root: Path, round_: upcoming.PredictionRound,
        corpus: Corpus, make_predictor: Callable[..., object]
    ) -> None:
        """A seal that could not be committed is not evidence yet, and has to say so rather than
        report success — `seal_violations` will say the same once the round has kicked off."""
        register(make_predictor())

        sealed = seal.run(round_, corpus, now=INSIDE_THE_WINDOW)

        assert sealed.commit is None
        assert "NOT COMMITTED" in sealed.describe()
