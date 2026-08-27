"""The live loop run against the real corpus and the real Predictor registry.

Everything else about stage 13 can be tested with two invented Clubs. This cannot: the claim that
matters is that **nothing here knows which Predictors can speak to an unplayed Fixture**, and the
only way to check that is to register all nine of them and see which ones answer.

The Fixtures are hand-built and the docstrings say so wherever they are used. What is real is
everything else — the corpus behind them, the Club slugs, the frozen Pundit dataset, the anchor
rule, and the nine registered Predictors. On 27 August 2026 Football-Data's rolling
``fixtures.csv`` held no Premier League row at all, so a real upcoming round is exactly what this
project cannot obtain; inventing the pairings and keeping everything else real is the honest way
to test the loop, and the reason the invention is safe is that no assertion here depends on who is
playing whom.

Needs a populated ``data/raw/``, which is gitignored, so these skip when it is absent:

    python -m epl.ingest fetch
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import epl.ledger as ledger
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import live as store
from epl.ledger import schema
from epl.live import seal, upcoming
from epl.predictors import Corpus, registered
from epl.windows import LIVE_SEASON

pytestmark = pytest.mark.cache

#: A hand-built second round of 2026/27, in the shape `epl.ingest.parse_fixtures` produces. The
#: twenty Clubs are the twenty that really played the first round, so every slug resolves and every
#: Club has a rating; the pairings are invented, and nothing below depends on them.
INVENTED_ROUND: tuple[tuple[str, str, float, float, float], ...] = (
    ("arsenal", "man_city", 2.10, 3.40, 3.60),
    ("chelsea", "brighton", 1.85, 3.60, 4.30),
    ("liverpool", "everton", 1.40, 4.75, 8.00),
    ("man_united", "tottenham", 2.55, 3.45, 2.80),
    ("aston_villa", "newcastle", 2.45, 3.35, 3.00),
    ("crystal_palace", "fulham", 2.30, 3.30, 3.30),
    ("leeds", "brentford", 2.60, 3.40, 2.75),
    ("sunderland", "nottm_forest", 3.00, 3.35, 2.45),
    ("bournemouth", "ipswich", 1.95, 3.55, 3.90),
    ("coventry", "hull", 2.20, 3.30, 3.45),
)

#: Friday 28 August 2026 is the anchor; the Fixtures are the Saturday after it.
KICKOFF_DAY = "2026-08-29"
AS_OF = pd.Timestamp("2026-08-28")
INSIDE_THE_WINDOW = pd.Timestamp("2026-08-28 14:00:00")

#: Who can speak to a Fixture nobody has played. Not a setting — the expected outcome of asking
#: every registered Predictor `covers`, written down so that a Predictor changing its mind about
#: unplayed Fixtures is a test failure rather than a silent change to what gets sealed.
EXPECTED_SPOKE: tuple[str, ...] = ("dixon_coles", "elo", "market_line", "naive_baseline")

#: Both lists are compared sorted. Which order a Predictor is asked in is the registration order,
#: which belongs to `epl.ledger.PREDICTOR_PACKAGES` rather than to anything here.
EXPECTED_SILENT: tuple[str, ...] = (
    "ceiling_line",
    "lawrenson",
    "margin_map_lawrenson",
    "margin_map_sutton",
    "sutton",
)


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    ledger.register_all()
    return load_matches()


@pytest.fixture(scope="module")
def rolling() -> pd.DataFrame:
    """The invented round in the rolling file's own shape — no Season, no result."""
    return pd.DataFrame(
        {
            "division": "E0",
            "date": pd.Timestamp(KICKOFF_DAY).date(),
            "time": ["12:30", *["15:00"] * 8, "17:30"],
            "home_club": [fixture[0] for fixture in INVENTED_ROUND],
            "away_club": [fixture[1] for fixture in INVENTED_ROUND],
            "prematch_odds_home": [fixture[2] for fixture in INVENTED_ROUND],
            "prematch_odds_draw": [fixture[3] for fixture in INVENTED_ROUND],
            "prematch_odds_away": [fixture[4] for fixture in INVENTED_ROUND],
        }
    )


@pytest.fixture(scope="module")
def upcoming_round(rolling: pd.DataFrame, matches: pd.DataFrame) -> upcoming.PredictionRound:
    return upcoming.next_round(
        upcoming.to_predict(rolling, matches), now=INSIDE_THE_WINDOW
    )


@pytest.fixture(scope="module")
def sealed(
    upcoming_round: upcoming.PredictionRound, matches: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Every registered Predictor asked for the round. Elo rebuilds its pool and Dixon-Coles
    refits, so this is the expensive fixture and it is built once for the module."""
    return seal.sealed_predictions(upcoming_round, Corpus(matches))


class TestWhoCanSpeakToAnUnplayedFixture:
    def test_the_models_and_the_market_speak_and_nothing_else_does(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]
    ) -> None:
        """The three that cannot are the three stage 12 predicted, and each says so itself: a
        Pundit has published no call the frozen dataset holds, a Calibrated Pundit has no call to
        read, and the Ceiling Line has no closing odds because the match has not closed."""
        _, spoke, silent = sealed

        assert tuple(sorted(spoke)) == EXPECTED_SPOKE
        assert tuple(sorted(silent)) == EXPECTED_SILENT

    def test_every_registered_predictor_is_accounted_for(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]
    ) -> None:
        """A Predictor added later must land in one list or the other rather than vanish."""
        _, spoke, silent = sealed

        assert set(spoke) | set(silent) == {one.name for one in registered()}

    def test_the_market_line_covers_only_the_fixtures_it_was_priced_at(
        self, upcoming_round: upcoming.PredictionRound, matches: pd.DataFrame
    ) -> None:
        """The rolling file prices most Fixtures and not always all of them, and an unpriced one
        must be absent rather than invented — the same rule as the Seasons with no market."""
        unpriced = upcoming_round.fixtures.copy()
        unpriced.loc[0, ["prematch_odds_home", "prematch_odds_draw", "prematch_odds_away"]] = None
        thinner = upcoming.PredictionRound(
            upcoming_round.prediction_round,
            upcoming_round.as_of,
            upcoming_round.first_kickoff,
            unpriced,
        )

        rows, spoke, _ = seal.sealed_predictions(thinner, Corpus(matches))

        assert "market_line" in spoke
        assert len(rows.loc[rows["predictor"] == "market_line"]) == len(INVENTED_ROUND) - 1
        assert len(rows.loc[rows["predictor"] == "elo"]) == len(INVENTED_ROUND)


class TestWhatGetsSealed:
    def test_the_rows_audit_clean(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]
    ) -> None:
        rows, _, _ = sealed

        assert schema.audit(rows) == []

    def test_every_prediction_is_stamped_at_the_rounds_own_instant(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]
    ) -> None:
        rows, _, _ = sealed

        assert set(rows["as_of_instant"]) == {AS_OF}
        assert set(rows["prediction_round"]) == {"2026-08-28"}
        assert set(rows["season"]) == {LIVE_SEASON}

    def test_nothing_the_models_read_reaches_into_the_round(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]], matches: pd.DataFrame
    ) -> None:
        """The project's one rule, checked off the rows rather than asserted about the code: the
        latest input any Predictor took is the last match played before the As-Of Instant."""
        rows, _, _ = sealed
        played = matches.loc[matches["season"] == LIVE_SEASON]

        assert (rows["latest_input"].dropna() < AS_OF).all()
        assert rows["latest_input"].max() == pd.Timestamp("2026-08-24 20:00")
        assert pd.Timestamp(played["date"].max()) < AS_OF

    def test_the_market_and_the_models_disagree_about_the_favourite_somewhere(
        self, sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]
    ) -> None:
        """A weak but load-bearing sanity check: if the models were reading the odds off the
        Fixture rather than the corpus, every column here would be identical."""
        rows, _, _ = sealed
        quotes = rows.pivot_table(
            index=["home_club", "away_club"], columns="predictor", values="prob_home"
        )

        assert not quotes["elo"].equals(quotes["market_line"])
        assert not quotes["dixon_coles"].equals(quotes["elo"])
        assert (quotes["naive_baseline"].nunique()) == 1


class TestTheRoundReachesTheStore:
    def test_sealing_writes_one_committed_file_that_audits(
        self,
        project_root: Path,
        upcoming_round: upcoming.PredictionRound,
        matches: pd.DataFrame,
        sealed: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]],
    ) -> None:
        """End to end, into a temporary project root so the real `outputs/live/` is untouched."""
        written = seal.run(
            upcoming_round, Corpus(matches), now=INSIDE_THE_WINDOW, commit=False
        )

        assert written.path.name == "2026-08-28.csv"
        assert len(store.sealed_rounds()) == 1
        assert schema.audit(store.read()) == []
        assert len(store.read()) == len(sealed[0])
