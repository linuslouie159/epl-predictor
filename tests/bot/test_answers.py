"""What the bot says, and the numbers it may not say — issue #20.

The messages themselves are ordinary rendering and most of what is below is about the second half.
Issue #20 carries a list headed "Things the bot must not say", and it opens by insisting these are
not style notes: each is a number this project has gone to some trouble to make hard to misquote,
and a chat message is the shortest path in the whole system from a real measurement to a sentence
that misrepresents it. There is no reviewer between a bot and a phone.

So the list is executable here. Where a rule can be enforced by construction it is, and the test
says which construction — a Predictor's `note` travelling with it is one function call in
`epl.bot.answers`, not a per-Predictor branch, for the same reason the scoreboard has none.

Everything runs against a machine shaped like the Pi: one sealed round, a match table with half of
it played, and **no** `outputs/backtest/`, because that store is gitignored and is written by a
command nothing schedules (`deploy/crontab`). A bot that quoted the README's headline numbers on
such a machine would be inventing a measurement, which is the failure this file is most concerned
with.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from log_blocks import THE_FIRE_THAT_PROVED_THE_SCHEDULE, block

from epl.bot import answers, fires
from epl.ledger import scoreboard
from epl.paths import outputs_dir
from epl.windows import EVALUATION_WINDOW, season_label

#: The three-way board's opponents and the two Pundits, whose numbers travel with a caveat or not
#: at all. `epl.predictors.note` is where the caveat comes from, so this is a list of names only.
CAVEATED = ("ceiling_line", "lawrenson", "sutton")


class TestTheSealedRound:
    """Criterion 1: the round, its Fixtures, and what each Predictor said."""

    def test_it_names_the_round_its_fixtures_and_every_predictor(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        text = answers.sealed_round()

        assert "2026-08-28" in text
        assert "Crystal Palace" in text and "Manchester City" in text
        assert "Liverpool" in text and "Nottingham Forest" in text
        for predictor in ("dixon_coles", "elo", "market_line", "naive_baseline"):
            assert predictor in text

    def test_it_quotes_the_probabilities_that_were_sealed(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """The Predictions are the evidence; a message about them that rounded away the disagreement
        between two Predictors would be a message about nothing."""
        text = answers.sealed_round()

        assert "18%" in text  # dixon_coles Home on the first Fixture
        assert "46%" in text  # naive_baseline Home, which is a different opinion entirely

    def test_it_says_who_was_silent_and_why(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """Five of nine say nothing about an unplayed Fixture, and a blank column reads as a bug.

        This is issue #20's "never a Pundit column on the live board" from the other direction: the
        absence is explained where it appears, rather than left for the reader to wonder about.
        """
        text = answers.sealed_round()

        assert "silent" in text.lower()
        assert "lawrenson" in text and "ceiling_line" in text

    def test_a_named_round_can_be_asked_for(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        assert "2026-08-28" in answers.sealed_round("2026-08-28")

    def test_a_round_that_was_never_sealed_is_said_so_rather_than_invented(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        text = answers.sealed_round("2026-09-04")

        assert "2026-09-04" in text
        assert "sealed" in text.lower()

    def test_an_empty_store_answers_rather_than_raising(self, project_root: Path) -> None:
        """A fresh clone before the first fire. The bot has to be startable there."""
        assert answers.sealed_round()


class TestTheLiveSeasonBoard:
    """Criterion 6, and two of the things it must not say."""

    def test_it_scores_only_the_fixtures_that_have_been_played(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        text = answers.live_board()

        assert season_label(2026) in text
        assert "market_line" in text

    def test_it_never_quotes_a_calibrated_live_figure(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """A calibration map needs a track record behind it and the Live Season has none.

        `epl.live.__main__._score` prints the pre-calibration half and nothing else for exactly this
        reason, and this is the same two lines rather than a second policy. A "calibrated" live
        column would be the raw one under another name — which is the sort of number that gets
        quoted, since it looks like it came from somewhere.
        """
        text = answers.live_board()

        assert scoreboard.CALIBRATED_PREFIX not in text
        for column in scoreboard.POST_CALIBRATION_COLUMNS:
            if column.startswith(scoreboard.CALIBRATED_PREFIX):
                assert column not in text

    def test_it_says_how_few_fixtures_are_behind_the_number(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """A live RPS over one round is noise, and noise beside 0.1975 invites the one comparison
        this project exists to prevent. The sample size travels with the number."""
        text = answers.live_board()

        assert "1 Fixture" in text
        assert str(len(EVALUATION_WINDOW)) in text or "Evaluation Window" in text

    def test_nothing_played_yet_is_a_sentence_rather_than_an_empty_table(
        self, sealed_store: pd.DataFrame, project_root: Path, registered_predictors: None
    ) -> None:
        from epl.paths import processed_dir

        processed_dir().mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["season", "division", "date", "home_club", "away_club", "outcome"]
        ).to_csv(processed_dir() / "matches.csv", index=False)

        text = answers.live_board()

        assert "none" in text.lower() or "not" in text.lower()

    def test_a_machine_with_no_match_table_says_which_command_builds_one(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """`data/processed/` is gitignored, so a clone has none until something ingests.

        It must not exit: `epl.ingest.match_table` raises `SystemExit` on a missing file, which is
        right for a command line and would take the bot's process down mid-answer.
        """
        text = answers.live_board()

        assert "epl.ingest" in text or "epl.live score" in text


class TestTheEvaluationWindowBoard:
    """Criterion 6 again, and the one answer this project cannot compute on the Pi."""

    def test_it_refuses_to_quote_a_board_that_is_not_on_this_machine(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        """`outputs/scoreboard.csv` is derived, regenerable and gitignored (ADR 0005's reasoning),
        and `backfill` is the one command `deploy/crontab` must never schedule. So the Pi has no
        Evaluation Window board, and the honest answer is to say so and name the command — rather
        than print the README's numbers, which would be a bot reciting a measurement it has not
        got."""
        text = answers.evaluation_board()

        assert "epl.ledger scoreboard" in text
        assert "0.19" not in text

    def test_it_reads_the_board_when_there_is_one(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        outputs_dir().mkdir(parents=True, exist_ok=True)
        board = pd.DataFrame(
            [
                {"predictor": "market_line", "fixtures": 7980, "rps": 0.1936, "brier": 0.5684,
                 "log_loss": 0.9582, "accuracy": 0.5471, "ece": 0.0061},
                {"predictor": "ceiling_line", "fixtures": 2660, "rps": 0.1968, "brier": 0.5717,
                 "log_loss": 0.9639, "accuracy": 0.5498, "ece": 0.0060},
            ]
        )
        board.to_csv(scoreboard.path(), index=False)

        text = answers.evaluation_board()

        assert "0.1936" in text
        assert "market_line" in text

    def test_it_says_when_the_board_it_is_quoting_was_written(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        """A derived file has no built-in date and this one is regenerated by hand.

        Quoting it without saying when it was made is how a bot comes to report last month's
        scoreboard as though it were this month's.
        """
        outputs_dir().mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"predictor": "elo", "fixtures": 7980, "rps": 0.1994}]).to_csv(
            scoreboard.path(), index=False
        )

        assert str(pd.Timestamp.now().year) in answers.evaluation_board()


class TestThingsTheBotMustNotSay:
    """Issue #20's list, made executable. Each of these is a real number, misquoted."""

    def _every_message(self) -> list[str]:
        """Every answer the bot can produce on this machine, for a sweep over all of them."""
        found = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)
        return [
            answers.help_text(),
            answers.sealed_round(),
            answers.live_board(),
            answers.evaluation_board(),
            answers.health(found, now=pd.Timestamp("2026-08-28 22:00", tz=fires.LOCAL_ZONE)),
        ]

    def test_the_sequential_diagnostic_is_never_quoted_anywhere(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """ADR 0002's diagnostic produces a number that must never be quoted as a score.

        Checked as an absence of the word because that is all a text check can do; the load-bearing
        version is `tests/bot/test_the_bot_is_read_only.py`, which walks the package's imports and
        finds that nothing here can reach `epl.ledger.backtest.sequential` at all.
        """
        for text in self._every_message():
            assert "sequential" not in text.lower()

    def test_accuracy_is_never_the_headline(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """RPS is primary; accuracy is for lay explanation only (CLAUDE.md).

        "Headline" is made concrete: on any board, RPS appears before accuracy — which is
        `epl.ledger.scoreboard.METRICS`' own order, and comes from using it rather than a second
        list written here.
        """
        for text in self._every_message():
            if "accuracy" in text and "rps" in text:
                assert text.index("rps") < text.index("accuracy")

    @pytest.mark.parametrize("predictor", CAVEATED)
    def test_a_predictor_whose_number_needs_a_caveat_never_appears_without_it(
        self, predictor: str, project_root: Path, registered_predictors: None
    ) -> None:
        """The Ceiling Line's 0.1968 is not worse than the Market Line's 0.1936 — different
        Fixtures — and a Pundit's ~0.334 is not comparable to the board's 7,980. Both say so in
        their own `note`, and the note is read off the registry rather than retyped, so a Predictor
        registered tomorrow gets the same treatment with no change here.
        """
        outputs_dir().mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"predictor": predictor, "fixtures": 2660, "rps": 0.1968}]
        ).to_csv(scoreboard.path(), index=False)

        from epl import predictors

        text = answers.evaluation_board()

        assert predictor in text
        # Whole, not truncated. An earlier draft cut a `note` at 180 characters, which on
        # `margin_map_lawrenson`'s 623 landed mid-sentence and dropped "It may beat a model on this
        # board, which would be a finding about the information in the calls and never a verdict on
        # the forecaster" — the exact clause standing between this bot and "Sutton beat the model"
        # (ADR 0003). A caveat shortened to fit a phone no longer says what it was written to say.
        assert " ".join(predictors.note(predictor).split()) in text

    def test_a_pundit_is_never_said_to_have_beaten_the_model(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """The Calibrated Pundits are deliberately not named after the forecasters (ADR 0003).

        "Sutton beat the model" is a sentence no output may support, and the reason it cannot be
        assembled here is structural: nothing in this package ranks Predictors against each other
        or writes a comparison in prose. A board is a board.
        """
        for text in self._every_message():
            assert "beat" not in text.lower()

    def test_a_pundit_column_is_never_shown_blank_on_the_live_board(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """It is empty by design and fills in retrospectively (issue #16), and a blank cell reads
        as a bug. The board holds the Predictors that spoke; the absence is explained in words."""
        text = answers.live_board()

        assert "lawrenson" not in text or "#16" in text


class TestTheHealthOfTheLastFire:
    """Criterion 6's fourth answer, and where criteria 3 and 5 surface on demand."""

    def test_it_reports_the_last_fire_of_each_subcommand(self, project_root: Path) -> None:
        found = fires.parse(
            block("2026-08-28 01:00:00 -0400", "score ", "nothing to score")
            + THE_FIRE_THAT_PROVED_THE_SCHEDULE
        )

        text = answers.health(found, now=pd.Timestamp("2026-08-28 22:00", tz=fires.LOCAL_ZONE))

        assert "seal" in text and "score" in text
        assert "exit 0" in text

    def test_a_schedule_that_has_never_fired_is_said_plainly(self, project_root: Path) -> None:
        text = answers.health((), now=pd.Timestamp("2026-08-28 22:00", tz=fires.LOCAL_ZONE))

        assert "never" in text.lower()

    def test_a_concern_reaches_the_health_answer(self, project_root: Path) -> None:
        """Criterion 5 on demand: the gaps are visible without waiting for a push."""
        found = fires.parse(block("2026-08-25 11:00:00 -0400", "seal --push", "worked"))

        text = answers.health(found, now=pd.Timestamp("2026-09-01 21:00", tz=fires.LOCAL_ZONE))

        assert "open risk 6" in text

    def test_a_failed_fire_shows_its_own_words(self, project_root: Path) -> None:
        """`epl.live.__main__._seal`'s exit-code contract distinguishes several failures, and the
        bot must not flatten them: `NOT PUSHED` is not a stale `LIVE_SEASON`."""
        found = fires.parse(
            block(
                "2026-08-28 11:00:00 -0400",
                "seal --push",
                "WARNING: the round is committed here and NOT PUSHED, proving nothing offsite.",
                exit_code=1,
            )
        )

        text = answers.health(found, now=pd.Timestamp("2026-08-28 22:00", tz=fires.LOCAL_ZONE))

        assert "NOT PUSHED" in text
        assert "exit 1" in text


class TestTheMessagesThePushHalfSends:
    def test_a_failure_carries_the_loop_s_own_output(self, project_root: Path) -> None:
        (fire,) = fires.parse(
            block("2026-08-28 11:00:00 -0400", "seal --push", "something broke", exit_code=1)
        )

        text = answers.failure(fire)

        assert "something broke" in text
        assert "exit 1" in text

    def test_an_unfinished_fire_is_not_reported_as_an_exit_code(self, project_root: Path) -> None:
        """A container killed mid-run has no exit code, and inventing one would be a claim."""
        (fire,) = fires.parse(
            block("2026-08-28 11:00:00 -0400", "seal --push", "half a run", exit_code=None)
        )

        text = answers.failure(fire)

        assert "exit" not in text.lower() or "no exit" in text.lower()

    def test_a_quiet_fire_says_which_silence_it_was(self, project_root: Path) -> None:
        """The two silences are different findings and the loop cannot tell them apart afterwards.

        This is where `epl.live.upcoming`'s two named constants earn their keep.
        """
        from epl.bot import watch
        from epl.live import upcoming

        (fire,) = fires.parse(
            block(
                "2026-08-25 11:00:00 -0400",
                "seal --push",
                f"{upcoming.NO_FIXTURE_TO_PREDICT} at 2026-08-25T21:00:00. ...",
            )
        )
        concern = watch.Concern(watch.STALE_UPSTREAM, "both fires read the same bytes", "…")

        text = answers.quiet(fire, concern)

        assert "rolling file" in text
        assert "open risk 7" in text
