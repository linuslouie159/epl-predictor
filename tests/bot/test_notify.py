"""When the push half speaks, and — much more often — when it does not.

Issue #20's first, second, third and fourth criteria are all here, and so is its last one: a notify
failure never breaks the run that triggered it.

The hard part of a notifier is silence, not speech. Two of the three crontab lines are
`seal --push` and the second is a retry designed to find the round already sealed, so a bot that
announced every fire would be one nobody reads by November — which is the same argument that made
`NothingToSeal` exit 0 in the first place. So most of what follows checks that nothing is sent.

What decides is an artefact rather than a sentence: a `seal` that sealed leaves a file under
`outputs/live/` newer than the fire, and a `score` that scored leaves `outputs/live_scoreboard.csv`
newer than the fire. That is deliberate — a reworded log line cannot make this go quiet.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest
from log_blocks import block

from epl.bot import api, fires, notify
from epl.bot.settings import Settings
from epl.ingest.fixtures import fixtures_dir
from epl.live import upcoming
from epl.paths import live_dir, outputs_dir

#: A fire that ran a moment ago, so a file written now counts as written by it.
JUST_NOW = "2026-08-28 11:00:00 -0400"


@pytest.fixture
def settings() -> Settings:
    return Settings(token="123:abc", allowed_ids=frozenset({7}), notify_ids=(7,))


class Recorder:
    """A transport that accepts everything and remembers it."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, method: str, payload: dict, *, timeout: float) -> dict:
        if method == "sendMessage":
            self.sent.append(payload["text"])
        return {"ok": True, "result": True}


def a_fire(stamp: str = JUST_NOW, *body: str, command: str = "seal --push", exit_code: int = 0):
    """One fire, stamped in the recent past so file times can be compared against it."""
    return fires.parse(block(stamp, command, *body, exit_code=exit_code))


def recently(path: Path) -> None:
    """Make ``path`` look like it was written after :data:`JUST_NOW`.

    Only the modification time is touched. Rewriting a sealed round to change its timestamp would
    be the one thing this whole project forbids, even in a test — and here it would also destroy
    the fixture the assertion is about.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))


def earlier(path: Path) -> None:
    """Make ``path`` look like it was already there before :data:`JUST_NOW` fired.

    Which is what the retry sees: the round was sealed two and a half hours ago by the 16:00 slot,
    and the 18:30 slot writes nothing. Without this the fixture's file carries the real clock's
    modification time and looks, to any test whose fire is stamped in 2026, brand new.
    """
    before = (pd.Timestamp(JUST_NOW) - pd.Timedelta(hours=1)).timestamp()
    os.utime(path, (before, before))


class TestASealThatSealed:
    def test_it_announces_the_round(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        recently(live_dir() / "2026-08-28.csv")

        message = notify.compose("seal", found=a_fire())

        assert message is not None
        assert "Sealed" in message
        assert "2026-08-28" in message
        assert "Crystal Palace" in message

    def test_the_message_names_every_predictor_that_spoke(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """Criterion 1 asks for what each Predictor said, not merely that a round was sealed."""
        recently(live_dir() / "2026-08-28.csv")

        message = notify.compose("seal", found=a_fire())

        for predictor in ("dixon_coles", "elo", "market_line", "naive_baseline"):
            assert predictor in message


class TestASealThatSealedNothing:
    def test_the_retry_finding_the_round_already_sealed_says_nothing(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """The commonest fire of all, and the one that most needs to be silent.

        The file is older than the fire, so nothing was written; there is no news in a retry that
        did exactly what it is scheduled to do.
        """
        earlier(live_dir() / "2026-08-28.csv")
        found = a_fire(JUST_NOW, "2026-08-28 is already sealed — nothing to do")

        assert notify.compose("seal", found=found) is None

    def test_an_ordinary_quiet_week_says_nothing(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        found = a_fire(
            JUST_NOW, f"{upcoming.NO_FIXTURE_TO_PREDICT} at 2026-08-28T21:00:00. ..."
        )

        assert notify.compose("seal", found=found) is None

    def test_a_stale_upstream_file_through_both_fires_does_speak(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        """Criterion 3, which is the single most valuable thing this bot does.

        Two silent fires in one window that read identical bytes: upstream did not regenerate, so
        nobody can tell an empty week from a lost round. That is the indistinguishability the bot
        exists to break.
        """
        cache = fixtures_dir()
        cache.mkdir(parents=True)
        for name in ("fixtures_a.csv", "fixtures_b.csv"):
            (cache / name).write_bytes(b"Div,Date,HomeTeam,AwayTeam\r\n")
        silence = f"{upcoming.NO_FIXTURE_TO_PREDICT} at 2026-08-28T21:00:00. ..."
        found = fires.parse(
            block("2026-08-28 11:00:00 -0400", "seal --push", "rolling file: fixtures_a.csv",
                  silence)
            + block("2026-08-28 13:30:00 -0400", "seal --push", "rolling file: fixtures_b.csv",
                    silence)
        )

        message = notify.compose("seal", found=found)

        assert message is not None
        assert "open risk 7" in message


class TestAFailure:
    def test_it_is_loud_and_quotes_the_loop(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        """Criterion 4. `_seal`'s exit-code contract distinguishes several failures and the bot
        must not flatten them, so the loop's own words go in the message."""
        found = a_fire(
            JUST_NOW,
            "WARNING: the round is committed here and NOT PUSHED, so it proves nothing offsite.",
            exit_code=1,
        )

        message = notify.compose("seal", found=found)

        assert message is not None
        assert "NOT PUSHED" in message
        assert "exit 1" in message

    def test_a_failed_score_is_reported_too(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        found = a_fire(JUST_NOW, "IngestError: the file changed shape", command="score ",
                       exit_code=1)

        message = notify.compose("score", found=found)

        assert message is not None and "changed shape" in message


class TestAScore:
    def test_it_announces_a_board_that_was_just_written(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """Criterion 2. `_score` writes the file only when there is a board to write, so the file
        being newer than the fire is exactly the question "was anything scored this time?"."""
        recently(outputs_dir() / "live_scoreboard.csv")

        message = notify.compose("score", found=a_fire(command="score "))

        assert message is not None
        assert "Scored" in message

    def test_a_score_with_nothing_played_yet_says_nothing(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """Twice a week through the summer, and every one of them is news to nobody."""
        assert notify.compose("score", found=a_fire(command="score ")) is None


class TestItNeverBreaksTheRunThatTriggeredIt:
    def test_a_missing_token_is_printed_and_not_raised(
        self, sealed_store: pd.DataFrame, registered_predictors: None,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("EPL_TELEGRAM_TOKEN", raising=False)
        recently(live_dir() / "2026-08-28.csv")

        assert notify.run("seal", found=a_fire()) is None
        assert "not sending" in capsys.readouterr().out

    def test_a_telegram_that_refuses_everything_is_survived(
        self, sealed_store: pd.DataFrame, settings: Settings, registered_predictors: None
    ) -> None:
        def refuse(method: str, payload: dict, *, timeout: float) -> dict:
            raise RuntimeError("no network")

        recently(live_dir() / "2026-08-28.csv")
        bot = api.Telegram(settings, caller=refuse)

        assert notify.run("seal", settings=settings, telegram=bot, found=a_fire()) is not None

    def test_a_broken_answer_does_not_propagate(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            notify, "compose", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert notify.run("seal") is None
        assert "could not compose" in capsys.readouterr().out

    def test_the_command_line_always_exits_zero(
        self, project_root: Path, registered_predictors: None
    ) -> None:
        """`deploy/run_live.sh` records the loop's exit code, not the notifier's, and this is the
        second half of making that true: there is no path here that returns anything else."""
        assert notify.main(["seal"]) == 0
        assert notify.main([]) == 0


class TestItSendsWhatItComposed:
    def test_a_sealed_round_reaches_every_notified_chat(
        self, sealed_store: pd.DataFrame, settings: Settings, registered_predictors: None
    ) -> None:
        recently(live_dir() / "2026-08-28.csv")
        recorder = Recorder()
        bot = api.Telegram(settings, caller=recorder)

        notify.run("seal", settings=settings, telegram=bot, found=a_fire())

        assert recorder.sent
        assert "2026-08-28" in "".join(recorder.sent)
