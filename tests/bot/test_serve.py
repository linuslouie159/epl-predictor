"""The pull half: who gets an answer, what the answers are, and why only one poller may run.

Three of issue #20's criteria live here.

**Only allow-listed ids get any answer at all.** Not "a reduced answer" and not "a polite refusal
containing the board" — nothing. A Telegram bot's username is guessable and this one can read a
machine's log.

**Two instances cannot poll one token.** The failure is worse than a crash and looks like neither:
Telegram hands each update to whichever poller asked first, so a forgotten instance steals half the
replies and the half that arrives makes everything look fine. Two guards, and they catch different
things — an OS lock catches a second process on this machine before it polls at all, and Telegram's
409 catches one on a machine no lock can see.

**The bot is read-only.** There is no handler that writes, and no argument that can become one.
Checked here at the dispatch table and, structurally, in `test_the_bot_is_read_only.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from epl.bot import api, serve
from epl.bot.settings import Settings

ALLOWED = 4242
STRANGER = 9999


@pytest.fixture
def settings() -> Settings:
    return Settings(token="123:abc", allowed_ids=frozenset({ALLOWED}), notify_ids=(ALLOWED,))


class FakeCaller:
    """Telegram, as far as anything above :mod:`epl.bot.api` can tell."""

    def __init__(self, *batches: list[dict], conflict: bool = False) -> None:
        self.batches = list(batches)
        self.conflict = conflict
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method: str, payload: dict, *, timeout: float) -> dict:
        self.calls.append((method, dict(payload)))
        if method == "getUpdates":
            if self.conflict:
                return {"ok": False, "error_code": 409, "description": "terminated by other"}
            return {"ok": True, "result": self.batches.pop(0) if self.batches else []}
        if method == "getMe":
            return {"ok": True, "result": {"username": "epl_predictor_bot"}}
        return {"ok": True, "result": True}

    def messages(self) -> list[str]:
        return [payload["text"] for method, payload in self.calls if method == "sendMessage"]


def message(text: str, *, user: int = ALLOWED, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 7,
            "from": {"id": user, "is_bot": False},
            "chat": {"id": user, "type": "private"},
            "text": text,
        },
    }


def telegram(settings: Settings, *batches: list[dict], conflict: bool = False) -> api.Telegram:
    return api.Telegram(settings, caller=FakeCaller(*batches, conflict=conflict))


class TestTheAllowlist:
    def test_a_listed_user_is_answered(self, settings: Settings, project_root: Path) -> None:
        caller = FakeCaller()
        bot = api.Telegram(settings, caller=caller)

        serve.handle(message("/help"), settings, bot, now=pd.Timestamp.now(tz="UTC"))

        assert any("read-only" in text for text in caller.messages())

    def test_a_stranger_gets_nothing_at_all(self, settings: Settings, project_root: Path) -> None:
        """Not a refusal message either. A reply confirms the bot exists and that the id was tried,
        which is the one thing an unlisted id should not learn."""
        caller = FakeCaller()
        bot = api.Telegram(settings, caller=caller)

        serve.handle(
            message("/board", user=STRANGER), settings, bot, now=pd.Timestamp.now(tz="UTC")
        )

        assert caller.messages() == []

    def test_a_message_from_nobody_is_refused(self, settings: Settings, project_root: Path) -> None:
        """A channel post has no `from`. There is no id to check, so there is no answer."""
        caller = FakeCaller()
        bot = api.Telegram(settings, caller=caller)
        anonymous = {"update_id": 1, "message": {"chat": {"id": ALLOWED}, "text": "/help"}}

        serve.handle(anonymous, settings, bot, now=pd.Timestamp.now(tz="UTC"))

        assert caller.messages() == []

    def test_an_empty_allowlist_never_reaches_a_running_bot(self) -> None:
        """Refused at the settings, so there is no state in which the bot is open."""
        from epl.bot.settings import BotError

        with pytest.raises(BotError, match="allowlist"):
            Settings.from_environment(
                {"EPL_TELEGRAM_TOKEN": "123:abc", "EPL_TELEGRAM_ALLOWED_IDS": ""}
            )


class TestTheCommands:
    def test_every_command_in_the_menu_can_be_dispatched(
        self, settings: Settings, project_root: Path
    ) -> None:
        """The "/" menu and the dispatch table are the same tuple, so they cannot disagree."""
        for command in serve.COMMANDS:
            assert serve.dispatch(f"/{command.name}", now=pd.Timestamp.now(tz="UTC")) is not None

    def test_an_unknown_command_says_so_without_pretending_to_answer(
        self, settings: Settings, project_root: Path
    ) -> None:
        answer = serve.dispatch("/backfill", now=pd.Timestamp.now(tz="UTC"))

        assert answer is not None
        assert "/help" in answer

    def test_plain_text_is_not_a_command_and_is_ignored(
        self, settings: Settings, project_root: Path
    ) -> None:
        assert serve.dispatch("how are the models doing", now=pd.Timestamp.now(tz="UTC")) is None

    def test_a_command_addressed_to_this_bot_by_name_still_works(
        self, settings: Settings, project_root: Path
    ) -> None:
        """Telegram writes `/help@epl_predictor_bot` in a group, and a bot that only matched the
        bare form would look broken in exactly the place several people can see it."""
        assert serve.dispatch("/help@epl_predictor_bot", now=pd.Timestamp.now(tz="UTC")) is not None

    def test_the_round_command_takes_the_round_as_an_argument(
        self, sealed_store: pd.DataFrame, settings: Settings
    ) -> None:
        answer = serve.dispatch("/round 2026-08-28", now=pd.Timestamp.now(tz="UTC"))

        # `/round` is now an alias for `/week`, and the round is named as a day rather than
        # as its id — the id is still what the argument takes.
        assert answer is not None and "Friday 28 August" in answer

    def test_no_command_writes_anything(self, settings: Settings, project_root: Path) -> None:
        """Criterion 8. The dispatch table is the bot's whole vocabulary, and none of it is a verb.

        A chat app must not be a second door into outputs/live/ (ADR 0005): a Prediction in that
        store is evidence because the loop wrote it before kickoff under a moment nobody chose, and
        a message from a phone is the easiest imaginable way to write one that was not.
        """
        forbidden = ("seal", "supersede", "backfill", "score", "fetch", "build", "write")
        assert [command.name for command in serve.COMMANDS if command.name in forbidden] == []


class TestOnlyOnePoller:
    def test_a_second_start_on_this_machine_fails_before_it_polls(
        self, project_root: Path
    ) -> None:
        with serve.single_instance():
            with pytest.raises(serve.AlreadyRunning):
                with serve.single_instance():
                    pass  # pragma: no cover - the point is that this is not reached

    def test_the_lock_is_released_when_the_bot_stops(self, project_root: Path) -> None:
        with serve.single_instance():
            pass
        with serve.single_instance():
            pass

    def test_a_poller_somewhere_else_is_fatal_rather_than_retried(
        self, settings: Settings, project_root: Path
    ) -> None:
        """No lock can see another machine, and Telegram's 409 is the only evidence there is.

        Retrying would be the worst of the three options: the two pollers would go on splitting the
        updates, and both would look alive.
        """
        bot = telegram(settings, conflict=True)

        exit_code = serve.run(settings, telegram=bot, forever=False)

        assert exit_code == 1


class TestThePollLoop:
    def test_it_answers_what_arrives(self, settings: Settings, project_root: Path) -> None:
        caller = FakeCaller([message("/help", update_id=11)])
        bot = api.Telegram(settings, caller=caller)

        serve.poll_once(bot, settings, offset=None, now=pd.Timestamp.now(tz="UTC"))

        assert any("read-only" in text for text in caller.messages())

    def test_it_asks_only_for_messages(self, settings: Settings, project_root: Path) -> None:
        caller = FakeCaller([])
        bot = api.Telegram(settings, caller=caller)

        serve.poll_once(bot, settings, offset=None, now=pd.Timestamp.now(tz="UTC"))

        (_, payload), = [call for call in caller.calls if call[0] == "getUpdates"]
        assert "message" in payload["allowed_updates"]

    def test_the_offset_moves_past_what_was_handled(
        self, settings: Settings, project_root: Path
    ) -> None:
        """Telegram redelivers anything not acknowledged, so an offset that did not advance is a
        bot answering the same message for ever."""
        caller = FakeCaller([message("/help", update_id=11)])
        bot = api.Telegram(settings, caller=caller)

        offset = serve.poll_once(bot, settings, offset=None, now=pd.Timestamp.now(tz="UTC"))

        assert offset == 12

    def test_an_empty_batch_leaves_the_offset_alone(
        self, settings: Settings, project_root: Path
    ) -> None:
        caller = FakeCaller([])
        bot = api.Telegram(settings, caller=caller)

        assert serve.poll_once(bot, settings, offset=5, now=pd.Timestamp.now(tz="UTC")) == 5

    def test_one_bad_handler_does_not_stop_the_loop(
        self, settings: Settings, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bot that died on a malformed answer would be monitoring that goes down on its own."""

        def explode(*_: Any, **__: Any) -> str:
            raise RuntimeError("the board is on fire")

        monkeypatch.setattr(serve, "dispatch", explode)
        caller = FakeCaller([message("/board", update_id=3)])
        bot = api.Telegram(settings, caller=caller)

        offset = serve.poll_once(bot, settings, offset=None, now=pd.Timestamp.now(tz="UTC"))

        assert offset == 4
        assert any("the board is on fire" in text for text in caller.messages())


class TestTheStartUpLine:
    def test_it_never_prints_the_token(self, settings: Settings) -> None:
        """A start-up line is the first thing anybody pastes into a chat when asking for help."""
        assert "123:abc" not in settings.redacted()
        assert settings.redacted().endswith("notified")


class TestTheFirstThingAnybodySends:
    """`/start` is the button Telegram shows on an unopened conversation.

    It is therefore the first message this bot will ever receive from anyone, and it was answered
    with "No such command" on the first real deployment — a poor account of a bot that works. Not a
    menu entry, because Telegram offers `/start` itself and `COMMANDS` is also the menu.
    """

    def test_start_answers_as_help(self, project_root: Path) -> None:
        now = pd.Timestamp.now(tz="UTC")

        assert serve.dispatch("/start", now=now) == serve.dispatch("/help", now=now)

    def test_it_is_not_offered_twice_in_the_menu(self) -> None:
        assert "start" not in {command.name for command in serve.COMMANDS}

    def test_every_alias_points_at_a_real_command(self) -> None:
        """An alias for a command that no longer exists would answer "No such command" again."""
        names = {command.name for command in serve.COMMANDS}
        assert set(serve.ALIASES.values()) <= names
