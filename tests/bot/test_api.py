"""The four calls, the split, and the two failures that must behave differently.

`epl.bot.api` is the only module here that knows Telegram exists, and it is small on purpose: the
Bot API surface this project needs is four methods over HTTPS, which is why there is no async
framework in an otherwise synchronous package.

Two behaviours in it carry weight out of proportion to their size.

**Sending fails soft and polling does not.** A notify failure must never break the run that
triggered it, so `send` swallows everything; but `Conflict` — a second poller on the same token —
has to stop the bot dead, because two pollers do not conflict visibly. Telegram gives each update
to whoever asked first, so the forgotten instance steals half the replies and the half that arrives
makes everything look fine.

**Long answers are split, never truncated.** A board cut off after the fourth Predictor is not a
shorter answer; it is a different and misleading one, which is the whole subject of
`test_answers.py`.
"""

from __future__ import annotations

import pytest

from epl.bot import api
from epl.bot.settings import Settings

SETTINGS = Settings(token="123:abc", allowed_ids=frozenset({7, 8}), notify_ids=(7, 8))


class Fake:
    """Telegram, answering however a test needs it to."""

    def __init__(self, answer: dict | None = None, raises: Exception | None = None) -> None:
        self.answer = answer or {"ok": True, "result": True}
        self.raises = raises
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method: str, payload: dict, *, timeout: float) -> dict:
        self.calls.append((method, dict(payload)))
        if self.raises is not None:
            raise self.raises
        return self.answer


class TestSendingFailsSoft:
    def test_an_unreachable_telegram_is_reported_and_not_raised(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bot = api.Telegram(SETTINGS, caller=Fake(raises=api.TransportError("no network")))

        assert bot.send(7, "hello") is False
        assert "failed" in capsys.readouterr().out

    def test_a_refusal_from_telegram_is_reported_and_not_raised(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """403 is what a chat that blocked the bot answers, and it is not the bot's problem."""
        bot = api.Telegram(SETTINGS, caller=Fake({"ok": False, "description": "bot was blocked"}))

        assert bot.send(7, "hello") is False
        assert "bot was blocked" in capsys.readouterr().out

    def test_one_blocked_chat_does_not_stop_the_others(self) -> None:
        """The people on the notify list are on it separately."""
        refused = {7}

        def selectively(method: str, payload: dict, *, timeout: float) -> dict:
            if payload.get("chat_id") in refused:
                return {"ok": False, "description": "blocked"}
            return {"ok": True, "result": True}

        bot = api.Telegram(SETTINGS, caller=selectively)

        assert bot.broadcast("hello") == 1

    def test_it_reaches_every_notified_chat_once(self) -> None:
        caller = Fake()
        bot = api.Telegram(SETTINGS, caller=caller)

        bot.broadcast("hello")

        assert [payload["chat_id"] for _, payload in caller.calls] == [7, 8]
        assert {payload["text"] for _, payload in caller.calls} == {"hello"}


class TestPollingDoesNot:
    def test_a_second_poller_raises_conflict(self) -> None:
        conflict = {"ok": False, "error_code": 409, "description": "terminated by other getUpdates"}
        bot = api.Telegram(SETTINGS, caller=Fake(conflict))

        with pytest.raises(api.Conflict, match="already polling"):
            bot.updates()

    def test_any_other_refusal_is_a_transport_error(self) -> None:
        bot = api.Telegram(SETTINGS, caller=Fake({"ok": False, "description": "unauthorized"}))

        with pytest.raises(api.TransportError, match="unauthorized"):
            bot.updates()

    def test_it_asks_for_messages_only_and_passes_the_offset(self) -> None:
        caller = Fake({"ok": True, "result": []})
        bot = api.Telegram(SETTINGS, caller=caller)

        bot.updates(42)

        (_, payload), = caller.calls
        assert payload["offset"] == 42
        assert payload["allowed_updates"] == '["message"]'

    def test_an_offset_of_none_is_left_off_rather_than_sent_as_null(self) -> None:
        """Telegram's `offset` means "acknowledge everything before this"; sending a null or a 0
        would ask for the whole backlog on the first poll of every restart."""
        caller = Fake({"ok": True, "result": []})
        bot = api.Telegram(SETTINGS, caller=caller)

        bot.updates(None)

        (_, payload), = caller.calls
        assert "offset" not in payload


class TestSplittingALongAnswer:
    def test_a_short_message_is_one_chunk_and_is_untouched(self) -> None:
        assert api.split("one line") == ["one line"]

    def test_it_splits_on_line_boundaries(self) -> None:
        text = "\n".join(f"line {number}" for number in range(100))

        chunks = api.split(text, limit=40)

        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)
        assert "\n".join(chunks) == text

    def test_nothing_is_lost_or_duplicated(self) -> None:
        """The property that matters: a board split in three is still the whole board."""
        text = "\n".join(f"predictor_{number:03d}  0.19{number:02d}" for number in range(60))

        assert "\n".join(api.split(text, limit=100)) == text

    def test_a_single_line_longer_than_the_limit_is_cut_rather_than_dropped(self) -> None:
        chunks = api.split("x" * 250, limit=100)

        assert [len(chunk) for chunk in chunks] == [100, 100, 50]

    def test_the_chunk_limit_is_under_telegram_s_own(self) -> None:
        assert api.CHUNK_LIMIT < api.MESSAGE_LIMIT

    def test_a_long_answer_arrives_as_several_messages(self) -> None:
        caller = Fake()
        bot = api.Telegram(SETTINGS, caller=caller)

        bot.send(7, "\n".join("line" for _ in range(2000)))

        assert len([call for call in caller.calls if call[0] == "sendMessage"]) > 1


class TestTheSurfaceIsWrittenDown:
    def test_every_method_the_bot_calls_is_named(self) -> None:
        """`METHODS` is the whole of this project's Telegram surface, and a fifth should be a
        decision rather than a diff."""
        assert set(api.METHODS) == {"getMe", "setMyCommands", "getUpdates", "sendMessage"}

    def test_the_cosmetic_calls_fail_soft(self) -> None:
        """The "/" menu and the username check are conveniences; neither may stop the bot."""
        bot = api.Telegram(SETTINGS, caller=Fake(raises=api.TransportError("no network")))

        assert bot.set_commands([("help", "This menu")]) is False
        assert bot.whoami() is None

    def test_the_token_is_not_in_anything_this_module_prints(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bot = api.Telegram(SETTINGS, caller=Fake({"ok": False, "description": "nope"}))

        bot.send(7, "hello")

        assert "123:abc" not in capsys.readouterr().out
