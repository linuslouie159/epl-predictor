"""The four Bot API calls this project makes, and the seam that lets tests make none of them.

A :class:`Caller` turns a method name and a payload into Telegram's decoded JSON. Everything above
it — the poller, the notifier, every answer — is written against that, so no test in this package
needs a token, a network or an event loop. Same shape and same reason as
:mod:`epl.ingest.fetcher`: patching a module global would leave the production path untested and
would break silently the moment the call site moved.

**Four methods, which is why there is no Telegram library here.** `getMe` proves the token,
`setMyCommands` fills the "/" menu, `getUpdates` long-polls and `sendMessage` answers. An async
framework for that would put an event loop inside a package whose every other module is
synchronous, and would add a dependency to a conda solve where two free version choices already
break the build outright (ADR 0009, and `arviz <1`).

**Sending fails soft and polling does not**, and the asymmetry is the whole design. A notify failure
must never break the run that triggered it — monitoring that can take down the thing it monitors is
worse than none — so :meth:`Telegram.send` swallows everything and reports a bool. Polling is the
bot's own main loop: a failure there is the bot's to handle, and one of them,
:class:`Conflict`, must stop it dead rather than be retried.

**Two pollers on one token silently eat each other's updates.** Telegram hands each update to
whichever asked first, so a forgotten instance does not conflict visibly — it steals half the
replies, and the half that arrives makes the bot look like it is working. Telegram answers the
second poller with a 409, which is raised here as :class:`Conflict` and is fatal at the far end
(:func:`epl.bot.serve.run`). The OS lock in `serve` is the belt to this braces: it catches the
second instance *before* it has polled at all, and this catches the one on another machine that no
lock can see.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from epl.bot.settings import BotError, Settings

#: Telegram's own limit on one message. Longer answers are split rather than truncated — a
#: scoreboard cut off at the fourth Predictor is a scoreboard that misleads.
MESSAGE_LIMIT = 4096

#: What one chunk is allowed to reach. Under the limit, because splitting happens on line
#: boundaries and the last line has to fit.
CHUNK_LIMIT = 3900

#: How long `getUpdates` is held open. Long polling: the request returns early when something
#: arrives, so this is idle time rather than latency.
POLL_SECONDS = 30

#: Every method this bot calls. Written down because it is the whole of its Telegram surface, and
#: because a fifth appearing here should be a decision rather than a diff.
METHODS: tuple[str, ...] = ("getMe", "setMyCommands", "getUpdates", "sendMessage")


class TransportError(BotError):
    """Telegram could not be reached, or answered with something that is not JSON."""


class Conflict(BotError):
    """Another process is polling this token. Fatal: two pollers silently split the updates."""


class Caller(Protocol):
    """Turns a Bot API method and its payload into Telegram's decoded JSON response."""

    def __call__(self, method: str, payload: Mapping[str, Any], *, timeout: float) -> dict: ...


@dataclass
class Telegram:
    """The bot's side of the Bot API. Holds the token; never logs it.

    Keeps no record of what it has sent. An earlier draft kept one so tests could assert on content
    without a fake of their own, which was production state shaped by a test — and worse in this
    package than most, because :mod:`epl.bot.serve` is the one long-lived process here and the list
    would have grown for as long as the bot ran. Tests assert against their own :class:`Caller`.
    """

    settings: Settings
    caller: Caller | None = None

    def __post_init__(self) -> None:
        self._call: Caller = self.caller or http_caller(self.settings.token)

    def send(self, chat_id: int, text: str) -> bool:
        """Send one message, split if it is too long. Never raises.

        Returns whether every chunk landed. A caller that ignores the answer is behaving
        correctly — the point of failing soft is that the run carries on — but the notifier prints
        it, because a bot that has silently stopped delivering is the failure this whole package is
        about.
        """
        landed = True
        for chunk in split(text):
            try:
                answer = self._call(
                    "sendMessage",
                    {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
                    timeout=20.0,
                )
            except Exception as unreachable:
                print(f"[epl.bot] send to {chat_id} failed: {type(unreachable).__name__}")
                return False
            if not answer.get("ok"):
                print(f"[epl.bot] send to {chat_id} refused: {answer.get('description')}")
                landed = False
        return landed

    def broadcast(self, text: str) -> int:
        """Send to everyone on the notify list, and return how many were reached.

        One failure does not stop the others: the people on this list are on it separately, and a
        blocked bot in one chat is no reason for the rest to hear nothing.
        """
        return sum(1 for chat_id in self.settings.notify_ids if self.send(chat_id, text))

    def updates(self, offset: int | None = None, *, timeout: int = POLL_SECONDS) -> list[dict]:
        """Long-poll for messages. Raises :class:`Conflict` when something else is polling too.

        ``allowed_updates`` asks for messages only. Everything else this bot would ignore, and
        ignoring it after Telegram has queued and delivered it is slower and no safer.
        """
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        if offset is not None:
            payload["offset"] = offset
        answer = self._call("getUpdates", payload, timeout=timeout + 10.0)
        if not answer.get("ok"):
            if answer.get("error_code") == 409:
                raise Conflict(
                    "another process is already polling this bot token. Two pollers split the "
                    "updates between them silently, so this one is stopping rather than stealing "
                    f"half the replies — {answer.get('description')}"
                )
            raise TransportError(f"getUpdates refused: {answer.get('description')}")
        result = answer.get("result") or []
        return list(result)

    def set_commands(self, commands: Sequence[tuple[str, str]]) -> bool:
        """Register the "/" menu. Cosmetic, so a failure is reported and not fatal."""
        try:
            answer = self._call(
                "setMyCommands",
                {
                    "commands": json.dumps(
                        [{"command": name, "description": summary} for name, summary in commands]
                    )
                },
                timeout=20.0,
            )
        except TransportError:
            return False
        return bool(answer.get("ok"))

    def whoami(self) -> str | None:
        """The bot's own username, which is how a start-up line proves the token works."""
        try:
            answer = self._call("getMe", {}, timeout=20.0)
        except TransportError:
            return None
        return (answer.get("result") or {}).get("username")


def split(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """One message as however many chunks Telegram will accept, split on line boundaries.

    Truncating instead would be worse than it sounds here: the things this bot sends are boards and
    Prediction Rounds, and one cut off after the fourth Predictor is not a shorter answer but a
    different and misleading one. A single line longer than the limit is cut, because there is
    nothing else to do with it.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                chunks.append("\n".join(current))
                current, length = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if length + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def http_caller(token: str) -> Caller:
    """The real one. ``requests`` is imported here so nothing else in the bot depends on it.

    A non-2xx answer is *not* raised on: Telegram puts its own reason in the body — a 409 for a
    second poller, a 403 for a chat that blocked the bot — and raising on the status would throw
    that away and leave the caller guessing at a network problem.
    """
    import requests

    def call(method: str, payload: Mapping[str, Any], *, timeout: float) -> dict:
        url = f"https://api.telegram.org/bot{token}/{method}"
        try:
            response = requests.post(url, data=dict(payload), timeout=timeout)
            return dict(response.json())
        except Exception as unreachable:
            raise TransportError(f"{method}: {type(unreachable).__name__}") from unreachable

    return call


__all__ = [
    "CHUNK_LIMIT",
    "MESSAGE_LIMIT",
    "METHODS",
    "POLL_SECONDS",
    "Caller",
    "Conflict",
    "Telegram",
    "TransportError",
    "http_caller",
    "split",
]
