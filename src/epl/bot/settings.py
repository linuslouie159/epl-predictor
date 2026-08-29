"""The token and the allowlist, read from the environment and from nowhere else.

Two rules, and both are refusals.

**The token is never in the repository.** It is read from the process environment, which on the Pi
is filled from `deploy/.env` — gitignored, like every `.env` in this tree. Nothing here reads a
file, so there is no path a token could be committed to by being put in the obvious place.

**An empty allowlist is a refusal, not an open bot.** The failure mode of the other spelling is
silent and total: a bot with no allowlist answers everyone, and a Telegram bot's username is
guessable. So :meth:`Settings.from_environment` raises rather than starting, and the process exits
before it has polled once.

The variables are prefixed ``EPL_`` for a reason that is specific to this machine. The Pi already
runs a Telegram bot for the paper-trading project, and that one reads ``TELEGRAM_BOT_TOKEN``
(docs/DECISIONS.md, "The schedule, and where it runs"). Two bots on one box sharing a variable name
is how this one would come up holding the other's token — and a bot that starts, polls successfully
and posts a football scoreboard into a trading chat has failed in a way that looks like working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Where the bot's own credentials live. Deliberately not the paper-trading bot's names.
TOKEN_VARIABLE = "EPL_TELEGRAM_TOKEN"
ALLOWLIST_VARIABLE = "EPL_TELEGRAM_ALLOWED_IDS"

#: Where a chat id may be listed to receive the push half's messages, when it is not simply every
#: allow-listed user. Optional: unset means "everyone on the allowlist", which is the common case
#: for a personal bot and the one that cannot go wrong by omission.
NOTIFY_VARIABLE = "EPL_TELEGRAM_NOTIFY_IDS"


class BotError(Exception):
    """The bot cannot start, or cannot answer. Never raised at a Telegram user."""


@dataclass(frozen=True)
class Settings:
    """Everything the bot needs that is not in this repository.

    ``notify_ids`` is who the push half writes to and defaults to ``allowed_ids``: the people who
    may ask are the people who may be told, unless somebody says otherwise.
    """

    token: str
    allowed_ids: frozenset[int]
    notify_ids: tuple[int, ...]

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> Settings:
        """Read the settings, or refuse with a message that says which variable is missing."""
        source = os.environ if environ is None else environ

        token = source.get(TOKEN_VARIABLE, "").strip()
        if not token:
            raise BotError(
                f"{TOKEN_VARIABLE} is not set, so there is no bot to run. Put it in deploy/.env, "
                "which is gitignored — a token in a tracked file is a token to revoke"
            )

        allowed = _ids(source.get(ALLOWLIST_VARIABLE, ""), ALLOWLIST_VARIABLE)
        if not allowed:
            raise BotError(
                f"{ALLOWLIST_VARIABLE} is empty. A bot with no allowlist answers everyone, and a "
                "bot username is guessable; refusing to start rather than run open"
            )

        notify = _ids(source.get(NOTIFY_VARIABLE, ""), NOTIFY_VARIABLE) or allowed
        return cls(
            token=token,
            allowed_ids=frozenset(allowed),
            notify_ids=tuple(sorted(notify)),
        )

    def allows(self, user_id: int | None) -> bool:
        """Whether this Telegram user gets an answer at all.

        ``None`` is a message with no sender — a channel post, or an edit Telegram attributes to
        nobody — and it is refused for the same reason an unlisted id is: there is no id to check.
        """
        return user_id is not None and user_id in self.allowed_ids

    def redacted(self) -> str:
        """The settings as they may safely be printed into a log the operator will paste."""
        return (
            f"token ...{self.token[-4:]}, {len(self.allowed_ids)} allowed, "
            f"{len(self.notify_ids)} notified"
        )


def _ids(raw: str, variable: str) -> set[int]:
    """Comma-separated Telegram user ids, with a complaint that names the variable it read."""
    found: set[int] = set()
    for item in raw.replace(" ", "").split(","):
        if not item:
            continue
        try:
            found.add(int(item))
        except ValueError as bad:
            raise BotError(
                f"{variable} holds {item!r}, which is not a Telegram user id. Ids are numeric — "
                "@userinfobot will tell you yours"
            ) from bad
    return found


__all__ = [
    "ALLOWLIST_VARIABLE",
    "NOTIFY_VARIABLE",
    "TOKEN_VARIABLE",
    "BotError",
    "Settings",
]
