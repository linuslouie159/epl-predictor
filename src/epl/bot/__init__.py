"""The Telegram bot: what the schedule did, what it did not do, and the boards on demand.

Built at stage 17 (issue #20). Two halves, and they are separate processes on purpose.

**The push half** (:mod:`epl.bot.notify`) is short-lived and is invoked by `deploy/run_live.sh`
after every scheduled fire. It says what happened — a round sealed, a round scored, a run that
failed loudly, and above all a run that sealed nothing on a day it plausibly should have.

**The pull half** (:mod:`epl.bot.serve`) is long-lived, polls Telegram for commands, and answers
from this project's own registry rather than from files somebody hopes are up to date.

Three properties are not features and must survive any change here.

**It is read-only with respect to both Prediction stores.** It cannot seal, supersede, backfill or
score. A chat app is not a second door into `outputs/live/`: the sealed store's whole value is that
a Prediction in it was written before kickoff by the loop, under a moment nobody chose (ADR 0005),
and a message from a phone is the easiest imaginable way to write one that was not.
`tests/bot/test_the_bot_is_read_only.py` checks that by walking every import in this package rather
than by trusting the absence of a call — the same technique `tests/v2/test_stubs_are_unreachable.py`
uses, and for the same reason.

**It never breaks the run that triggered it.** Every send fails soft. A notifier that could take
down the loop would be a monitoring system that causes the outage it exists to report.

**There are numbers it may not quote**, and they are listed in :mod:`epl.bot.answers` where the
messages are built. Each is one this project has gone to some trouble to make hard to misquote, and
a chat message is the least ceremonious place in the system — the shortest path from a real
measurement to a sentence that misrepresents it.

**Why a second bot rather than an extension of the paper-trading bot on the same Pi.** That was the
repository owner's call and it has a consequence in its favour: a bot living here can `import epl`
inside the live loop's own image and answer from the registry, where a bot in the other project's
venv could only read files this repository happened to have written. The two projects deliberately
share no interpreter (docs/DECISIONS.md, "The schedule, and where it runs"), so the prior art on
that machine — `live/telegram_bot.py` and `live/notify.py` — was read and its patterns reused, and
neither file is imported.

**Why no Telegram library.** The Bot API surface this needs is four endpoints over HTTPS, and
`requests` is already in `environment.yml`. Adding an async framework would put an event loop
inside a package whose every other module is synchronous, and would add a dependency to a solve
where two free version choices already break the build outright (ADR 0009, and `arviz <1`). The
cost is :mod:`epl.bot.api`, which is small, and the gain is that every answer in this package is a
plain function returning a string — testable without a network, a token or an event loop.
"""

from __future__ import annotations

from epl.bot.settings import BotError, Settings

__all__ = ["BotError", "Settings"]
