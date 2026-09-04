"""Markup, layout and vocabulary: the one place a measurement becomes something to read.

:mod:`epl.bot.answers` decides *what* a message says and this module decides what it looks like.
The split matters because the rules about what may be said are subtle and per-message — no
calibrated figure for the Live Season, a sample size beside every RPS, a Predictor's `note` quoted
whole — while the rules about how it looks are uniform and mechanical, and mixing the two is how a
formatting change comes to drop a caveat.

**Telegram HTML, not Markdown, and not plain text.** This module's predecessor sent plain text on
purpose: a Markdown parse error on a Club name with an apostrophe in it is a message that does not
arrive, and the one thing this bot must not do is go quiet. That reasoning was right and is what
chose HTML here rather than overruling it. MarkdownV2 needs eighteen characters escaped in running
text; HTML needs three, and :func:`html.escape` is total and mechanical rather than a judgement made
per string. Every value that reaches a message goes through :func:`escape` in this module and
nowhere else, so there is one place to check the claim.

The remaining risk is handled at the other end: :meth:`epl.bot.api.Telegram.send` retries once as
plain text if Telegram refuses to parse a message. Going quiet is still the failure that matters.

**Tables live inside a fixed-width block and are the whole reason for the parse mode.** Telegram
renders running text in a proportional font, so a column of probabilities does not line up and is
unreadable. A fixed-width block fixes that and introduces the one new way to be wrong: a line too
long for a narrow phone wraps, and a wrapped line in a fixed-width block is worse than no block at
all. :data:`PRE_WIDTH` is the budget, and it is checked by a test over every message this bot can
send rather than trusted to whoever writes the next one.

**No emoji, and no non-ASCII at all.** The repository owner asked for the first; the second is how
it is kept, because a bullet character is not an emoji and reads as one on half the devices that
render it. :func:`ascii_only` is the checkable form, and `tests/bot/test_render.py` applies it.

**Club short names are here rather than in `clubs.csv`, and that is deliberate.**
`src/epl/clubs/build.py` regenerates the canonical table from its own mapping, so a `short_name`
column added by hand would be silently erased by the next `python -m epl.clubs.build`; and
:class:`epl.clubs.ClubResolver` pins that table's columns exactly. More than either: what to call
Wolverhampton Wanderers in a chat window is a fact about the chat window. The canonical name stays
the authority and is what this falls back to.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence

import pandas as pd

from epl.clubs import ClubResolver

#: Widest line allowed inside a fixed-width block.
#:
#: A budget rather than a preference: a wrapped line inside a fixed-width block loses the alignment
#: the block was for, and the message most likely to be read on the narrowest screen in the house is
#: a ten-match digest. Every builder that lays out a table asks :func:`fits` rather than assuming.
PRE_WIDTH = 44

#: The parse mode every message is sent under. Named here rather than in :mod:`epl.bot.api` because
#: this module is what produces the markup; the transport should not be the place that decides what
#: dialect its payload is written in.
PARSE_MODE = "HTML"

#: Trailing words a Club's canonical name can lose without becoming ambiguous.
#:
#: Order matters only in that the longest match wins, which is why the tuple is scanned in order
#: rather than split on: "North End" has to be tried before "End" would be, and "& Hove Albion"
#: before "Albion".
DROPPABLE_SUFFIXES: tuple[str, ...] = (
    " & Hove Albion",
    " North End",
    " Wanderers",
    " Alexandra",
    " Athletic",
    " Hotspur",
    " Rangers",
    " Stanley",
    " Rovers",
    " Albion",
    " County",
    " Argyle",
    " Orient",
    " United",
    " City",
    " Town",
)

#: Clubs whose short name is not what the rule would produce.
#:
#: Two kinds live here and they are worth telling apart. Most are what a broadcaster says — Spurs,
#: Wolves, Palace, Forest — where the rule's answer would be correct and nobody's spoken English.
#: The rest are the pairs the rule *cannot* separate: dropping " City" from Manchester City and
#: " United" from Manchester United gives one name for two Clubs. Those are caught anyway by
#: :func:`_unambiguous`, which reverts a collision to the full canonical name; they are named here
#: because "Man City" is better than "Manchester City" and the collision check would not know that.
SHORT_NAMES: dict[str, str] = {
    "afc_wimbledon": "AFC Wimbledon",
    "brighton": "Brighton",
    "crystal_palace": "Palace",
    "man_city": "Man City",
    "man_united": "Man United",
    "middlesbrough": "Boro",
    "milton_keynes_dons": "MK Dons",
    "nottm_forest": "Forest",
    "qpr": "QPR",
    "sheffield_united": "Sheff Utd",
    "sheffield_weds": "Sheff Wed",
    "tottenham": "Spurs",
    "west_brom": "West Brom",
    "wolves": "Wolves",
}

#: How the middle Outcome is named. The two ends are named after the Club that would win them.
#:
#: "Home", "Draw" and "Away" are the terms everywhere else in this project (CONTEXT.md) and two of
#: the three are the wrong words for a phone: a reader wants to know whether Newcastle win, not
#: whether the away side does.
DRAW_LABEL = "Draw"

#: Typography this project's own prose uses that a phone should not have to render.
#:
#: A closed set, and deliberately punctuation only. Text written elsewhere in the repository reaches
#: a message unaltered — a Predictor's `note` is quoted whole, and the loop's own output is quoted
#: rather than summarised — and both are written with em dashes. Mapping those to ASCII changes no
#: word, which is the whole test of whether a quotation is still a quotation.
#:
#: Anything outside this set is left exactly as it is. A Club with an accent in its name should
#: appear with the accent rather than be mangled into something no reader recognises, and
#: :func:`ascii_only` failing on it is the right outcome: it means somebody should look.
# Each entry is a character ruff calls ambiguous, which is exactly why it is here: the
# whole job of this table is turning those into the plain one beside them.
TYPOGRAPHY: dict[str, str] = {
    "—": "-",
    "–": "-",  # noqa: RUF001
    "‘": "'",  # noqa: RUF001
    "’": "'",  # noqa: RUF001
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",  # noqa: RUF001
    "•": "-",
}

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"<pre>.*?</pre>", re.S)

_CACHED: tuple[dict[str, str], dict[str, str]] | None = None


def escape(text: object) -> str:
    """One value, safe to put inside a message. The only door markup-bearing text goes through."""
    return html.escape(str(text), quote=False)


def block(lines: Iterable[str]) -> str:
    """A fixed-width table. Contents are escaped here, so callers hand it plain strings.

    Empty input returns an empty string rather than an empty block: Telegram renders a block with
    nothing in it as a visible grey box, which is a table saying nothing where the caller meant to
    say nothing at all.
    """
    body = [str(line).rstrip() for line in lines]
    if not any(line.strip() for line in body):
        return ""
    return "<pre>" + escape("\n".join(body)) + "</pre>"


def document(parts: Iterable[str]) -> str:
    """The finished message: the parts that have anything in them, separated by a blank line.

    Blank parts are dropped rather than rendered, so a builder can hand over a section that turned
    out to be empty — no disagreement worth reporting, no Predictor with a `note` — without having
    to decide whether the blank line before it should still be there.

    **Running text is escaped here, and that is the half a caller would forget.** A table goes
    through :func:`block`, which escapes what it is given; a heading is an f-string a builder wrote,
    and "Brighton & Hove Albion" in one is invalid HTML that Telegram may refuse to parse. Escaping
    at each call site would work until the first call site that did not, so it is done once, here,
    on everything that is not already inside a block — which is the only markup this module emits.
    """
    return "\n\n".join(
        _escape_around_blocks(str(part).rstrip()) for part in parts if str(part).strip()
    )


def combine(*documents: str) -> str:
    """Join finished messages, without escaping them a second time.

    :func:`document` escapes the running text it is handed, so handing it the output of another
    :func:`document` would turn ``&amp;`` into ``&amp;amp;`` in front of the reader. The two
    operations are therefore different functions rather than one that tries to guess which it was
    given: `document` takes plain text and returns markup, and this takes markup and returns markup.

    An announcement is the case that needs it — a heading in front of a whole digest — and a helper
    that builds a section with a table in it is the other; that one hands back its parts instead,
    which is why there are only two callers here.
    """
    return "\n\n".join(part.rstrip() for part in documents if part.strip())


def _escape_around_blocks(part: str) -> str:
    """Escape everything outside a fixed-width block, and leave the blocks exactly as they are.

    A block's contents were escaped by :func:`block` when it was built, so escaping them again would
    render ``&amp;`` to the reader. Splitting on the markers rather than escaping the whole string
    is what keeps both halves right.
    """
    out: list[str] = []
    at = 0
    for found in _BLOCK.finditer(part):
        out.append(escape(part[at : found.start()]))
        out.append(found.group(0))
        at = found.end()
    out.append(escape(part[at:]))
    return "".join(out)


def strip_tags(text: str) -> str:
    """The same message as plain text, for a transport that refused it and for a terminal.

    Both callers need exactly this and for the same reason. :meth:`epl.bot.api.Telegram.send` falls
    back to it when Telegram rejects the markup, because a message that arrives unformatted beats
    one that does not arrive; `python -m epl.bot answer` prints it, because a terminal shows tags
    rather than rendering them.
    """
    return html.unescape(_TAG.sub("", text))


def asciify(text: str) -> str:
    """Text from elsewhere in the repository, with its typography flattened and its words intact.

    Applied at the boundary where prose written for a source file — a Predictor's `note`, the loop's
    own log output — becomes a chat message. Only :data:`TYPOGRAPHY` is touched, so this can never
    turn a caveat into a different caveat: an em dash becomes a hyphen and nothing else moves.
    """
    return "".join(TYPOGRAPHY.get(character, character) for character in text)


def ascii_only(text: str) -> bool:
    """Whether this message is free of emoji and of everything else non-ASCII.

    The repository owner asked for the first. This is the checkable form of it: a bullet character
    is not an emoji and renders as a box, a diamond or nothing at all depending on the device, which
    is the same problem wearing a different hat.
    """
    return text.isascii()


def fits(line: str, width: int = PRE_WIDTH) -> bool:
    """Whether a line will sit inside a fixed-width block without wrapping."""
    return len(line.rstrip()) <= width


def short(slug: object) -> str:
    """A Club's slug as the name to print, in as few characters as stays unambiguous.

    An unknown slug comes back as itself. A message with a slug in it is legible; a message that
    raised is not, and this is called from inside a bot whose own docstring says the one thing it
    must not do is go quiet.
    """
    return _resolved()[0].get(str(slug), str(slug))


def full(slug: object) -> str:
    """A Club's canonical name, for the one place a message can afford the width.

    Used in a single-match heading, where there are two Clubs and a whole line for them, and never
    in a table.
    """
    return _resolved()[1].get(str(slug), str(slug))


def percentages(home: float, draw: float, away: float) -> tuple[int, int, int]:
    """Three probabilities as whole percentages that sum to exactly 100.

    Largest remainder, not three independent roundings. A card reading 59 / 23 / 19 invites the
    reader to wonder what happened to the other one percent, and the honest answer — that each
    number was rounded on its own — is not one anybody wants in a message about football.

    The three are re-normalised first, because a Prediction is guaranteed to sum to one only within
    :data:`epl.metrics.SUM_TOLERANCE` and a stored row has been through a nine-decimal round trip.
    """
    raw = [float(home), float(draw), float(away)]
    total = sum(raw)
    if total <= 0:
        return (0, 0, 0)
    scaled = [value * 100.0 / total for value in raw]
    floors = [int(value) for value in scaled]
    short_by = 100 - sum(floors)
    order = sorted(range(3), key=lambda index: scaled[index] - floors[index], reverse=True)
    for index in order[:short_by]:
        floors[index] += 1
    return (floors[0], floors[1], floors[2])


def kickoff_long(when: object) -> str:
    """``Sunday 30 August, 17:30`` — a heading, where the day is worth spelling out."""
    moment = pd.Timestamp(when)
    return f"{moment:%A} {moment.day} {moment:%B}, {moment:%H:%M}"


def kickoff_short(when: object) -> str:
    """``Sun 17:30`` — a table cell."""
    moment = pd.Timestamp(when)
    return f"{moment:%a} {moment:%H:%M}"


def time_only(when: object) -> str:
    """``17:30`` — a table that is already grouped by day."""
    return f"{pd.Timestamp(when):%H:%M}"


def day_heading(when: object) -> str:
    """``SATURDAY 30 AUGUST`` — the separator between a digest's days."""
    moment = pd.Timestamp(when)
    return f"{moment:%A} {moment.day} {moment:%B}".upper()


def relative(when: object, now: object) -> str:
    """How long until kickoff, in the words somebody would use out loud.

    Deliberately coarse. This is read beside an exact kickoff time, so what it adds is the shape of
    the answer rather than a precision the reader can already see.
    """
    minutes = int((pd.Timestamp(when) - pd.Timestamp(now)).total_seconds() // 60)
    if minutes < 0:
        return "under way"
    if minutes < 45:
        return f"in {minutes} minutes"
    if minutes < 90:
        return "in about an hour"
    if minutes < 60 * 24:
        return f"in about {round(minutes / 60)} hours"
    days = round(minutes / (60 * 24))
    return "in about a day" if days == 1 else f"in {days} days"


def outcome_table(
    home_club: object,
    away_club: object,
    model: Sequence[float],
    market: Sequence[float] | None = None,
) -> list[str]:
    """The three-line body every match card is built around, as fixed-width lines.

    Ordered by the model's own probability rather than home, draw, away. The glossary's order is
    right for a ledger and wrong here: a reader wants the likeliest outcome first, and leading with
    the home side puts the answer to "who wins this?" in a different place on every card.

    ``market`` is optional because the Market Line covers a Fixture only when the rolling file
    carried odds for it, and a card with one column is better than no card.
    """
    home, draw, away = percentages(*model)
    labels = [f"{short(home_club)} win", DRAW_LABEL, f"{short(away_club)} win"]
    values = [home, draw, away]
    quoted = list(percentages(*market)) if market is not None else [0, 0, 0]

    label_width = min(max(max(len(label) for label in labels), 12), 22)
    header = f"{'':<{label_width}}{'MODEL':>7}"
    if market is not None:
        header += f"{'MARKET':>8}"

    lines = [header]
    for index in sorted(range(3), key=lambda position: values[position], reverse=True):
        line = f"{labels[index][:label_width]:<{label_width}}{f'{values[index]}%':>7}"
        if market is not None:
            line += f"{f'{quoted[index]}%':>8}"
        lines.append(line)
    return lines


def summary_line(
    home_club: object, away_club: object, model: Sequence[float], width: int = PRE_WIDTH
) -> str:
    """One match's odds on one indented line, shortened until it fits rather than left to wrap.

    Three attempts, widest first: both Clubs and the draw, then both Clubs, then both Clubs with
    their names cut. The indent is included because the caller is laying out a two-line entry and
    the indent is part of what has to fit.
    """
    home, draw, away = percentages(*model)
    indent = "       "
    named = [(short(home_club), home), (DRAW_LABEL.lower(), draw), (short(away_club), away)]
    ordered = sorted(named, key=lambda pair: pair[1], reverse=True)

    whole = indent + "  ".join(f"{name} {value}" for name, value in ordered)
    if fits(whole, width):
        return whole

    without_draw = [pair for pair in ordered if pair[0] != DRAW_LABEL.lower()]
    trimmed = indent + "  ".join(f"{name} {value}" for name, value in without_draw)
    if fits(trimmed, width):
        return trimmed

    room = max((width - len(indent) - 2 - len(" 00") * 2) // 2, 3)
    return indent + "  ".join(f"{name[:room]} {value}" for name, value in without_draw)


def _resolved() -> tuple[dict[str, str], dict[str, str]]:
    """The short-name and canonical-name tables, worked out once."""
    global _CACHED
    if _CACHED is None:
        clubs = ClubResolver.load().clubs
        canonical = {slug: club.name for slug, club in clubs.items()}
        _CACHED = (_unambiguous(canonical), canonical)
    return _CACHED


def _unambiguous(canonical: dict[str, str]) -> dict[str, str]:
    """Apply the rule, then take back every short name two Clubs would have shared.

    The rule is what keeps this from being 115 hand-written rows that go stale the first time a Club
    is promoted. The collision check is what makes the rule safe to apply to Clubs nobody looked at,
    which is most of the fourth tier: Bristol City and Bristol Rovers would both become "Bristol",
    so both keep their canonical names instead.
    """
    candidates = {
        slug: SHORT_NAMES.get(slug) or _shortened(name) for slug, name in canonical.items()
    }
    claimed: dict[str, list[str]] = {}
    for slug, name in candidates.items():
        claimed.setdefault(name, []).append(slug)
    return {
        slug: name if len(claimed[name]) == 1 else canonical[slug]
        for slug, name in candidates.items()
    }


def _shortened(name: str) -> str:
    """A canonical name with one droppable suffix removed, longest match first."""
    for suffix in DROPPABLE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


__all__ = [
    "DRAW_LABEL",
    "DROPPABLE_SUFFIXES",
    "PARSE_MODE",
    "PRE_WIDTH",
    "SHORT_NAMES",
    "TYPOGRAPHY",
    "ascii_only",
    "asciify",
    "block",
    "combine",
    "day_heading",
    "document",
    "escape",
    "fits",
    "full",
    "kickoff_long",
    "kickoff_short",
    "outcome_table",
    "percentages",
    "relative",
    "short",
    "strip_tags",
    "summary_line",
    "time_only",
]
