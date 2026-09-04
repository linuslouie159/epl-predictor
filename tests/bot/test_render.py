"""How a message looks, and the three ways looking wrong makes it say the wrong thing.

Formatting tests are usually not worth writing. These are, because each of the three properties
below fails *silently* on the one device the bot is read on and on none of the machines it is
developed on:

* **A line too wide for a phone wraps**, and a wrapped line inside a fixed-width block has lost the
  column alignment the block existed for. A digest of ten Fixtures becomes twenty ragged lines.
* **Three percentages that do not sum to 100** invite the reader to wonder where the missing point
  went. The honest answer — three independent roundings — is not one anybody wants in a message
  about football.
* **A `&` in a Club's name is invalid HTML**, and Telegram answers a message it cannot parse with a
  refusal rather than a best effort. Brighton & Hove Albion play twenty times a season.

So the sweep below renders every message this bot can produce against a machine shaped like the Pi
and asserts all three over the lot, rather than against whichever message somebody remembered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from log_blocks import THE_FIRE_THAT_PROVED_THE_SCHEDULE

from epl.bot import answers, api, fires, render, serve

#: Every command, as a Telegram user would type it. Read off the dispatch table rather than listed,
#: so a command added tomorrow is covered here without anybody remembering to add it — the same
#: reason `epl.bot.answers` reads a Predictor's `note` off the registry.
COMMANDS: tuple[str, ...] = tuple(f"/{command.name}" for command in serve.COMMANDS)

#: A moment inside the sealed round the fixtures build, so `/next` and `/club` have something to
#: answer with. Zone-aware, because that is what `epl.bot.fires.uk_now` hands the dispatcher.
NOW = pd.Timestamp("2026-08-28T12:00:00", tz=fires.LOCAL_ZONE)

#: A moment after the last Fixture of that round has been played, which is the state `/week` is in
#: for three days of every week and renders a different message for. Swept as well as :data:`NOW`
#: because the width and ASCII budgets are properties of a *message*, and this is a second message
#: from the same command — the one that would otherwise be checked by nobody until a Monday.
AFTER_THE_ROUND = pd.Timestamp("2026-08-31T12:00:00", tz=fires.LOCAL_ZONE)

#: The one message exempt from the width budget, and why.
#:
#: :func:`epl.bot.answers.failure` quotes the loop's own output rather than summarising it, because
#: the exit-code contract distinguishes several failures and "the loop failed" is not one of them.
#: Those lines are as long as they are. Re-wrapping them would move the line breaks in a log
#: somebody is reading to work out what broke, and truncating them would drop the end of the
#: complaint — so the block is allowed to be wide and an operator scrolls. Named here rather than
#: quietly left out of the sweep.
WIDTH_EXEMPT: frozenset[str] = frozenset({"failure"})

_BLOCK = re.compile(r"<pre>(.*?)</pre>", re.S)


def every_message() -> dict[str, str]:
    """Every command's answer, plus the two the push half sends unprompted."""
    found = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)
    messages = {
        command: serve.dispatch(command, now=NOW) or "" for command in COMMANDS
    }
    messages["/week (round over)"] = serve.dispatch("/week", now=AFTER_THE_ROUND) or ""
    messages["/club arsenal"] = serve.dispatch("/club arsenal", now=NOW) or ""
    messages["/club nonsense"] = serve.dispatch("/club nonsense", now=NOW) or ""
    messages["sealed_announcement"] = answers.sealed_announcement()
    messages["scored_announcement"] = answers.scored_announcement()
    messages["failure"] = answers.failure(found[-1])
    return messages


def block_lines(message: str) -> list[str]:
    """Every line inside a fixed-width block, as the reader sees it."""
    return [
        line
        for block in _BLOCK.findall(message)
        for line in render.strip_tags(block).splitlines()
    ]


class TestEveryMessageFitsAPhone:
    def test_no_block_line_is_wider_than_the_budget(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """The budget is :data:`epl.bot.render.PRE_WIDTH` and it is a hard one.

        Running text is exempt and is meant to be: Telegram wraps a paragraph sensibly, which is
        exactly what it cannot do to a table.
        """
        too_wide = {
            command: [line for line in block_lines(message) if len(line) > render.PRE_WIDTH]
            for command, message in every_message().items()
            if command not in WIDTH_EXEMPT
        }
        assert {command: lines for command, lines in too_wide.items() if lines} == {}

    def test_the_exemption_is_the_one_message_that_quotes_the_loop(
        self, project_root: Path
    ) -> None:
        """A blanket exemption would be a way to stop this test ever failing again.

        So the exempt message is checked to be the one whose whole job is quoting: it does carry a
        wide block, and it carries it because the loop's own words are in it.
        """
        found = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)
        widest = max(len(line) for line in block_lines(answers.failure(found[-1])))

        assert WIDTH_EXEMPT == {"failure"}
        assert widest > render.PRE_WIDTH

    def test_there_are_blocks_to_check(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """A sweep that found no tables would pass the test above by vacuum."""
        assert sum(len(block_lines(message)) for message in every_message().values()) > 20


class TestEveryMessageIsPlainAscii:
    def test_nothing_carries_an_emoji_or_any_other_non_ascii(
        self, sealed_store: pd.DataFrame, corpus: Path, registered_predictors: None
    ) -> None:
        """The repository owner asked for no emoji, and this is the checkable form of it.

        Wider than asked, on purpose: a bullet character is not an emoji and renders as a box on the
        devices that lack the glyph, which is the same problem wearing a different hat. Prose from
        elsewhere in the repository — a Predictor's `note`, the loop's own output — is flattened
        through :func:`epl.bot.render.asciify`, which touches punctuation and never words.
        """
        offenders = {
            command: sorted({character for character in message if not character.isascii()})
            for command, message in every_message().items()
        }
        assert {command: bad for command, bad in offenders.items() if bad} == {}


class TestMarkupIsEscapedExactlyOnce:
    def test_an_ampersand_in_a_club_name_survives_as_itself(
        self, sealed_store: pd.DataFrame, registered_predictors: None
    ) -> None:
        """Brighton & Hove Albion, which is the whole reason running text is escaped at all.

        Escaped once so Telegram parses it, and once only — a message showing the reader
        ``&amp;amp;`` is the failure the other way, and it is what a nested `document` would have
        produced before :func:`epl.bot.render.combine` existed.
        """
        heading = "Chelsea v Brighton & Hove Albion"
        message = render.document([heading, render.block(["a & b"])])

        assert "Brighton &amp; Hove Albion" in message
        assert "&amp;amp;" not in message
        assert render.strip_tags(message).count("&") == 2

    def test_composing_two_finished_messages_does_not_escape_them_again(self) -> None:
        inner = render.document(["Brighton & Hove Albion"])

        assert render.combine("A HEADING", inner).count("&amp;") == 1

    def test_a_block_is_left_alone_by_the_running_text_escape(self) -> None:
        """`block` escapes its own contents, so `document` must not reach inside one."""
        assert render.document([render.block(["x & y"])]) == "<pre>x &amp; y</pre>"


class TestThreePercentagesAlwaysSumToOneHundred:
    @pytest.mark.parametrize(
        "probabilities",
        [
            (0.180396028, 0.232513614, 0.587090358),
            (1 / 3, 1 / 3, 1 / 3),
            (0.005, 0.005, 0.99),
            (0.456926188, 0.247927199, 0.295146613),
            (0.5, 0.25, 0.25),
        ],
    )
    def test_they_sum_to_one_hundred(self, probabilities: tuple[float, float, float]) -> None:
        assert sum(render.percentages(*probabilities)) == 100

    def test_a_third_each_is_the_case_three_roundings_would_get_wrong(self) -> None:
        """33 + 33 + 33 is 99. Largest remainder is what makes it 100, and which one gains the
        point is arbitrary rather than wrong — what matters is that the reader is not shown 99."""
        assert sorted(render.percentages(1 / 3, 1 / 3, 1 / 3)) == [33, 33, 34]

    def test_an_all_zero_row_does_not_divide_by_zero(self) -> None:
        """A Fixture the Market Line did not cover has no odds, and a card must still render."""
        assert render.percentages(0.0, 0.0, 0.0) == (0, 0, 0)


class TestClubNamesAreShortAndStillUnambiguous:
    def test_no_two_clubs_share_a_short_name(self) -> None:
        """The rule drops a suffix; the collision check is what makes the rule safe to apply to the
        ninety-odd Clubs nobody looked at. Bristol City and Bristol Rovers are the pair that proves
        it — both would become "Bristol", so both keep their canonical names."""
        names = render._resolved()[0]

        assert len(set(names.values())) == len(names)

    @pytest.mark.parametrize(
        ("slug", "expected"),
        [
            ("man_city", "Man City"),
            ("man_united", "Man United"),
            ("nottm_forest", "Forest"),
            ("tottenham", "Spurs"),
            ("newcastle", "Newcastle"),
            ("ipswich", "Ipswich"),
            ("preston", "Preston"),
            ("bristol_city", "Bristol City"),
            ("bristol_rovers", "Bristol Rovers"),
        ],
    )
    def test_the_names_a_reader_would_recognise(self, slug: str, expected: str) -> None:
        assert render.short(slug) == expected

    def test_an_unknown_slug_comes_back_as_itself(self) -> None:
        """A message with a slug in it is legible; one that raised is not, and this is called from
        inside a bot whose whole job is not going quiet."""
        assert render.short("not_a_club") == "not_a_club"


class TestSplittingALongMessage:
    def test_a_block_is_closed_and_reopened_across_a_chunk_boundary(self) -> None:
        """Without this, a long digest becomes one chunk with an unclosed tag and one with an
        unopened one, and Telegram refuses both — so the message goes missing at exactly the
        length that makes it worth sending."""
        message = render.document(
            ["A HEADING", render.block([f"row {n:<30}" for n in range(400)]), "a note"]
        )

        chunks = api.split(message, limit=900)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.count(api.BLOCK_OPEN) == chunk.count(api.BLOCK_CLOSE)

    def test_nothing_is_lost_in_the_split(self) -> None:
        message = render.document([render.block([f"row {n}" for n in range(300)])])

        rejoined = "".join(
            render.strip_tags(chunk) for chunk in api.split(message, limit=500)
        )

        assert rejoined.count("row ") == 300

    def test_a_short_message_is_left_exactly_as_it_was(self) -> None:
        assert api.split("hello") == ["hello"]


class TestSummaryLinesShortenRatherThanWrap:
    @pytest.mark.parametrize(
        ("home", "away"),
        [
            ("crystal_palace", "man_city"),
            ("bournemouth", "everton"),
            ("bristol_rovers", "bristol_city"),
            ("wolves", "nottm_forest"),
            ("kidderminster", "peterborough"),
        ],
    )
    def test_the_line_always_fits(self, home: str, away: str) -> None:
        """Three attempts, widest first: both Clubs and the draw, then both Clubs, then both Clubs
        with their names cut. The last is ugly and is still better than a wrapped table."""
        line = render.summary_line(home, away, [0.45, 0.27, 0.28])

        assert render.fits(line)
        assert line.strip()

    def test_the_draw_is_what_gets_dropped_first(self) -> None:
        """It is the outcome a reader is least likely to be looking for, and dropping it is what
        keeps two long Club names on one line."""
        assert "draw" not in render.summary_line(
            "bristol_rovers", "bristol_city", [0.45, 0.27, 0.28]
        )
