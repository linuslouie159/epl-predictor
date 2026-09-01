"""The bot cannot write to either Prediction store, checked structurally rather than by reading.

Issue #20's eighth criterion, and the one worth the most: **a chat app must not be a second door
into `outputs/live/`** (ADR 0005). A Sealed Prediction is evidence because the loop wrote it before
kickoff, under a moment nobody could choose (`epl.live.__main__.clock` is a function and not a
flag), and committed it so git records when. A message from a phone is the shortest route in the
whole system to a row that has none of those properties.

"There is no handler that writes" is a claim about *today's* code and the sort of thing a helpful
change quietly breaks — a `/reseal` for convenience, a `/score` because it is only a report, an
import of `epl.ledger.backtest` because the scoreboard lives near it. So it is checked the way
`tests/v2/test_stubs_are_unreachable.py` checks its own claim: by walking the package with `ast`.

**Be precise about what that proves, because the tempting stronger claim is false.** Nothing makes
a Python process *incapable* of calling a function in a module something else already imported, and
plenty is already imported here — `epl.bot.answers` reads the sealed store through
`epl.ledger.live`, which is the module `seal` and `supersede` live in. What is checked, and what is
worth checking, is narrower: **no file under `src/epl/bot/` names a writing door**, by import or by
call. Adding one would therefore be a diff that fails this test rather than a line nobody notices,
which is exactly the failure mode a chat interface invites — a `/reseal` added for convenience by
somebody who had not read ADR 0005.

The same sweep does two more jobs, because they are the same question asked of different names:

* **The sequential diagnostic** produces a number that must never be quoted as a score (ADR 0002),
  and the only ironclad way to not quote it is to have no way of computing it.
* **The ingest's fetchers** would make the bot a second thing that pulls from Football-Data. That is
  not forbidden anywhere, but it would make a `/live` a network call, and a bot that hangs for
  thirty seconds on a phone is a bot nobody uses. Reading `epl.ingest`'s *parsers* is fine and is
  what `answers` does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from epl import bot

#: Every door into either Prediction store that writes. Named as `module.function`, which is what
#: `_imported_names` records, and as the bare module, since importing the module is enough to be
#: one import away from calling anything in it.
WRITING_DOORS: tuple[str, ...] = (
    "epl.ledger.live.seal",
    "epl.ledger.live.supersede",
    "epl.ledger.live.commit",
    "epl.ledger.live.push",
    "epl.ledger.schema.write_csv",
    "epl.ledger.readings.record",
    "epl.ledger.backtest.backfill",
    "epl.ledger.backtest.sequential",
    "epl.live.seal",
    "epl.live.upcoming.next_round",
    "epl.live.prematch.run",
)

#: Whole modules the bot has no business importing at all.
FORBIDDEN_MODULES: tuple[str, ...] = (
    "epl.ledger.backtest",
    "epl.live.seal",
    "epl.live.prematch",
    "epl.ingest.fetcher",
    "epl.ingest.cache",
)

#: Functions that reach upstream. A `/live` that fetched would be a chat command that hangs.
FETCHERS: tuple[str, ...] = (
    "epl.ingest.fetch_all",
    "epl.ingest.fetch_fixtures",
    "epl.ingest.build_tables",
    "epl.ingest.fixtures.fetch_fixtures",
    "epl.ingest.football_data.fetch_all",
)

#: The package under guard.
PACKAGE_DIR = Path(bot.__file__).parent


def _files() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _imported_names(path: Path) -> set[str]:
    """Every ``epl.*`` name one file imports, as both the module and the module-plus-attribute.

    Read from the syntax tree rather than by importing, because importing would answer a different
    question — what happens to be reachable at runtime today — and the claim is about the source.
    Both spellings of a `from` import are recorded for the reason `tests/v2` records both: writing
    `from epl.ledger.live import seal` binds the function just as surely as reaching it through the
    module, and recording only the module half would miss the most natural spelling of the mistake.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name == "epl" or name.startswith("epl.")}


def _all_imports() -> dict[str, set[str]]:
    return {path.name: _imported_names(path) for path in _files()}


#: The bare names of the functions above, for the second half of the sweep. An import is not the
#: only way to name a writing door: `from epl.ledger import live as store` is allowed and correct —
#: `answers` reads the store through it — and `store.seal(rows)` would then be a write with no
#: forbidden import anywhere in the file.
WRITING_CALLS: frozenset[str] = frozenset(
    {
        "seal",
        "supersede",
        "backfill",
        "sequential",
        "write_csv",
        "fetch_all",
        "build_tables",
        # The Pre-Match Reading store's writer. The bot *reads* that store — `notify` sends the
        # cards a `prematch` fire wrote — so `epl.ledger.readings` is a correct import here, and
        # `readings.record(rows)` under it would be a write with no forbidden import in the
        # file. Named deliberately unlike anything a reader does, so this entry cannot collide.
        "record",
    }
)


def _called_attributes(path: Path) -> set[str]:
    """Every ``something.name(...)`` this file calls, by the attribute's own name.

    Deliberately blind to what ``something`` is. A test that tried to resolve the receiver would be
    a small type checker, and would be defeated by the alias that makes the mistake easy in the
    first place; asking only "does this file call anything named `seal`?" is cruder, has no false
    negatives, and its false positives are names this package has no business using either.
    """
    called: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    return called


class TestNothingHereCanWriteAPrediction:
    @pytest.mark.parametrize("door", WRITING_DOORS)
    def test_no_module_imports_a_writing_door(self, door: str) -> None:
        offenders = {name: sorted(names & {door}) for name, names in _all_imports().items()}
        assert {name: found for name, found in offenders.items() if found} == {}

    @pytest.mark.parametrize("module", FORBIDDEN_MODULES)
    def test_no_module_imports_a_forbidden_module(self, module: str) -> None:
        offenders = {
            name: module
            for name, names in _all_imports().items()
            if module in names
        }
        assert offenders == {}

    @pytest.mark.parametrize("fetcher", FETCHERS)
    def test_no_module_reaches_upstream(self, fetcher: str) -> None:
        offenders = {
            name: fetcher for name, names in _all_imports().items() if fetcher in names
        }
        assert offenders == {}

    def test_no_module_calls_anything_named_like_a_writing_door(self) -> None:
        """The half an import sweep cannot see.

        `from epl.ledger import live as store` is a correct and necessary import — `answers` reads
        the sealed store through it — and `store.seal(rows)` underneath it would be a write with no
        forbidden import in the file at all.
        """
        offenders = {
            path.name: sorted(_called_attributes(path) & WRITING_CALLS)
            for path in _files()
            if _called_attributes(path) & WRITING_CALLS
        }
        assert offenders == {}

    def test_the_dispatch_table_holds_no_verb(self) -> None:
        """The other half of the same claim, at the only surface a Telegram user can reach."""
        from epl.bot import serve

        assert {command.name for command in serve.COMMANDS} == {
            "next", "week", "bet", "disagree", "club", "results", "record", "board",
            "explain", "health", "help",
        }
        # The aliases are the same claim from the other side: they may only ever point at a
        # command in the table above, so a verb cannot be reached by a second spelling either.
        assert set(serve.ALIASES.values()) <= {command.name for command in serve.COMMANDS}


class TestTheSweepWouldNoticeAViolation:
    """The tests above pass today because the package is clean, and would also pass if the detector
    had quietly stopped finding imports. So it is pointed at files where the answer is known."""

    def test_it_finds_the_imports_the_package_does_have(self) -> None:
        every = set().union(*_all_imports().values())

        assert "epl.ledger.live" in every  # answers reads the sealed store
        assert "epl.paths" in every

    def test_it_finds_a_writing_door_when_one_is_there(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text("from epl.ledger.live import seal\n", encoding="utf-8")

        assert "epl.ledger.live.seal" in _imported_names(offender)

    def test_it_finds_a_module_import_too(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text("import epl.ledger.backtest\n", encoding="utf-8")

        assert "epl.ledger.backtest" in _imported_names(offender)

    def test_it_finds_a_write_hidden_behind_an_alias(self, tmp_path: Path) -> None:
        """The realistic version of the mistake, which no import check would catch."""
        offender = tmp_path / "offender.py"
        offender.write_text(
            "from epl.ledger import live as store\n\n\ndef go(rows):\n    store.seal(rows)\n",
            encoding="utf-8",
        )

        assert _imported_names(offender) & set(WRITING_DOORS) == set()
        assert _called_attributes(offender) & WRITING_CALLS == {"seal"}

    def test_there_is_something_to_check(self) -> None:
        """A glob that found no files would pass every test above by vacuum."""
        assert len(_files()) >= 7
        assert {path.stem for path in _files()} >= {
            "__init__", "__main__", "answers", "api", "fires", "notify", "serve", "settings",
            "watch",
        }


class TestTheStoreIsNotWrittenEvenByAccident:
    def test_reading_every_answer_leaves_the_sealed_store_byte_identical(
        self, sealed_store: object, corpus: Path, registered_predictors: None
    ) -> None:
        """The belt to the import sweep's braces, and it catches a different thing: a helper that
        rewrote a file it had merely opened. `outputs/live/` is append-only and never rewritten
        (CLAUDE.md), so the bytes are the assertion."""
        from epl.bot import answers, fires
        from epl.paths import live_dir

        before = {path: path.read_bytes() for path in live_dir().glob("*.csv")}

        answers.round_digest()
        answers.live_record()
        answers.evaluation_board()
        answers.health(fires.read(), now=fires.uk_now())

        assert {path: path.read_bytes() for path in live_dir().glob("*.csv")} == before
