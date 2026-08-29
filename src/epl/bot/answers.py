"""Every message the bot sends, and the numbers it may not put in one.

This module is the whole of the bot's contact with the project's results. Nothing in
:mod:`epl.bot.serve` or :mod:`epl.bot.notify` formats a figure; they choose which of these to call
and hand the string to Telegram. That is the point: there is one place where a measurement becomes
a sentence, and it is this one.

**Things this must not say**, from issue #20. Each is a number this project has gone to some
trouble to make hard to misquote, and a chat message is the shortest path in the system from a real
measurement to a sentence that misrepresents it — there is no reviewer between a bot and a phone.
Where a rule can be kept by construction it is kept that way rather than by care:

* **No calibrated figure for the Live Season.** :func:`live_board` selects
  :data:`epl.ledger.scoreboard.PRE_CALIBRATION_COLUMNS`, which is the same two lines
  :func:`epl.live.__main__._score` uses and for the same reason: a calibration map needs a track
  record behind it, the Live Season has none, and the column would be the raw one renamed.
* **No RPS from a handful of Fixtures as though it were a track record.** The sample size travels
  with the number, in the same message, always.
* **Never the sequential diagnostic.** Nothing here imports it, which
  `tests/bot/test_the_bot_is_read_only.py` checks by walking imports rather than by trusting this
  sentence.
* **Never accuracy as the headline.** Boards are rendered in
  :data:`epl.ledger.scoreboard.METRICS` order, which puts RPS first — from that tuple rather than
  from a second list written here.
* **Never a Predictor without its `note`.** :func:`epl.predictors.note` is read by name for every
  row of every board, so the Ceiling Line's caveat and a Pundit's travel with them and a Predictor
  registered tomorrow is covered with no change here. There is no branch per Predictor, the same
  way the scoreboard has none.
* **Never a blank Pundit column on the live board.** The board holds the Predictors that spoke, and
  the absence of the rest is explained in words where it appears.

**What it refuses to answer, and why that is the answer.** The Evaluation Window's board is derived
and regenerable, so it is gitignored (ADR 0005's reasoning), and `python -m epl.ledger backfill` is
the one command `deploy/crontab` must never run. The Pi therefore has no such board, and
:func:`evaluation_board` says so and names the command instead of reciting the README's numbers. A
bot that recited them would be reporting a measurement it had not got, on a machine where it might
no longer be true.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from epl import predictors
from epl.bot import fires, watch
from epl.clubs import ClubResolver
from epl.ingest.football_data import match_table
from epl.ledger import live as store
from epl.ledger import scoreboard
from epl.paths import processed_dir
from epl.windows import EVALUATION_WINDOW, LIVE_SEASON, season_label

#: How many lines of a failed run's output to quote. Enough to see the complaint, short of pasting
#: a traceback into a chat window.
TAIL_LINES = 14

#: Telegram renders nothing here, so the messages are plain text. Deliberate: a Markdown parse
#: error on a Club name with an apostrophe in it — Nott'm Forest — is a message that does not
#: arrive, and the one thing this bot must not do is go quiet.
BULLET = "  •"


def help_text() -> str:
    """The command menu, and what the bot is not allowed to do."""
    return "\n".join(
        [
            "EPL Predictor — read-only monitor",
            "",
            "/round     the current Sealed Prediction Round, and what each Predictor said",
            "/live      the Live Season board, scored on the Fixtures played so far",
            "/board     the Evaluation Window scoreboard",
            "/health    the last scheduled fires, and anything worth worrying about",
            "/help      this menu",
            "",
            "It cannot seal, supersede, backfill or score — a chat app is not a second door",
            "into outputs/live/ (ADR 0005). RPS is the headline everywhere; accuracy is not.",
        ]
    )


def sealed_round(prediction_round: str | None = None) -> str:
    """One Sealed Prediction Round: its Fixtures, and what every Predictor said about each.

    Defaults to the most recent round in the store. A round that was never sealed is said to be
    missing rather than rendered as an empty table — on this store the difference is the whole
    point, since a round the loop failed to seal is exactly the event worth hearing about.
    """
    sealed = store.read()
    if sealed.empty:
        return (
            "Nothing has been sealed yet. outputs/live/ is empty on this machine — either the "
            "schedule has not run inside a round's window, or this is a fresh clone."
        )

    wanted = str(prediction_round or sealed["prediction_round"].max())
    rows = sealed.loc[sealed["prediction_round"] == wanted]
    if rows.empty:
        held = ", ".join(sorted(set(sealed["prediction_round"]))[-4:])
        return f"{wanted} has never been sealed. The store holds: {held}"

    names = _club_names()
    lines = [
        f"Prediction Round {wanted}",
        f"as-of {rows['as_of_instant'].min()} · {_count(rows, 'Fixture')} · "
        f"{len(rows)} Predictions from {rows['predictor'].nunique()} Predictors",
    ]
    for (kickoff, home, away), fixture in rows.groupby(
        ["kickoff", "home_club", "away_club"], sort=True
    ):
        lines.append("")
        lines.append(f"{names(home)} v {names(away)} — {pd.Timestamp(kickoff):%a %d %b %H:%M}")
        lines.append("   (home / draw / away)")
        for _, quote in fixture.sort_values("predictor").iterrows():
            lines.append(
                f"{BULLET} {quote['predictor']:<16} {quote['prob_home']:>4.0%} /"
                f" {quote['prob_draw']:>4.0%} / {quote['prob_away']:>4.0%}"
            )

    silent = _silent(set(rows["predictor"]))
    if silent:
        lines += [
            "",
            f"silent (cover none of this round): {', '.join(silent)}",
            "A Pundit cannot be part of a Sealed Prediction — the only permitted source",
            "transcribes a matchday after it is played (issue #16). Their column on the",
            "live board fills in retrospectively rather than staying blank.",
        ]
    return "\n".join(lines)


def live_board() -> str:
    """The Live Season, scored on its own board over the Fixtures that have been played.

    Pre-calibration only, and the sample size is in the message. Both are issue #20's, and the
    first is :func:`epl.live.__main__._score`'s rule rather than a second one: the Live Season has
    no track record for a calibration map to be fitted on, so a calibrated column here would be the
    raw one under a name that makes it look like it came from somewhere.
    """
    sealed = store.read()
    if sealed.empty:
        return "Nothing has been sealed yet, so there is nothing to score."

    matches = _matches()
    if matches is None:
        return (
            "No match table on this machine, so nothing can be scored yet.\n"
            "data/processed/ is gitignored and is rebuilt by:\n"
            "    python -m epl.live score      (which ingests first)\n"
            "    python -m epl.ingest fetch && python -m epl.ingest build"
        )

    board = scoreboard.build(sealed, matches, seasons=[LIVE_SEASON])
    if board.empty:
        return (
            f"{season_label(LIVE_SEASON)}: {len(sealed)} Sealed Predictions, none of whose "
            "Fixtures has been played and ingested yet."
        )

    fixtures = int(board["fixtures"].max())
    lines = [
        f"{season_label(LIVE_SEASON)} — sealed and scored",
        f"{len(sealed)} Predictions on {_count_of(fixtures, 'Fixture')} played so far",
        "",
        _table(board, scoreboard.PRE_CALIBRATION_COLUMNS),
        "",
        f"RPS over {_count_of(fixtures, 'Fixture')} is noise, not a track record. The",
        f"Evaluation Window is {len(EVALUATION_WINDOW)} closed Seasons, is scored separately,",
        "and does not move when this does (epl.windows). /board has its numbers.",
        "Pre-calibration only: a calibration map needs a track record behind it and the",
        "Live Season has none, so a calibrated column here would be the raw one renamed.",
        "A Pundit's column fills in retrospectively, once the archive catches up (#16).",
    ]
    return "\n".join(lines + _notes(board))


def evaluation_board() -> str:
    """The Evaluation Window's scoreboard, as `python -m epl.ledger scoreboard` last wrote it.

    Read rather than recomputed, and refused rather than recited. Recomputing needs
    `outputs/backtest/`, which is gitignored and is written by the one command `deploy/crontab` must
    never schedule (ADR 0005) — so on the Pi there is nothing to recompute from. Reciting the
    README's numbers instead would be a bot reporting a measurement it does not hold, on a machine
    where it may since have changed.

    **Both halves, unlike :func:`live_board`, and the difference is not an inconsistency.** ADR 0006
    publishes every metric twice and says neither may be reported alone: the shared calibration
    layer makes every Predictor slightly worse over this window and is kept anyway, and showing only
    the better column is exactly how that finding gets lost. The live board is pre-calibration only
    for the opposite reason — there is no track record for a map to have been fitted on at all, so
    there is no second column, rather than one being withheld.

    The file's own modification time goes with it. A derived file carries no date and this one is
    regenerated by hand, which is how a bot comes to report last month's board as this month's.
    """
    board_path = scoreboard.path()
    if not board_path.exists():
        return (
            "No Evaluation Window board on this machine.\n"
            "outputs/scoreboard.csv is derived and regenerable, so it is not in git\n"
            "(ADR 0005), and rebuilding it needs outputs/backtest/, which nothing on a\n"
            "schedule writes.\n"
            "On a machine that has one:\n"
            "    python -m epl.ledger backfill\n"
            "    python -m epl.ledger scoreboard"
        )

    board = pd.read_csv(board_path)
    written = pd.Timestamp(board_path.stat().st_mtime, unit="s")
    lines = [
        f"Evaluation Window — {len(EVALUATION_WINDOW)} closed Seasons",
        f"written {written:%Y-%m-%d %H:%M}, by python -m epl.ledger scoreboard",
        "",
        "pre-calibration",
        _table(board, scoreboard.PRE_CALIBRATION_COLUMNS),
        "",
        "post-calibration",
        _table(board, scoreboard.POST_CALIBRATION_COLUMNS),
        "",
        "Both halves, because neither may be reported alone (ADR 0006). The shared",
        "calibration layer makes every Predictor slightly worse here and is kept anyway;",
        "publishing only the better column is how that finding gets lost.",
        "RPS is the headline. Accuracy is for lay explanation only and is never it.",
    ]
    return "\n".join(lines + _notes(board))


def health(found: Sequence[fires.Fire], *, now: pd.Timestamp) -> str:
    """What the schedule has been doing, and anything worth worrying about.

    The last fire of each subcommand rather than the last fire overall, because they answer
    different questions: `seal` costs a round when it fails and `score` costs a day's delay.
    """
    lines = ["Schedule health"]
    if not found:
        lines.append(
            "The loop has never fired on this machine — deploy/logs/live_loop.log is empty\n"
            "or absent. If the crontab is installed, that is open risk 6."
        )
    for subcommand in ("seal", "score"):
        last = fires.latest(found, subcommand=subcommand)
        if last is None:
            if found:
                lines.append(f"{BULLET} {subcommand}: never")
            continue
        lines.append(f"{BULLET} {subcommand}: {_describe(last)}")
        if last.failed:
            lines += [f"     {line}" for line in last.tail(4).splitlines()]

    raised = watch.concerns(found, now=now)
    if raised:
        lines.append("")
        lines += [concern.message() for concern in raised]
    elif found:
        lines.append("")
        lines.append("Nothing else to report: no missed sealing day, no unchanged upstream file.")
    return "\n".join(lines)


def failure(fire: fires.Fire) -> str:
    """A run that needs somebody, in its own words.

    `epl.live.__main__._seal`'s exit-code contract distinguishes several failures — `NOT PUSHED`,
    a stale `LIVE_SEASON`, a rolling file that changed shape — and this must not flatten them into
    "the loop failed". So the loop's own output is quoted rather than summarised.

    A run with no exit code was killed rather than having failed, and is said that way: inventing
    an exit code would be making a claim about what the loop decided.
    """
    verdict = (
        "was killed before it could finish (no exit code)"
        if fire.exit_code is None
        else f"failed with exit {fire.exit_code}"
    )
    return "\n".join(
        [
            f"🚨 {fire.command} {verdict}",
            f"at {fire.local:%Y-%m-%d %H:%M %Z}",
            "",
            fire.tail(TAIL_LINES) or "(the run printed nothing)",
        ]
    )


def quiet(fire: fires.Fire, concern: watch.Concern) -> str:
    """A fire that sealed nothing, and which of the two silences it was.

    Most weeks this is the true and boring answer and nobody needs to be told. The push half
    decides that (:func:`epl.bot.notify.run`); what this does is make the two distinguishable at
    all, which they are not from the exit code — both are 0 on purpose (issue #19).
    """
    silence = {
        fires.NOTHING_IN_FILE: (
            "the rolling file held no row in a tier this project predicts"
        ),
        fires.OUTSIDE_EVERY_WINDOW: (
            "no Prediction Round was inside its sealing window (ADR 0005)"
        ),
    }.get(fire.verdict, "nothing was sealed")
    lines = [
        f"{fire.command} at {fire.local:%a %d %b %H:%M %Z}: "
        "sealed nothing",
        f"— {silence}.",
    ]
    if concern is not None:
        lines += ["", concern.message()]
    return "\n".join(lines)


def sealed_announcement(prediction_round: str | None = None) -> str:
    """A round has just been sealed. The body is :func:`sealed_round`'s."""
    return f"🔒 Sealed\n\n{sealed_round(prediction_round)}"


def scored_announcement() -> str:
    """A sealed round has been scored. The body is :func:`live_board`'s."""
    return f"📊 Scored\n\n{live_board()}"


def _matches() -> pd.DataFrame | None:
    """The match table, or ``None`` when this machine has not built one.

    :func:`epl.ingest.football_data.match_table` raises :class:`SystemExit` on a missing file, which
    is right for a command line and would take the bot's process down in the middle of answering a
    message. So the file is checked for rather than the exception caught: a bot that exited on a
    `/live` would be the monitoring going down before the thing it monitors.
    """
    if not (processed_dir() / "matches.csv").exists():
        return None
    return match_table()


def _club_names() -> Callable[[str], str]:
    """A slug to the Club's own name, from the canonical table rather than by un-slugging it.

    `nottm_forest` is "Nottingham Forest" and `man_city` is "Manchester City"; no rule over the slug
    turns one into the other, and the table that does is the one every other part of this project
    already agrees with. An unknown slug comes back as itself, because a message with a slug in it
    is legible and a message that raised is not.
    """
    clubs = ClubResolver.load().clubs

    def name(slug: str) -> str:
        found = clubs.get(str(slug))
        return found.name if found is not None else str(slug)

    return name


def _silent(spoke: set[str]) -> list[str]:
    """The registered Predictors that said nothing about this round.

    Derived by difference rather than listed, for the same reason `epl.live.seal` has no branch per
    Predictor: which ones can speak to an unplayed Fixture is each Predictor's own business, and a
    list here would be a second opinion about it that could quietly go out of date.
    """
    return sorted(
        predictor.name for predictor in predictors.registered() if predictor.name not in spoke
    )


def _table(board: pd.DataFrame, columns: Sequence[str]) -> str:
    """A scoreboard, rendered in :data:`epl.ledger.scoreboard.METRICS`' own order.

    RPS comes first because that tuple puts it first, which is the project's rule stated once
    rather than a second ordering written here — and it is what keeps accuracy off the headline
    without anybody having to remember to.
    """
    shown = [name for name in columns if name in board.columns]
    return board[shown].to_string(index=False, float_format=lambda value: f"{value:.4f}")


def _notes(board: pd.DataFrame) -> list[str]:
    """Each Predictor's caveat, read off the registry by name and quoted **whole**.

    The Ceiling Line's 0.1968 is not worse than the Market Line's 0.1936 — they are measured over
    different Fixtures — and a Pundit's as-stated ~0.334 is not comparable to the board's much
    longer track record. Both say so in their own `note`. Read here rather than retyped, so a
    Predictor registered tomorrow arrives with its caveat and nothing in this module changes.

    **Not truncated, and an earlier draft that cut them at 180 characters was wrong in the specific
    way this rule exists to prevent.** `margin_map_lawrenson`'s note runs to 623 characters and the
    cut landed mid-sentence, dropping "It may beat a model on this board, which would be a finding
    about the information in the calls and never a verdict on the forecaster" — which is precisely
    the clause standing between this bot and "Sutton beat the model" (ADR 0003). A caveat shortened
    to fit a phone is a caveat that no longer says the thing it was written to say; a long message
    is split rather than cut, for the same reason (:func:`epl.bot.api.split`).

    Whitespace is flattened because a `note` is written to sit in a CSV cell and carries the line
    breaks of the source that defined it, not of the message quoting it.
    """
    lines: list[str] = []
    for name in board.get("predictor", pd.Series(dtype="object")):
        caveat = predictors.note(str(name))
        if caveat:
            lines.append(f"{BULLET} {name}: {' '.join(caveat.split())}")
    return ["", "Read these with their numbers:", *lines] if lines else []


def _describe(fire: fires.Fire) -> str:
    """One fire as a line: when it ran, in UK time, and how it came out."""
    when = fire.local.strftime("%a %d %b %H:%M %Z")
    ending = "no exit code" if fire.exit_code is None else f"exit {fire.exit_code}"
    return f"{when} — {ending}, {fire.verdict}"


def _count(rows: pd.DataFrame, noun: str) -> str:
    return _count_of(rows.groupby(["home_club", "away_club"]).ngroups, noun)


def _count_of(many: int, noun: str) -> str:
    return f"{many} {noun}" if many == 1 else f"{many} {noun}s"


__all__ = [
    "TAIL_LINES",
    "evaluation_board",
    "failure",
    "health",
    "help_text",
    "live_board",
    "quiet",
    "scored_announcement",
    "sealed_announcement",
    "sealed_round",
]
