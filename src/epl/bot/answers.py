"""Every message the bot sends, and the numbers it may not put in one.

This module is the whole of the bot's contact with the project's results. Nothing in
:mod:`epl.bot.serve` or :mod:`epl.bot.notify` chooses a word or a column; they choose which of these
to call and hand the string to Telegram. That is the point: there is one place where a measurement
becomes a sentence, and it is this one. How that sentence *looks* is :mod:`epl.bot.render`'s, and
the split is what stops a formatting change from quietly dropping a caveat.

**Who this is written for changed, and the rules did not.** It was built as a monitor: it answered
"did the schedule fire?" and printed boards. It is now also read by somebody who wants to know who
is going to win, which is why a match card leads with the likeliest Outcome named after the Club
that would win it rather than with `prob_home`. None of that relaxes anything below.

**Things this must not say**, from issue #20. Each is a number this project has gone to some
trouble to make hard to misquote, and a chat message is the shortest path in the system from a real
measurement to a sentence that misrepresents it — there is no reviewer between a bot and a phone.
Where a rule can be kept by construction it is kept that way rather than by care:

* **No calibrated figure for the Live Season.** :func:`live_record` selects
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
  from a second list written here. :func:`last_results` counts correct picks, which *is* accuracy,
  and says in the same message that it is the lay reading and not the metric.
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

from collections.abc import Sequence

import pandas as pd

from epl import predictors
from epl.bot import fires, render, watch
from epl.clubs import ClubResolver
from epl.ingest.fixtures import latest_fixtures_path, parse_fixtures
from epl.ingest.football_data import match_table
from epl.ledger import live as store
from epl.ledger import scoreboard
from epl.paths import processed_dir
from epl.windows import EVALUATION_WINDOW, LIVE_SEASON, season_label

#: How many lines of a failed run's output to quote. Enough to see the complaint, short of pasting
#: a traceback into a chat window.
TAIL_LINES = 14

#: Which Predictor a message means when it says "the model", and which it means by "the market".
#:
#: A **chosen** pair, not a view of the registry, and the same decision as
#: :data:`epl.pundits.report.OPPONENTS` for the same reason. Nine Predictors are registered and a
#: match card has room for two columns; reading the registry here would mean a Predictor added
#: tomorrow silently became the thing a message calls "the model", which is an argument nobody made.
#: Dixon-Coles because it is the best-scoring model on the Evaluation Window (0.19752 RPS); the
#: Market Line because it is the benchmark ADR 0001 chose. Boards still show every Predictor that
#: spoke — this pair is for the single-match messages, where a table would not fit anyway.
HEADLINE_MODEL = "dixon_coles"
MARKET = "market_line"

#: How far apart a Prediction and the Market Line must be before a message points it out.
#:
#: On any Outcome, in percentage points. Below this the two are agreeing within the noise either
#: could carry, and a bot that announced a two-point gap as a disagreement would be crying wolf
#: twice a week. Measured on the one sealed round the store holds: seven of its ten Fixtures sit
#: under this and the three above it are the ones worth reading.
DISAGREEMENT_POINTS = 6

#: How far a Pre-Match Reading must have moved from what was sealed before a card mentions it.
#:
#: Same units and the same argument. A Reading is computed from a corpus that has grown by a couple
#: of results, so it moves a little on every Fixture; the interesting case is the one where it moved
#: enough to change what a reader would do with it.
MOVEMENT_POINTS = 4

#: How many Fixtures :func:`disagreements` reports. Three, because the message is read on a phone
#: and a ranked list whose tail is noise teaches its reader to stop at the top anyway.
DISAGREEMENTS_SHOWN = 3

#: The expected return, per unit staked, below which :func:`value` says nothing about an Outcome.
#:
#: Five percent, and it is a noise floor rather than a target. The model's edge over the offered
#: price is the difference between two numbers that are each uncertain, and the smaller of the two
#: uncertainties is not small: over the Evaluation Window this model is measured **worse** than the
#: market it would be betting against (0.19752 RPS against 0.19362). An edge of one or two percent
#: is comfortably inside that gap and reporting it would be reporting the model's error as a
#: finding.
VALUE_THRESHOLD = 0.05

#: What the Evaluation Window says about the two Predictors a value bet is a disagreement between.
#:
#: Carried as data rather than as prose so it cannot drift from the board, and quoted in every
#: message :func:`value` produces. This is the single most important number in that message: the
#: market outscores the model over 7,980 Fixtures, so an Outcome where the model thinks the price is
#: wrong is the model contradicting the better-scoring of the two.
MARKET_RPS = 0.19362
MODEL_RPS = 0.19752

#: The list marker. An ASCII hyphen rather than a bullet character: the repository owner asked for
#: no emoji, and a bullet is not an emoji but renders as a box on the devices that lack the glyph,
#: which is the same problem (:func:`epl.bot.render.ascii_only`).
MARK = "  -"


def help_text(commands: Sequence[object] = ()) -> str:
    """The command menu, and what the bot is not allowed to do.

    Built from :data:`epl.bot.serve.COMMANDS`, passed in rather than imported: `serve` imports this
    module, so reading it back from here would close the loop. Passing it keeps the menu and the
    dispatch table the same object, which is what stops a command from being offered and not
    answered, or answered and never offered.
    """
    listing = [
        f"/{getattr(command, 'name', command):<9} {getattr(command, 'summary', '')}".rstrip()
        for command in commands
    ]
    return render.document(
        [
            "EPL PREDICTOR",
            render.block(listing) if listing else "",
            "Every number is a probability, not a tip. /explain says how to read them.",
            "This bot is read-only: it cannot seal, score or change anything. A chat app is not "
            "a second door into the sealed store (ADR 0005). RPS is the headline metric "
            "everywhere here; getting the winner right is the lay reading and never the score.",
        ]
    )


def explain() -> str:
    """How to read the numbers, for somebody who has never opened this repository.

    Written out rather than linked. The one thing every other message assumes is that the reader
    knows a percentage here is a probability and not a prediction, and there is nowhere else in the
    system that says so to a person holding a phone.
    """
    return render.document(
        [
            "HOW TO READ THIS",
            "Every match gets three numbers that add up to 100: one for each side winning, "
            "and one for the draw.",
            "A 59 next to Manchester City means that in 100 matches like this one, the model "
            "expects City to win 59 of them. It is not a claim about this match, and a 59 that "
            "loses is not a wrong number.",
            "MODEL is this project's own forecast. MARKET is what the bookmakers' odds imply once "
            "their margin is taken out. They usually agree closely, and the interesting matches "
            "are the ones where they do not.",
            "The model sees results, and only results: every match played in the top four English "
            "divisions, weighted so recent ones count for more. It does not see injuries, "
            "team news, transfers or weather, which is most of why the bookmakers are still "
            "slightly better than it is.",
            "A forecast is sealed and committed before the round kicks off, so it can never be "
            "quietly improved afterwards. That sealed number is the one this project is scored "
            "on. A message an hour before kick-off may show a fresher reading beside it.",
        ]
    )


def round_digest(prediction_round: str | None = None) -> str:
    """One Sealed Prediction Round: every Fixture in it, and what the model makes of each.

    Defaults to the most recent round in the store. A round that was never sealed is said to be
    missing rather than rendered as an empty table — on this store the difference is the whole
    point, since a round the loop failed to seal is exactly the event worth hearing about.

    This is both `/week` and the body of the announcement sent when a round seals, which is what
    makes the weekly preview free: the loop already fires a notifier at the moment the round is
    written, and there is nothing a separate schedule would know that this does not.
    """
    sealed = store.read()
    if sealed.empty:
        return render.document(
            [
                "NOTHING SEALED YET",
                "The sealed store is empty on this machine. Either the schedule has not run "
                "inside a round's window yet, or this is a fresh clone.",
            ]
        )

    wanted = str(prediction_round or sealed["prediction_round"].max())
    rows = sealed.loc[sealed["prediction_round"] == wanted]
    if rows.empty:
        held = ", ".join(sorted(set(sealed["prediction_round"]))[-4:])
        return render.document(
            [f"{wanted} has never been sealed.", f"The store holds: {held}"]
        )

    fixtures = _fixtures(rows)
    as_of = pd.Timestamp(rows["as_of_instant"].min())
    first, last = fixtures["kickoff"].min(), fixtures["kickoff"].max()

    spoke = sorted(set(rows["predictor"]))
    parts = [
        _headline(rows, wanted, as_of, first, last),
        *_days(fixtures),
        _biggest_gap(fixtures),
        # Issue #20's first criterion asks for what each Predictor said, and the body above shows
        # one of them. Naming the rest here is what keeps the criterion met without putting a
        # four-column table under ten Fixtures; a single-match card carries their numbers.
        f"Sealed by {', '.join(spoke)}. The numbers above are the model's ({HEADLINE_MODEL}); "
        "/next and /club show every Predictor that spoke on one match.",
        _silence(set(spoke)),
        "You will get a message about an hour before each kick-off.",
    ]
    return render.document(parts)


def next_match(now: pd.Timestamp) -> str:
    """The next Fixture to kick off that has a Sealed Prediction, and what was forecast for it."""
    upcoming = _upcoming(now)
    if upcoming.empty:
        return _nothing_upcoming(now)

    first = upcoming.iloc[0]
    return render.document(
        [
            "NEXT MATCH",
            _card_heading(first, now),
            render.block(_table_for(first)),
            _verdict(first),
            *_also_sealed(first),
            _sealed_line(first),
        ]
    )


def disagreements(prediction_round: str | None = None) -> str:
    """The Fixtures where this project's model and the bookmakers are furthest apart.

    The one thing on this bot that is not available from anywhere else: the whole scoreboard is an
    argument about whether the model can keep up with the market, and this is where it is currently
    saying something different. Ranked by the largest gap on any single Outcome rather than by a
    summed distance, because that is the number a reader can check against the two columns in front
    of them.
    """
    sealed = store.read()
    if sealed.empty:
        return "Nothing has been sealed yet, so there is nothing to compare."

    wanted = str(prediction_round or sealed["prediction_round"].max())
    fixtures = _fixtures(sealed.loc[sealed["prediction_round"] == wanted])
    ranked = _ranked_by_gap(fixtures)
    if not ranked:
        return render.document(
            [
                "MODEL AND MARKET AGREE",
                _round_line(wanted),
                f"No Fixture in this round has the two more than {DISAGREEMENT_POINTS} points "
                "apart on any outcome. That is the usual answer and it is a good sign: the "
                "market is the benchmark this project is measured against.",
            ]
        )

    parts = ["WHERE THE MODEL DIFFERS FROM THE BOOKMAKERS", _round_line(wanted)]
    for row, gap in ranked[:DISAGREEMENTS_SHOWN]:
        parts.append(
            f"{render.short(row['home_club'])} v {render.short(row['away_club'])}, "
            f"{render.kickoff_short(row['kickoff'])}"
        )
        parts.append(render.block(_table_for(row)))
        parts.append(f"{_leaning(row)} The largest gap is {gap} points.")
    parts.append(
        "The bookmakers score better than the model over the 21 closed Seasons this project "
        "measures. A gap is where the model would take the other side, not where it is right."
    )
    return render.document(parts)


def value(now: pd.Timestamp) -> str:
    """Which Outcome the model thinks is mispriced, and which side it simply expects to win.

    **Two questions, and they are not the same one.** "Who will win?" is answered by the model's
    highest probability and is usually a favourite whose price reflects exactly that. "What is worth
    backing?" is answered by the model's probability against the price actually on offer, and the
    answer is usually somewhere else entirely — often a longshot. Reporting one under the other's
    name is the single easiest way for this message to mislead, so it reports both, separately, and
    says which is which.

    **Against the offered price, not against the Market Line.** The stored Market Line has had the
    bookmaker's margin removed (ADR 0001), which is right for scoring two Predictors against each
    other and wrong here: nobody can bet at a vig-free price. So the expected return is computed
    from the raw decimal odds in the rolling fixtures file, and already carries the ~5% overround
    (`python -m epl.benchmarks overround`). Using the vig-removed number would overstate every edge
    on this board by about that much.

    **What the message must always carry, and does.** Over the Evaluation Window's 7,980 Fixtures
    the market scores :data:`MARKET_RPS` and this model scores :data:`MODEL_RPS` — the market is
    *better*. Every line below is therefore the model disagreeing with something that has outscored
    it over twenty-one Seasons, which is a fact about the model's opinion and not a demonstrated
    edge. That sentence is not a disclaimer bolted on: it is the most informative thing in the
    message, and :data:`VALUE_THRESHOLD` exists because an edge smaller than that gap is noise.
    """
    upcoming = _upcoming(now)
    if upcoming.empty:
        return _nothing_upcoming(now)

    offered = _offered_odds()
    priced = [
        (row, best)
        for _, row in upcoming.iterrows()
        if (best := _best_return(row, offered)) is not None
    ]
    worth_it = sorted(
        (pair for pair in priced if pair[1][1] >= VALUE_THRESHOLD),
        key=lambda pair: pair[1][1],
        reverse=True,
    )

    return render.document(
        [
            "WHO WINS, AND WHAT IS WORTH BACKING",
            _likeliest(upcoming),
            _value_section(worth_it, priced, offered),
            f"The bookmakers score better than this model over the {len(EVALUATION_WINDOW)} "
            f"Seasons it has been measured on - {MARKET_RPS:.4f} against {MODEL_RPS:.4f}, lower "
            "being better. So everything above is the model disagreeing with something that has "
            "beaten it, which is its opinion rather than an edge it has been shown to have.",
            "Returns are worked against the market-average odds in the fixtures file, so the "
            "bookmaker's margin is already in them. They are what the model's numbers imply, not "
            "what anyone has won.",
        ]
    )


def _likeliest(upcoming: pd.DataFrame) -> str:
    """The Fixtures the model is most confident about, and whether the market agrees.

    Confidence and value point in opposite directions almost by construction — a side the model
    makes a heavy favourite is a side the market has priced as one — so this section exists to be
    read *against* the one below it rather than beside it.
    """
    ranked: list[tuple[pd.Series, int, int, bool]] = []
    for _, row in upcoming.iterrows():
        model = render.percentages(*_model(row))
        pick = max(range(3), key=lambda index: model[index])
        agrees = _has_market(row) and pick == max(
            range(3), key=lambda index: render.percentages(*_market(row))[index]
        )
        ranked.append((row, pick, model[pick], agrees))
    ranked.sort(key=lambda entry: entry[2], reverse=True)

    lines: list[str] = []
    for row, pick, confidence, agrees in ranked[:DISAGREEMENTS_SHOWN]:
        lines.append(f"{render.short(row['home_club'])} v {render.short(row['away_club'])}")
        # Whether the bookmakers pick the same side is the useful half here. The model being sure
        # is worth much less on its own than the model and the market being sure together, and a
        # reader deciding what to back needs to be able to see which of the two they have.
        verdict = "market agrees" if agrees else "market disagrees"
        lines.append(f"       {_outcome_name(row, pick, short=True)} {confidence}%  ({verdict})")
    return render.document(["MOST LIKELY TO WIN", render.block(lines)])


def _value_section(
    worth_it: list[tuple[pd.Series, tuple[int, float, float]]],
    priced: list[tuple[pd.Series, tuple[int, float, float]]],
    offered: dict[tuple[str, str], tuple[float, float, float]],
) -> str:
    """The mispriced Outcomes, or a plain statement that there are none."""
    if not offered:
        return render.document(
            [
                "NO PRICES TO COMPARE AGAINST",
                "The cached fixtures file carries no odds for these Fixtures, so nothing here can "
                "be priced. It is refreshed by the loop's own fires, not by this bot.",
            ]
        )
    if not priced:
        return render.document(
            [
                "NO PRICES TO COMPARE AGAINST",
                "None of the upcoming Fixtures appears in the cached fixtures file with a "
                "price on it.",
            ]
        )
    if not worth_it:
        return render.document(
            [
                "NOTHING WORTH BACKING",
                f"No Outcome clears {VALUE_THRESHOLD:.0%} expected return against the offered "
                "price. That is the ordinary answer and the reassuring one: it means the model and "
                "the bookmakers are reading these matches the same way.",
            ]
        )

    lines: list[str] = []
    for row, (index, expected, price) in worth_it[:DISAGREEMENTS_SHOWN]:
        lines.append(f"{render.short(row['home_club'])} v {render.short(row['away_club'])}")
        lines.append(f"       {_outcome_name(row, index, short=True)} at {price:.2f}")
        # The price's own implied probability rather than the stored Market Line's, so the three
        # numbers on this line are all about the same thing: the price above them. The Market Line
        # has had the margin taken out and would be a percentage point or so lower, which is a
        # distinction worth not making in the middle of a betting message.
        lines.append(
            f"       model {render.percentages(*_model(row))[index]}%"
            f"  price {round(100 / price)}%"
            f"  returns {expected:+.0%}"
        )
    tail = (
        f"{len(worth_it)} Outcomes clear {VALUE_THRESHOLD:.0%}; the {DISAGREEMENTS_SHOWN} best are "
        "above. A round where most of the card looks like value is a round where the model simply "
        "disagrees with the book, which is worth reading as a warning rather than as an "
        "opportunity."
        if len(worth_it) > DISAGREEMENTS_SHOWN
        else ""
    )
    return render.document(["WHERE THE MODEL SEES VALUE", render.block(lines), tail])


def _best_return(
    row: pd.Series, offered: dict[tuple[str, str], tuple[float, float, float]]
) -> tuple[int, float, float] | None:
    """The Outcome of this Fixture with the best expected return, and what that return is.

    ``(index, expected return per unit staked, the decimal price)``, or ``None`` when this Fixture
    has no price. The arithmetic is the whole feature and is one line: a stake returns ``price``
    when the Outcome lands and nothing when it does not, so its expected value is
    ``probability * price - 1``. Positive means the model thinks the price is too long.
    """
    prices = offered.get((str(row["home_club"]), str(row["away_club"])))
    if prices is None or any(price is None or price <= 1.0 for price in prices):
        return None
    model = _model(row)
    returns = [model[index] * prices[index] - 1.0 for index in range(3)]
    best = max(range(3), key=lambda index: returns[index])
    return best, returns[best], prices[best]


def _outcome_name(row: pd.Series, index: int, *, short: bool = False) -> str:
    """An Outcome named after the Club it wins for, because "away win" is not what a reader
    is looking for."""
    name = render.short if short else render.full
    return [f"{name(row['home_club'])} win", "Draw", f"{name(row['away_club'])} win"][index]


def _offered_odds() -> dict[tuple[str, str], tuple[float, float, float]]:
    """The decimal prices in the most recently cached fixtures file, keyed by the two Clubs.

    Read from the cache and never fetched — this is a bot, and a `/bet` that went to the network
    would be a chat command that hangs (`tests/bot/test_the_bot_is_read_only.py`). The file is
    refreshed by the loop's own fires, so these are the prices as of the last one.

    An empty mapping when the cache holds nothing, which is an ordinary state rather than an error:
    `data/raw/` is gitignored, so a fresh clone has no fixtures file at all.
    """
    path = latest_fixtures_path()
    if path is None:
        return {}
    try:
        rolling = parse_fixtures(path)
    except Exception:  # pragma: no cover - a malformed cache must not take the bot down
        return {}
    return {
        (str(row["home_club"]), str(row["away_club"])): (
            _price(row["prematch_odds_home"]),
            _price(row["prematch_odds_draw"]),
            _price(row["prematch_odds_away"]),
        )
        for _, row in rolling.iterrows()
    }


def _price(offered: object) -> float:
    """One decimal price, or 0.0 where the book carried none for that Outcome."""
    if not isinstance(offered, (int, float)) or pd.isna(offered):
        return 0.0
    return float(offered)


def for_club(argument: str, now: pd.Timestamp) -> str:
    """One Club's next Fixture and the forecast for it.

    The argument is matched against short names, canonical names and Football-Data's own spellings
    before it is given up on, because the three differ for exactly the Clubs somebody is most likely
    to type: Spurs, Wolves, Forest.
    """
    if not argument.strip():
        return "Name a club, like /club arsenal."

    upcoming = _upcoming(now)
    if upcoming.empty:
        return _nothing_upcoming(now)

    slug = _slug_for(argument, set(upcoming["home_club"]) | set(upcoming["away_club"]))
    if slug is None:
        playing = sorted({render.short(name) for name in upcoming["home_club"]}
                         | {render.short(name) for name in upcoming["away_club"]})
        return render.document(
            [
                f"No club here matches {argument.strip()!r}.",
                "Playing in this round:",
                render.block(playing),
            ]
        )

    theirs = upcoming.loc[
        upcoming["home_club"].eq(slug) | upcoming["away_club"].eq(slug)
    ]
    if theirs.empty:
        return f"{render.full(slug)} have no upcoming Fixture in the sealed store."

    row = theirs.iloc[0]
    return render.document(
        [
            render.full(slug).upper(),
            _card_heading(row, now),
            render.block(_table_for(row)),
            _verdict(row),
            *_also_sealed(row),
            _sealed_line(row),
        ]
    )


def last_results() -> str:
    """The most recent round whose Fixtures have been played, against what was forecast for them.

    The count of correct picks in here **is** accuracy, which this project never reports as a
    headline (CLAUDE.md). It is kept because it is the one reading a person can check against their
    own memory of the weekend, and it is labelled in the same message as the lay reading rather than
    the score. `/record` is where the score is.
    """
    sealed = store.read()
    matches = _matches()
    if sealed.empty:
        return "Nothing has been sealed yet, so there is nothing to look back on."
    if matches is None:
        return _no_match_table()

    model = sealed.loc[sealed["predictor"].eq(HEADLINE_MODEL)]
    played = model.merge(
        matches[["season", "division", "home_club", "away_club", "home_goals", "away_goals",
                 "outcome"]],
        on=["season", "division", "home_club", "away_club"],
        how="inner",
    )
    if played.empty:
        return render.document(
            [
                "NO RESULTS YET",
                f"{len(sealed)} Predictions are sealed and none of their Fixtures has been "
                "played and ingested yet.",
            ]
        )

    wanted = str(played["prediction_round"].max())
    round_rows = played.loc[played["prediction_round"] == wanted].sort_values("kickoff")

    lines: list[str] = []
    right = 0
    for _, row in round_rows.iterrows():
        home, draw, away = render.percentages(
            row["prob_home"], row["prob_draw"], row["prob_away"]
        )
        called = max(
            (("H", home), ("D", draw), ("A", away)), key=lambda pair: pair[1]
        )
        hit = called[0] == str(row["outcome"])
        right += int(hit)
        lines.append(
            f"{render.short(row['home_club'])} {int(row['home_goals'])}-"
            f"{int(row['away_goals'])} {render.short(row['away_club'])}"
        )
        lines.append(
            f"       called {_named(row, called[0])} {called[1]}"
            f"{'' if hit else ' - wrong'}"
        )

    return render.document(
        [
            "LAST ROUND",
            f"Round of {wanted}, {len(round_rows)} matches played",
            render.block(lines),
            f"{right} of {len(round_rows)} picks were right. That is the lay reading and not "
            "the score: this project is judged on how well-calibrated its probabilities are "
            "(RPS), which /record has, and a 40 that comes in is a good forecast.",
        ]
    )


def live_record() -> str:
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
        return _no_match_table()

    board = scoreboard.build(sealed, matches, seasons=[LIVE_SEASON])
    if board.empty:
        return render.document(
            [
                f"{season_label(LIVE_SEASON)} - NOTHING SCORED YET",
                f"{len(sealed)} Sealed Predictions, none of whose Fixtures has been played and "
                "ingested yet.",
            ]
        )

    fixtures = int(board["fixtures"].max())
    return render.document(
        [
            "LIVE TRACK RECORD",
            f"{season_label(LIVE_SEASON)}, sealed and scored on "
            f"{_count_of(fixtures, 'Fixture')} played so far",
            render.block(_table(board, scoreboard.PRE_CALIBRATION_COLUMNS)),
            "RPS first, and lower is better. Anything above the naive baseline has no value.",
            f"RPS over {_count_of(fixtures, 'Fixture')} is noise and not a track record. The "
            f"Evaluation Window is {len(EVALUATION_WINDOW)} closed Seasons, is scored separately "
            "and does not move when this does; /board has its numbers. Pre-calibration only, "
            "because a calibration map needs a track record behind it and this Season has none. "
            "A pundit's column fills in retrospectively, once the archive catches up.",
            _notes(board),
        ]
    )


def evaluation_board() -> str:
    """The Evaluation Window's scoreboard, as `python -m epl.ledger scoreboard` last wrote it.

    Read rather than recomputed, and refused rather than recited. Recomputing needs
    `outputs/backtest/`, which is gitignored and is written by the one command `deploy/crontab` must
    never schedule (ADR 0005) — so on the Pi there is nothing to recompute from. Reciting the
    README's numbers instead would be a bot reporting a measurement it does not hold, on a machine
    where it may since have changed.

    **Both halves, unlike :func:`live_record`, and the difference is not an inconsistency.**
    ADR 0006 publishes every metric twice and says neither may be reported alone: the shared
    calibration layer makes every Predictor slightly worse over this window and is kept anyway,
    and showing only
    the better column is exactly how that finding gets lost. The live board is pre-calibration only
    for the opposite reason — there is no track record for a map to have been fitted on at all, so
    there is no second column, rather than one being withheld.

    The file's own modification time goes with it. A derived file carries no date and this one is
    regenerated by hand, which is how a bot comes to report last month's board as this month's.
    """
    board_path = scoreboard.path()
    if not board_path.exists():
        return render.document(
            [
                "NO EVALUATION WINDOW BOARD ON THIS MACHINE",
                "That board is derived and regenerable, so it is not in git (ADR 0005), and "
                "rebuilding it needs outputs/backtest/, which nothing on a schedule writes.",
                "On a machine that has one:",
                render.block(
                    ["python -m epl.ledger backfill", "python -m epl.ledger scoreboard"]
                ),
            ]
        )

    board = pd.read_csv(board_path)
    written = pd.Timestamp(board_path.stat().st_mtime, unit="s")
    return render.document(
        [
            f"EVALUATION WINDOW - {len(EVALUATION_WINDOW)} CLOSED SEASONS",
            f"written {written:%Y-%m-%d %H:%M} by python -m epl.ledger scoreboard",
            "Pre-calibration",
            render.block(_table(board, scoreboard.PRE_CALIBRATION_COLUMNS)),
            "Post-calibration",
            render.block(_table(board, scoreboard.POST_CALIBRATION_COLUMNS)),
            "Both halves, because neither may be reported alone (ADR 0006). The shared "
            "calibration layer makes every Predictor slightly worse here and is kept anyway; "
            "publishing only the better column is how that finding gets lost. RPS is the "
            "headline. Accuracy is for lay explanation only and is never it.",
            _notes(board),
        ]
    )


def health(found: Sequence[fires.Fire], *, now: pd.Timestamp) -> str:
    """What the schedule has been doing, and anything worth worrying about.

    The last fire of each subcommand rather than the last fire overall, because they answer
    different questions: `seal` costs a round when it fails, `score` costs a day's delay, and
    `prematch` costs one message about one match.
    """
    lines: list[str] = []
    if not found:
        lines.append("The loop has never fired on this machine.")
        lines.append("If the crontab is installed, that is open risk 6.")
    for subcommand in ("seal", "score", "prematch"):
        last = fires.latest(found, subcommand=subcommand)
        if last is None:
            if found:
                lines.append(f"{MARK} {subcommand}: never")
            continue
        lines.append(f"{MARK} {subcommand}: {_describe(last)}")
        if last.failed:
            lines += [f"       {line}" for line in last.tail(4).splitlines()]

    raised = watch.concerns(found, now=now)
    tail = (
        "\n\n".join(concern.message() for concern in raised)
        if raised
        else ("Nothing else to report: no missed sealing day, no unchanged upstream file."
              if found else "")
    )
    return render.document(["SCHEDULE HEALTH", "\n".join(lines), tail])


def failure(fire: fires.Fire) -> str:
    """A run that needs somebody, in its own words.

    `epl.live.__main__._seal`'s exit-code contract distinguishes several failures — `NOT PUSHED`,
    a stale `LIVE_SEASON`, a rolling file that changed shape — and this must not flatten them into
    "the loop failed". So the loop's own output is quoted rather than summarised.

    A run with no exit code was killed rather than having failed, and is said that way: inventing
    an exit code would be making a claim about what the loop decided.

    **This is the one message exempt from :data:`epl.bot.render.PRE_WIDTH`**, and the exemption is
    the same decision as quoting rather than summarising. The loop's lines are as long as they are;
    re-wrapping them to fit a phone would move the line breaks in a log somebody is reading to work
    out what broke, and truncating them would drop the end of the complaint. An operator reading a
    failure will scroll. `tests/bot/test_render.py` names this message as the exception rather than
    silently not covering it.
    """
    verdict = (
        "was killed before it could finish (no exit code)"
        if fire.exit_code is None
        else f"failed with exit {fire.exit_code}"
    )
    return render.document(
        [
            "SCHEDULE FAILURE",
            f"{fire.command} {verdict} at {fire.local:%Y-%m-%d %H:%M %Z}",
            render.block(
                render.asciify(
                    fire.tail(TAIL_LINES) or "(the run printed nothing)"
                ).splitlines()
            ),
        ]
    )


def quiet(fire: fires.Fire, concern: watch.Concern) -> str:
    """A fire that sealed nothing, and which of the two silences it was.

    Most weeks this is the true and boring answer and nobody needs to be told. The push half
    decides that (:func:`epl.bot.notify.run`); what this does is make the two distinguishable at
    all, which they are not from the exit code — both are 0 on purpose (issue #19).
    """
    silence = {
        fires.NOTHING_IN_FILE: "the rolling file held no row in a tier this project predicts",
        fires.OUTSIDE_EVERY_WINDOW: "no Prediction Round was inside its sealing window",
    }.get(fire.verdict, "nothing was sealed")
    return render.document(
        [
            "NOTHING SEALED",
            f"{fire.command} at {fire.local:%a %d %b %H:%M %Z} sealed nothing: {silence}.",
            concern.message() if concern is not None else "",
        ]
    )


def sealed_announcement(prediction_round: str | None = None) -> str:
    """A round has just been sealed. The body is :func:`round_digest`'s."""
    return render.combine("PREDICTIONS SEALED", round_digest(prediction_round))


def scored_announcement() -> str:
    """A sealed round has been scored. The body is :func:`live_record`'s."""
    return render.combine("ROUND SCORED", live_record())


def prematch_card(reading: pd.DataFrame, *, now: pd.Timestamp) -> str:
    """One Fixture, about an hour before it kicks off, from a Pre-Match Reading.

    The Reading was computed after the round was sealed, from a corpus that now holds the results of
    matches played earlier in the same round — which is genuinely more than the sealed forecast knew
    and is the whole reason this message exists. The sealed row is quoted beside it only when the
    two have moved apart by :data:`MOVEMENT_POINTS`, so an ordinary card is three numbers and not a
    comparison nobody asked for.

    What is never blurred is which of the two is the record. The sealed Prediction is what gets
    scored; this is a reading, and the message says so every time.
    """
    fixtures = _fixtures(reading)
    if fixtures.empty:
        return ""
    first = fixtures.iloc[0]
    return render.document(
        [
            f"KICK-OFF {render.relative(first['kickoff'], fires.wall_clock(now)).upper()}",
            _card_heading(first, now, exact=False),
            render.block(_table_for(first)),
            _verdict(first),
            _movement(first),
            f"Read at {pd.Timestamp(first['as_of_instant']):%H:%M} today. The forecast sealed "
            "before this round began is unchanged, and is the one that gets scored.",
        ]
    )


def _round_line(prediction_round: str) -> str:
    """A round id as a day somebody would say out loud.

    A Prediction Round is identified by the ISO date it anchors to, which is right in a filename and
    in a commit message and is a number in a chat window. Rendered here rather than at three call
    sites so that a round is named the same way in every message it appears in.
    """
    return f"Round of {pd.Timestamp(prediction_round):%A %d %B}"


def _headline(
    rows: pd.DataFrame,
    wanted: str,
    as_of: pd.Timestamp,
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> str:
    """The three lines at the top of a digest: which round, how big, and when it was sealed."""
    span = (
        f"{pd.Timestamp(first):%A}"
        if pd.Timestamp(first).normalize() == pd.Timestamp(last).normalize()
        else f"{pd.Timestamp(first):%A} to {pd.Timestamp(last):%A}"
    )
    return "\n".join(
        [
            _round_line(wanted),
            f"{_count(rows, 'match')}, {span}",
            f"Sealed {as_of:%H:%M on %A}, before any kick-off",
        ]
    )


def _days(fixtures: pd.DataFrame) -> list[str]:
    """A digest's body: one heading and one fixed-width block per day of the round."""
    parts: list[str] = []
    for day, playing in fixtures.groupby(fixtures["kickoff"].dt.normalize(), sort=True):
        lines: list[str] = []
        for _, row in playing.sort_values("kickoff").iterrows():
            lines += _entry(row)
        parts.append(render.day_heading(day))
        parts.append(render.block(lines))
    return parts


def _entry(row: pd.Series) -> list[str]:
    """One Fixture in a digest, as lines that are guaranteed to fit.

    The heading is split over two lines rather than allowed to wrap when two long Club names meet.
    A wrapped line inside a fixed-width block loses the alignment the block was for, and the fix has
    to be one this code makes rather than one the phone makes.
    """
    home, away = render.short(row["home_club"]), render.short(row["away_club"])
    heading = f"{render.time_only(row['kickoff'])}  {home} v {away}"
    lines = (
        [heading]
        if render.fits(heading)
        else [f"{render.time_only(row['kickoff'])}  {home}", f"       v {away}"]
    )
    return [*lines, render.summary_line(row["home_club"], row["away_club"], _model(row))]


def _biggest_gap(fixtures: pd.DataFrame) -> str:
    """The one Fixture in a round where the model and the bookmakers are furthest apart."""
    ranked = _ranked_by_gap(fixtures)
    if not ranked:
        return ""
    row, gap = ranked[0]
    return (
        f"Biggest gap with the bookmakers: {render.short(row['home_club'])} v "
        f"{render.short(row['away_club'])}, {gap} points. {_leaning(row)}"
    )


def _ranked_by_gap(fixtures: pd.DataFrame) -> list[tuple[pd.Series, int]]:
    """Fixtures where the model and the Market Line differ, worst first.

    Compared as whole percentages rather than as the stored floats, because that is what the reader
    is shown: a message saying two Predictors are six points apart, above a table where they read
    the same, would be right and useless.
    """
    ranked: list[tuple[pd.Series, int]] = []
    for _, row in fixtures.iterrows():
        if not _has_market(row):
            continue
        model = render.percentages(*_model(row))
        market = render.percentages(*_market(row))
        gap = max(abs(one - other) for one, other in zip(model, market, strict=True))
        if gap >= DISAGREEMENT_POINTS:
            ranked.append((row, gap))
    return sorted(ranked, key=lambda pair: pair[1], reverse=True)


def _leaning(row: pd.Series) -> str:
    """Which way a disagreement runs, said in the direction a reader thinks in."""
    model = render.percentages(*_model(row))
    market = render.percentages(*_market(row))
    names = [render.short(row["home_club"]), "a draw", render.short(row["away_club"])]
    above = [one - other for one, other in zip(model, market, strict=True)]
    keenest = max(range(3), key=lambda index: above[index])
    return f"The model rates {names[keenest]} higher than the bookmakers do."


def _verdict(row: pd.Series) -> str:
    """One sentence naming the likeliest Outcome, and whether the bookmakers agree."""
    model = render.percentages(*_model(row))
    names = [
        f"{render.full(row['home_club'])} to win",
        "a draw",
        f"{render.full(row['away_club'])} to win",
    ]
    pick = max(range(3), key=lambda index: model[index])
    sentence = f"The model's pick is {names[pick]}, at {model[pick]} percent."
    if not _has_market(row):
        return sentence + " No bookmakers' odds were carried for this Fixture."

    market = render.percentages(*_market(row))
    if max(range(3), key=lambda index: market[index]) != pick:
        theirs = names[max(range(3), key=lambda index: market[index])]
        return sentence + f" The bookmakers make it {theirs}."
    gap = max(abs(one - other) for one, other in zip(model, market, strict=True))
    return sentence + (
        " The bookmakers agree closely."
        if gap < DISAGREEMENT_POINTS
        else f" The bookmakers agree on the winner, {gap} points apart on how likely."
    )


def _movement(row: pd.Series) -> str:
    """How far this Reading has moved from what was sealed, when it has moved at all."""
    sealed = _sealed_for(row)
    if sealed is None:
        return ""
    fresh = render.percentages(*_model(row))
    before = render.percentages(*_model(sealed))
    moved = [abs(one - other) for one, other in zip(fresh, before, strict=True)]
    if max(moved) < MOVEMENT_POINTS:
        return ""
    names = [render.short(row["home_club"]), "the draw", render.short(row["away_club"])]
    # The biggest mover, and on a tie the Outcome the model now picks. Two Outcomes moving the same
    # distance in opposite directions is the ordinary shape of a shift — one side gains what the
    # other loses — and of the two, the one the reader is looking at is the one being predicted.
    index = max(range(3), key=lambda position: (moved[position], fresh[position]))
    direction = "up" if fresh[index] > before[index] else "down"
    return (
        f"Moved since this round was sealed: {names[index]} {direction} from "
        f"{before[index]} to {fresh[index]}, on results since."
    )


def _sealed_for(row: pd.Series) -> pd.Series | None:
    """The sealed Prediction this Reading is a later look at, if the store holds one."""
    sealed = store.read()
    if sealed.empty:
        return None
    same = sealed.loc[
        sealed["predictor"].eq(row["predictor"])
        & sealed["season"].eq(row["season"])
        & sealed["division"].eq(row["division"])
        & sealed["home_club"].eq(row["home_club"])
        & sealed["away_club"].eq(row["away_club"])
    ]
    return None if same.empty else same.sort_values("as_of_instant").iloc[0]


def _card_heading(row: pd.Series, now: pd.Timestamp, *, exact: bool = True) -> str:
    """Two lines: who is playing, spelt out, and when."""
    when = render.kickoff_long(row["kickoff"])
    if not exact:
        return f"{render.full(row['home_club'])} v {render.full(row['away_club'])}\n{when}"
    return (
        f"{render.full(row['home_club'])} v {render.full(row['away_club'])}\n"
        f"{when}, {render.relative(row['kickoff'], fires.wall_clock(now))}"
    )


def _sealed_line(row: pd.Series) -> str:
    return (
        f"Sealed {pd.Timestamp(row['as_of_instant']):%H:%M on %A %d %B}, before any of this "
        "round had kicked off."
    )


def _also_sealed(row: pd.Series) -> list[str]:
    """What every *other* Predictor said about this Fixture.

    A card shows two Predictors because two columns is what a phone has room for, and the digest
    shows one. Neither is the whole of what was sealed: four spoke on the first real round, and Elo
    disagreeing with Dixon-Coles by four points is a fact about the forecast rather than clutter.
    So the rest go here, on the one message that has room for them — which is also what keeps this
    package's promise that a message about the sealed store does not quietly narrow it.

    Home, draw and away in the ledger's own order, with the order named in the caption. The card
    above sorts by probability, which is right when each row is labelled with the Club it belongs
    to and would be unreadable here, where three numbers share a line.
    """
    every = store.read()
    if every.empty:
        return []
    others = every.loc[
        every["season"].eq(row["season"])
        & every["division"].eq(row["division"])
        & every["home_club"].eq(row["home_club"])
        & every["away_club"].eq(row["away_club"])
        & ~every["predictor"].isin([HEADLINE_MODEL, MARKET])
    ].sort_values("predictor")
    if others.empty:
        return []

    width = int(others["predictor"].str.len().max())
    lines = [
        f"{quote['predictor']:<{width}}  "
        + " / ".join(f"{value:2d}" for value in render.percentages(*_model(quote)))
        for _, quote in others.iterrows()
    ]
    return [
        f"Also sealed for this match, as home / draw / away, by the other "
        f"{_count_of(len(lines), 'Predictor')} that spoke:",
        render.block(lines),
    ]


def _table_for(row: pd.Series) -> list[str]:
    """A match card's fixed-width body, with the market column only when there is one."""
    return render.outcome_table(
        row["home_club"],
        row["away_club"],
        _model(row),
        _market(row) if _has_market(row) else None,
    )


def _fixtures(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per Fixture, carrying every Predictor's numbers as its own columns.

    A flat frame rather than a group-by at each call site: every message about a Fixture wants the
    model and the market side by side, and a Predictor that said nothing about it has to be an
    absent column rather than a missing row.
    """
    if rows.empty:
        return rows
    keys = ["season", "division", "kickoff", "home_club", "away_club"]
    wide = rows.loc[rows["predictor"].eq(HEADLINE_MODEL)].copy()
    market = rows.loc[rows["predictor"].eq(MARKET)].set_index(keys)
    for outcome in ("home", "draw", "away"):
        wide[f"market_{outcome}"] = wide.set_index(keys).index.map(
            market[f"prob_{outcome}"]
        )
    return wide.sort_values("kickoff").reset_index(drop=True)


def _model(row: pd.Series) -> tuple[float, float, float]:
    return (float(row["prob_home"]), float(row["prob_draw"]), float(row["prob_away"]))


def _market(row: pd.Series) -> tuple[float, float, float]:
    return (
        float(row.get("market_home", 0.0)),
        float(row.get("market_draw", 0.0)),
        float(row.get("market_away", 0.0)),
    )


def _has_market(row: pd.Series) -> bool:
    return pd.notna(row.get("market_home")) and float(row.get("market_home") or 0.0) > 0.0


def _upcoming(now: pd.Timestamp) -> pd.DataFrame:
    """Sealed Fixtures that have not kicked off, soonest first.

    ``now`` arrives as :func:`epl.bot.fires.uk_now`'s aware instant and a kickoff is naive UK
    wall-clock, so it is converted rather than compared — see :func:`epl.bot.fires.wall_clock`.
    """
    sealed = store.read()
    if sealed.empty:
        return sealed
    fixtures = _fixtures(sealed)
    if fixtures.empty:
        return fixtures
    return fixtures.loc[fixtures["kickoff"] > fires.wall_clock(now)].reset_index(drop=True)


def _nothing_upcoming(now: pd.Timestamp) -> str:
    sealed = store.read()
    if sealed.empty:
        return "Nothing has been sealed yet, so there is no next match to report."
    return render.document(
        [
            "NO MATCH COMING UP",
            "Every Fixture in the sealed store has kicked off. The next round is sealed on its "
            "own Tuesday or Friday, and a message goes out when it is.",
            "/results has the last one.",
        ]
    )


def _slug_for(argument: str, playing: set[str]) -> str | None:
    """Whatever somebody typed, as a Club slug — short name, canonical name or source spelling.

    Clubs in the round come first. "United" matches four Clubs in the corpus and one in any given
    Premier League round, and answering about the one that is actually playing is right far more
    often than refusing.
    """
    wanted = argument.strip().casefold().replace("_", " ")
    resolver = ClubResolver.load()

    def candidates(slug: str) -> set[str]:
        return {slug.replace("_", " "), render.short(slug).casefold(), render.full(slug).casefold()}

    for pool in (sorted(playing), sorted(resolver.clubs)):
        for slug in pool:
            if wanted in {name.casefold() for name in candidates(slug)}:
                return slug
    for pool in (sorted(playing), sorted(resolver.clubs)):
        for slug in pool:
            if any(wanted in name.casefold() for name in candidates(slug)):
                return slug
    for source in resolver.sources():
        if resolver.knows(argument.strip(), source):
            return resolver.resolve(argument.strip(), source)
    return None


def _named(row: pd.Series, outcome: str) -> str:
    return {
        "H": render.short(row["home_club"]),
        "D": "the draw",
        "A": render.short(row["away_club"]),
    }[outcome]


def _matches() -> pd.DataFrame | None:
    """The match table, or ``None`` when this machine has not built one.

    :func:`epl.ingest.football_data.match_table` raises :class:`SystemExit` on a missing file, which
    is right for a command line and would take the bot's process down in the middle of answering a
    message. So the file is checked for rather than the exception caught: a bot that exited on a
    `/record` would be the monitoring going down before the thing it monitors.
    """
    if not (processed_dir() / "matches.csv").exists():
        return None
    return match_table()


def _no_match_table() -> str:
    return render.document(
        [
            "NO MATCH TABLE ON THIS MACHINE",
            "data/processed/ is gitignored and is rebuilt by:",
            render.block(
                [
                    "python -m epl.live score",
                    "python -m epl.ingest fetch && python -m epl.ingest build",
                ]
            ),
        ]
    )


def _silence(spoke: set[str]) -> str:
    """The registered Predictors that said nothing about this round, and why that is expected.

    Derived by difference rather than listed, for the same reason `epl.live.seal` has no branch per
    Predictor: which ones can speak to an unplayed Fixture is each Predictor's own business, and a
    list here would be a second opinion about it that could quietly go out of date.
    """
    silent = sorted(
        predictor.name for predictor in predictors.registered() if predictor.name not in spoke
    )
    if not silent:
        return ""
    return (
        f"Said nothing about this round: {', '.join(silent)}. A pundit cannot be part of a "
        "sealed forecast - the only source this project may read transcribes a matchday after "
        "it is played - so their column fills in retrospectively rather than staying blank."
    )


def _table(board: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """A scoreboard, rendered in :data:`epl.ledger.scoreboard.METRICS`' own order.

    RPS comes first because that tuple puts it first, which is the project's rule stated once
    rather than a second ordering written here — and it is what keeps accuracy off the headline
    without anybody having to remember to.

    **One Predictor per entry rather than one per row, and that is forced.** The board this reads
    from is nine Predictors wide by seven columns, which `DataFrame.to_string` lays out at 71
    characters pre-calibration and **135** post — against the 44 a phone can show without wrapping,
    and a wrapped line inside a fixed-width block is worse than no block at all. Turning the table
    on its side is what makes the whole board fit; nothing is dropped, and the order the columns
    arrive in is the order they are printed in, so RPS is still first and accuracy is still not.

    The ``calibrated_`` prefix is dropped from the labels because the block sits under a heading
    that already says which half it is, and eleven characters of prefix on every line of a
    44-character budget is the difference between fitting and not.
    """
    shown = [name for name in columns if name in board.columns]
    if len(shown) < 2:
        # A board that carries neither of this half's first two columns is a board this half is not
        # about — the post-calibration selection over a pre-calibration file, which
        # `evaluation_board` reads straight off disk and cannot assume the shape of. Nothing rather
        # than a partial table: half a board under a heading naming the other half is worse than an
        # absent one, and the caller drops an empty block (:func:`epl.bot.render.document`).
        return []
    name_column, count_column, metrics = shown[0], shown[1], shown[2:]
    width = max((len(_metric_label(metric)) for metric in metrics), default=0)

    lines: list[str] = []
    for _, entry in board.iterrows():
        lines.append(f"{entry[name_column]}  ({int(entry[count_column])} {count_column})")
        pairs = [
            f"{_metric_label(metric):<{width}} {_metric_value(entry[metric])}"
            for metric in metrics
        ]
        lines += [f"  {'   '.join(pairs[at:at + 2])}" for at in range(0, len(pairs), 2)]
    return lines


def _metric_label(column: str) -> str:
    return column.removeprefix("calibrated_")


def _metric_value(value: object) -> str:
    """One metric, to four places, or a dash where a board has no value for it.

    A dash rather than `nan`: a Predictor the calibration layer never reached has an empty cell on
    the post-calibration half, and printing the float's own spelling of "no answer" reads as a
    number that went wrong rather than one that was never taken.
    """
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _notes(board: pd.DataFrame) -> str:
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
            lines.append(f"{MARK} {name}: {render.asciify(' '.join(caveat.split()))}")
    return "\n".join(["Read these with their numbers:", *lines]) if lines else ""


def _describe(fire: fires.Fire) -> str:
    """One fire as a line: when it ran, in UK time, and how it came out."""
    when = fire.local.strftime("%a %d %b %H:%M %Z")
    ending = "no exit code" if fire.exit_code is None else f"exit {fire.exit_code}"
    return f"{when} - {ending}, {fire.verdict}"


def _count(rows: pd.DataFrame, noun: str) -> str:
    return _count_of(rows.groupby(["home_club", "away_club"]).ngroups, noun)


def _count_of(many: int, noun: str) -> str:
    """``1 match`` / ``10 matches``. The sibilant rule, because "10 matchs" reads as a typo."""
    if many == 1:
        return f"{many} {noun}"
    return f"{many} {noun}es" if noun.endswith(("s", "x", "ch", "sh")) else f"{many} {noun}s"


__all__ = [
    "DISAGREEMENTS_SHOWN",
    "DISAGREEMENT_POINTS",
    "HEADLINE_MODEL",
    "MARK",
    "MARKET",
    "MARKET_RPS",
    "MODEL_RPS",
    "MOVEMENT_POINTS",
    "TAIL_LINES",
    "VALUE_THRESHOLD",
    "disagreements",
    "evaluation_board",
    "explain",
    "failure",
    "for_club",
    "health",
    "help_text",
    "last_results",
    "live_record",
    "next_match",
    "prematch_card",
    "quiet",
    "round_digest",
    "scored_announcement",
    "sealed_announcement",
    "value",
]
