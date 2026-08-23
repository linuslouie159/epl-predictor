"""The three-way scoreboard, the cost of stating certainty, and every call ranked by its miss.

Issue #12's last four acceptance criteria, and the artifact the project was actually for: a model,
the market and a Pundit scored on identical metrics over identical Fixtures, with the as-stated
number published beside the calibrated one.

**Every table here is a report over :mod:`epl.ledger.scoreboard`'s own output, and the scoreboard
is untouched.** It "has no branch per Predictor and must not grow one" (issue #7, spec user story
16), so the Pundit-shaped questions live here instead — reading the same calibrated frame every
other report reads, and scoring it with the same code. Nothing in this module computes a metric.

## Why the slate is cut and the calibration is not

A Pundit called 1,896 Fixtures of the Evaluation Window's 7,980, and their margin map covers fewer
still. Comparing an RPS over 1,896 against one over 7,980 is the mistake ADR 0001 records for the
Ceiling Line, so every Predictor is cut to the Fixtures they all reached
(:func:`shared`) — but each is cut *after* the shared isotonic layer has run over its own whole
track record. The Market Line's calibrated form on this board is therefore the one the Market Line
really has, not a weaker map refitted on a Pundit's slate. :func:`epl.ledger.scoreboard.lines`
exists to make that split possible.

One thing that follows from cutting in that order, and is worth knowing before reading the
post-calibration half of a board. The Fixtures are identical and the metrics are identical, but
``corrected`` is **not** the same fraction for every row: the shared layer needs
:data:`epl.calibration.MINIMUM_SAMPLE` of a Predictor's own Predictions behind it, and by the time
a Pundit's slate begins the market and Elo are thousands of Predictions into their records while
the Pundit's has barely started. So some of the two Pundit rows are raw pass-through where none of
the market's are. That is the honest arrangement — the alternative is refitting the market's map on
a Pundit's Fixtures — but it means ``calibrated_rps`` compares differently-corrected Predictions
over the same Fixtures, and the command line says so under the table rather than leaving it in a
column. An unread caveat is the Ceiling Line's whole lesson (ADR 0001).

## What "the cost of stating certainty" is

The same calls, read two ways. `lawrenson` is the Scoreline taken at face value as `[1, 0, 0]`;
`margin_map_lawrenson` is the same Scoreline read as what a call of that predicted goal margin has
historically been worth. Neither is a trick: the first is what the Pundit literally said, and the
second is what it was worth. The gap between them is the price of being asked for a scoreline
instead of a probability, and ADR 0003 is the argument that publishing either alone is dishonest.

It is published as its own file with the gap in a column called ``cost_of_certainty``, rather than
left for a reader to subtract two rows and guess what the difference means.

## Plain files, no presentation logic

Four tables, four files, no formatting in any of them (issue #12's seventh criterion). The
command line prints the head and tail of the calls table; the file holds every call, so a frontend
picks its own N rather than inheriting one chosen here.

* ``outputs/three_way.csv``   — the board, per Pundit slate
* ``outputs/certainty.csv``   — the two readings and the gap
* ``outputs/pundit_calls.csv``— every call, ranked by miss
* ``outputs/margin_map.csv``  — the map itself, as it stood at each Pundit's last round
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from epl import metrics, predictors
from epl.ledger import schema, scoreboard
from epl.paths import outputs_dir
from epl.predictors import Corpus, Evidence
from epl.pundits.calibrated import CalibratedPundit
from epl.pundits.margin import MAP_COLUMNS, margins_of
from epl.rounds import as_of_instant

#: The Predictors every three-way table carries beside the two readings of the Pundit. The model
#: and the market are the "three-way" of issue #12; the floor is there because a Pundit as stated
#: scores *below* it, which is the finding ADR 0003 is built on and is invisible without it.
#:
#: Written out, where :func:`calibrated_pundits` reads the registry instead — and the difference is
#: the point rather than an inconsistency. The scoreboard is *every* registered Predictor, so a
#: list there would be a second copy of the registry waiting to fall behind it. This is a **chosen
#: comparison**: three named opponents, picked because each says something different about the two
#: Pundit readings. Registering a fifth Predictor should put it on the scoreboard and should not
#: silently rewrite an argument ADR 0003 is making.
OPPONENTS: tuple[str, ...] = ("elo", "market_line", "naive_baseline")

#: Canonical column order for the three-way board: the scoreboard's own columns, with the Pundit
#: whose Fixtures define the slate named in front of them.
THREE_WAY_COLUMNS: tuple[str, ...] = ("slate", *scoreboard.SCOREBOARD_COLUMNS)

#: Canonical column order for the cost of stating certainty.
CERTAINTY_COLUMNS: tuple[str, ...] = (
    "slate",
    "as_stated",
    "calibrated",
    "fixtures",
    "as_stated_rps",
    "calibrated_rps",
    "cost_of_certainty",
    "as_stated_accuracy",
    "calibrated_accuracy",
)

#: Canonical column order for one call ranked by its miss.
CALL_COLUMNS: tuple[str, ...] = (
    "pundit",
    "season",
    "prediction_round",
    "home_club",
    "away_club",
    "pred_home_goals",
    "pred_away_goals",
    "margin",
    "outcome",
    "prob_home",
    "prob_draw",
    "prob_away",
    "miss",
    "as_stated_rps",
    "cost_of_certainty",
)

#: Canonical column order for the published maps — one map per Pundit, so the Pundit names it.
PUBLISHED_MAP_COLUMNS: tuple[str, ...] = ("pundit", *MAP_COLUMNS)


def calibrated_pundits() -> tuple[CalibratedPundit, ...]:
    """Every registered Calibrated Pundit, in registration order.

    Read off the registry rather than listed here, so that registering one is what puts it in these
    reports — the same rule the scoreboard follows, and the reason neither has a table of names to
    fall out of date.
    """
    return tuple(
        one for one in predictors.registered() if isinstance(one, CalibratedPundit)
    )


def shared(scored: pd.DataFrame, named: Sequence[str]) -> pd.DataFrame:
    """The rows of ``scored`` on the Fixtures every one of ``named`` has a Prediction for.

    An intersection, not a filter on the narrowest Predictor: which of them is narrowest is a fact
    about the data rather than something a caller should have to know, and a Predictor that is
    missing one Fixture in the middle of its span would otherwise pass unnoticed.

    A name nothing was stored under contributes an empty intersection, which is the honest answer —
    a three-way comparison missing one of its three is not a comparison to publish.
    """
    keys = _fixture_keys(scored)
    slates = [set(keys[scored["predictor"] == name]) for name in named]
    everywhere = set.intersection(*slates) if slates else set()
    return scored.loc[keys.isin(everywhere) & scored["predictor"].isin(list(named))]


def _fixture_keys(rows: pd.DataFrame) -> pd.Series:
    """:data:`epl.ledger.schema.FIXTURE_KEY` read off a frame as tuples, keeping its index."""
    return pd.Series(
        list(zip(*(rows[name] for name in schema.FIXTURE_KEY), strict=True)),
        index=rows.index,
        dtype=object,
    )


def three_way(
    scored: pd.DataFrame,
    pundit: CalibratedPundit,
    *,
    opponents: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One Pundit's slate: the model, the market, the floor, and the two readings of their calls.

    ``scored`` is :func:`epl.ledger.scoreboard.calibrated_predictions` over the whole ledger. Five
    Predictors are cut to the Fixtures all five reached and scored there, every metric twice.

    ``opponents`` defaults to :data:`OPPONENTS` and is resolved in the body rather than in the
    signature, so that the constant is read at every call. A default argument would bind it once at
    import and quietly ignore anything that changed it afterwards.
    """
    named = (*(opponents or OPPONENTS), pundit.pundit.name, pundit.name)
    return scoreboard.lines(shared(scored, named)).assign(slate=pundit.pundit.name)[
        list(THREE_WAY_COLUMNS)
    ]


def boards(
    scored: pd.DataFrame, pundits: Iterable[CalibratedPundit] | None = None
) -> pd.DataFrame:
    """Every Pundit's three-way board, stacked and labelled by whose slate each one is.

    One file rather than one per Pundit, because the two are the same table asked twice and a
    reader comparing them should not have to open two things to do it.
    """
    tables = [three_way(scored, one) for one in (pundits or calibrated_pundits())]
    if not tables:
        return pd.DataFrame(columns=list(THREE_WAY_COLUMNS))
    return pd.concat(tables, ignore_index=True)


def certainty(
    boards_table: pd.DataFrame, pundits: Iterable[CalibratedPundit] | None = None
) -> pd.DataFrame:
    """What stating a Scoreline instead of a probability cost each Pundit, over their own slate.

    Read off the three-way board rather than recomputed, so the two published files cannot disagree
    about a number that appears in both. ``cost_of_certainty`` is the as-stated RPS minus the
    calibrated one: positive means the format of the question was charged to the forecaster.

    Accuracy is carried too, and it is not a formality. It is the one metric on which the as-stated
    reading is *not* obviously unfair — it asks only who they picked — so the pair of columns says
    whether the map is finding real information or merely padding a one-hot into a distribution.
    """
    rows = [
        _certainty_line(boards_table, one) for one in (pundits or calibrated_pundits())
    ]
    return pd.DataFrame(
        [row for row in rows if row is not None], columns=list(CERTAINTY_COLUMNS)
    )


def _certainty_line(
    boards_table: pd.DataFrame, pundit: CalibratedPundit
) -> dict[str, object] | None:
    """One slate's pair of readings, or ``None`` where the board has no rows for it."""
    slate = boards_table.loc[boards_table["slate"] == pundit.pundit.name]
    stated = slate.loc[slate["predictor"] == pundit.pundit.name]
    fair = slate.loc[slate["predictor"] == pundit.name]
    if stated.empty or fair.empty:
        return None

    as_stated, calibrated = stated.iloc[0], fair.iloc[0]
    return {
        "slate": pundit.pundit.name,
        "as_stated": pundit.pundit.name,
        "calibrated": pundit.name,
        "fixtures": int(as_stated["fixtures"]),
        "as_stated_rps": float(as_stated["rps"]),
        "calibrated_rps": float(calibrated["rps"]),
        "cost_of_certainty": float(as_stated["rps"]) - float(calibrated["rps"]),
        "as_stated_accuracy": float(as_stated["accuracy"]),
        "calibrated_accuracy": float(calibrated["accuracy"]),
    }


def calls_by_miss(
    scored: pd.DataFrame, pundit: CalibratedPundit, calls: pd.DataFrame
) -> pd.DataFrame:
    """Every call this Pundit made that their map reached, best first (spec, user story 34).

    **The miss is the RPS of the fair reading** — what the call was still wrong by once the
    Scoreline had been read as the distribution such a call has historically produced. RPS because
    it is the project's primary metric and because it is ordinal: calling Home when Away happens
    misses by more than calling Home when it is a Draw, which is exactly the distinction a "best
    and worst calls" list has to get right.

    ``as_stated_rps`` and ``cost_of_certainty`` ride along per call, so the list can be read the
    other way too: the calls that certainty cost the most are the bold ones that came off, where
    the as-stated reading scored zero and the fair one did not.

    Sorted ascending, so ``head`` is the best calls and ``tail`` the worst. Every call is returned;
    choosing how many to show is the caller's business, not this function's.
    """
    fair = _per_call(scored, pundit.name, "miss")
    stated = _per_call(scored, pundit.pundit.name, "as_stated_rps")
    if fair.empty:
        return pd.DataFrame(columns=list(CALL_COLUMNS))

    together = fair.merge(
        stated[[*schema.FIXTURE_KEY, "as_stated_rps"]],
        on=list(schema.FIXTURE_KEY),
        how="left",
        validate="1:1",
    ).merge(
        _scorelines(calls, pundit.pundit.name),
        on=list(schema.FIXTURE_KEY),
        how="left",
        validate="1:1",
    )
    ranked = together.assign(
        pundit=pundit.pundit.name,
        margin=margins_of(together["pred_home_goals"], together["pred_away_goals"]),
        cost_of_certainty=together["as_stated_rps"] - together["miss"],
    )
    return (
        ranked.sort_values(["miss", *schema.FIXTURE_KEY], kind="stable")
        .reset_index(drop=True)[list(CALL_COLUMNS)]
    )


def ranked_calls(
    scored: pd.DataFrame,
    calls: pd.DataFrame,
    pundits: Iterable[CalibratedPundit] | None = None,
) -> pd.DataFrame:
    """Every Pundit's calls, ranked by miss and stacked — :func:`calls_by_miss`, published.

    One file rather than one per Pundit, for the reason :func:`boards` gives: the two are the same
    table asked twice, and ``pundit`` is a column so either can be read out of it.
    """
    tables = [
        table
        for one in (pundits or calibrated_pundits())
        if not (table := calls_by_miss(scored, one, calls)).empty
    ]
    if not tables:
        return pd.DataFrame(columns=list(CALL_COLUMNS))
    return pd.concat(tables, ignore_index=True)


def _per_call(scored: pd.DataFrame, name: str, column: str) -> pd.DataFrame:
    """One Predictor's scored rows, with the RPS of each raw quote under ``column``.

    Raw rather than post-shared-calibration on both sides, because the pair being compared here is
    "as stated" against "read through the margin map". Putting the shared isotonic layer on one
    side of that subtraction and not the other would make the gap measure two things at once.
    """
    rows = scored.loc[scored["predictor"] == name].reset_index(drop=True)
    if rows.empty:
        return rows.assign(**{column: pd.Series(dtype="float64")})
    return rows.assign(
        **{
            column: metrics.rps_per_prediction(
                rows[list(scoreboard.PROBABILITY_COLUMNS)].to_numpy(float),
                rows["outcome"].tolist(),
            )
        }
    )


def _scorelines(calls: pd.DataFrame, pundit: str) -> pd.DataFrame:
    """This Pundit's published Scorelines, keyed the way the ledger keys a Fixture.

    The frozen dataset has no ``division`` in its Fixture key because the source publishes one tier
    (:data:`epl.pundits.dataset.FIXTURE_KEY`); the ledger's key has one. Reconciled here rather
    than by loosening either, since both are right about their own file.
    """
    mine = calls.loc[calls["pundit"] == pundit]
    return mine[
        ["season", "division", "home_club", "away_club", "pred_home_goals", "pred_away_goals"]
    ].astype({name: schema.DTYPES[name] for name in schema.FIXTURE_KEY})


def published_maps(
    matches: pd.DataFrame, pundits: Iterable[CalibratedPundit] | None = None
) -> pd.DataFrame:
    """Each Pundit's margin map as it stood at their final Prediction Round, ready to publish.

    "What a 3-0 from this Pundit is worth" is the question a reader of this project arrives with,
    and it deserves a file rather than a docstring.

    At their **final round** rather than over their whole record, and that is not fussiness: a map
    fitted on every call would include the calls it was quoting, which is exactly the leak this
    project exists to refuse. The map published here is one a Prediction was really made from — it
    is asked for through :meth:`epl.pundits.calibrated.CalibratedPundit.map_at`, the same call the
    walk makes, cut at the same kind of instant.
    """
    corpus = Corpus(matches)
    tables = [
        _final_map(one, corpus).assign(pundit=one.pundit.name)[list(PUBLISHED_MAP_COLUMNS)]
        for one in (pundits or calibrated_pundits())
    ]
    if not tables:
        return pd.DataFrame(columns=list(PUBLISHED_MAP_COLUMNS))
    return pd.concat(tables, ignore_index=True)


def _final_map(pundit: CalibratedPundit, corpus: Corpus) -> pd.DataFrame:
    """One Pundit's map at the As-Of Instant of the last Fixture they called."""
    last = max(pundit.pundit.calls["date"])
    return pundit.map_at(Evidence.before(corpus, as_of_instant(last))).table()


def path(name: str) -> Path:
    """Where one of this module's four tables is published."""
    return outputs_dir() / f"{name}.csv"


def write(table: pd.DataFrame, name: str) -> Path:
    """Publish one table, and return where it went.

    Deterministic float formatting, shared with the scoreboard, so a regenerated report is
    byte-identical to the last one and a rebuild is never mistaken for a change.
    """
    destination = path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        destination, index=False, float_format=schema.FLOAT_FORMAT, lineterminator="\n"
    )
    return destination
