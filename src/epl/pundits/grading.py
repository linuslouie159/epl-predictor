"""Grading a published Scoreline: the strict reading and the lenient one, side by side.

A Pundit publishes a Scoreline, and there are two honest ways to mark it. **Exact score** is what
the call literally said. **Correct Outcome** is what the call amounts to once the Scoreline is
reduced to Home, Draw or Away. Issue #11 asks for both rather than a choice, because each on its
own reads as an argument: the strict number makes any pundit look hopeless, and the lenient one
quietly discards the part of the claim they actually took a risk on.

Neither is the scoreboard. RPS over the three Outcomes is the metric every Predictor is compared
on (CLAUDE.md), and these two are the lay explanation beside it — the same role accuracy plays for
the models. What they are genuinely for is the surrounding facts: how often a pundit nails a
scoreline, and how that compares to how often they merely pick the winner.

Grading needs an Outcome, so it happens here rather than in the frozen dataset. No row of
`predictions.csv` knows how its Fixture finished, and that is deliberate (ADR 0005) — the join to
the corpus is made at reporting time, exactly as the scoreboard joins the ledger.

The Scoreline-to-Outcome arrow itself is :func:`epl.pundits.dataset.outcomes_of`, beside the calls
rather than here: the as-stated Predictor needs it too, and it should not have to import a
reporting module to know what its own call means.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from epl.pundits.dataset import CALL_COLUMNS, FIXTURE_KEY, outcomes_of
from epl.windows import season_label

#: Canonical column order for one graded call.
GRADE_COLUMNS: tuple[str, ...] = (
    *CALL_COLUMNS,
    "home_goals",
    "away_goals",
    "predicted_outcome",
    "outcome",
    "exact_score",
    "correct_outcome",
)

#: Canonical column order for the report over them.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "pundit",
    "season",
    "season_label",
    "calls",
    "exact_scores",
    "exact_rate",
    "correct_outcomes",
    "outcome_rate",
)


def grade(calls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Every call that has a played Fixture behind it, graded both ways.

    A call on a Fixture that has not been played yet grades to nothing rather than to a miss —
    the current Season always has some, and marking them wrong would be marking the calendar.
    """
    played = matches[[*FIXTURE_KEY, "division", "home_goals", "away_goals", "outcome"]]
    joined = calls.merge(
        played,
        on=[*FIXTURE_KEY, "division"],
        how="inner",
        validate="one_to_one",
    )
    joined = joined.loc[joined["outcome"].notna()].reset_index(drop=True)

    predicted = outcomes_of(joined["pred_home_goals"], joined["pred_away_goals"])
    graded = joined.assign(
        predicted_outcome=predicted,
        exact_score=(joined["pred_home_goals"] == joined["home_goals"])
        & (joined["pred_away_goals"] == joined["away_goals"]),
        correct_outcome=predicted == joined["outcome"].astype("string"),
    )
    return graded[list(GRADE_COLUMNS)]


def summary(
    graded: pd.DataFrame, by: Sequence[str] = ("pundit", "season")
) -> pd.DataFrame:
    """How often each reading was right, grouped as asked.

    ``by=("pundit",)`` gives the career line; the default gives a Season at a time, which is where
    a pundit's good and bad years are visible rather than averaged away.
    """
    grouped = graded.groupby(list(by), sort=True, dropna=False)
    report = grouped.agg(
        calls=("exact_score", "size"),
        exact_scores=("exact_score", "sum"),
        correct_outcomes=("correct_outcome", "sum"),
    ).reset_index()
    report["exact_rate"] = report["exact_scores"] / report["calls"]
    report["outcome_rate"] = report["correct_outcomes"] / report["calls"]
    if "season" in report.columns:
        report["season_label"] = report["season"].map(lambda year: season_label(int(year)))
    return report[[name for name in SUMMARY_COLUMNS if name in report.columns]]
