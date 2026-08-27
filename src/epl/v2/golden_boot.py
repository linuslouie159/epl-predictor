"""A per-player Season goals model and the top-scorer market, deferred out of v1 (decision 12).

**What it would do.** Produce, for each player, a distribution over how many league goals they
finish the Season with, and from those a probability of winning the Golden Boot — the same shape of
answer a Season Projection gives for the title, one level down. Simulating the remaining Fixtures
already happens (:mod:`epl.simulate.projection`); this would sample goals *within* each simulated
match and attribute them, so a player's top-scorer probability and their Club's title probability
would come out of the same ten thousand simulated Seasons and be mutually consistent.

The shape would follow the projection's: fit at Season Projection checkpoints only, walk the draws,
report each player's distribution with its own recorded seed. Dixon-Coles already gives a match's
expected goals for each Club; the missing half is dividing those goals among the eleven players on
the pitch, and knowing who is on it.

**Why it is not in v1.** Not because it is hard — because the corpus cannot express it. This
project's whole data foundation is Football-Data's E0-E3 match files, which carry Clubs, goals,
odds and match statistics, and **no player at all**. There is no player column to model, no
appearance to condition on, and no squad. A Golden Boot model is not a further step along the
existing pipeline; it is a second ingest, a second canonical entity with its own Aliases, and a
second class of leakage question, all before the first probability.

That second entity is the real cost. :mod:`epl.clubs` exists because "Man United", "Manchester Utd"
and "Man Utd" are one Club, and the table that says so is generated, checked against the cache, and
fails loudly when a spelling appears that it does not cover (decision 5). Players need exactly that
and are far worse: there are two orders of magnitude more of them, they move between Clubs
mid-Season, they share names, and the sources spell them inconsistently and sometimes differently
across a single Season. Nothing about the Club machinery generalises for free.

**What it would be worth.** More than the top-scorer market itself, which is small. A player-level
goals model is the natural place to put team news — the thing the Ceiling Line knows and the model
does not, and the stated reason closing odds are a labelled reference bound rather than the headline
opponent (ADR 0001). Knowing a Club's leading scorer is out is how a forecaster would close some of
the **0.0013 RPS** the Ceiling Line beats the Market Line by on the 2,660 Fixtures the two share.

That is the number to quote and the subtraction of the two headline figures is not: the Ceiling Line
scores 0.1968 against the Market Line's 0.1936 and is nonetheless the better of them, because they
are measured over different Fixtures (CLAUDE.md; ADR 0001). Taking the difference would size this
feature off a figure that points the wrong way.

**The As-Of rule applies unchanged and is harder here.** A lineup is published about an hour before
kickoff, which is *after* the As-Of Instant of the round it belongs to (ADR 0002). So a Predictor
built on this may use squad availability known at the cut — a long-term injury, a transfer, a
suspension already served — and may not use the teamsheet. That distinction has to be enforced by
the ingest, because by the time it reaches a model both look like a column.
"""

from __future__ import annotations

#: What would have to exist before this is a build rather than a wish.
#:
#: The first two are the whole of the cost. Everything after them is ordinary work that the existing
#: simulation machinery mostly already does.
WHAT_IT_NEEDS: tuple[str, ...] = (
    "a player-level source: appearances, minutes and goals per match. Understat or FBref through"
    " soccerdata, which decision 5 already reserves for exactly this and nothing else",
    "a canonical Player table with per-source Aliases, generated and checked the way epl.clubs is —"
    " including a rule for two players sharing a name and for one changing Club mid-Season",
    "squad membership over time, so a simulated Fixture knows who could plausibly appear; a"
    " transfer the corpus cannot see would otherwise keep scoring for the wrong Club",
    "an availability feed whose facts are timestamped, so injury and suspension news can be cut at"
    " an As-Of Instant and a published teamsheet can be excluded by construction",
    "a way to divide a Club's simulated goals among its players — a shares model fitted on minutes"
    " and finishing rate, itself needing a Burn-In-only tuning budget under ADR 0008",
    "a decision about what counts: the Golden Boot is league goals only, so cup goals in a shared"
    " source have to be excluded rather than silently summed",
)

__all__ = ["WHAT_IT_NEEDS"]
