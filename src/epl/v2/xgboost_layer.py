"""A learned Outcome model, deferred out of v1 (decision 12).

**What it would do.** Emit Predictions like any other Predictor — three probabilities per Fixture,
stamped with an As-Of Instant — from a gradient-boosted classifier over engineered features rather
than from a rating or a goals process. It would register through :mod:`epl.predictors` and appear
on the scoreboard beside Elo, Dixon-Coles and the Market Line, scored on the same Fixtures by the
same metrics. Nothing downstream would need to change to accommodate it, which is the point of the
contract having been written first.

**Why it is not in v1.** Three reasons, in order of how much they matter.

The first is that the target was met without it. The README asked for ≤0.1986 RPS and Dixon-Coles
returned **0.19752** over 7,980 Fixtures — **0.0039** short of the Market Line's 0.19362, where Elo
was 0.0058 short. So an ML layer would not be answering "can this project forecast the Premier
League"; it would be answering "can a learned model close the last four thousandths of RPS between a
goals model and the market", which is a different and much narrower question, and one that deserves
its own justification rather than inheriting v1's.

The second is leakage. Every Predictor here is handed :class:`epl.predictors.Evidence` — the corpus
already cut at its As-Of Instant — so the leak-free property is structural and cheap. A feature
matrix breaks that cheapness: each feature is a fresh opportunity to compute something over a window
that reaches past the cut, and a leak of that kind does not raise, it just returns a better number.
Every feature below would need its own As-Of-respecting builder, and the audit that makes
``inputs_seen`` and ``latest_input`` meaningful would need to reach into the feature layer too.

The third is that ADR 0008 makes tuning expensive on purpose. Hyperparameters may be fitted only in
the Burn-In Window — 2000/01-2004/05, 1,520 scored Fixtures, and no Market Line to compare against
in it. A boosted model has considerably more knobs than Elo's three, and this is the entire budget
for setting them. Tuning against anything later would be the exact failure the Windows exist to
prevent, and it would still report a perfectly respectable number.

**What would make it worth building.** Not "we have not tried ML yet". A concrete claim that some
signal in the list below is real, is not already inside a Dixon-Coles strength, and is worth more
than the leak surface it opens. Rest days is the most plausible; the rest are guesses until
measured, and measuring them one at a time against the existing models is cheaper than the layer.
"""

from __future__ import annotations

#: The features that would justify the layer, and roughly why each is thought to carry signal.
#:
#: Deliberately not a wish list. Elo and Dixon-Coles already carry Club strength, home advantage
#: and — through the time decay — recent form, so anything here that merely restates strength adds
#: leak surface and no information. What is listed is what the two current models genuinely cannot
#: see.
FEATURES: tuple[str, ...] = (
    "rest days since each Club's previous match — the models see kickoff order but not fatigue",
    "Fixture congestion: matches played in the preceding fortnight, including the cup and European"
    " ties the corpus does not currently ingest",
    "promotion or relegation status this Season — a Club's first months in a new tier",
    "the Dixon-Coles attack and defence strengths themselves, as inputs rather than as a rival",
    "Supremacy from the ratings, so the learned model starts where the fitted one finished",
    "days into the Season — the corpus can measure whether early rounds behave differently",
    "distance travelled by the away Club, which is constant per pairing and cheap to precompute",
)

#: What would have to exist before this is a build rather than an experiment.
WHAT_IT_NEEDS: tuple[str, ...] = (
    "a feature builder that takes an As-Of Instant and can be audited the way Evidence is — every"
    " feature computed from data strictly before the cut, with a receipt",
    "cup and European Fixtures in the corpus, for the congestion features; Football-Data's E0-E3"
    " files carry league matches only",
    "a hyperparameter budget that fits inside the Burn-In Window under ADR 0008, chosen and frozen"
    " before a single Evaluation Window Fixture is scored",
    "a measured reason to expect signal beyond the two fitted models, rather than the absence of a"
    " measured reason not to",
    "xgboost added to environment.yml from conda-forge (ADR 0009 anticipated this dependency)",
)

__all__ = ["FEATURES", "WHAT_IT_NEEDS"]
