# EPL Predictor

A self-taught English Premier League forecasting system. It produces calibrated probabilities for
match outcomes, projects the final league table by simulation, and scores itself against the betting
market and against public pundits on identical metrics.

## Language

### Time and scope

**Season**:
One league campaign, identified by its start year (`2019` means 2019/20). Seasons before 1995/96 had
22 clubs and 462 fixtures; from 1995/96 onward, 20 clubs and 380 fixtures.

**Burn-In Window**:
Seasons 2000/01–2004/05. Ratings warm up and every hyperparameter is chosen here. Nothing in this
window is ever scored, which is what keeps the Evaluation Window free of contamination.
_Avoid_: training set, warm-up period

**Evaluation Window**:
Seasons 2005/06–2025/26 — the span over which Predictors are scored against each other. Begins where
the Market Line first exists, so every scored Fixture has a market to be compared against. Closed
Seasons only, so the numbers over it mean the same thing this week as last.
_Avoid_: test set, holdout

**Live Season**:
The Season being played. Ingested, so a Predictor can see it and a Sealed Prediction can be joined to
a result — and in neither Window: never fitted on, never backfilled, and scored on its own board.
The last Season ingested is the Live Season, which is what makes moving it one deliberate act.
_Avoid_: current season, this season, in-season

**As-Of Instant**:
The moment a Prediction was made. Everything the Predictor knew is, by definition, data timestamped
strictly before this instant. The central guard against leakage. Derived as the most recent Tuesday
or Friday preceding kickoff, mirroring the market's own odds-sampling convention.
_Avoid_: cutoff, snapshot date

**Prediction Round**:
All Fixtures sharing an As-Of Instant, predicted together as one batch. Replaces the notion of a
"matchweek", which does not exist in the source data and does not cleanly slice time.
_Avoid_: gameweek, matchweek, round

### Clubs and fixtures

**Club**:
A team, identified by a canonical slug (`man_united`) stable across Seasons, tiers and sources.
Rated across all four English tiers, so a Club promoted into the Premier League arrives with a
rating it earned rather than a guess.
_Avoid_: team, side

**Alias**:
One source's spelling of a Club (`Man United`, `Manchester Utd`, `Wolverhampton Wanderers`). Aliases
are data, held in one authoritative table; they never appear in model code.

**Fixture**:
A scheduled match between a home Club and an away Club in a Season. Exists before it is played and
carries no result until it is.
_Avoid_: game, match (reserve "match" for a Fixture that has been played)

**Outcome**:
The result of a played Fixture from the home Club's perspective: Home, Draw, or Away. Ordinal — Draw
sits between Home and Away, which is why RPS is the primary metric.
_Avoid_: result, W/D/L, 1X2

**Scoreline**:
An exact goals pair, e.g. 2-1. A Scoreline implies an Outcome; an Outcome does not imply a Scoreline.

**Supremacy**:
How far apart two Clubs are judged to be, as the gap between their Home and Away probabilities.
Draw probability falls monotonically with Supremacy — measured at 32% for evenly matched Clubs and
13% for the widest mismatches.

### Predictions

**Prediction**:
A probability distribution over the three Outcomes for one Fixture, attributed to one Predictor and
stamped with an As-Of Instant. Never a bare label.

**Predictor**:
Anything that emits Predictions and can be scored: a model, the Market Line, a Pundit, or the Naive
Baseline. All Predictors are scored on the same metrics over the same Fixtures.

**Sealed Prediction**:
A Prediction written before its Fixture kicked off and never afterwards altered. Evidence of what was
actually forecast, not a claim about what the code would produce today. Only Sealed Predictions can
support a live track record.
_Avoid_: live prediction, logged prediction

**Backtest Prediction**:
A Prediction reproduced from history by re-running the pipeline. Regenerable and disposable — its
value is in the aggregate score, never in any individual row.

**Pre-Match Reading**:
A Prediction computed roughly an hour before one Fixture's kickoff, after its Prediction Round was
already sealed, from a corpus that by then holds the results of matches played earlier in the same
round. Recorded in its own store and never scored: the Sealed Prediction is what the track record
is made of. Exists so a message sent before a match quotes the best available forecast without that
forecast quietly replacing the one made before the round.
_Avoid_: live prediction, updated prediction, refreshed forecast

**Season Projection**:
A distribution over final league tables, produced by simulating every remaining Fixture many times.
Yields each Club's probability of the title, of European places, and of relegation.
_Avoid_: forecast table, predicted table

### Benchmarks

**Market Line**:
The primary market benchmark — vig-removed implied probabilities from the market-average *pre-match*
odds. Chosen because its information set matches the model's and the Pundits'. Scores ~0.194 RPS.
_Avoid_: the odds, bookmaker probability

**Ceiling Line**:
The vig-removed market-average *closing* odds, available from 2019/20. Reported as a reference upper
bound only, never as the headline opponent — it knows team news the model cannot.
_Avoid_: closing odds benchmark

**Naive Baseline**:
The floor Predictor: base-rate Outcome frequencies with no knowledge of which Clubs are playing.
Scores ~0.229 RPS. Any Predictor that fails to beat it has no value.

**Pundit**:
A named public forecaster whose published Scorelines are collected and scored as a Predictor. Scored
as-stated, treating the Scoreline as a claim of certainty.

**Calibrated Pundit**:
A Predictor derived from a Pundit — their Scoreline mapped onto the Outcome frequencies that such a
call has historically produced. A one-feature model, not a person, and always named as such.
