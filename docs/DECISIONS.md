# Decisions, evidence and open risks

Everything settled in the design session. Decisions with lasting architectural consequence also have
an ADR in [adr/](./adr/); the rest are recorded only here.

## Settled decisions

| # | Decision | ADR |
|---|---|---|
| 1 | Success = leak-free, well calibrated, beats Naive Baseline and Pundits, within market + 0.005. Does **not** need to beat the market. | — |
| 2 | Training from 2000/01 — 26 seasons, four tiers | — |
| 3 | Market Line = market-average **pre-match**, `BbAv*` spliced to `Avg*`; closing odds are a labelled Ceiling Line only | [0001](./adr/0001-pre-match-odds-as-market-benchmark.md) |
| 4 | Weekly Prediction Rounds; As-Of = most recent Tue/Fri before kickoff | [0002](./adr/0002-weekly-prediction-rounds.md) |
| 5 | Fetch Football-Data CSVs directly; soccerdata only for Understat/FBref; own canonical Club table with per-source Aliases, exported to `teamname_replacements.json` | — |
| 6 | Pundits scored via Calibrated Pundit (headline) with as-stated published beside it | [0003](./adr/0003-calibrated-pundit-predictor.md) |
| 7 | Pundit data: MyFootballFacts backfill + BBC for the live season | — |
| 8 | Build order changed — Pundits early (need no model), goals model before the Season Projection (needs goal difference) | — |
| 9 | Split ledger: regenerable `outputs/backtest/`, sealed `outputs/live/` | [0005](./adr/0005-split-prediction-ledger.md) |
| 10 | Elo runs across E0–E3 so promoted Clubs arrive with earned ratings | [0004](./adr/0004-rate-the-whole-pyramid.md) |
| 11 | Ordered logit for Outcomes, behind a shared isotonic calibration layer built now | [0006](./adr/0006-ordered-logit-with-shared-calibration.md) |
| 12 | v1 = stages 1–8. Deferred with written stubs: XGBoost/ML, Golden Boot, API-Football | — |
| 13 | 2026/27 gameweek 1 (21 Aug 2026) deliberately not chased; backtest power matters more than 10 live matches | — |
| 14 | Season Projections use a Bayesian posterior. Within-season strength drift is **not** modelled — measured at zero | [0007](./adr/0007-mle-for-matches-bayesian-for-projections.md) |
| 15 | Dixon-Coles: MLE at all 1,189 Prediction Rounds, Bayesian only where a Season Projection is produced | [0007](./adr/0007-mle-for-matches-bayesian-for-projections.md) |
| 16 | Miniforge + conda-forge; `environment.yml` is the source of truth; PyMC + nutpie | [0009](./adr/0009-conda-forge-toolchain.md) |
| 17 | Burn-In Window 2000/01–2004/05 for warm-up and tuning, then frozen; scoring from 2005/06 | [0008](./adr/0008-burn-in-prefix-frozen-hyperparameters.md) |

## Smaller calls

- **Vig removal**: implement normalisation, power and Shin behind one interface; **Shin is the default**. Empirically they differ by 0.0002 RPS, so the choice is near-immaterial for benchmarking — but power and Shin correct favourite-longshot bias, which matters if value betting is ever explored.
- **Monte Carlo**: 10,000 simulated seasons per projection, fixed deterministic seed recorded in the output.
- **Storage**: CSV everywhere. Every output is small (~60k rows at most), and CSV is git-diffable, human-readable and works as evidence. No Parquet in v1.
- **Tests**: pytest. The metrics module is unit-tested against hand-worked examples before any model uses it.
- **Calibration**: isotonic regression, fitted walk-forward on out-of-sample Predictions only; reliability diagrams use 10 bins.
- **Reporting**: every metric is reported both pre-calibration and post-calibration, so a large correction is a visible warning about the underlying model rather than a silent fix.
- **Tiebreakers**: the Season Projection implements the full chain — points, goal difference, goals scored, head-to-head points, head-to-head away goals. The neutral-ground play-off is treated as a coin flip.

Added at stage 1:

- **`data/processed/` is gitignored.** It is rebuilt from the raw cache by `python -m epl.ingest build` and is deterministic, so it is code output rather than evidence — the same reasoning ADR 0005 applies to `outputs/backtest/`.
- **A refreshed raw file is superseded, not overwritten.** Re-fetching a Season whose upstream bytes have changed moves the previous copy to `superseded/` first. Overwriting would destroy the only record of what the cache held when a Sealed Prediction was made, which is the failure ADR 0005 exists to prevent.
- **The Club table is generated, not hand-maintained.** `python -m epl.clubs.build` rebuilds `clubs.csv` and `aliases.csv` from a single spelling-to-Club mapping, and fails if the raw cache holds a spelling it does not cover or names a Club the cache never fielded. It also writes `teamname_replacements.json`.
- **ruff and mypy** are in `environment.yml` and configured in `pyproject.toml`, so style and typing are tooling concerns rather than review concerns.
- **`arviz` is pinned below 1.0.** arviz 1.x moved to xarray's DataTree and removed `InferenceData`, which pymc 5.x imports at load; solved unpinned, `import pymc` fails outright. `gxx` is likewise pinned in, because without it PyTensor silently falls back to a slow Python implementation — the exact failure ADR 0009 chose conda-forge to avoid.

## Measured facts

Derived from the source data during design. Recorded so they need not be re-derived — but re-verify
before relying on any of them.

**Re-verified at stage 1 (21 Aug 2026)** against the full ingested corpus — 52,672 matches over
26 Seasons and four tiers. `tests/ingest/test_raw_cache_integrity.py` re-derives these from the data
on every run rather than trusting the numbers below. Confirmed exactly: the 7,980-fixture Evaluation
Window; Naive Baseline H 45.56% / D 24.32% / A 30.11%; mean overround 1.05616; the era boundaries for
`BbAv*`/`Avg*` (2005/06) and `AvgC*` (2019/20); 380 Premier League fixtures in every Season.
Three corrections:

- Match stats are **99.98% populated, not 100%** — nine rows across the whole pyramid lack them
  (six in 2002/03, two in 2016/17, one in 2018/19).
- **Referee is only 18.7% populated in 2012/13**, and ~97% overall. No v1 model uses it.
- The lower tiers are 552 rows per Season **except 2019/20**, where COVID curtailed League One
  (400 rows) and League Two (440). The Premier League completed that Season.

**Re-verified at stage 2 (21 Aug 2026)** against the 9,880 Premier League fixtures of 2000/01–2025/26.
`tests/test_rounds.py` re-derives these on every run. One correction:

- **Prediction Rounds are 1,189, not 1,332.** The design recorded 1,332 rounds, 51.2 per season and
  7.42 fixtures per round; the anchor rule below yields **1,189 rounds, 45.7 per season and 8.31
  fixtures per round**. The corpus is not in doubt — 2024/25 spans 109 distinct kickoff dates,
  exactly as recorded. Every Tue/Fri anchoring variant was tried (on-or-before, strictly-before,
  and every subset of anchor weekdays) and none reproduces 1,332, so the original figure could not
  be sourced. The rule is stated as executable code in the spec and below, so the rule was kept and
  the count corrected. Round sizes run from 1 to 20 fixtures; seasons hold between 41 and 52 rounds.

### Football-Data era boundaries (English tiers)

| From | What appears |
|---|---|
| 1993/94 | Result only. 1993/94–1994/95 have 462 fixtures (22 clubs) |
| 2000/01 | Match stats: shots, shots on target, corners, fouls, cards, referee. 99.98% populated — see the re-verification note below |
| 2002/03 | Bet365 pre-match odds |
| 2005/06 | Market-average pre-match (`BbAvH/D/A`) |
| 2012/13 | Pinnacle closing (`PSC*`) |
| 2019/20 | `BbAv*` replaced by `Avg*`; market-average closing (`AvgC*`) appears |

There is **no xG anywhere** in Football-Data. Understat starts 2014/15; FBref advanced stats ~2017/18.

### Benchmark scores (2005/06–2025/26, 7,980 fixtures)

| Predictor | RPS |
|---|---|
| Naive Baseline (H 45.6% / D 24.3% / A 30.1%) | 0.2292 |
| Elo through an ordered logit | 0.19943 |
| Market Line, normalised | 0.19379 |
| Market Line, Shin | 0.19362 |
| Market Line, power | 0.19359 |

Mean overround on the market-average pre-match line: **1.0562** (5.62% margin).

**Re-verified at stage 4 (22 Aug 2026)**, walked end to end through the ledger rather than computed
in a notebook. `tests/benchmarks/test_market_over_the_corpus.py` re-derives all of it on every run
against a populated raw cache. Every figure above confirmed to the places it is stated in: 0.193789
normalised, 0.193622 Shin, 0.193587 power, a spread of 0.00020 RPS, and a mean overround of
1.05616. The Naive Baseline scores 0.22938 walk-forward, against the 0.2292 the whole-window rates
give — the difference is the leak it refuses.

The **Ceiling Line** measures for the first time here, and it needs reading carefully:

| Span | Fixtures | Market Line | Ceiling Line |
|---|---|---|---|
| 2005/06–2025/26 | 7,980 | 0.19362 | — (no closing book before 2019/20) |
| 2019/20–2025/26 | 2,660 | 0.19810 | **0.19676** |

Its headline 0.1968 is *worse* than the Market Line's 0.1936, which invites exactly the wrong
conclusion. The two numbers are measured over different Fixtures: on the 2,660 they share, the
closing book beats the pre-match one by **0.0013 RPS**, which is what a few hours of team news is
worth. That is small beside the 0.036 RPS the market takes out of the Naive Baseline — evidence
for ADR 0001's claim that benchmarking against closing odds would say little about model quality.
The caveat travels with the Ceiling Line onto the scoreboard as its `note`, because the bare
number reads as the opposite of what it means.

**Four books in the corpus have an overround below one** — 2025/26 League One *closing* averages,
as low as 0.955, which is not a price anyone was offered. No Premier League Fixture and no
pre-match book anywhere is affected, so no scored line loses a Fixture to it. They are excluded by
`epl.benchmarks.vig.is_book` rather than special-cased: a book that pays more than it takes is not
a book, and the per-tier overround report walks the whole pyramid.

### Other measurements

- **Prediction Rounds**: **1,189** across 2000/01–2025/26 (mean 45.7 per season, 8.31 fixtures per round). Corrected at stage 2 — the design recorded 1,332 / 51.2 / 7.42, which the anchor rule below does not produce over this corpus. See the note under *Measured facts*.
- **Draw rate vs Supremacy**: 32.3% when evenly matched, falling monotonically to 13.4% at the widest mismatch. A 2.4x range. This is the *observed* rate bucketed by **market** Supremacy; re-measured at stage 5 over deciles it is 32.0% → 13.7%. Elo's own curve is different and is recorded under *Measured at stage 5* — bucketing by a noisier Supremacy makes the most-even decile a less even set of Fixtures.
- **Promoted clubs** (75 club-seasons): mean 36.9 points vs a league average of 52.0; lose 51.5% of matches; 35 of 75 finished on 35 points or fewer. The penalty is unstable — 44.0 points for the 2010/11 intake, 19.7 for 2024/25.
- **Within-season drift** (520 club-seasons): first-half vs second-half PPG correlation 0.686; observed SD of the difference 0.390 against 0.400 expected from sampling noise alone. Implied true drift: **zero**.
- **Points ties**: 24 of 26 seasons had at least one, averaging 3.3 tied pairs. This is why the Season Projection needs goals, not just Outcomes.
- **Home advantage is declining**: ~47% home wins in the 2000s, ~43% since 2020, 40.8% in 2024/25. 2020/21 is an outlier — 37.9% home vs 40.3% away, the empty-stadium season.

## Source URLs

Football-Data season CSVs: `https://www.football-data.co.uk/mmz4281/{SSEE}/{DIV}.csv`
where `{SSEE}` is e.g. `2526` and `{DIV}` is `E0`, `E1`, `E2` or `E3`.

Forward fixtures with the Market Line: `https://www.football-data.co.uk/fixtures.csv` (rolling ~1 week).

Pundit backfill, all under `https://www.myfootballfacts.com/premier-league/all-time-premier-league/predictions/`:

| Season | Pundit | Path |
|---|---|---|
| 2025/26 | Sutton | `chris-sutton-predictions-for-premier-league-2025-26/` |
| 2024/25 | Sutton | `chris-sutton-predictions-for-premier-league-2024-25/` |
| 2023/24 | Sutton | `chris-sutton-predictions-for-premier-league-2023-24/` |
| 2022/23 | Sutton | `chris-sutton-predictions-for-premier-league-2022-23/` |
| 2021/22 | Lawrenson | `lawros-predictions-premier-league-2021-22/` |
| 2020/21 | Lawrenson | `lawros-predictions-premier-league-2020-21/` |
| 2019/20 | Lawrenson | `lawros-predictions-premier-league-2019-20/` |
| 2018/19 | Lawrenson | `mark-lawrensons-predictions-2018-19/` |
| 2017/18 | Lawrenson | `mark-lawrensons-predictions-2017-18/` |

Store only the facts — fixture, predicted scoreline, predictor, date — never the prose, and attribute
BBC as the origin.

## The As-Of Instant rule

```
anchor(kickoff_date):
    wd = kickoff_date.weekday()          # Mon=0 .. Sun=6
    if wd >= 4:  return date - (wd - 4)  # Fri/Sat/Sun -> that Friday
    if wd == 0:  return date - 3         # Mon         -> previous Friday
    if wd == 1:  return date             # Tue         -> that Tuesday
    return date - (wd - 1)               # Wed/Thu     -> that Tuesday
```

Matches Football-Data's stated collection convention: pre-match odds sampled Friday afternoon for
weekend fixtures, Tuesday afternoon for midweek ones.

The **As-Of Instant** is midnight at the start of that anchor day, not the afternoon. No football is
played between the two, so the information sets are identical for every input this project holds,
and the earlier instant can only ever withhold data, never admit it. It is also the only choice that
is checkable: kickoff times are absent before 2019/20, and the earliest kickoff recorded on any
Tuesday or Friday is 12:30.

Football-Data records no kickoff time before 2019/20, so only 2,660 of the 9,880 Premier League
fixtures carry one. That does not weaken the guarantee for most of the rest: a fixture kicking off
on a *later* day than its anchor is strictly after the instant whatever time it started.

The gap is the **437 fixtures — 4.4%, all Tue/Fri kickoffs between 2000/01 and 2018/19** — that kick
off on their own anchor day with no recorded time. For those, the assertion is that the As-Of Instant
lands at or before the start of the kickoff day, which, since no fixture in the corpus kicks off at
midnight, still gives strictness. `tests/test_rounds.py` pins that count at exactly 437 so the
carve-out cannot quietly grow, and asserts the strict form over the other 9,443.

## The Prediction ledger

Added at stage 3 (issue #7), which put one Predictor through the whole pipeline and out the other
side scored.

- **A Predictor is handed Evidence, not an As-Of Instant.** The contract could have been "here is
  the instant, go and read what you need". It is not, because that would put the project's one rule
  in as many places as there are Predictors. `epl.predictors.Evidence` is the corpus already cut at
  the instant, and it *records what it handed over*, so `inputs_seen` and `latest_input` on every
  stored row are a receipt rather than an assertion.
- **A Predictor is handed a Fixture, not the match row it was drawn from.** `Evidence` guards the
  corpus; this guards the *other* argument. The corpus is a table of played matches, so the row a
  Fixture comes from also carries the Outcome, the goals and the match statistics — and a Predictor
  handed that row could score perfectly with every stored row still auditing clean, because nothing
  about it would be inconsistent. `schema.VISIBLE_FIXTURE_COLUMNS` is an allow-list, not a
  deny-list: a column nobody has thought about is excluded rather than included. The pre-match odds
  are on it (sampled at the As-Of Instant itself, ADR 0001); the closing odds are not.
- **The seal check reads git for what is *missing*, too.** A check that walks the working tree
  cannot report its own deletions — once a sealed file is gone there is nothing left to notice, and
  deleting a round is the most destructive rewrite available. Git still holds the record, so
  `live.seal_violations` asks git which rounds were ever sealed and complains about any that are no
  longer on disk. Append-only means nothing leaves either.
- **A match row is timestamped at its kickoff**, not at full time. Full time is when its result
  became knowable, so kickoff is the loose direction in principle. Every As-Of Instant is a
  midnight and the latest kickoff anywhere in the corpus is **20:15**, so no match is in progress
  when one falls, and the two can never land on opposite sides of an instant.
  `tests/ledger/test_the_corpus.py` re-derives that from the data.
- **The ledger stores no Outcome.** A Prediction is sealed before kickoff, so it cannot know one;
  the scoreboard joins to the match table on the Club pairing within a Season and tier. Not on the
  date, so a postponed Fixture is still the Fixture that was predicted.
- **The audit is two-tier on kickoff, exactly as the As-Of rule is.** A Prediction whose kickoff
  time is recorded must be made strictly before it; one whose kickoff time is not recorded sits at
  midnight on its own day, and 313 of the Evaluation Window's Fixtures are played on the very
  Tuesday or Friday they anchor to. Equality is allowed there and nowhere else.
- **`outputs/scoreboard.csv` sits beside the two stores, not inside one.** It summarises both, so
  it belongs to neither — and a store must never pick up a report while globbing its own files.
  Gitignored, being derived and regenerable.

### Measured at stage 3 (21 Aug 2026)

Re-derived on every run by `tests/ledger/test_the_corpus.py`, which skips when `data/raw/` is absent.

- The Naive Baseline scores **0.22938 RPS** over the Evaluation Window — 7,980 Fixtures across 952
  Prediction Rounds — with Brier 0.6430, log loss 1.0642 and accuracy 0.4556.
- The published **0.2292** is the *whole-window* figure, computed from rates that already know how
  the window turned out. Estimating them walk-forward costs **0.0002 RPS**. That difference is the
  leak being refused, not a bug — and it is in the expected direction, since a floor that knew the
  future would be a slightly better floor and would make every Predictor measured against it look
  slightly worse than it is.
- The first scored Prediction sees exactly **1,900** Premier League matches: the five Burn-In
  Seasons, warmed up and never scored.
- The Naive Baseline's top pick is a Home win at every one of the 952 rounds, so its accuracy is
  just the Home-win rate. Which is why accuracy is never the headline.

## The market benchmarks

Added at stage 4 (issue #8), which put the opponent on the board. Three of these are extensions to
the Predictor contract rather than facts about the market, and all three exist because the Ceiling
Line needs something the ledger deliberately does not give every Predictor.

- **A Predictor may declare which Fixtures it `covers`.** The Ceiling Line's closing book begins in 2019/20 and a Pundit publishes only in the Seasons they worked, so both would otherwise have to invent Predictions for Fixtures they know nothing about. The walk asks before it assigns rounds, so a round nobody covers never becomes an empty round. A Predictor that declares nothing covers everything, which is every other Predictor.
- **A Predictor may claim extra Fixture columns by naming them in `also_sees`.** Only what `epl.ledger.schema.PRIVILEGED_FIXTURE_COLUMNS` permits — the three closing-odds columns — and only the Ceiling Line claims any. Appending them to the ordinary allow-list would have handed team news from after the As-Of Instant to every Predictor in the project, which is the leak the allow-list exists to prevent (ADR 0001). Making the claim in the Predictor's own source is what keeps the exception visible where it is used rather than buried where it is granted.
- **A Predictor may carry a `note`, and the scoreboard prints it.** The Ceiling Line needs one: its RPS is measured over a shorter, harder span than everything else on the board, so the bare number reads as the opposite of what it means. Generic rather than a branch — the scoreboard looks a note up by name and has no idea which Predictor a row belongs to.
- **The overround report is a command, not a column.** `python -m epl.benchmarks overround` writes `outputs/overround.csv` and prints the per-Season margin. It belongs outside the ledger's row schema because it is a fact about a book rather than about a Prediction, and adding a column that only two Predictors could fill would be the per-Predictor branch the ledger refuses.

The vig removal itself is `epl.benchmarks.vig`: `normalise`, `power` and `shin` behind
`remove(book, method=...)`, solved by fixed-step bisection rather than to a tolerance so that a
rebuilt backtest file is byte-identical to the last one (ADR 0005).

## Elo through an ordered logit

Added at stage 5 (issue #9), the first real model. `epl.models.elo` is the rating pool,
`epl.models.ordered_logit` the mapping from one edge to three probabilities, `epl.models.burn_in`
the only place either is fitted.

- **The fitted draw band is symmetric, and the fit is not allowed to move its centre.** An edge already carries what playing at home is worth, so an edge of zero is a genuinely even contest — and at an even contest Home and Away must be equally likely, or the model is claiming a home advantage it has already counted. Fitted free, the band's centre becomes a second home-advantage parameter pointing the other way, and the two are then only weakly told apart. Measured on the Burn-In Window two ways: held at one arbitrary pair of Elo constants (K 30, home advantage 100) the free band scores 0.20552 against the centred band's 0.20583, a gap of 0.0003 RPS; searched end to end, where K and the home advantage move too, the free band reaches 0.20550 and the centred one **0.20554** — **0.00004 RPS**, for a home-advantage constant that goes from 80 rating points to **155** with the band shifted back to cancel most of it. One parameter, in the place where it also moves the ratings, is the honest version of the same model.
- **2000/01 warms the ratings up and is not fitted on.** Every Club starts the corpus at the same conventional 1500, so a Season in which the model knows nothing about anybody would mostly teach the fit to learn fast. That is open risk 4 surfacing *inside* the Burn-In Window. The choice is made on that reasoning alone and never compared against an Evaluation Window score, which would be the leak ADR 0008 exists to stop.
- **The fit refuses a winner sitting on the wall of its own grid.** The home-advantage grid originally stopped at 140 rating points, the answer sat on it, and the refinement passes then re-centred on the boundary and reported 155 as though it were fitted. A search that stops at the edge of what it was allowed to consider has not found an optimum. Two candidates have no interior, so the check has nothing to say there — which is what lets a test afford a grid at all.
- **RPS is the objective, not log loss.** RPS is what the scoreboard reports and what the project is judged on (CLAUDE.md), and it is strictly proper for ordinal Outcomes, so minimising it is not a shortcut. Log loss is reported beside it and agrees.
- **Elo rebuilds its rating pool at every Prediction Round rather than folding one forward.** That is 53 seconds for a full backfill instead of about one. It is bought deliberately: a pool carried between calls has to judge whether the Evidence it was just handed extends the one it folded last, and getting that wrong is the one failure this project cannot see — the ratings would be built from the wrong matches while every stored row still audited clean, because `inputs_seen` and `latest_input` are a receipt from `Evidence` and not from the model.
- **No regression to the mean between Seasons, and no cross-tier offset.** Issue #9 names three hyperparameters and a summer decay would be a fourth that had to be fitted like the others. The offset is unnecessary by construction: Elo is zero-sum, so a tier's level is expressed by the Clubs promoted into and relegated out of it.

### Measured at stage 5 (22 Aug 2026)

Re-derived on every run by `tests/models/test_elo_over_the_corpus.py`, which skips when `data/raw/`
is absent.

Fitted on 2001/02–2004/05's 1,520 Premier League Fixtures, warmed from 2000/01, over 10,180
matches across all four tiers: **K 28.5, home advantage 80 rating points, logit scale 186.92,
cutpoints ±0.62318**. It scores 0.20554 RPS there against the base rate's 0.22383.

| Predictor | Fixtures | RPS | Brier | Log loss | Accuracy |
|---|---|---|---|---|---|
| Market Line | 7,980 | 0.19362 | 0.5684 | 0.9582 | 0.5471 |
| Ceiling Line | 2,660 | 0.19676 | 0.5717 | 0.9639 | 0.5498 |
| **Elo** | **7,980** | **0.19943** | 0.5810 | 0.9771 | 0.5380 |
| Naive Baseline | 7,980 | 0.22938 | 0.6430 | 1.0642 | 0.4556 |

- Elo takes **0.0300** of the **0.0358** RPS the market takes out of the floor — 84% of the
  available edge, from ratings and nothing else. It sits 0.0058 behind the market and 0.0008 above
  the README's ≤0.1986 target, which is what issues #10 and #13 are for. An Elo that *beat* the
  market would be evidence of a leak, and the corpus test says so.
- **The draw curve.** Across ten Supremacy deciles Elo's predicted draw rate falls **30.2% → 14.5%**,
  monotonically at every step, and the observed rate over its own buckets falls **27.6% → 13.8%**.
  The design's 32.3% → 13.4% is the *observed* curve bucketed by **market** Supremacy, and it still
  measures at 32.0% → 13.7%. Elo's most-even decile is a genuinely less even set of Fixtures than
  the market's, which is why its top end is lower — the shape is reproduced, the endpoints are the
  model's own.
- **Elo quotes draws a little too often in every bucket** — the market *under*-quotes them at the
  even end, 28.6% against 32.0% observed. Both are what issue #10's shared calibration layer was
  for; what it did to them when it arrived is "Measured at stage 6" below. Elo's is the
  frozen-hyperparameter drift ADR 0008 accepts by name: the band was fitted on
  2001/02–2004/05 and the draw rate has moved since. Pinned by a test so it cannot be forgotten.
- **Promoted Clubs arrive with ratings that differ**, in all 21 scored Seasons. For 2005/06:
  Sunderland 1679.8 from 206 matches, Wigan 1623.9 from 230, West Ham 1565.6 from 206. Not one is
  at the 1500 starting value.
- **By the first scored Prediction Round the thinnest Premier League rating rests on 190 matches**
  — five Burn-In Seasons of football. This is what closes open risk 4.

## The shared calibration layer

Added at stage 6 (issue #10). `epl.calibration` is the isotonic step and the walk that fits it;
`epl.ledger.scoreboard` is where it is applied and both sides of it are published.

- **It lives at the top level, not in `epl.models`.** The README's layout and three docstrings said
  `epl.models` before it existed. It is not a model — it takes Predictions and returns Predictions,
  and it wraps the Market Line and the Pundits exactly as it wraps Elo — and putting it there would
  have made `epl.models` and `epl.ledger` import each other, since the ledger holds the Predictions
  it walks over. It sits beside `predictors.py`, which is the contract it wraps. It cannot live in
  `epl.metrics` either: no function in that package may take or produce a Prediction, which is what
  makes the three-way scoreboard structurally incapable of being apples-to-oranges.
- **Nothing is stored.** A calibrated Prediction is a function of a stored Prediction *and of
  Outcomes that happened after it*, so it is derived at scoring time. No row in either store knows
  an Outcome (ADR 0005), and a store whose rows were built from Outcomes would lose the property
  that makes a leaked Prediction distinguishable from a recorded one.
- **One isotonic map per Outcome, fitted one-versus-rest, then renormalised.** Per Outcome rather
  than pooled across all three, because the defect it was built for is Outcome-specific: a pooled
  map cannot lower Elo's draw quotes without lowering every Home and Away quote in the same
  probability band. The reliability diagram pools because it asks one pooled question; the
  correction does not, because it answers three.
- **The training cut is `kickoff < as_of`, strictly.** An Outcome is not knowable until its Fixture
  has been played, so this is the same cut `Evidence` applies to the corpus and it is applied for
  the same reason. This is the one thing in the project fitted on results, so it is the one place a
  leak could enter with every stored row still auditing clean.
- **380 Predictions are needed before a map is fitted at all**, one Season of Premier League
  Fixtures, stated rather than tuned. An isotonic map has as many knots as the Predictor has
  distinct quotes, so on a smaller sample it can rest one match on a knot and hand back the Outcome
  that happened as though it were a probability. The cost is visible rather than hidden: the first
  380 Predictions of each track record are uncorrected, and the scoreboard reports `corrected`.
- **The headline numbers stay pre-calibration**, for the reason measured below.

### Measured at stage 6 (23 Aug 2026)

Re-derived on every run by `tests/test_calibration_over_the_corpus.py`, which skips when `data/raw/`
is absent.

**The layer makes every Predictor worse.** Walk-forward over the Evaluation Window:

| Predictor | Fixtures | corrected | RPS | calibrated RPS | ten-bin error | calibrated | mass moved |
|---|---|---|---|---|---|---|---|
| Market Line | 7,980 | 7,600 | 0.19362 | 0.19450 | 0.0061 | 0.0124 | 0.034 |
| Ceiling Line | 2,660 | 2,280 | 0.19676 | 0.19800 | 0.0060 | 0.0084 | 0.033 |
| Elo | 7,980 | 7,600 | 0.19943 | 0.20037 | 0.0055 | 0.0097 | 0.031 |
| Naive Baseline | 7,980 | 7,600 | 0.22938 | 0.23087 | 0.0061 | 0.0161 | 0.046 |

Two effects add up to that, and the numbers below separate them rather than reporting the sum as one
fact about the corpus.

- **Most of it is knot resolution.** An isotonic map gets a knot per distinct quote, and market odds
  and Elo edges are nearly continuous — **7,909 distinct Home quotes across 7,980 Fixtures** — so
  most knots rest on a single Fixture and the map fits noise. Cutting the knots at ten equal-width
  probability bands instead takes Elo from 0.20037 to **0.19968** and the Market Line from 0.19450
  to **0.19404**, recovering 73% and 52% of the loss. That variant is not shipped: the band count
  would be a hyperparameter, and ADR 0008 wants those fitted inside a Burn-In Window that holds no
  stored Prediction. It is re-derived by the corpus test from `Curve` and `Isotonic`, the layer's
  own public API, so the alternative is measured rather than asserted.
- **The rest is the corpus.** Even coarse, both stay worse than raw. All four sit at a pooled
  ten-bin calibration error of about 0.006 before the layer touches them, so there is little real
  miscalibration left for a monotone map to find.
- **At full resolution it leaves each Predictor less calibrated than it found it.** Every
  `calibrated` error above is larger than the one beside it. That is what rules out "the correction
  is right and RPS is judging it unfairly" — the shipped layer raises the very number it exists to
  lower.
- **This is overfitting, not a wiring fault.** A clean split half of the Market Line's 7,980
  Fixtures, with none of the walk's machinery involved: fitted on the older half by kickoff, the map
  improves that half by **0.0017** RPS and costs the later half **0.0005**.
- **The correction points the right way; the noise around it is bigger.** Elo's draw quote at even
  Supremacy moves **30.2% → 29.3%** against 27.6% observed, which is exactly the defect issue #9
  handed this layer. At the widest Supremacy it overshoots: 14.5% → 13.3% against 13.8% observed.
- **So the double reporting is what earned its place.** Publishing only the post-calibration column
  would have applied a silent ~0.001 RPS tax to every Predictor on the board, and nobody would have
  had a number to notice it with. ADR 0006 wrote that rule expecting the warning to point at a
  model; here it points at the layer.
- **Re-measure at issue #11.** A Pundit scored as-stated publishes `[1, 0, 0]` (ADR 0003), which is
  the most miscalibrated Prediction there is. That is the first Predictor this layer has something
  real to correct, and none of the numbers above should be assumed to survive it.

## The Pundit backfill

Added at stage 7 (issue #11). `epl.pundits.myfootballfacts` fetches and parses the nine archive
pages, `epl.pundits.dataset` reconciles them with the corpus and freezes `predictions.csv` beside
the code, `epl.pundits.grading` marks each call two ways, and `epl.pundits.predictor` registers the
Pundits as Predictors scored as-stated.

- **Two named Pundits, not one pundit slot.** A Pundit is "a *named* public forecaster"
  (CONTEXT.md), the spec asks for "each Pundit's best and worst calls" (user story 34), and
  ADR 0003 spends its Consequences section on keeping a person distinguishable from a model built
  out of their calls. Mark Lawrenson worked 2017/18–2021/22 and Chris Sutton 2022/23–2025/26, so
  `lawrenson` and `sutton` are registered separately and each `covers` its own Seasons. A single
  averaged line would be a Predictor that is nobody, and neither of them could be held to it.
- **The dataset is committed, and it is provably what the build produces.** `predictions.csv` ships
  with the package exactly as the Club table does, so a fresh clone can score the Pundits without
  touching MyFootballFacts. "Frozen" is only worth something if the frozen file is the file the
  build makes, so the corpus test rebuilds it from the raw cache and compares bytes.
- **Only facts are stored** — Fixture, predicted Scoreline, Pundit, date. Not the prose, not the
  matchday heading, and **not the result**, which the page publishes beside every call. The result
  is read at build time to check the parse and then discarded: a stored Prediction that knew its
  own Outcome is the thing ADR 0005 exists to prevent, and it is what makes a leaked Prediction
  distinguishable from a recorded one.
- **The date comes from Football-Data, not from the page.** The matchday headings carry typos —
  2024/25's opening matchday is headed `16/08/25`, a year out — and a Fixture is identified by its
  Club pairing inside a Season anyway, which is what the ledger keys on. Every call is located
  against the corpus, and one the corpus has no Fixture for is refused rather than invented.
- **A Fixture listed twice keeps the call published for the date it was played.** Fifteen Fixtures
  across the nine Seasons were postponed or abandoned, re-listed later, and called twice — 2022/23's
  Leicester against Aston Villa is 1-2 on the original date and 2-2 on the rearranged one. The one
  that stands is the one whose listing is not marked `PP`, because that is the date the Fixture
  actually has and the date its As-Of Instant is derived from. The order the page lists them in
  decides nothing: 2020/21 puts the played listing first and 2022/23 puts it second.
- **Annotations are stripped; misspellings are Alias rows.** The pages hang notes on Club names — a
  trailing `*`, or `(14.02)` naming a rearranged date — and those are facts about the Fixture, so
  they never reach the Alias table. `Wolverhampton Wand` and `Brighton & Hove Alb` are the source
  genuinely misspelling a Club, and a spelling is exactly what that table holds. 38 spellings across
  32 Clubs, added to `epl.clubs.build` so the committed CSVs keep their provenance.
- **Twice the Scoreline landed inside a Club's name.** `Burnley v Brighton 1-2 Hove Albion` is the
  source writing `Burnley v Brighton & Hove Albion` and dropping the call where the `&` belongs.
  Read literally that is a Club called `Burnley v Brighton`; handled in the parser rather than as
  two Alias rows, because it is not a spelling of anything.
- **The as-stated score is published with three caveats attached to it.** Each Pundit's `note`
  carries all of them onto the scoreboard, because the scoreboard line is the artifact a reader
  actually receives: who published the calls (the BBC, archived by MyFootballFacts — issue #11 asks
  that the origin be attributed); that the number is measured over the Fixtures that Pundit called
  and so is **not comparable to a full-window RPS**, which is the Ceiling Line's lesson applied
  again (ADR 0001); and that a Scoreline read as `[1, 0, 0]` is a claim of certainty nobody made,
  which is what the number really measures (ADR 0003).
- **Log loss is not a meaningful number for these two.** A one-hot Prediction that is wrong has its
  probability floored at `epl.metrics.LOG_LOSS_FLOOR` before the log is taken, so the 16.9 and 17.5
  on the scoreboard are the miss rate times the floor and say nothing about the Pundit. The metrics
  module named this case when it was written; it is part of why RPS is the headline.

### Measured at stage 7 (23 Aug 2026)

Re-derived on every run by `tests/pundits/test_over_the_corpus.py`, which skips when `data/raw/` is
absent, and by `tests/pundits/test_the_frozen_dataset.py`, which does not need it.

**3,408 calls of a possible 3,420**, all nine pages parsed:

| Season | Pundit | Calls | Exact scores | Correct Outcomes |
|---|---|---|---|---|
| 2017/18 | Lawrenson | 378 | 48 (12.7%) | 193 (51.1%) |
| 2018/19 | Lawrenson | 380 | 42 (11.1%) | 204 (53.7%) |
| 2019/20 | Lawrenson | 379 | 35 (9.2%) | 186 (49.1%) |
| 2020/21 | Lawrenson | 380 | 45 (11.8%) | 189 (49.7%) |
| 2021/22 | Lawrenson | 379 | 39 (10.3%) | 194 (51.2%) |
| 2022/23 | Sutton | 372 | 32 (8.6%) | 174 (46.8%) |
| 2023/24 | Sutton | 380 | 28 (7.4%) | 204 (53.7%) |
| 2024/25 | Sutton | 380 | 47 (12.4%) | 193 (50.8%) |
| 2025/26 | Sutton | 380 | 31 (8.2%) | 173 (45.5%) |
| **Lawrenson** | | **1,896** | **209 (11.0%)** | **966 (50.9%)** |
| **Sutton** | | **1,512** | **138 (9.1%)** | **744 (49.2%)** |

Twelve Fixtures have no call — the archive never listed them, eight of them in 2022/23, the Season
of the Queen's death and the winter World Cup. They are named in the corpus test and each Pundit's
`covers` keeps them off the ledger, because a made-up Prediction that scores is worse than an
absent one.

**The as-stated scoreboard, and the same Fixtures for comparison.** Each Pundit is measured only
over the Fixtures they spoke to, so the other Predictors are cut to the same slate — the Ceiling
Line's lesson, applied again:

| Over | Fixtures | Market Line | Elo | Naive Baseline | Pundit |
|---|---|---|---|---|---|
| Lawrenson's, RPS | 1,896 | 0.1946 | 0.2019 | 0.2356 | **0.3341** |
| Lawrenson's, accuracy | 1,896 | 0.5530 | 0.5385 | 0.4388 | **0.5095** |
| Sutton's, RPS | 1,512 | 0.1968 | 0.2028 | 0.2319 | **0.3343** |
| Sutton's, accuracy | 1,512 | 0.5453 | 0.5351 | 0.4431 | **0.4921** |

**That pair of rows is ADR 0003's whole argument, measured.** On RPS a Pundit is 0.14 behind the
market and a tenth of a point *below the floor* — worse than a Predictor that does not know which
Clubs are playing. On accuracy, which asks only who they picked and not how sure they claimed to be,
the same calls are four points behind the market and seven ahead of the floor. Nothing about the
Pundit changes between those two lines; only the question does. ADR 0003 predicted "~0.36 against a
market at ~0.19" on illustrative rates and the measured gap is 0.334 against 0.195.

**The published results cross-check against Football-Data on 3,402 of 3,406 rows.** This is what
closes open risk 3, and it is a far stronger check than the ticket's "at least one row": the page
prints the real score beside every call, so agreement confirms the one thing no unit test can reach
— that two spellings became the right two Clubs, the right way round. The denominator is 3,406 and
not 3,408 because two Fixtures — 2022/23's Southampton against Brentford and Crystal Palace against
Manchester United — were only ever listed as postponed, so each carries a call and no score;
counting those as agreements would report a check that was never made. The four disagreements are
MyFootballFacts one goal out in one row, and Football-Data is the authority:

| Season | Fixture | MyFootballFacts | Football-Data |
|---|---|---|---|
| 2017/18 | Bournemouth v Southampton | 1-0 | 1-1 |
| 2024/25 | Ipswich v Bournemouth | 1-1 | 1-2 |
| 2024/25 | Ipswich v Arsenal | 0-3 | 0-4 |
| 2025/26 | Bournemouth v West Ham | 1-2 | 2-2 |

A Season that *stopped* agreeing would be a column shift or a swapped home and away, which agrees
only where the score is symmetric — about a quarter of the time. `MIN_AGREEMENT` sits at 95%, in
the gap between those two, and the worst real Season is 99.5%. Separately, the grading agrees with
the tally MyFootballFacts keeps for itself: the page says "48 Correct Scores" for 2017/18 and this
grading finds 48.

**The shared calibration layer finally meets a Predictor it can help.** Stage 6 measured it costing
every Predictor 0.0009–0.0015 RPS and CLAUDE.md asked for this to be re-measured rather than
assumed. It is not close:

| Predictor | Predictions | corrected | RPS | calibrated RPS | ten-bin error | calibrated |
|---|---|---|---|---|---|---|
| Lawrenson | 1,896 | 1,508 | 0.3341 | **0.2374** | 0.3270 | 0.0792 |
| Sutton | 1,512 | 1,130 | 0.3343 | **0.2473** | 0.3386 | 0.0988 |

On the Fixtures a fitted map actually reached, Lawrenson goes 0.3385 → **0.2169** against a Naive
Baseline of 0.2381 on the same 1,508, and Sutton 0.3301 → **0.2137** against 0.2330 on the same
1,130. So the layer takes a Predictor that is far below the floor as stated and lifts it above the
floor — a gain of about 0.09 RPS where the other four paid about 0.001.

**This confirms stage 6's diagnosis rather than contradicting it.** The layer was never broken; it
had nothing to find. All four earlier Predictors arrive at a ten-bin calibration error of about
0.006, and a monotone map fitted on that finds noise. A Pundit arrives at 0.33 — the most
miscalibrated Prediction there is — and the same layer, unchanged, recovers most of it. The
headline numbers stay pre-calibration for the reason ADR 0006 gives, and both columns stay
published for the reason this row demonstrates.

**None of this is the Calibrated Pundit.** Issue #12 fits a *different* map, bucketed by predicted
goal margin so that a 3-0 call is treated as the stronger claim it is, on past calls only. The
shared layer is generic and sees a one-hot Prediction as a two-valued input; the Calibrated Pundit
sees the Scoreline. They must not be collapsed into each other, and the numbers above are not a
preview of #12's.

## Open risks

1. **BBC live scraping is unproven.** `www.bbc.co.uk` was unreachable during design, article URLs are opaque IDs (`/sport/football/articles/cvg0e92ezz4o`, legacy `/sport/football/28859459`) and there is no index page. Needs a spike at stage 5. If it fails, live pundit data has no confirmed source — MyFootballFacts' update latency during a season is unknown.
2. **The live path is only half tested.** Re-checked 21 Aug 2026: `fixtures.csv` is reachable and parses, but it still holds a single English row (one E2 fixture) and no E0 rows, and `mmz4281/2627/E0.csv` still does not exist. The transport is proven; the E0 live path is not. `pytest --run-network` exercises what can be exercised, including the check that no new Club spelling has appeared upstream — the check that must pass before a live Prediction Round can be sealed.
3. ~~**MyFootballFacts parseability is unverified.** Content correctness was confirmed — a 2025/26 result cross-checked exactly against Football-Data — but the HTML has not been parsed across all nine season pages.~~ **Closed at stage 7.** All nine parse, yielding **3,408 calls** of a possible 3,420, and the cross-check went far past the one row the ticket asked for: the page prints the real score beside every call, and **3,402 of the 3,406 that carry one match Football-Data**, with the four exceptions named above. The HTML is hand-maintained and reads like it — annotated names, six misspellings, two Scorelines dropped inside a Club's name, and which table holds the predictions moving between pages — so the parser recognises a call by its shape rather than by where it sits, and refuses a page that yields fewer than 360 or more than 420. `tests/pundits/test_over_the_corpus.py` re-derives all of it.
4. ~~**Cross-tier Elo has no burn-in before 2000/01**, so early ratings linking E0 to E3 will be unreliable.~~ **Closed at stage 5.** Measured: by the first scored Prediction Round the thinnest Premier League rating rests on **190 matches**, and every Club promoted into the Premier League in every scored Season arrives with a distinct rating built from more than 200. The cold start is real and is confined to 2000/01, which is why that Season warms the ratings and is not fitted on either. `tests/models/test_elo_over_the_corpus.py` re-derives both numbers.
5. **Frozen hyperparameters will drift out of date** by the late Evaluation Window, given the measured decline in home advantage. Accepted deliberately; see ADR 0008.
