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
| 12 | v1 = stages 1–8. Deferred with written stubs: XGBoost/ML, Golden Boot, API-Football — written at stage 14 as `src/epl/v2/`, prose and constants, imported by nothing | — |
| 13 | 2026/27 gameweek 1 (21 Aug 2026) deliberately not chased; backtest power matters more than 10 live matches. **Superseded at stage 13**: the live loop cannot predict a Season it cannot see or score what it sealed, so 2026/27 is now ingested — as a third span outside both Windows, which is the part that mattered | [0010](./adr/0010-live-season-outside-both-windows.md) |
| 14 | Season Projections use a Bayesian posterior. Within-season strength drift is **not** modelled — measured at zero | [0007](./adr/0007-mle-for-matches-bayesian-for-projections.md) |
| 15 | Dixon-Coles: MLE at all 1,189 Prediction Rounds, Bayesian only where a Season Projection is produced | [0007](./adr/0007-mle-for-matches-bayesian-for-projections.md) |
| 16 | Miniforge + conda-forge; `environment.yml` is the source of truth; PyMC + nutpie | [0009](./adr/0009-conda-forge-toolchain.md) |
| 17 | Burn-In Window 2000/01–2004/05 for warm-up and tuning, then frozen; scoring from 2005/06 | [0008](./adr/0008-burn-in-prefix-frozen-hyperparameters.md) |
| 18 | The Season in progress is ingested, in neither Window, never backfilled, and scored on its own board | [0010](./adr/0010-live-season-outside-both-windows.md) |

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

Added at stage 9:

- **The BLAS provider is pinned to OpenBLAS**, and it is load-bearing rather than a preference. Solved without the pin on 24 Aug 2026, conda picks MKL 2026, whose Intel-OpenMP threading layer cannot resolve its symbols against the `libiomp5md.dll` shim that llvm-openmp 22 supplies. The failure is not a warning or a slow path: every LAPACK call aborts the interpreter with `0xc06d007f`, taking `numpy.linalg`, `scipy.linalg`, L-BFGS-B — and so the Dixon-Coles fit — and PyMC with it. Found while building stage 9, and it would have blocked #14 as surely as #13. This is the second place a free version choice breaks the build rather than merely drifting; `arviz` above is the first.

## Measured facts

Derived from the source data during design. Recorded so they need not be re-derived — but re-verify
before relying on any of them.

**Re-verified at stage 1 (21 Aug 2026)** against the full ingested corpus — 52,672 matches over
26 Seasons and four tiers. Every count in this section is over those **closed** Seasons and stays
there: from stage 13 the corpus also holds the Season in progress, which grows every Saturday
([ADR 0010](./adr/0010-live-season-outside-both-windows.md)).
`tests/ingest/test_raw_cache_integrity.py` re-derives these from the data
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
| Dixon-Coles by MLE | 0.19752 |
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
preview of #12's — measured, #12 gets to 0.2127 and 0.2111 where the shared layer gets to 0.2374
and 0.2473.

## The Calibrated Pundit

Added at stage 8 (issue #12). `epl.pundits.margin` is the map, `epl.pundits.calibrated` is the
Predictor built on it, and `epl.pundits.report` is the three-way board and the two tables beside it.

- **The map is bucketed by predicted goal margin, and by nothing else.** A call is reduced to
  `pred_home_goals - pred_away_goals`, and the map answers "when this Pundit said that before, how
  often was it Home, Draw and Away?". Reading a call as the Outcome it implies would throw away
  the part the Pundit took a risk on: a +1 call goes Home 42% of the time and a +3 call 83%.
- **No cap is chosen and no boundary is tuned.** The margins the Pundit actually called are the
  buckets, and the one rule is that a bucket too thin to carry a rate merges with its neighbour
  nearer zero. That is doing real work rather than tidying: over nine Seasons the two Pundits
  called +3 or better 213 times and −3 or worse 50, so any fixed symmetric cap would either lose
  the +3 bucket the ticket asks for by name or invent a −3 rate out of two dozen calls. The merge
  settles it from the sample — Lawrenson ends with `-3,-2 | -1 | 0 | 1 | 2 | 3,4` and Sutton with
  `-5,-4,-3,-2 | -1 | 0 | 1 | 2 | 3,4,5,6`, and nobody picked either.
- **`MINIMUM_SAMPLE = 40` is stated rather than fitted, because it could not be fitted.** ADR 0008
  permits a hyperparameter to be tuned only inside the Burn-In Window (2000/01–2004/05) and no
  Pundit in this project published a call before 2017/18. The reasoning is structural instead: a
  bucket is a claim about three Outcomes, the rarest is the Draw at about a quarter of Fixtures,
  and forty calls expect ten Draws where twenty expect five. Its cost is visible — each Pundit's
  first forty calls have no map behind them and are not covered, so 1,856 of 1,896 and 1,472 of
  1,512 reach the board.
- **Nothing enforces monotonicity, and the map comes out monotone anyway.** The isotonic layer of
  ADR 0006 imposes it because it is correcting a *scale* and must not touch a ranking; here the
  ranking is the thing being measured, and imposing it would turn a finding into an assumption.
  Both Pundits' Home rates rise at every step. `tests/pundits/test_calibrated_over_the_corpus.py`
  re-derives that rather than asserting it.
- **A Calibrated Pundit is a Predictor, not a scoring step**, and that is the one structural
  difference from `epl.calibration`. The shared layer is fitted on the Outcomes of the very
  Predictions it corrects, so it can only run at scoring time and stores nothing. This map is
  fitted on the Outcomes of matches that had *already kicked off* at the As-Of Instant, and its
  input — the Scoreline — was published before it. So on any Friday it really can be computed and
  quoted for Saturday. It goes through the ledger like everything else, reads its history through
  `Evidence`, and every stored row carries `inputs_seen` and `latest_input` — which is what makes
  the walk-forward claim checkable off the file rather than asserted by a test.
- **It refits at every Prediction Round** rather than folding one map forward, for the reason
  `epl.models.elo.Elo` gives at far greater cost: a map carried between calls would have to decide
  whether the Evidence it was handed extends the one it fitted last time, and getting that wrong is
  the one kind of bug this project cannot see.
- **One map per Pundit, never one shared between them.** A map is a statement about one
  forecaster's own calls, and pooling two people's would correct each with the other's habits —
  the same reasoning `epl.ledger.scoreboard.calibrated_predictions` gives for fitting the shared
  layer per Predictor. It costs the opening of each record separately.
- **The Predictors are named for the map, not the person.** `margin_map_lawrenson` and
  `margin_map_sutton`, with a `note` that says "a one-feature model fitted on Mark Lawrenson's
  published Scorelines — not Mark Lawrenson". ADR 0003 spends its Consequences section on this and
  the scoreboard line is the artifact a reader actually receives, so the distinction lives there.
- **The comparison is cut; the calibration is not.** `epl.ledger.scoreboard.lines` was split out of
  `build` so a caller can score a narrower slate over an already-calibrated frame. Cutting first
  would give the Market Line a calibrated form fitted on a Pundit's 1,900 Fixtures rather than its
  own 7,980 — a post-calibration number that exists nowhere else and belongs to nobody. Narrow the
  comparison, never the Predictor; ADR 0001's rule, applied to the map.

### Measured at stage 8 (23 Aug 2026)

Re-derived on every run by `tests/pundits/test_calibrated_over_the_corpus.py`, which skips when
`data/raw/` is absent.

**The three-way board**, over the Fixtures every Predictor in it reached:

| Over | Fixtures | Market Line | Elo | Calibrated Pundit | Naive Baseline | Pundit as-stated |
|---|---|---|---|---|---|---|
| Lawrenson's | 1,856 | 0.1943 | 0.2016 | **0.2127** | 0.2356 | 0.3335 |
| Sutton's | 1,472 | 0.1968 | 0.2031 | **0.2111** | 0.2322 | 0.3346 |

**The cost of stating certainty: 0.1209 and 0.1235 RPS.** The deliverable of ADR 0003, in one
number per forecaster — what being asked for a scoreline instead of a probability charged them.

**Read fairly, the same calls beat the floor they were a tenth of a point below.** That single
sentence is what issue #12 existed to make true. And **accuracy barely moves** across the two
readings — 0.5102 → 0.5116 and 0.4925 → 0.4993 — so the 0.12 RPS is the format of the question
rather than a different set of opinions. The map changes the top pick only on draw calls, where no
bucket has the Draw as its mode.

**Neither Calibrated Pundit beats Elo.** ADR 0003 anticipated that one might, and the naming rule
is in the code regardless of how it landed. It sits between Elo and the floor.

**The shared calibration layer's 0.09 RPS gain disappears once the margin map runs first.** It
costs the two Calibrated Pundits 0.0014 and 0.0015 — exactly what it costs Elo and the market.
Ten-bin calibration error goes 0.327 → 0.019 and 0.338 → 0.020. That confirms stage 6's diagnosis
from the other direction: the layer was never broken, and it only ever had something to find
because an as-stated Pundit was the most miscalibrated Prediction on the board. The residual 0.019
against the other four's 0.006 is the map's seven buckets showing — it is coarse by construction.

**The fitted maps**, at each Pundit's final Prediction Round:

| Margin | Lawrenson: calls, H/D/A | Sutton: calls, H/D/A |
|---|---|---|
| −3 and worse | with −2 | 134, 0.13 / 0.25 / 0.62 |
| −2 | 315, 0.21 / 0.18 / 0.62 | with −3 and worse |
| −1 | 150, 0.33 / 0.23 / 0.45 | 258, 0.31 / 0.22 / 0.47 |
| 0 | 391, 0.33 / 0.30 / 0.38 | 375, 0.37 / 0.29 / 0.35 |
| +1 | 344, 0.42 / 0.28 / 0.30 | 374, 0.48 / 0.25 / 0.26 |
| +2 | 571, 0.60 / 0.21 / 0.19 | 264, 0.65 / 0.21 / 0.14 |
| +3 and better | 115, 0.83 / 0.10 / 0.07 | 97, 0.81 / 0.12 / 0.06 |
| pooled | 1,886, 0.44 / 0.23 / 0.33 | 1,502, 0.44 / 0.24 / 0.32 |

**The best and worst calls, by miss** (user story 34, at `outputs/pundit_calls.csv`). The miss is
the RPS of the fair reading. Both ends are bold calls: the best twenty are all |margin| ≥ 3 that
came off — where stating certainty *paid*, so the per-call cost is negative — and the worst are the
same boldness missing, where the as-stated reading scored a flat 1.00. Lawrenson's best is 2018/19
Tottenham 3-0 Huddersfield (0.004) and his worst 2018/19 Bournemouth 3-0 Fulham (0.890), which
Fulham won; Sutton's are 2023/24 Arsenal 3-0 Burnley (0.017) and 2023/24 Liverpool 3-0 Crystal
Palace (0.774), which Palace won.

## Dixon-Coles by maximum likelihood

Added at stage 9 (issue #13), the first model that predicts goals. `epl.models.likelihood` is the
likelihood, the weighted sample and the Scoreline grid; `epl.models.dixon_coles` is the maximum-
likelihood fit and the Predictor over it; `epl.models.burn_in.fit_decay` is where the one
hyperparameter is chosen.

- **One likelihood, in a module that knows nothing about optimisers or Predictors.** ADR 0007's
  whole justification for fitting one model two ways is that "both paths share one likelihood
  function, so the models cannot drift apart", and a shared function that lived inside the MLE path
  would be shared only until someone needed it to be. `epl.models.likelihood` holds the rates, the
  low-score correction, the decay and the Scoreline grid, and imports nothing from either fit;
  `epl.simulate` (issue #14) is expected to fit the same `Sample` and return the same `Strengths`.
- **The whole pyramid again, and this time it was measured rather than inherited.** ADR 0004 is an
  argument about Elo, and Elo is zero-sum: a rating carried across a promotion is comparable by
  construction. This model has no such guarantee — no Club ever plays outside its own tier, so
  nothing in the likelihood knows a division exists and the four tiers are joined *only* by the
  Clubs that changed tier inside the decay horizon. Fitting all four scores **0.20165** on the
  Burn-In Window against a Premier-League-only **0.20382**, and the bridge holds: at the first
  scored round mean attack falls monotonically E0 → E3 (+0.45, +0.11, −0.09, −0.31), and it still
  does in 2015/16 and 2025/26. Nothing orders those tiers; the promoted Clubs do.
- **The half-life is a well-determined region, not a well-determined number.** 322.5 days is what
  the grid finds, and anything from 270 to 480 days scores within 0.0001 RPS of it. What the data
  *does* exclude is the short end: a 60-day half-life costs 0.007 RPS, because a fit that remembers
  two months is fitting form rather than strength.
- **The weight floor is a tolerance and was checked to be one.** It decides how far back a sample
  reaches, which makes it look like a hyperparameter. Five times the floor — 0.05 instead of 0.01,
  a horizon of 1,394 days instead of 2,143 — moves the Burn-In score by 0.00001 RPS. It is stated
  rather than fitted, and `tests/models/test_dixon_coles_over_the_corpus.py` re-derives that.
- **Dixon-Coles' low-score correction has all but vanished from this corpus, and is kept anyway.**
  The 1997 paper fitted about −0.13 on four Seasons of one division. Here the fitted value wanders
  around zero and changes sign — −0.058 at the first scored round, +0.003 in 2015/16, −0.010 in
  2025/26 — and pinning it at zero, which is two independent Poissons, costs **0.00011 RPS**. It
  stays for two reasons: it is a parameter of the shared likelihood rather than a hyperparameter, so
  dropping it would change the model rather than simplify the code; and 0.00011 in the right
  direction is a measurement, not nothing. It is emphatically *not* why this model beats Elo.
- **The fit is refitted from cold at every Prediction Round**, for the reason `epl.models.elo.Elo`
  gives: a fit carried between calls would have to judge whether the Evidence it was handed extends
  the one it fitted last, and getting that wrong is invisible. Here it is nearly free — about 300 ms
  a round, five minutes for the whole Evaluation Window.
- **The likelihood is flat along one direction, and the fit is put in a gauge before it leaves.**
  Adding a constant to every attack *and* every defence changes no rate, so a fitted `Strengths` is
  an arbitrary point on a line until `centred()` picks the one where the mean attack is zero. Two
  fits are not comparable — and no attack table is readable — without it.
- **The gradient is analytic and checked against central differences.** Two hundred parameters
  differenced is two hundred extra evaluations per step, which is the difference between the five
  minutes above and most of a day. A hand-written derivative that had drifted from its own function
  would converge slightly wrong and look completely normal, so `tests/models/test_likelihood.py`
  differences the very function it belongs to.
- **A Club with no matches in the sample keeps neutral strengths rather than being dropped.** Zero
  weight is zero gradient, so the optimiser leaves it where it started. The alternative makes a
  Fixture unanswerable rather than uncertain. Over this corpus it reaches only the Clubs entering
  League Two from outside the Football League, four tiers from anything scored.

### Measured at stage 9 (24 Aug 2026)

Re-derived on every run by `tests/models/test_dixon_coles_over_the_corpus.py`, which skips when
`data/raw/` is absent.

| Predictor | Fixtures | RPS | Brier | Log loss | Accuracy | Ten-bin error |
|---|---|---|---|---|---|---|
| Market Line | 7,980 | 0.19362 | 0.5684 | 0.9582 | 0.5471 | 0.0061 |
| Ceiling Line | 2,660 | 0.19676 | 0.5717 | 0.9639 | 0.5498 | 0.0060 |
| **Dixon-Coles** | **7,980** | **0.19752** | 0.5768 | 0.9707 | 0.5360 | 0.0080 |
| Elo | 7,980 | 0.19943 | 0.5810 | 0.9771 | 0.5380 | 0.0055 |
| Naive Baseline | 7,980 | 0.22938 | 0.6430 | 1.0642 | 0.4556 | 0.0061 |

- **It clears the README's ≤0.1986 target**, which Elo missed by 0.0008, and it takes **0.0319 of
  the 0.0358** RPS the market takes out of the floor — 89% of the available edge against Elo's 84%.
  It is 0.0039 short of the market, and a goals model that *beat* the book on the book's own
  information set would be evidence of a leak rather than of a good model.
- **Reading the goals is worth 0.0019 RPS over reading the Outcomes**, which is the whole of what
  issue #13 set out to find. Note it is not worth much on *accuracy* — 0.5360 against Elo's 0.5380,
  slightly worse. The two models pick nearly the same winners; this one is better calibrated about
  how sure it is, which is exactly what RPS measures and accuracy does not (CLAUDE.md).
- **The shared calibration layer costs it 0.0004 RPS**, 0.19752 → 0.19793, against 0.0009 for Elo
  and 0.0009 for the market. Stage 6's finding survives a fifth Predictor: the layer is not broken,
  it has nothing to find. Its pre-calibration ten-bin error of 0.0080 is the highest of the four
  full-window Predictors and still small.
- **Playing at home is worth about a third of a goal, and falling.** +0.2964 log-goals at the first
  scored round (x1.345), +0.2117 in 2015/16, +0.1972 in 2025/26. That is the same structural drift
  ADR 0008 accepts by name, visible here in a unit anyone can read.
- **A full walk takes about five minutes**, at 952 refits of roughly 230 parameters over some
  12,000 weighted matches each. Issue #13 asked for "minutes rather than hours". A fit needs a mean
  of 251 L-BFGS-B iterations and a worst of 1,454 over a 136-round sample, and the tail is longer
  than that — one of the diagnostic's 3,130 per-kickoff cuts needed more than 2,000, which is why
  the iteration ceiling sits at 10,000 rather than near the observed worst case. The tolerances are
  untouched by that: measured over the same sample, the *objective* tolerance terminates every fit
  and loosening the gradient tolerance tenfold does not change a single iteration.

### What the weekly batch gives up (ADR 0002's diagnostic)

`epl.ledger.backtest.sequential` predicts every Fixture twice with the same Predictor — once from
its Prediction Round's As-Of Instant, exactly as the ledger stores it, and once from an Evidence cut
at its own kickoff. ADR 0002 promised this measurement and forbade quoting it as a score; the
`sequential_rps` column is what a model that broke the three-way comparison would get, and it is on
the record so the size of the choice is known rather than argued about.

| Predictor | Over | Fixtures | Batch RPS | Sequential RPS | Withheld |
|---|---|---|---|---|---|
| Elo | whole window | 7,980 | 0.19943 | 0.19942 | **+0.00001** |
| Elo | 2019/20 on | 2,660 | 0.20525 | 0.20523 | **+0.00002** |
| Dixon-Coles | whole window | 7,980 | 0.19752 | 0.19749 | **+0.00003** |
| Dixon-Coles | 2019/20 on | 2,660 | 0.20343 | 0.20338 | **+0.00006** |

**Withholding Saturday's results from Monday night's call costs essentially nothing** — 0.00001 RPS
for Elo and 0.00003 for Dixon-Coles, two to three orders of magnitude below the 0.0019 that reading
the goals is worth, and below the resolution anything in this project is reported at. ADR 0002
traded accuracy for comparability and it turns out the trade was very nearly free. The goals model
uses the withheld information about three times as well as Elo does, which is the ordering one would
expect and is still nothing.

The batch column reproduces each Predictor's scoreboard RPS exactly, which is what makes the two
columns comparable at all: the diagnostic re-derives the stored Prediction rather than a different
one.

Two limits on that number, pointing opposite ways, and both in
`epl.ledger.backtest.sequential`'s docstring. Football-Data records no kickoff time before 2019/20,
so an untimed Fixture sits at midnight and cannot see the earlier kickoffs of its own day — which
*understates* the gap over most of the window. And `Evidence` timestamps a match at its kickoff
rather than its final whistle, so inside the timed era a 17:30 Fixture is handed the 16:00 match
that was still being played — which *overstates* it. Both readings come out negligible, which is
what makes this a finding rather than an artefact of missing timestamps. No attempt is made to
sharpen the second: a match-length constant subtracted from every kickoff would be a hyperparameter
invented to flatter a diagnostic.

The Elo figures are re-derived by `tests/models/test_dixon_coles_over_the_corpus.py`. Elo is used
there rather than Dixon-Coles as an economy — the question is about the As-Of rule, which is the
same for every Predictor, and the walk takes one fit per distinct kickoff instead of one per round,
3,130 of them against 952. Elo pays six minutes for the whole window where Dixon-Coles pays
thirty-four, and half an hour inside a test suite is a test nobody runs. The Dixon-Coles row above
came from `python -m epl.models sequential`.

## The Bayesian Dixon-Coles posterior

Added at stage 10 (issue #14), the second of ADR 0007's two fits. `epl.simulate.posterior` samples
the same likelihood `epl.models.dixon_coles` maximises; `epl.simulate.checkpoints` decides the
handful of Prediction Rounds it is allowed to run at. The Monte Carlo Season Projection over these
draws is issue #15.

- **The sampler is handed the numpy likelihood itself, not a PyTensor copy of it.** The obvious way
  to write this module is to re-express the rates, the Poissons and the low-score correction in
  PyTensor so NUTS can differentiate them — and that produces exactly the second implementation
  ADR 0007 exists to prevent, undetectably: two Dixon-Coles likelihoods disagreeing in the fourth
  decimal both look entirely reasonable, and only the Season Projection would quietly be a different
  model from the match probabilities. Instead `negative_log_likelihood` is wrapped as a single
  opaque `Op`. It already returns its own analytic gradient, which is exactly the pair a PyTensor
  `Op` needs, so the sampler differentiates the arithmetic the optimiser descends rather than a
  faithful copy of it.
- **The gauge has to be inside the model here, where the MLE can apply it afterwards.** The
  likelihood is flat along "add a constant to every attack *and* every defence". An optimiser walks
  to an arbitrary point on that ridge and `centred()` picks the readable one on the way out; a
  sampler handed the same ridge has an improper posterior and wanders it forever. `ZeroSumNormal`
  on attack is that gauge expressed where a sampler can see it, and it is not a stylistic choice: a
  plain Normal there does not give a wrong number, it gives a fit that never finishes converging.

### The cliff under the low-score correction

The single hardest thing found at this stage, and it is a property of the *shared* likelihood that
only the Bayesian path can trip over.

Dixon-Coles' correction multiplies four Scorelines by factors that go **negative** once `rho` is
large enough — beyond about 0.35 on a typical sample. There the model is not a probability
distribution at all, and `epl.models.likelihood.CORRECTION_FLOOR` clamps the factor at 1e-12 before
its logarithm so the optimiser's line search cannot fall off the edge. That clamp is right for
L-BFGS-B, which only accepts an improving step and so never lingers there. It is a trap for a
sampler, because it keeps the log-density *smooth* while putting `1 / 1e-12` into the gradient:

| `rho` | smallest correction factor | log-density | largest gradient component |
|---|---|---|---|
| 0.30 | 0.071 | −1145 | 2.3e2 |
| 0.35 | **−0.083** | −1226 | **9.3e12** |
| 0.50 | −0.548 | −1316 | 1.4e13 |

The log-density moves by less than a hundred while the gradient gains ten orders of magnitude. NUTS
takes one leapfrog step against 1e12, throws the position to infinity, overflows `exp`, and produces
a `nan` — which is worse than an error, because divergence is tested with a comparison and every
comparison against `nan` is false. The trajectory is never rejected: it doubles to the tree-depth
ceiling on **every** draw. Measured on an 18-parameter fixture that is 1,023 leapfrog steps per draw
against 7, a 55× slowdown, and a posterior mean 0.85 log-goals from the MLE — a fit that looks
merely slow and is actually wrong.

`epl.simulate.posterior.log_likelihood_at` therefore refuses that region outright, returning `-inf`
with a zero gradient. That is not a patch on the shared likelihood; it is the model's actual support
written down, since a Scoreline probability cannot be negative and those parameter values have zero
likelihood. The check is built from `low_score_factor` — the same function the likelihood corrects
with — so it asks the model where it is valid rather than restating any part of it. `rho = 0.35` is
only three and a half prior standard deviations out, so this is reached during ordinary tuning
rather than in some pathological corner.

### The priors are scaffolding, and had to be measured to prove it

`epl.models.dixon_coles` says of the MLE path that "nothing here regresses a Club to the mean, and
nothing carries a prior — the time decay is the only thing that forgets". A Bayesian fit with
informative priors shrinks every Club toward the mean. That is ordinary practice and it is **wrong
here**, because it makes the posterior a different model from the MLE, which is the one thing
ADR 0007's shared likelihood exists to prevent. The priors exist only to make the posterior proper.

Measured at the first Prediction Round of 2015/16 — 11,873 weighted matches, 102 Clubs, 206
parameters — as the regression slope of posterior mean on MLE, where one is no shrinkage:

| strength prior | attack slope | defence slope | Fixture quote off by |
|---|---|---|---|
| 0.5 | 0.649 | 0.557 | 0.0791 |
| 1.0 | 0.856 | 0.769 | 0.0332 |
| 2.0 | 1.005 | 0.861 | 0.0082 |
| 5.0 | 1.155 | 0.896 | 0.0052 |

At 0.5 — a perfectly reasonable-looking default — Darlington's attack goes from −0.645 to −0.127 and
Manchester City's from +0.960 to +0.748. So `strength_sigma` is set from what the corpus has ever
produced rather than from what seems reasonable: fitted attack has a spread of about 0.32 and a range
inside ±1, so **2.0** puts every strength this project has fitted within half a standard deviation
and the likelihood's own `STRENGTH_BOUND` at two and a half.

**Widening was only half of it.** Attack recovered to 1.005 while defence stalled at 0.861, and the
asymmetry was structural rather than a matter of width: attack was a `ZeroSumNormal` and defence a
plain Normal centred at zero, which constrains not only the *spread* of defence but its *mean* — and
that mean is the pyramid's overall goal rate, which the likelihood determines from every Scoreline
in the sample and the MLE has no opinion about. `model_for` now builds both as zero-sum deviations
from a `scoring_level` that carries the rate itself, under a wide prior. Home advantage comes back
at **+0.2118** against the MLE's **+0.2117**.

**Every number in this section comes from an Evaluation-Window Season, and none of it is tuning.**
ADR 0008 confines hyperparameter fitting to the Burn-In Window, and the measurements below were all
taken at the first Prediction Round of 2015/16. The distinction that makes that legitimate is what
they compare: this fit against *the MLE of the same model on the same matches*, chains against each
other, and the clock. **None of them looks at an Outcome**, and nothing here was chosen because it
scored better. Same argument as `epl.pundits.margin.MINIMUM_SAMPLE`, stated for the same reason —
an Evaluation-Window season number sitting in a constant's docstring should make a reader stop.

### Measured at stage 10 (25 Aug 2026)

At the first Prediction Round of 2015/16, four chains of 1,000 draws after 1,000 tuning steps:

- **One posterior fit takes about four minutes** — 231 s over 206 parameters and 11,873 weighted
  matches, against the MLE's 0.22 s at the same round. That is roughly a thousand times the cheap
  fit, which is precisely why ADR 0007 confines it to Season Projection points: six checkpoints
  across 21 historical Seasons is 126 fits and about eight hours, where every round would be some
  20,000 fits and the overnight-to-two-day job the ADR refuses by name.
- **The two fits agree on a real Fixture to 0.0090**, which is issue #14's stated acceptance
  criterion. Bournemouth v Aston Villa comes out H 0.5340 / D 0.2381 / A 0.2279 by maximum
  likelihood and H 0.5430 / D 0.2381 / A 0.2190 at the posterior mean. This is the measurement that
  licenses the whole split: every match probability the project publishes is the cheap fit.
- **The chains converge** — R-hat 1.0000 and 2,303 effectively independent draws of the least-mixed
  parameter, from 4,000 total.
- **Defence still comes back at 0.875 times the MLE's, and that is not shrinkage.** It is stable
  across chain counts, and the Clubs carrying it are the ones the two fits disagree about most:
  Grimsby, Stockport, Darlington and Lincoln, on 38 to 84 weighted matches each, all pushed *further*
  from zero by the posterior rather than toward it. That is what a posterior mean does to a
  weakly-identified log-rate whose marginal is skewed, and it is the opposite sign to shrinkage.
  Every Club affected is in the fourth tier, four promotions from anything scored.
- **The draws genuinely disagree, which is the entire point.** Across the posterior, that one
  Fixture's Home probability spans 0.42 to 0.77 with a standard deviation of 0.073, and home
  advantage has a standard deviation of 0.025. An MLE reports none of that, and it is what compounds
  across 380 simulated Fixtures into the difference between a 34% title probability and a 48% one.
- **Running the chains in parallel is nearly free and buys nearly nothing.** `cores` does not change
  the draws — PyMC derives one seed per chain from the seed and the chain count alone, and four
  chains on four cores come back bit-identical to the same four on one — but it saves 6% rather than
  the fourfold a chain count suggests: 58 s against 62 s. The likelihood is a Python function called
  through numba's object mode, so the chains queue for the GIL. That is the standing cost of not
  writing a second likelihood in PyTensor, and it is what issue #15 should plan its validation run
  around rather than assuming chains scale.
- **About 3.3% of draws are divergent, and they are all the same dozen Clubs.** The correlation
  between a Club's log weight in the sample and the width of its posterior is **−0.977**. At the
  first round of 2015/16 Grimsby carries **half a weighted match** — against 52.9 for a
  fully-observed Club — a posterior standard deviation of 1.18, and draws reaching an attack of
  **5.85**, which is a rate of 330 goals a match. Those draws walk into the region the support
  check refuses, and each refusal is counted as a divergence. This is the wide prior working rather
  than failing: a Club with no football has a posterior that *is* the prior, which is the honest
  answer, and narrowing the prior to stop it would reintroduce the shrinkage on the Clubs that do
  have football. Every Club involved is in the fourth tier. `target_accept` is 0.95 because 0.99
  did not finish a corpus-scale fit in twenty minutes and 0.9 diverges more.

## The Monte Carlo Season Projection

Added at stage 11 (issue #15), the last modelling stage. `epl.simulate.table` turns Fixtures and
goals into a final league table; `epl.simulate.projection` walks 10,000 Seasons over the posterior's
draws; `epl.simulate.validation` runs the whole thing across completed Seasons and asks whether the
answers were calibrated or merely plausible.

- **The tiebreaker chain is issue #15's, not the Premier League's, and the difference is a
  refinement.** The competition's own regulation runs points, goal difference, goals scored, and
  then — with **no head-to-head step at all** — declares the Clubs to occupy the same position,
  holding a play-off at a neutral ground only if the championship, relegation or a European place
  turns on it. The ticket asks for head-to-head points and head-to-head away goals between goals
  scored and that play-off, so `TIEBREAKERS` has six steps where the regulation has four. It changes
  nothing the regulation decides and replaces some coin flips with a rule, which is why it was
  built as asked rather than argued about. See the measurement below for how often it matters.
- **The chain lives in a module that has never heard of a posterior.** `epl.simulate.table` takes
  Fixtures and goals and returns positions, so every step of it is exercised by hand-built leagues
  of three and four Clubs in `tests/simulate/test_league_table.py`. A chain that could only be reached
  through a ten-thousand-Season Monte Carlo run is a chain nobody checks, and the tests that matter
  most are the ones where two steps disagree — a league where goal difference says one thing and
  the head-to-head record says the opposite is the only kind that tells a correct chain from a
  plausible one.
- **A `Slate` holds the results of the played Fixtures and cannot express any others.** This is the
  project's one rule made structural rather than checked. Validating a historical Season means
  handing the projection a corpus that already contains every result it is supposed to be
  forecasting, so `Slate.of(played, remaining)` reads goals off the first frame and only the two
  Club columns off the second. There is no argument through which a projection can give itself the
  answer.
- **Both seeds are recorded, because a projection is random twice.** The sampler's is on every
  `Diagnostics` and the walk's is on `Simulation`, and `Projection.describe()` prints both. They are
  deliberately different numbers so that no reader can mistake one for the other being reused.
- **What is printed and what is written are two different tables, and only one of them can be
  re-run.** `Projection.table()` is the twenty rows a person reads; `Projection.published()` puts
  the Season, the As-Of Instant and the Prediction Round in front of them and the two seeds, the
  simulated-Season count and the posterior draw count behind. The file gets the second, because
  "a fixed deterministic seed recorded in the output" means the output rather than the terminal,
  and a projection that cannot say which Season it is of is twenty numbers. `published()` refuses
  an unattributed projection rather than writing one.
- **Draws are spread, not sampled with replacement, and never walked in order.** `fit` concatenates
  its chains, so the first quarter of a posterior's draws are one chain's; a walk over the first
  10,000 would explore a quarter of what it paid nine minutes for. `draw_order` therefore lays down
  whole copies of every draw and hands out the remainder at random. Truncating a permutation of
  enough whole copies — the obvious version — leaves some draws used three times and others once.
- **Goals are drawn from the Scoreline grid, not from two independent Poissons.** The low-score
  correction lives in exactly four cells of that grid, so a walk that sampled the two rates
  independently would be simulating Dixon-Coles with the Dixon-Coles taken out.
- **The European band is a stated simplification and says so.** Relegation is the bottom three and
  the title is first place; both are facts about the league. "European places" is the top four,
  which is the Champions League for most of the Evaluation Window but not all of it — England has
  had a fifth place in some recent Seasons on UEFA's coefficient, and a European place can also be
  won by lifting a cup, which is not a league position and which nothing here models.
- **The real final table is the one the results give.** The corpus has no column for an
  administrative points deduction — Portsmouth's nine in 2009/10, Everton's and Nottingham Forest's
  in 2023/24 — so both sides of the validation are the table the football alone produced. That is
  self-consistent, since the projection is not being marked against a table it was never asked to
  predict, but it does mean the relegation column of those two Seasons describes a table that is
  not the published one.
- **One reliability diagram, two things measured with it.** `epl.metrics.diagram` is the ten bins,
  the right-open edges and the rounding; `epl.metrics.reliability` one-hots three ordinal Outcomes
  into it and `Validation.reliability` pools three binary events into it. Only the front differs,
  and it has to — but the middle must not, or the projection's 0.008 and the scoreboard's 0.006
  would be two measurements rather than one measurement of two things. The count column is named by
  the caller (`predictions` against `projections`) for the reason `f714c28` renamed a clashing
  column at stage 8: a Club-projection is not a Prediction, and a reader with both files open should
  not find one word meaning two things.
- **The Season-and-tier cut is written once**, in `epl.simulate.checkpoints.season_fixtures`, and
  the three places that want it call it: where a projection is taken, where the Season is split at
  that instant, and where the table it eventually produced is read. Three copies is three places for
  `PROJECTED_DIVISION` to quietly become the literal `"E0"` — which had already happened once, in
  the command line's default season list, and is why the helper exists. One consequence is worth
  knowing: a Season the corpus does not hold now raises `CheckpointError` from all three rather than
  each module's own error, which is what that exception already meant.

### Measured at stage 11 (26 Aug 2026)

- **Points ties are routine and goal difference settles all of them.** Over the 26 ingested
  Seasons, **24 had at least one pair of Clubs level on points** and there were **85 tied pairs in
  all, 3.3 a Season** — which reproduces the design's figure exactly and is why the Season
  Projection could not be built on Elo. And then: **not one pair in 26 years was still level after
  goals scored.** The two head-to-head steps and the coin flip beneath them have never been needed
  by a real Premier League final table. They are there for the simulated ones, and a projection
  reports its own `level_pairs` so that "how often did the lower half of the chain decide
  anything?" is a number rather than an assumption.
- **The walk is not the expensive half, by three orders of magnitude.** Ten thousand Seasons over a
  real half-Season of Fixtures and 4,000 posterior draws takes **2.7 seconds**. The posterior fit
  in front of it took **537 seconds** at the shipped sampler settings — rather more than stage 10's
  231 s at the first round of 2015/16, which is worth knowing before planning a validation run.
- **Drawing from the posterior widens the table exactly as ADR 0007 says it should.** At 2011/12's
  first checkpoint, with 331 Fixtures still to play, Manchester United's title probability comes out
  **0.853 from the posterior mean, 0.850 from the MLE and 0.772 from the draws** — entropy across
  the twenty Clubs 0.52 against 0.75. The point estimate and the mean agree with each other and
  disagree with the honest answer, which is the shape of ADR 0007's 48%-versus-34% claim on real
  football rather than in the abstract.
- **The same effect, isolated.** Over a synthetic posterior that cannot decide which of two Clubs
  is the strong one, the other two Clubs take the title **6.3% of the time from the draws and 20.4%
  from their mean**; and the Club the draws are undecided about finishes bottom 17.4% of the time
  from the draws against 11.1% from the mean. A mean of "a is dominant" and "b is dominant" is "a
  and b are both fairly good", which is a claim neither draw made — in both directions.
- **2015/16 at Christmas, which is the projection's own worked example.** Leicester led on 38 points
  with 210 Fixtures left. The projection gives them **0.0577** for the title, behind Arsenal at
  0.5165 and Manchester City at 0.3572, and gives them 0.63 for a European place. They won it. That
  is not a failure of the projection so much as the whole reason the Season is remembered, and it is
  the number the command prints, so it is the number this file quotes.

### The validation: where the real champion landed

`python -m epl.simulate validate --seasons 2005 2024 --checkpoints 3 --draws 300 --tune 500
--chains 2`, run 26 Aug 2026 — **60 projections across the 20 completed Evaluation Window Seasons,
10,000 simulated Seasons each, 98 minutes**. The sampler settings are below what the module ships
with, and that is stated rather than buried: 60 fits at the shipped settings is nine hours. The full
six-checkpoint, full-sampler run is the overnight job and the command for it is the same one.

- **The eventual champion was the projection's own favourite 73% of the time, and in its top three
  97%** — 58 of 60. The three it put outside its top three are 2007/08 (Manchester United third
  behind Chelsea in September), 2015/16 and 2016/17.
- **It gave the eventual champion a mean title probability of 0.581, and its favourite 0.666.** The
  gap between those two is the honest cost of being wrong about the favourite a quarter of the time.
- **Ten-bin calibration error: title 0.012, European places 0.012, relegation 0.018, pooled 0.008.**
  For comparison, every match Predictor on the scoreboard sits at about 0.006 pre-calibration, and
  the margin map at 0.019. So a Season Projection is about as well calibrated as the Predictors it
  is built from — which is what "calibrated and not merely plausible" was asking.
- **The distribution tightens exactly as it should, and this is the number the six checkpoints were
  spread across a Season to produce:**

  | checkpoint | Fixtures left | probability given the eventual champion | favourite's own | champion was the favourite |
  |---|---|---|---|---|
  | 1 | 331 | 0.405 | 0.590 | 55% |
  | 2 | 207 | 0.609 | 0.651 | 85% |
  | 3 | 102 | 0.729 | 0.758 | 80% |

- **Leicester is the whole exercise in one row.** 2015/16 at the first checkpoint: **0.0008**, eighth
  favourite of twenty. At Christmas: 0.0599, third. In March with 92 Fixtures left: 0.5258, and the
  favourite at last. A projection that had said anything else in September would have been wrong
  about every other Season, and one that could not get to 0.53 by March would not be reading the
  table in front of it.
- **The largest miss is 2011/12**, where Manchester City won on goal difference on the final day and
  the projection had Manchester United at 0.777 in September against City's 0.065. It never made
  City favourite at any of the three checkpoints. That is a real miss rather than a rounding one,
  and it is what a 73% hit rate looks like from the inside.
- **These points are not independent and no summary above should be read as though they were.**
  Within one projection the twenty Clubs' title probabilities sum to one; the three checkpoints of a
  Season concern the same champion. 3,600 Club-projections carry far less information than 3,600
  independent forecasts, so the diagram is a shape rather than a significance test. `validate` prints
  that sentence under every diagram it produces.

## The BBC spike, and why a Pundit cannot be sealed

Added at stage 12 (issue #16), which is a spike: the deliverable is a decision backed by evidence,
and the evidence changed the question. `epl.pundits.myfootballfacts.discover_pages` is the code that
survived it, `epl.pundits.live` is the finding written as a module, and `python -m epl.pundits live`
runs it. Everything below was measured on **27 August 2026**.

**The BBC is reachable, and that is not the blocker.** Open risk 1 recorded `www.bbc.co.uk` as
unreachable during design. It answers in 0.1 s. The opaque article IDs resolve —
`/sport/football/articles/cvg0e92ezz4o` is *Premier League opening weekend predictions: Chris Sutton
v singer Tom Grennan*, published 2025-08-15 — and the page carries both `application/ld+json` and a
`window.__INITIAL_DATA__` blob, so it is machine-readable rather than something that would have to be
scraped out of prose. On the technical question the risk asked, the answer is yes.

**The blocker is permission, and it is explicit.** `bbc.co.uk/robots.txt` opens with a plain-English
statement of the BBC's terms: *"No scraping, crawling, or systematic extraction of content"*, *"No
creating datasets from BBC content"*, *"No text and data mining"*, and *"No business use without
permission"*. It then disallows `ClaudeBot`, `Claude-Web` and `anthropic-ai` from the entire site.
Issue #16 would build a fetcher producing a committed dataset of BBC pundit calls, which is the named
case twice over. **So the BBC is out** — and this is a better answer than "unreachable" was, because
it does not change when a network does. Article discovery was therefore not pursued past this point;
there is no index page, and finding one would not make the terms say anything different.

**MyFootballFacts wins, and permits what the BBC forbids.** Its `robots.txt` is `Allow: /` for
`ClaudeBot`, `anthropic-ai`, `CCBot` and a dozen other agents, named individually. It also has the
one thing the BBC lacks: **an index**. That mattered immediately — the archive has now used *four*
slug conventions in eighteen Seasons, and 2026/27 arrived as
`chris-sutton-predictions-premier-league-2026-27`, dropping the `for-` its four predecessors carried.
A URL built from the Season would have 404ed, so `discover_pages` follows the index's own
`rel="next"` rather than naming or constructing anything.

**The index links eighteen Season pages, and the backfill uses nine.** 2009/10 through 2026/27,
consecutively. The eight before 2017/18 are Mark Lawrenson calls this project has never scored —
roughly 3,000 more, on the same source, in the same shape. That is not in scope here and is recorded
because it is worth an issue rather than a rediscovery.

**The latency, which is what the ticket asked to be stated.** It cannot be measured from a page
fetched today — a Season page shows what the archive holds *now*, not when each call arrived — so it
was measured against Wayback Machine snapshots of the 2025/26 page, asking at each snapshot the
question a Prediction Round asks: *are the next round's calls already published?*

| Snapshot | Calls on page | Next round kicks off | Its Fixtures | Already called |
|---|---|---|---|---|
| 2025-08-15 | 10 | same day | 10 | **10** |
| 2026-02-04 | 240 | 2 days later | 10 | **0** |
| 2026-03-30 | 309 | 11 days later | 10 | **0** |

The Wayback Machine holds a fourth distinct-content snapshot, 2025-12-21, and it is **left out
because it is not evidence**: the round it would be measured against had already kicked off the day
before, so its 10-of-10 says only that the archive records the past, which is the thing not in
dispute. Four snapshots exist; three can be asked this question. Re-derive the set with
`https://web.archive.org/cdx/search/cdx?url=myfootballfacts.com/premier-league/all-time-premier-league/predictions/chris-sutton-predictions-for-premier-league-2025-26/&output=json&collapse=digest`
and fetch each as `https://web.archive.org/web/<timestamp>id_/<url>`.

Only the season opener was ever covered before kickoff, and it is the one round the archive publishes
in advance. The live 2026/27 page said the same on 27 Aug 2026: one day before the second round, it
held the first round only, **with every result already filled in**. That is the shape of the whole
finding — the archive transcribes a matchday *after* it has been played.

**Why none of the BBC evidence is a test, unlike everything else here.** This repo's habit is a
`--run-network` test that re-checks an upstream fact rather than trusting a note, and
`tests/pundits/test_live_upstream.py` does exactly that for MyFootballFacts. There is deliberately no
equivalent for the BBC: a test that fetched BBC pages on every network run would be the automated,
repeated access their terms refuse, so building one to prove they refuse it would be self-refuting.
The BBC findings are therefore prose with the URLs and the date they were checked, and
`bbc.co.uk/robots.txt` is a one-line manual check for anyone who wants to confirm them.

**So a Pundit cannot be part of a Sealed Prediction**, and the reason is worth stating precisely: not
that the calls do not exist when they are needed — the BBC publishes them days before kickoff — but
that the only source this project is *permitted* to read has not transcribed them yet. The
consequence for issue #17 is concrete: the live loop seals the models and the Market Line, and a
Pundit's column on the three-way board is filled in retrospectively once the archive catches up. Note
this is a constraint on the live loop only; it takes nothing away from the committed backfill, which
is nine complete Seasons and is what every Pundit number on the scoreboard already comes from.

**Two things that had to change to make a live page readable at all.** Neither is a workaround:

- **The backfill's size floor cannot apply.** `MIN_CALLS = 360` is right for a complete Season and
  wrong for every live page there has ever been — the 2026/27 page held ten calls. `epl.pundits.live`
  lifts the floor and keeps the ceiling, and reconciles against the corpus instead, which is a
  stronger check than a row count.
- **A live page brings Club spellings the Alias table cannot have held.** 2026/27 promoted Coventry
  and Hull into a Premier League the archive had never covered them in, so `Coventry City` and
  `Hull City` were unknown to `myfootballfacts` and every call on the page refused to resolve. Both
  are now in `epl.clubs.build.MYFOOTBALLFACTS`, and `tests/pundits/test_live_upstream.py` asks the
  Pundit source the same "has a new spelling appeared" question the ingest already asks Football-Data.
  **That check failing is the ordinary way a promoted Club arrives**, not a defect.

**A finding that confirms the freeze, and it was proven by the pipeline rather than asserted.** All
nine cached pages had drifted from what upstream serves — every one 280–450 bytes different — while
**every call on all nine parses identically**, 3,408 of them. Then `python -m epl.pundits live
--season 2025` refreshed one of them for real: `epl.ingest.cache` archived the 23 Aug bytes into
`superseded/` under their own fetch stamp, wrote the new ones, and
`tests/pundits/test_the_frozen_dataset.py` still rebuilt `predictions.csv` byte-for-byte from them.
So the archive's HTML churns continuously while its content does not — which is exactly the case for
the dataset being committed and frozen rather than re-scraped on every run (issue #11), and the
supersede rule earning its keep on a file that mattered.

## The live loop, and the input it is still waiting for

Stage 13 (issue #17) built the other half of the ledger: `epl.live` seals a Prediction Round before
its first kickoff, commits it, and scores it retrospectively once the results arrive. The code is
built and tested end to end. **What it reads is not proven, and that is the finding.**

### The Season in progress is ingested, and belongs to neither Window

`epl.windows.LAST_SEASON` moved from 2025 to 2026 — which CLAUDE.md flagged as a deliberate act
rather than a config bump, because that module is the leakage protocol. Three consequences, and the
third is the one that needed deciding.

- **A Predictor can now see the live Season.** Without it, Elo and Dixon-Coles would predict round
  two of 2026/27 having never seen round one. Not a leak — just a worse forecast that got worse
  every week.
- **A sealed Prediction can be joined to a result.** `scoreboard.scored_predictions` needs the match
  table to hold the Fixture, so retrospective scoring is impossible without the Season being
  ingested. This is acceptance criterion 5.
- **`EVALUATION_WINDOW` did not move with it**, and that is the decision. The Evaluation Window is
  2005/06–2025/26 and stays there. Folding the live Season in would mean the headline RPS grew by a
  handful of Fixtures every Saturday, so the number on the scoreboard would stop being comparable
  with the same number last week — and, worse, `python -m epl.ledger backfill` would *regenerate*
  Predictions for a Season the live loop had sealed, which is precisely the fiction ADR 0005 exists
  to prevent. So `epl.windows.LIVE_SEASON` is a third span: ingested, never fitted on, never
  backfilled, scored on its own board at `outputs/live_scoreboard.csv`.

The corpus therefore **grows every Saturday**, and that broke the shape of the corpus tests rather
than any of their numbers. Every pinned count — 52,672 matches, 42,792 below the Premier League, nine
rows missing match statistics — is now taken over the 26 *closed* Seasons, and the Season in progress
is checked for being present and partial instead. A count that included it would be a test that
failed weekly and told nobody anything.

### Superseding had to be built, because it could not be expressed

ADR 0005 says "correcting a genuine bug in a sealed round means adding a superseding row with a new
As-Of Instant, never editing history", and the *reading* side already honoured it: the audit's
duplicate check keys on `(predictor, Fixture, as_of_instant)`, and the scoreboard keeps the latest
instant per Fixture. The **writing** side could not produce such a row at all.
`schema.predictions_for` derived the As-Of Instant from the Fixture's own date, so every Prediction
for a given Fixture was stamped at the same midnight, and `live.seal` refused to write a round twice.
There was no way to correct anything.

So two things were added and both are narrow:

- `schema.predictions_for(..., as_of=...)` — the one door to a later instant, which may move forward
  and never back.
- `live.supersede(rows)` — writes a new **revision** of the round's file, `2026-08-28.1.csv` beside
  `2026-08-28.csv`, and refuses any row stamped at or before what the store already holds for that
  Predictor and Fixture. A replacement claiming the same instant is indistinguishable from the
  original having been rewritten.

**A superseding Prediction genuinely knows more than the one it replaces**, because its Evidence is
cut at its own later instant. That is uncomfortable next to ADR 0002's comparability argument, and
recording it at the round's original midnight to keep the comparison tidy would be worse: it would
be a false claim about when a Prediction was made, in the one store whose entire value is that such
claims are true. Superseding is for correcting a bug, not for refreshing a quote.

Revisions also cost `sealed_rounds` its sort: `2026-08-28.1.csv` sorts *before* `2026-08-28.csv` as
a string, so the store now sorts on the round and revision it parses out of the name.

`seal` gained the other end of its window at the same time. It already refused a round whose first
Fixture had kicked off; it now also refuses one whose As-Of Instant has not arrived. Sealing on
Thursday under Friday's midnight instant claims a moment that has not happened, and reads odds that
do not exist yet.

### Nothing in the loop knows which Predictors can speak

Stage 12's finding is that a Pundit cannot be part of a Sealed Prediction. It reaches the loop as
zero lines of code. Asked about an unplayed Fixture, `lawrenson` and `sutton` find no call in the
frozen dataset, `margin_map_*` find no call to read, and `ceiling_line` finds no closing odds because
the match has not closed — so all five answer `schema.covered` with nothing, exactly as they do for
a Season they never covered. The run *records* who was silent; it is never told in advance.
`tests/live/test_live_loop_over_the_corpus.py` registers all eight against the real corpus and pins
which four speak, so a Predictor changing its mind about unplayed Fixtures is a test failure rather
than a quiet change to what gets sealed.

### Measured at stage 13 (27 Aug 2026)

**Ingesting the Season in progress changed nothing that is scored, and that was checked rather than
assumed.** With 2026/27 in the corpus — 82 matches across the four tiers, 10 of them Premier League
— `python -m epl.ledger backfill` rewrote all nine Backtest Prediction files **byte for byte
identically**, and the scoreboard printed the same 0.1936 / 0.1975 / 0.1994 / 0.2294. The mechanism
is that the last Fixture of 2025/26 was played in May and the first of 2026/27 in August, so no
scored round's As-Of Instant reaches the new Season. That is why the separation is free *today*, and
is not a reason to rely on it: the guard is `EVALUATION_WINDOW` staying put, not the calendar.

The live loop was then run for real, and what it found is the state of open risk 2.

| Fetched | Upstream `Last-Modified` | Rows | E0 rows |
|---|---|---|---|
| 2026-08-21 06:33 UTC — round 1 kicked off that evening | not recorded | 3 | **0** |
| 2026-08-27 06:12 UTC — round 2 kicked off the next day | Tue 25 Aug 09:59 GMT | 5 | **0** |
| 2026-08-27 14:21 UTC — same day, eight hours later | Tue 25 Aug 09:59 GMT | 5 | **0** |
| 2026-08-28 15:12 UTC — **from the Pi**, the Friday of a round | Fri 28 Aug 12:01 GMT | 197 | **10** |

The fourth row is stage 16's and is discussed below; the three above it are what stage 13 measured,
and they are the reason this section reads the way it does. **Each of those three** found a file
holding only Fixtures dated on or before the day it was generated: one League One tie (`E2`,
Sheffield Wed v Bradford City) and two Spanish on the first, one National League tie (`EC`) and four
Spanish on the other two. The second fetch's file was **two days stale** —
the `Last-Modified` header says it was written on the Tuesday morning, and even that batch, generated
three days before a Premier League round, carried no Premier League row.

**The third fetch was taken at stage 15 and came back byte-identical to the second** — same md5, same
`Last-Modified`, eight hours apart on the eve of a Premier League round. That closes off the most
hopeful reading of the first two, which was that a fetch timed later in the day might catch a fresher
batch. It would not have: upstream had not regenerated the file in two and a half days, across a
matchday. The horizon is short *and* the refresh is irregular, and neither is a fetching mistake.

The `E2` row on the first fetch is worth noticing: this is not a file that omits English football.
It has carried an English tier this project ingests. Across these three fetches it had simply not
been seen carrying the one tier this project predicts.

On those three fetches, then, `fixtures.csv` was regenerated irregularly and its forward horizon at
the moment of generation was a couple of days — shorter than a Prediction Round's own sealing window.

**The fourth fetch closed it, the other way.** Taken from the Pi at stage 16 on Friday 28 Aug 2026 at
15:12 UTC — the Friday of a Premier League round, which is the timing every note above asked for —
it came back with **197 rows, 10 of them `E0`**: the entire round beginning that evening, out to a
Fixture three days later, each row carrying the `AvgH`/`AvgD`/`AvgA` market averages the Market Line
is built from. `Last-Modified` was 12:01 GMT the same morning, three hours before the fetch.

So the short horizon measured above was a property of *those files* rather than of the feed. Upstream
regenerates the rolling file irregularly; a fetch landing between regenerations sees a stale and
nearly empty file, and a fetch landing after one sees the round. **That is a sampling hazard, not an
absent forward horizon** — and it is the argument for fetching on a schedule rather than by hand,
because the schedule is what puts a fetch inside every round's window instead of whenever somebody
remembers. Three fetches could not prove the negative, and the fourth did not need to.

`python -m epl.live upcoming` remains what answers the question on any given day, writes nothing, and
distinguishes the two silences that matter: "no round is inside its window right now" and "the file
held no Premier League Fixture at all".

The consequence reaches issue #18. The stated reason for deferring an API-Football client is that
"`fixtures.csv` already carries upcoming Fixtures with the Market Line, free and unauthenticated" —
the premise that stood in doubt for three fetches and that the fourth confirmed. So the client stays
deferred, and now **for its original reason rather than in spite of it**: the case for a paid,
authenticated, key-carrying source is weaker than it was while the doubt stood, not stronger.

**Stage 14 wrote it to be updated that way.** `epl.v2.api_football` carries the table above as
`FETCHES_MEASURED`, the count that settles the question as `PREMIER_LEAGUE_ROWS_SEEN` — now `10` —
and the conditions that would revive the client as `WHAT_WOULD_REVIVE_IT`. A fourth fetch was an
append to a tuple and a one-digit edit, which is the reason the fetches are data there and prose
here. The fourth fetch falsified two of the four revival conditions — "keeps reporting no E0 rows"
and "the forward horizon stays shorter than a sealing window" — and both were removed. Two survive
untouched: the live Season Projection, which needs the rest of the campaign rather than three days
of it, and the file dropping the market-average odds columns. One was added, and it is open risk 7:
that the file is unreliable *in time* rather than absent. Three conditions now, not four.

### Added at stage 14 (the deferred-v2 stubs, issue #18)

**The stubs are Python modules, not a docs page, and the reason is testability.** Issue #18's fifth
acceptance criterion is that none of them is imported or executed by the pipeline. A Markdown file
satisfies that trivially and unfalsifiably; a module under `src/epl/v2/` satisfies it in a way
`tests/v2/test_stubs_are_unreachable.py` can check by walking every import in `src/epl`. Deleting the
directory breaks no import and moves no number, and that is now a test rather than an intention.

**Each stub carries its entry price as a `WHAT_IT_NEEDS` tuple rather than a closing paragraph.**
Criterion 4 asks that a stub say what it needs in order to be picked up, not merely that it was
deferred — and that is precisely the sentence a docstring loses first, because losing it leaves prose
that still reads fine. A named constant a test holds onto cannot be edited away quietly.

**A stub defines no function and no class, and that is checked too.** Decision 12 says three stubs
and *no implementation*. The failure mode is not someone building the ML layer by accident; it is a
helper added "while I am in there", then an import, and then a deferred feature the pipeline depends
on half of. `test_it_defines_no_function_and_no_class` is where that stops.

**The import detector reads `from epl import v2` as well as `import epl.v2`.** Recording only the
module half of a `from X import Y` would have let the most natural spelling of the violation through
the one test that exists to catch it. `tests/metrics/test_module_contract.py` has the same shape and
does not need the same care, because no module in `epl.metrics`' blast radius is a submodule of the
package being guarded — but the two should not drift, and this note is why they differ.

### What stage 13 deliberately did not build

**A live Season Projection.** CLAUDE.md expected #17 to turn the projection weekly, via
`projection_rounds(..., live=True)`. It cannot, and the reason is the same input: a Season Projection
simulates every *remaining* Fixture of the campaign, and `slate_at` says in its own docstring that
for the live Season those have to be handed in from `fixtures.csv`. A file whose horizon is two days
cannot supply the rest of a season. This is blocked on the same missing source as the seal itself,
and is worth an issue rather than a workaround.

**A Pundit column on the live board.** Filling it in retrospectively means folding the Season in
progress into `predictions.csv`, which is committed and frozen precisely so that "frozen" cannot come
to mean "whatever was written last" (issue #11). The right moment is when the archive's 2026/27 page
is complete, not weekly.

**A schedule.** Issue #17 asks that the scoreboard maintain itself "without manual intervention", and
the loop is built for that: both write commands are idempotent inside a round, `score` refreshes the
Live Season and rebuilds the match table itself, and a second run of `seal` in the same round is a
no-op with exit 0. No cron entry or workflow is committed, for two reasons. There is nothing to seal
yet — a schedule would fire weekly onto a file with no Premier League row in it — and a schedule that
fetches from upstream and pushes commits to this repository is a decision for whoever owns it.

**A correction after kickoff.** `supersede` refuses a round whose first Fixture has kicked off, which
is worth stating because that is when a bug is most likely to be *found*. The refusal is the rule
rather than a limitation: this store holds what was forecast before kickoff, so a row written after
it could not be evidence of that whatever it said. The sealed row stands and is scored as made, and
the fix goes into the code so the next round is right. A track record that could be tidied up
afterwards would not be a track record.

## The schedule, and where it runs

Issue #19 asked for two decisions to be made deliberately rather than coded around, and required the
reason to be recorded here — "in particular whether an automated push to this repository is
acceptable". Both were made by the repository owner on 27 August 2026.

### Where it runs: a container on a Raspberry Pi, not GitHub Actions

**Rejected: GitHub Actions.** It fires whether or not a machine at home is awake, which is the one
thing that genuinely matters for a deadline that cannot be recovered — a round whose window has shut
is gone, and `supersede` refuses a round after its first kickoff on purpose. Against that: every run
would rebuild the conda environment and re-fetch the whole raw cache, because `data/raw/` and
`data/processed/` are gitignored and `score` rebuilds the match table from all 108 files. That is a
cold ingest of the entire corpus twice a week, in perpetuity, against a free file host that asks for
nothing in return.

**Chosen: a Raspberry Pi 5 the owner already runs, with the loop in Docker.** The raw cache lives on
it and persists, so a fire is a few files rather than a hundred. The cost accepted in exchange is
real and should not be glossed: **if the Pi is off on a Friday afternoon, that round is lost and
cannot be sealed later.** That is a worse failure mode than Actions', and it was chosen anyway,
because at the time the input this loop waits for had never been seen, and a schedule firing onto an
empty file loses nothing at all. **That reasoning has since expired.** `fixtures.csv` carried a full
round on 28 Aug 2026 and the loop sealed it, which is the condition this paragraph said to revisit
on — so the cost is now live rather than hypothetical. It is being carried rather than escalated,
on measured grounds; see open risk 6.

**Why a container, on a machine that already runs Python on bare metal.** The Pi's other tenant is a
paper-trading loop installed as a venv of prebuilt aarch64 wheels, chosen that way specifically so
that nothing on the box needs a toolchain. This project cannot join it there: ADR 0009 chose
conda-forge for the prebuilt scientific stack, and `environment.yml` pins the BLAS provider because
the wrong one aborts the interpreter on every LAPACK call. Two package managers competing for one
machine's system libraries is the failure the image exists to prevent, and it is the only reason it
exists. The two projects therefore share the Pi's crontab and a log-block convention and nothing
else — deliberately not an interpreter, not a virtualenv, and not a process.

The image is the *environment*; the repository is bind-mounted over it. So `git pull` updates the
code with no rebuild, and a sealed round lands in a real checkout rather than inside a container
layer that evaporates. That last part is not convenience: `outputs/live/` is evidence (ADR 0005),
and evidence written somewhere nobody can inspect is not evidence.

### An automated push is acceptable, and a push that fails is loud

**Yes.** The loop commits and pushes to `origin` unattended, over a deploy key, on `seal --push`.

The reasoning is ADR 0005's rather than a convenience. A commit proves *when* a Prediction existed
to anyone who can reach the machine holding it. On the desktop that was everyone who mattered; on a
Pi in a cupboard it is nobody, and the claim collapses to "trust the box". So an unattended loop that
does not push has not finished sealing anything, and `epl.ledger.live.push` sits beside `commit` for
the reason that module already gives for `commit` being there: one module knowing about git is what
makes "git history is the proof" checkable in one place.

Pushing is **opt-in** (`--push`, off by default). An outward-facing act on somebody's repository is
the schedule's to ask for, not a default a person at a keyboard should discover by accident.

A push that fails exits 1 and says `NOT PUSHED`, as loudly as a seal that could not be committed,
because the sealed file on disk looks identical either way. The 18:30 fire retries it: the round is
already sealed by then, so pushing is the whole of what is left for that run to do — which is
exactly what makes a second fire worth scheduling, since the run that seals a round is the run that
may have been unable to reach the network.

### When it fires: 16:00 and 18:30 UK, Tuesday and Friday

The window opens at the round's As-Of Instant — midnight at the start of its anchor day — and shuts
at its first kickoff. Football-Data samples the pre-match odds on the *afternoon* of that day, so
both ends are real: fire at midnight and the odds the Market Line needs do not exist yet; fire after
the first kickoff and the store refuses the round outright. 16:00 is after the sample and comfortably
before a 19:45 or 20:00 first kickoff.

18:30 is a retry and is free, because sealing is idempotent inside a round. `score` runs at 06:00 on
the same two days: a Tuesday round is played Tuesday to Thursday and a Friday round Friday to
Monday, so each morning fire scores the round that has just finished, hours before the next is
sealed.

### Two things the schedule forced into the code

**The clock was reading the wrong country, and had been all along.** `epl.live.__main__.clock`
returned `pd.Timestamp.now()` — the *machine's* naive local time — and every instant it is compared
against comes off Football-Data in UK local time. On the desktop this project was built on, which is
eight hours ahead, a Friday afternoon inside a round's window reads as Saturday morning and the round
is refused as having kicked off; a container defaulting to UTC gets the British Summer Time hour
wrong in the other direction. Neither says anything when it happens — the loop simply seals nothing,
forever. `epl.ledger.live.uk_now` converts explicitly and both the store's defaults and the command
line now read it.

Measured on the build machine while fixing it, 27 Aug 2026: `pd.Timestamp.now()` returned
**2026-08-28 00:17** where the UK read **2026-08-27 17:17**. Not an offset — a different *day*, and
the wrong side of a Friday round's midnight As-Of Instant. The old code would have judged Friday's
window open on Thursday evening and sealed a round under a moment that had not happened, reading
odds Football-Data had not yet sampled. That is the early end of the window (`NOT_OPEN`) being
defeated by the clock rather than by anybody's decision, and it is the more dangerous of the two
directions: sealing nothing is visible eventually, and sealing something under a false instant is
not visible at all. It is a latent bug the loop had while it was run by hand from one desk, and a
schedule is what made it certain. **`clock` is still a function and still not a `--now` flag**
(criterion 7); `tests/live/test_unattended.py` fails if either changes.

**"Nothing to seal" had to stop being a failure.** `seal` exited 1 whenever no round could be
identified, which is correct for a person and ruinous for a schedule: no round is inside its window
most of the week, and until upcoming Fixtures have a source it is *every* fire. A job that goes red
twice a week for a season is a job whose owner stops reading it, and the next thing they do not read
is the one that mattered. `epl.live.upcoming.NothingToSeal` now separates the two silences that need
nobody — an empty file, and a clock outside every window — from the refusals that do: a rolling file
that changed shape, and a `LIVE_SEASON` gone stale in either direction. The first pair exit 0, the
second exit 1. That distinction is drawn by an exception type rather than by reading messages,
because a message is not a contract.

An upstream shape change stays an uncaught `IngestError` with its traceback, deliberately. The shape
of the rolling file is `epl.ingest`'s to complain about, and widening `epl.live`'s handler to catch
it would put "upstream changed" behind the same exit code as "it is Wednesday".

## Bringing it up on the Pi, and the first Sealed Prediction Round

Issue #21. Its deliverable was evidence rather than code: `deploy/` had been written, reasoned about
and statically checked, and never once executed. This is what happened when it was, on 28 August
2026, on a Raspberry Pi 5 (8 GB, NVMe, Raspberry Pi OS, system zone `America/New_York`).

### The aarch64 build was the named risk and was not a risk at all

The conda solve on `linux-aarch64` was the one step nothing could prove from an x86-64 machine, and
the amd64 build had made it look expensive.

| | amd64 desktop, 27 Aug | **Pi 5, 28 Aug** |
|---|---|---|
| `conda env create` | 1450 s | **73 s** |
| whole build | ~27 min | **271 s** |
| image | 6 GB | 5.76 GB on disk, 1.3 GB content |

`environment.yml` needed **no aarch64 exception**. conda-forge resolved it in about seven seconds
and every pin held, OpenBLAS included, so ADR 0009 is now measured on both architectures rather than
one. The desktop's 1450 s was almost certainly its VM filesystem rather than anything about x86-64.

**The consequence is that SETUP.md's advice to cross-build was wrong and has been removed.** It
recommended `docker buildx --platform linux/arm64` on a desktop and `docker save | ssh docker load`,
on the strength of that 27-minute figure. The Pi was six times faster than the machine it was being
told to borrow, so the advice traded a four-minute native build for a multi-gigabyte image transfer
plus QEMU. This is the clearest case in the project of a number measured on the wrong machine
turning into guidance.

### Three defects that only a real run could find

**`deploy/run_live.sh` was committed `100644`.** Every clone landed it non-executable, so all three
crontab entries would have failed with **exit 126** — a schedule that looks installed, fires on time
and does nothing, wrapped in a `===== RUN =====` block containing one line of `Permission denied`.
It had been that way since stage 15 and nothing looked. It was invisible from the development
machine because `core.filemode` is `false` on Windows, where the checkout reports `rwxr-xr-x`
whatever the tree holds — which is why `tests/deploy/test_the_schedule_is_runnable.py` asks
`git ls-files -s` rather than the filesystem. A test against the working tree would have passed on
the machine where the mistake is made and failed only on the Pi.

**ssh in the container was reading the wrong home directory, and this was the one real bug.** It
surfaced at SETUP.md step 7 — the push proof — with the image built, the corpus ingested, the clock
right, the round sealed and every other check green. `git push --dry-run` inside the container
answered `Host key verification failed`, which is step 7's own first troubleshooting row, whose
stated cause is a skipped `ssh -T`. That was not the cause: `ssh -T git@github.com` had been run on
the host and had authenticated.

**OpenSSH resolves `~` from the passwd database for the running uid, not from `$HOME`.**
`docker-compose.yml` mounts the key at `/home/epl/.ssh` and sets `HOME=/home/epl`, reasoning in a
comment that `user:` "gives the container a uid with no passwd entry, and HOME then defaults to
`/`". That premise is false for this base image: `condaforge/miniforge3:latest` carries a real
`ubuntu` user at uid 1000, homed at `/home/ubuntu`. So ssh looked in `/home/ubuntu/.ssh`, found no
`known_hosts`, and refused the host key — while the correct file sat mounted, readable, correctly
owned, one directory away. Every diagnostic pointed at the mount, and the mount was perfect.

The fix is `GIT_SSH_COMMAND` naming the key and `known_hosts` outright rather than relying on any
`~` expansion, which also makes it independent of a floating base image that is free to add or
rename a uid-1000 user between builds. Setting `HOME` is kept, because it is still right for the
case the original comment described. **This is the argument for step 7 existing as its own step**:
nothing else in the runbook exercises the key, and without it the failure would have surfaced on the
first Friday a round needed pushing.

**`CRON_TZ=Europe/London` is ignored by Raspberry Pi OS's cron**, which `deploy/crontab` had
asserted that Debian's honours. Measured rather than assumed, because grepping the documentation
proves nothing: an entry at `16:29` under `CRON_TZ` did not fire at 11:29 EDT, and a control entry
at `11:32` fired at 11:32 EDT exactly. Cron read the system zone and nothing else.

**The old comment's assessment of that failure was wrong in a way worth keeping a record of.** It
said the consequence was mild — "the loop would fire outside the window and exit 0 having sealed
nothing". That is true of a schedule that runs *early* and false of one that runs *late*, and five
hours behind the UK is late: uncorrected, `18:30` fires at 23:30 London, past a 20:00 first kickoff,
and `supersede` refuses a round after its first kickoff on purpose. The round is not deferred, it is
gone. What remains true is the other half — the window is judged by `epl.ledger.live.uk_now`
whatever cron believes, so a wrong hour cannot seal something *wrongly*.

The fallback applied was to convert the three times into the Pi's own zone (06:00/16:00/18:30 UK to
01:00/11:00/13:30 EDT) and delete the `CRON_TZ` line, so that a future cron which starts honouring
it cannot shift already-converted times a second time. **SETUP.md's other suggested fallback — set
the whole Pi to `Europe/London` — was rejected and should not have been offered first.** This Pi's
other tenant is a paper-trading loop whose own crontab is written around the US market open, and
moving the system zone would silently reschedule somebody else's production job.

### The clock, checked on a machine where it could actually be wrong

Inside the container `uk_now()` read **16:13 BST** while the Pi's own `date` read **11:13 EDT**. The
amd64 run could only show these agreeing, because that desktop was effectively in UK time; here the
host is five hours behind and the container's `TZ=Europe/London` is doing real work. This is the
concrete version of what `tests/live/test_unattended.py` documents in `TestWhichClock` — and note
that inside the container `pd.Timestamp.now()` and `uk_now()` still return the same value, which is
exactly the vacuousness that test derives its expectation from UTC to work around.

### And then the thing nobody expected: it sealed

Every document in this repository said the loop would seal nothing, and named the reason — open risk
2. The first `upcoming` run from the Pi, at 15:12 UTC on the Friday of a round, returned **10
Premier League Fixtures** and reported `sealable now: 2026-08-28`. See open risk 2, now closed.

The round was sealed by hand before the schedule could reach it, deliberately, so that the first
write into an append-only store was watched rather than unattended. What it wrote:

- **40 Predictions over 10 Fixtures from 4 Predictors** — `market_line`, `naive_baseline`,
  `dixon_coles`, `elo` — into `outputs/live/2026-08-28.csv`, as-of `2026-08-28T00:00:00`.
- **5 Predictors silent**: `ceiling_line`, `lawrenson`, `sutton`, `margin_map_lawrenson`,
  `margin_map_sutton`. Exactly the five stage 12 predicted, each answering `schema.covered` with
  nothing for its own reason, through zero lines of code that know which they are.
- `market_line` recorded `inputs_seen = 0` and the models 52,754, which is the receipt ADR 0005 asks
  for: a Predictor that consumes no history has no history to leak.
- `latest_input` was `2026-08-24T20:00:00`, strictly before the As-Of Instant.
- `python -m epl.ledger audit` answered **"both stores audit clean; nothing under outputs/live/ has
  been rewritten"**.

The rows were inspected before the write, through `seal.sealed_predictions`, which returns what
would be sealed without touching the store — the property `run(commit=False)` documents and the
reason it exists. Dixon-Coles gave Manchester City 0.587 away at Crystal Palace against the Market
Line's 0.576; the two disagree by more on Bournemouth v Everton, 0.452 against 0.456 with Elo at
0.560. Nothing about the numbers needed defending, which is the only reason it was allowed to write.

### The schedule fired, unattended, and pushed

The last thing #21 asked for was a *real* scheduled fire rather than a hand-run command. It came at
**13:30:01 in the Pi's zone on 28 Aug 2026 — 18:30 UK**, the converted retry slot, with nobody at a
terminal:

```
===== RUN 2026-08-28 13:30:01 -0400  (seal --push) =====
rolling file: fixtures_20260828T173004Z.csv
2026-08-28: 10 Fixtures, as-of 2026-08-28T00:00:00, first kickoff 2026-08-28T20:00:00
2026-08-28 is already sealed — nothing to do
  a genuine correction goes in with --supersede (ADR 0005)
pushed to origin/main
===== END  2026-08-28 13:30:08 -0400  (exit 0) =====
```

Seven seconds, and it settles more than the criterion asked. **The converted times are right**: an
entry written `30 13` fired at 18:30 UK, which is what the `CRON_TZ` fallback was for and the only
way to confirm the arithmetic. **The deploy key works unattended**, through `GIT_SSH_COMMAND` and a
read-only mount, from cron's environment rather than an interactive shell — the push proof at step 7
was a `--dry-run` by a human, and this was not. And **the retry did precisely what it is scheduled
for**: it found the round already sealed, wrote nothing, added no commit, and pushed anyway, because
pushing is the whole of what is left for a second fire (`deploy/crontab`, "WHY TWICE").

It also took a fifth fetch of `fixtures.csv` on its own, three hours after the fourth, and that one
still held the round. Fetches the schedule takes are not appended to
`epl.v2.api_football.FETCHES_MEASURED` — the log is their record, and the tuple is for fetches taken
by hand that tell you something the log does not.

**All eight of #21's acceptance criteria are now met.** What is not proven, and cannot be until it
happens, is a fire that has to seal a round *from cold* on a schedule — every seal so far has been
inside a window a person was watching.

## The Telegram bot, and the silence it was built to read

Stage 17, issue #20. `src/epl/bot/`, and a second and third service in `deploy/docker-compose.yml`.

### The ticket asked for a different thing than it was written for

Its first acceptance criterion, when it was filed, was that **nothing announces the day the answer
changes** — the day `fixtures.csv` first carried a Premier League row. That day was 28 Aug 2026 and
it has been and gone, noticed only because a person happened to be running the loop by hand while
deploying it. The criterion was rewritten before this stage was built, and what replaced it is
narrower and recurs:

> A quiet exit 0 is correct for the loop and is not enough for a human: distinguishing "no Premier
> League football this week" from "upstream served us a stale file through both fires" is the single
> most valuable thing this bot does.

That is open risk 7, and it is worth being exact about what can and cannot be known. **Nothing can
tell an empty week from a lost round from one fetch.** The rolling file says what it says. What
*can* be known is whether upstream regenerated the file between the two fires, and that is the same
question in a form that has an answer: every fetch is cached under the instant it was taken
(`epl.ingest.fixtures.raw_fixtures_path`), so two fires naming two copies whose bytes match saw a
file upstream had not touched across the whole window. `epl.bot.watch.stale_upstream` is that
comparison and nothing more.

The ticket suggested `Last-Modified` for this. Bytes were used instead, because the fetcher does not
record response headers and the cached copies are already on disk — the measurement that closed open
risk 2 was itself "two fetches, byte-identical eight hours apart". Adding header capture to
`epl.ingest` to answer a question the bytes already answer would have put a change in the ingest for
the benefit of a chat app.

### Open risk 6 is reported retrospectively, and that limit is stated rather than glossed

A Pi that is off writes no `===== RUN` block. It also runs no bot. So **this is not a dead man's
switch and cannot be one on a single machine**: a process that is not running cannot report that it
is not running, and closing that gap needs a second host — which is the same argument
docs/DECISIONS.md already weighed and lost when it chose a Pi over GitHub Actions.

What is buildable is a bot that, when it comes back, says which anchor days went by unfired.
`epl.bot.watch.absent` does that, bounded at both ends: measured from the first fire in the log, so
a fresh clone is not reported as an outage, and limited to the last fortnight, so a bot restarted
after a month away reports the part its reader can still act on.

The days it expects are **Tuesdays and Fridays from `epl.rounds`**, not from `deploy/crontab`. The
schedule fires on those days *because* a Prediction Round anchors to one, and reading cron would
make the check agree with the schedule rather than with the thing the schedule was derived from — a
hand-edited crontab being exactly the case where those differ and somebody should hear about it.

### Which half says what, and why the split is not arbitrary

The push half (`epl.bot.notify`, fired by `deploy/run_live.sh`) speaks about the fire that just
happened: a round sealed, a round scored, a loud failure, and a stale upstream file. The pull half
(`epl.bot.serve`, long-lived) speaks about the fires that *did not* happen, and dedupes.

That is not tidiness. A notifier is started by a fire, so "there was no fire" is not something it
can observe; and open risk 6's finding covers a fortnight, so a notifier reporting it would repeat
the same gap three times a day until it scrolled out of the lookback. Nobody reads that channel by
November, which is the failure issue #19's exit-code contract was written against, arriving by
another route.

**Most fires say nothing, and the tests are mostly about that.** Two of the three crontab lines are
`seal --push` and the second is a retry designed to find the round already sealed.

### It decides from artefacts, not from the loop's prose

A `seal` that sealed leaves a file under `outputs/live/` newer than the fire; a `score` that scored
leaves `outputs/live_scoreboard.csv` newer than the fire. Both are facts on disk. The alternative —
matching on "sealed 40 Predictions from 4 Predictors" — would make a reworded log line silence the
bot, which is the one failure a notifier must not have.

Three lines of the loop's output *are* read, and all three are now named constants in
`epl.live.upcoming`: `NO_FIXTURE_TO_PREDICT`, `NONE_INSIDE_A_WINDOW` and `ROLLING_FILE_PREFIX`. The
first two are the whole of what distinguishes the two silences, and they are constants so that
rewording a message cannot quietly blind the only thing watching for the risk. `ROLLING_FILE_PREFIX`
moved out of `epl.live.__main__` and into `upcoming` at the same time: a library module importing a
`__main__` is the wrong direction, and it dragged the ingest's fetchers into the bot's process.

### Read-only, checked by walking the source rather than by promising

Issue #20's eighth criterion is that the bot cannot seal, supersede, backfill or score. A chat app
must not be a second door into `outputs/live/` (ADR 0005): a Sealed Prediction is evidence because
the loop wrote it before kickoff under a moment nobody could choose, and a message from a phone is
the shortest route in the system to a row with none of those properties.

`tests/bot/test_the_bot_is_read_only.py` checks it the way `tests/v2` checks its own claim, and is
**precise about what that proves**. Nothing makes a Python process incapable of calling a function
in a module something else already imported, and plenty is already imported — `epl.bot.answers`
reads the sealed store through `epl.ledger.live`, which is where `seal` and `supersede` live. What
is checked is that **no file under `src/epl/bot/` names a writing door**, by import *or* by call.
The call sweep matters and the import sweep alone would miss the realistic mistake: `from
epl.ledger import live as store` is a correct import, and `store.seal(rows)` beneath it would be a
write with nothing forbidden in the file's imports at all.

### The numbers it may not say, made structural where possible

Issue #20 carries a list headed "Things the bot must not say", and insists these are not style
notes. They are enforced in `epl.bot.answers`, which is the one place a measurement becomes a
sentence:

- **No calibrated Live Season figure.** `live_board` selects `PRE_CALIBRATION_COLUMNS` — the same
  two lines `epl.live.__main__._score` uses, rather than a second policy that could drift from it.
- **No RPS from a handful of Fixtures as a track record.** The sample size is in the same message,
  always, beside the Evaluation Window's 21 closed Seasons.
- **Never the sequential diagnostic.** Nothing in the package imports it, which the sweep above
  enforces; the word-absence check in `test_answers.py` is the weaker half.
- **Never accuracy as the headline.** Boards render in `epl.ledger.scoreboard.METRICS` order, from
  that tuple rather than a second list, so RPS is first because the project says it is.
- **Never a Predictor without its `note`.** Read off the registry by name for every row of every
  board, so the Ceiling Line's caveat and a Pundit's travel with them, and a Predictor registered
  tomorrow is covered with no change here. No branch per Predictor, the same as the scoreboard.

### The Evaluation Window's board is refused rather than recited, and that is the answer

`outputs/scoreboard.csv` is derived and regenerable, so it is gitignored (ADR 0005's reasoning), and
rebuilding it needs `outputs/backtest/`, which is written by the one command `deploy/crontab` must
never schedule. **So the Pi has no Evaluation Window board**, and `/board` says so and names the two
commands rather than printing the README's numbers. A bot reciting them would be reporting a
measurement it does not hold, on a machine where it may since have changed. When the file *is*
there, its modification time goes in the message: a derived file carries no date, and this one is
regenerated by hand.

### No Telegram library, and no event loop

The Bot API surface used is four methods over HTTPS — `getMe`, `setMyCommands`, `getUpdates`,
`sendMessage` — and `requests` is already in `environment.yml`. An async framework would put an
event loop inside a package whose every other module is synchronous, and would add a dependency to a
conda solve where two free version choices already break the build outright (ADR 0009's BLAS pin,
and `arviz <1`). `epl.bot.api` is the cost; the gain is that every answer in the package is a plain
function returning a string, testable with no network, no token and no event loop.

Sending fails soft and polling does not, and the asymmetry is deliberate. A notify failure must
never break the run that triggered it, so `send` swallows everything and returns a bool. Polling is
the bot's own main loop, and one of its failures — `Conflict` — must stop it dead: **two pollers on
one token do not conflict visibly.** Telegram hands each update to whoever asked first, so a
forgotten instance steals half the replies and the half that arrives makes the bot look like it is
working. There are two guards because they catch different things: an OS lock catches a second
process on this machine before it polls at all, and the 409 catches one on a machine no lock can
see.

### A compose service rather than a systemd unit, and no deploy key

Issue #20 asks for the reason to be recorded either way. **The image is the environment.** A systemd
unit on the host would need its own route to conda-forge's stack with the pinned BLAS provider,
which is the precise thing the container exists to stop two package managers fighting over on a Pi
that already runs a venv-based tenant. `restart: unless-stopped` also does what its name says across
a reboot, and leaves a bot switched off on purpose switched off.

The bot does **not** mount the deploy key. The ssh mount is on the `live` service alone: the bot
cannot push, and it is the only long-lived process here and the only one taking input from outside
the machine.

### What is not built, and is not an oversight

- **No live Season Projection command.** It cannot be built at all (see open risk 7's entry), and a
  `/projection` that answered from a stale file would be worse than its absence.
- **No Pundit column on the live board.** Empty by design, filling in retrospectively (issue #16);
  the absence is explained in words where it would appear, because a blank cell reads as a bug.
- **No state file.** The push half needs none — every decision it makes is a comparison against a
  file's modification time — and the pull half's deduplication is in memory, lost on a restart. That
  is the right way round: a restart is when open risk 6 is most likely to be true, since the
  commonest reason this process stopped is the machine having stopped.

### What is untested until it runs on the Pi

The same gap stage 16 left, one layer up. Every message, every silence and every refusal here is
tested against a fake transport; **no message has been sent to Telegram from this repository**, and
the things that can only fail on the real thing are the ones deployment always breaks: a token
pasted with a trailing newline, an allowlist read from a `deploy/.env` compose does not load, a
container that cannot resolve `api.telegram.org` on the Pi's bridge network. `python -m epl.bot
check` exists for exactly that, and sends nothing — proving the wiring by sending a real message is
the worst available way to discover it is wrong.

## The pre-match message, and a third store

Issue #20's bot was built as a monitor: it answered "did the schedule fire?" and printed boards. It
is now also read by somebody who wants to know who is going to win, and that reader was asking for
two things it did not do. Every message is reformatted for them; the second thing needed a decision.

### The reader asked for a fresh number before every match, and that number cannot be a Sealed Prediction

The request was a message about an hour before each kickoff carrying the odds on that one match. The
obvious implementation is to re-read the row already sealed for the round, and that would have been
safe, honest and slightly disappointing — the sealed number never changes between Friday and Sunday,
so eight of the ten messages in a weekend would be a re-run of the same forecast.

The repository owner asked for a **fresh run** instead, and the reason it is worth building is not
convenience. A Prediction Round anchored to Friday is played Friday to Monday. A forecast recomputed
at four o'clock on Sunday has seen Friday's and Saturday's results; the sealed one had not. That is
a genuinely larger information set, and the numbers really do move on it.

It is also exactly why such a number **cannot go in `outputs/live/`**. In that store a later As-Of
Instant for the same Predictor and Fixture means one thing and one thing only: a superseding
revision correcting a bug (ADR 0005). The scoreboard keeps the latest instant per Fixture. So a
Reading written there would silently replace the honest before-the-round forecast with one taken
after two results were in — on every Fixture, every week — and the live track record would improve
for a reason no reader could see and no audit would flag. That is not a variant of ADR 0005's
failure mode; it is that failure mode, arriving through a door the store did not previously have.

### So there are three stores now, and the third is not regenerable either

`outputs/backtest/` is gitignored because a Backtest Prediction is reproducible: rerun the pipeline
and it comes back identical, which is what makes it disposable. A **Pre-Match Reading** (CONTEXT.md)
is not. The corpus it was cut from has grown since it was taken, so re-running the same code
tomorrow gives a different number and there is no way back to the old one. Evidence that cannot be
regenerated is evidence, so `outputs/prematch/` is committed and pushed — beside the sealed store
rather than inside it.

Three rules keep it from becoming a fourth way to write a Prediction:

- One file per calendar day, appended to as that day's matches come up. Named after the kickoff day
  rather than the round, because a round spans four afternoons and four separate fires.
- The same row schema as both other stores, so `epl.ledger.schema.audit` applies unchanged and a
  Reading that had somehow seen its own Fixture's result would be refused by the existing check.
- **Nothing in it reaches the scoreboard.** `epl.ledger.stored` concatenates the backtest and sealed
  stores and does not know this one exists. Checked rather than asserted:
  `tests/live/test_prematch.py` scores the Live Season with Readings on disk and with none, and
  compares the two boards.

The second fire of a pair is quiet because the store answers the question — a Fixture already read
is no longer due — rather than because a marker file somebody has to keep in step says so. Same
argument as `epl.ledger.live.is_sealed`. And where the sealed store insists a later row supersede an
earlier one, this store keeps the **first** Reading and drops the second: there a later row is a
correction somebody made deliberately, here it is the schedule doing its job twice, and the first
Reading is the one that was actually sent.

**Whether a Reading actually beats the Prediction it was taken after is an open question**, and
deliberately not answered here. It needs a season of them. The store exists so that it *can* be
asked later; measuring it is its own ticket.

### The schedule fires forty times harder, and had to get cheaper to match

`prematch` runs every half hour from 11:00 to 22:00 against a window of 45 to 75 minutes before
kickoff. That cadence and that window are one decision: every Premier League kickoff falls on a
quarter-hour, so each is inside at least one fire's window — 20:00 caught at 19:00, 17:30 at 16:30,
12:30 at 11:30 — and most are inside two, which the store's dedupe absorbs.
`tests/live/test_prematch.py` checks the cadence against the window rather than trusting the
arithmetic in this paragraph.

Most of those twenty-two daily fires have nothing to do, so the expensive half sits behind a cheap
one. `epl.live.__main__._prematch` asks the sealed store first — one small file, no network, no fit
— and exits 0 having printed one line. Only a fire with a Fixture in its window pays for a refresh
of the Live Season, a rebuild of the match table and a run of every Predictor.

Two consequences worth writing down. It **shares the loop's flock**, deliberately: two containers
committing into one checkout is a corrupt index, so a prematch fire landing inside a slow `score`
stands down, costing at most one of the two chances a Fixture gets. And it writes to its **own log**,
`deploy/logs/prematch.log`, because `live_loop.log` is the file a person opens when something has
gone wrong and it holds three fires a week; burying those under a thousand lines saying nothing
kicks off within the hour would cost that file the thing it is for. Both logs share a format and a
parser, and `/health` reports the last fire of each.

### The messages themselves, and the three ways formatting fails silently

Every message is now composed through `epl.bot.render`, which owns the markup, and sent as Telegram
HTML rather than plain text. That reverses stage 17's choice, for stage 17's own stated reason:
plain text was chosen because a Markdown parse error on a Club name with an apostrophe is a message
that does not arrive. HTML needs three characters escaped where MarkdownV2 needs eighteen,
`html.escape` is mechanical rather than a judgement made per string, and `api.Telegram.send` now
retries once as plain text if Telegram refuses to parse — so the failure lands on the formatting
rather than on the delivery. Going quiet is still the thing that must not happen.

The parse mode buys one thing: a fixed-width block, so a column of probabilities lines up. That
introduces the one new way to be wrong, and it is invisible from every machine this is developed on.
**A line too wide for a phone wraps, and a wrapped line inside a fixed-width block has lost the
alignment the block existed for.** `render.PRE_WIDTH` is a 44-character budget and
`tests/bot/test_render.py` applies it to every message the bot can send, with exactly one exemption:
`answers.failure`, which quotes the loop's own output and must not re-wrap somebody's log. The
exemption is named in the test rather than left as a message the sweep quietly does not cover.

The same sweep also enforces that every message is pure ASCII. The repository owner asked for no
emoji; ASCII is the checkable form of that, because a bullet character is not an emoji and renders
as a box on the devices that lack the glyph. Prose written elsewhere in the repository — a
Predictor's `note`, quoted whole, and the loop's own output — passes through `render.asciify`, which
maps a closed set of typographic punctuation and never touches a word.

Two smaller decisions made for the same reader. Club short names live in `epl.bot.render` and not in
`clubs.csv`, because `epl.clubs.build` regenerates that table from its own mapping and would erase
them, and because what to call Wolverhampton Wanderers in a chat window is a fact about the chat
window. They are derived by a rule — drop a droppable suffix — with a collision check that reverts
to the canonical name, so Bristol City and Bristol Rovers keep their full names rather than both
becoming "Bristol". And three probabilities are rounded by largest remainder so they always sum to
100: a card reading 59 / 23 / 19 invites a question whose honest answer is "three separate
roundings", which is not something anybody wants in a message about football.

### What the commands became

Five commands became ten, and two of the five were renamed rather than kept. `/round` printed every
Predictor's three probabilities under every Fixture, which is the table that made it unreadable; it
is now `/week`, one line per match. `/live` is now `/record`. Both old names survive as aliases,
because they are in one person's muscle memory and in this repository's own documentation, and a bot
that answers "No such command" to the name it used last week looks broken rather than tidied.

The detail `/round` used to carry did not disappear, it moved. A single-match card has room for it,
so `/next` and `/club` list every other Predictor that spoke on that Fixture beneath the model and
the market. That keeps issue #20's first criterion — what each Predictor said — met, without putting
a four-column table under ten Fixtures.

`/disagree` is the one command with no equivalent anywhere else in the project: the Fixtures where
the model and the Market Line are furthest apart. The whole scoreboard is an argument about whether
the model can keep up with the market, and this is where it is currently saying something different.
`/board` survives unchanged and still refuses on the Pi, which is its correct answer.

## Open risks

1. ~~**BBC live scraping is unproven.** `www.bbc.co.uk` was unreachable during design, article URLs are opaque IDs (`/sport/football/articles/cvg0e92ezz4o`, legacy `/sport/football/28859459`) and there is no index page. Needs a spike at stage 5. If it fails, live pundit data has no confirmed source — MyFootballFacts' update latency during a season is unknown.~~ **Closed at stage 12, and the answer is no.** Both halves were tested for real on 27 Aug 2026 and both came back differently from the way the risk was written. See "The BBC spike" below: the BBC is *reachable* and its articles are machine-readable, and it is nonetheless **unusable**, because its terms forbid the thing this ticket would build; MyFootballFacts is permitted and is the source, and its measured latency means **a Pundit cannot be part of a Sealed Prediction**. That last sentence is a constraint on issue #17 rather than a gap left open.
2. ~~**`fixtures.csv` has never been seen carrying a Premier League row, and that is now the only unproven link in the live path.**~~ **Closed at stage 16, in the affirmative.** On 28 Aug 2026 at 15:12 UTC the fourth fetch ever taken — the first from the Pi, on the Friday of a Premier League round — returned **197 rows, 10 of them `E0`**: the whole of that evening's round out to a Fixture three days later, each carrying the `AvgH`/`AvgD`/`AvgA` market averages. The round was sealable, and was sealed. So the two-day horizon measured at stage 13 was a property of *those three files* rather than of the feed: upstream regenerates the rolling file irregularly, and a fetch landing between regenerations sees a stale and nearly empty one. **What replaces this risk is narrower and is now open risk 7** — the file is reliable in shape but not in *time*, and a round is lost if every fetch inside its window happens to land on a stale copy. Everything else this risk named had already closed at stage 13: `mmz4281/2627/E0.csv` parses, `LIVE_SEASON` is ingested and in neither Window, and `epl.live` seals, refuses to rewrite, supersedes and scores — all tested end to end. Issue #18's premise is confirmed rather than in doubt, so `epl.v2.api_football` stays a stub for its original reason and now carries `PREMIER_LEAGUE_ROWS_SEEN = 10`.
3. ~~**MyFootballFacts parseability is unverified.** Content correctness was confirmed — a 2025/26 result cross-checked exactly against Football-Data — but the HTML has not been parsed across all nine season pages.~~ **Closed at stage 7.** All nine parse, yielding **3,408 calls** of a possible 3,420, and the cross-check went far past the one row the ticket asked for: the page prints the real score beside every call, and **3,402 of the 3,406 that carry one match Football-Data**, with the four exceptions named above. The HTML is hand-maintained and reads like it — annotated names, six misspellings, two Scorelines dropped inside a Club's name, and which table holds the predictions moving between pages — so the parser recognises a call by its shape rather than by where it sits, and refuses a page that yields fewer than 360 or more than 420. `tests/pundits/test_over_the_corpus.py` re-derives all of it.
4. ~~**Cross-tier Elo has no burn-in before 2000/01**, so early ratings linking E0 to E3 will be unreliable.~~ **Closed at stage 5.** Measured: by the first scored Prediction Round the thinnest Premier League rating rests on **190 matches**, and every Club promoted into the Premier League in every scored Season arrives with a distinct rating built from more than 200. The cold start is real and is confined to 2000/01, which is why that Season warms the ratings and is not fitted on either. `tests/models/test_elo_over_the_corpus.py` re-derives both numbers.
5. **Frozen hyperparameters will drift out of date** by the late Evaluation Window, given the measured decline in home advantage. Accepted deliberately; see ADR 0008.
6. **A round whose sealing window passes while the Pi is off is lost, and cannot be recovered.** This is the price of choosing a machine at home over GitHub Actions at stage 15, and it was chosen knowingly — see "The schedule, and where it runs". `supersede` refuses a round after its first kickoff on purpose, so there is no catching up afterwards. **Its trigger has now fired.** This risk said to revisit it "the moment `fixtures.csv` starts carrying Premier League rows", and that moment was 28 Aug 2026: the Pi's uptime is now load-bearing for the one store in this project that cannot be regenerated. Measured on the day it was deployed, and the reason it is being *kept* rather than escalated: the Pi had been up **32 days continuously**, `vcgencmd get_throttled` reported `0x0` — no undervoltage or thermal event since boot — it runs from NVMe rather than an SD card, and it already carries a second production tenant whose owner would notice an outage independently. The loop also fires twice per window, so only an outage spanning both loses a round. That is judged sufficient for now and it is a judgement, not a proof: a single home machine has no redundancy, and one long outage over a Friday is all it takes. The mitigation short of moving is to watch `deploy/logs/live_loop.log` for a week with no `===== RUN` block in it, which is what an off Pi looks like from here. **Stage 17 automated that half and did not close the risk.** `epl.bot.watch.absent` reports the anchor days that went by unfired, and it is explicitly *not* a dead man's switch: the bot runs on the Pi, so an outage still in progress is reported by nobody. It speaks when the machine comes back. Closing this needs a second host, which is the argument this risk lost in the first place.
7. **The rolling fixtures file is reliable in shape but not in time, and a round is lost if every fetch inside its window lands on a stale copy.** This is what open risk 2 became when it closed. Upstream regenerates `fixtures.csv` irregularly: three fetches across 21–27 Aug 2026 found a file that had not been rewritten in two and a half days *across a matchday*, and the fourth, on 28 Aug, found one written three hours earlier carrying the whole round. Nothing distinguishes the two cases except when the fetch happened to land. The schedule is the mitigation — two fires per window, at 16:00 and 18:30 UK, so a single stale sample does not lose the round — and it is a mitigation rather than a fix, because both fires read the same upstream file and a copy stale for a whole afternoon defeats both. **Do not confuse this with open risk 6.** That one is about this machine being off; this one happens with the Pi up, the loop green and the log reporting exit 0, because "no Premier League row in the file" is a quiet success by design and is indistinguishable from a genuinely empty week. **Stage 17 made it visible without closing it.** `epl.bot.watch.stale_upstream` reports when both of a round's fires read cached copies with identical bytes, which means upstream did not regenerate across the window — the honest claim being that nobody can then tell an empty week from a lost round, rather than that a round was lost. Knowing costs nothing and changes nothing: `supersede` still refuses a round after kickoff. What would close it is a source with a stated refresh guarantee, which is one of the two surviving arguments in `epl.v2.api_football.WHAT_WOULD_REVIVE_IT`. The other, and the stronger of the pair, is the live Season Projection: this risk costs a round when it bites, and that one cannot be built at all from a file three days wide.
