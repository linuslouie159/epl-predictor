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
- **Draw rate vs Supremacy**: 32.3% when evenly matched, falling monotonically to 13.4% at the widest mismatch. A 2.4x range.
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

## Open risks

1. **BBC live scraping is unproven.** `www.bbc.co.uk` was unreachable during design, article URLs are opaque IDs (`/sport/football/articles/cvg0e92ezz4o`, legacy `/sport/football/28859459`) and there is no index page. Needs a spike at stage 5. If it fails, live pundit data has no confirmed source — MyFootballFacts' update latency during a season is unknown.
2. **The live path is only half tested.** Re-checked 21 Aug 2026: `fixtures.csv` is reachable and parses, but it still holds a single English row (one E2 fixture) and no E0 rows, and `mmz4281/2627/E0.csv` still does not exist. The transport is proven; the E0 live path is not. `pytest --run-network` exercises what can be exercised, including the check that no new Club spelling has appeared upstream — the check that must pass before a live Prediction Round can be sealed.
3. **MyFootballFacts parseability is unverified.** Content correctness was confirmed — a 2025/26 result cross-checked exactly against Football-Data — but the HTML has not been parsed across all nine season pages.
4. **Cross-tier Elo has no burn-in before 2000/01**, so early ratings linking E0 to E3 will be unreliable. This sits inside the Burn-In Window so it should not reach scored results, but it is worth watching.
5. **Frozen hyperparameters will drift out of date** by the late Evaluation Window, given the measured decline in home advantage. Accepted deliberately; see ADR 0008.
