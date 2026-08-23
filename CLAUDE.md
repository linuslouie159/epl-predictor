# EPL Predictor — read this first

A calibrated Premier League forecasting system: match Outcome probabilities, a simulated final table,
and an honest three-way scoreboard against the betting market and public pundits.

**Before writing any code, read these in order:**

1. [CONTEXT.md](./CONTEXT.md) — the glossary. Use these exact terms in code and in conversation.
2. [docs/DECISIONS.md](./docs/DECISIONS.md) — every decision, the measured evidence behind it, and the open risks.
3. [docs/adr/](./docs/adr/) — nine ADRs explaining the choices that look wrong until you know why.
4. [README.md](./README.md) — target numbers, data sources, build order, layout.

## The rule that outranks everything

**No future data, ever.** A Prediction may use only data timestamped strictly before its As-Of
Instant. If a change would make the model faster, simpler, or more accurate but blurs that line,
the answer is no.

## Things that look like bugs but are deliberate

Do not "fix" these without reading the linked ADR first:

- **Lower-league results are ingested** into an EPL predictor — E0 through E3. [ADR 0004](./docs/adr/0004-rate-the-whole-pyramid.md)
- **The benchmark is pre-match odds, not closing odds**, against convention. [ADR 0001](./docs/adr/0001-pre-match-odds-as-market-benchmark.md)
- **The model predicts weekly in batches** instead of per-fixture, deliberately using less information. [ADR 0002](./docs/adr/0002-weekly-prediction-rounds.md)
- **26 seasons are ingested but only 21 are scored.** [ADR 0008](./docs/adr/0008-burn-in-prefix-frozen-hyperparameters.md)
- **Dixon-Coles is fitted two different ways** — MLE and Bayesian. [ADR 0007](./docs/adr/0007-mle-for-matches-bayesian-for-projections.md)
- **The pundit's published scoreline is transformed before scoring.** [ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)
- **Two prediction stores exist, not one**, and one of them must never be rewritten. [ADR 0005](./docs/adr/0005-split-prediction-ledger.md)
- **The Ceiling Line scores *worse* than the Market Line on the scoreboard** — 0.1968 against
  0.1936 — and this is not evidence that closing odds are worse. The two are measured over
  different Fixtures; on the ones they share the Ceiling Line wins by 0.0013 RPS. Its `note` says
  so, and must keep saying so. [ADR 0001](./docs/adr/0001-pre-match-odds-as-market-benchmark.md)
- **The Market Line's stored rows say `inputs_seen = 0`** and carry no `latest_input`. Correct: it
  reads its odds off the Fixture, and a Predictor that consumes no history has no history to leak.
  Do not make it touch the corpus. A Pundit will record the same.
- **The shared calibration layer makes every Predictor slightly worse**, and is kept anyway. All
  four are already well calibrated, so a monotone map finds noise and charges ~0.001 RPS for it.
  The headline numbers are therefore pre-calibration, both columns are published, and the finding
  is the point. [ADR 0006](./docs/adr/0006-ordered-logit-with-shared-calibration.md)
- **`epl.calibration` is at the top level, not in `epl.models`**, and older docstrings said
  otherwise. It is not a model, and putting it there would make `epl.models` and `epl.ledger`
  import each other. See docs/DECISIONS.md, "The shared calibration layer".
- **Two Pundits are registered, not one**, and their as-stated RPS of ~0.334 is *worse than the
  Naive Baseline*. Both are correct. A Pundit is "a named public forecaster" (CONTEXT.md) and two
  people worked these nine Seasons; and a Scoreline read as `[1, 0, 0]` is a claim of certainty
  nobody made, which is what the number measures (ADR 0003). Their `note` says so, and must keep
  saying so. On accuracy the same calls beat the floor by seven points.
- **The Pundit dataset is committed to git and the pages it came from are not.**
  `src/epl/pundits/predictions.csv` ships with the code like the Club table; `data/raw/` is
  gitignored like everything else in it. That is the ticket's "committed and frozen rather than
  re-scraped on every run", and the corpus test rebuilds the file from the cache and compares
  bytes so "frozen" cannot come to mean "whatever was written last".
- **Elo's fitted draw band is symmetric and the fit cannot move its centre**, and **Elo rebuilds
  its whole rating pool at every one of the 952 Prediction Rounds** rather than folding one
  forward. Both are explained in `src/epl/models/__init__.py`; the second costs a minute per
  backfill and buys the one kind of leak this project cannot see.
- **Refreshing a cached raw file archives the old bytes** into `superseded/` instead of replacing
  them, and **`Wimbledon`, `Milton Keynes Dons` and `AFC Wimbledon` are three separate Clubs**.
  Both are explained where they live — `src/epl/ingest/cache.py`, which both the Football-Data
  ingest and the Pundit fetch write through, and `src/epl/clubs/table.py`.

## Never do this

- Never edit anything under `data/raw/` — it is a byte-identical cache of upstream files.
- Never rewrite a file under `outputs/live/` after its round's first kickoff. Supersede with a new
  As-Of Instant instead. That directory is evidence, not output.
- Never tune a hyperparameter using data from outside the Burn-In Window (2000/01–2004/05).
- Never report accuracy as the headline metric. RPS is primary; accuracy is for lay explanation only.

## Status

Design is complete and grilled.

**Stage 1 is built**: the Miniforge environment (`environment.yml`), the Football-Data ingester
(`src/epl/ingest/`) and the Club/Alias table (`src/epl/clubs/`, 115 Clubs across four tiers).

**Stage 2 is built**: Prediction Rounds (`src/epl/rounds.py`, issue #5) and the metrics module
(`src/epl/metrics/`, issue #6) — RPS, Brier, log loss, accuracy and the 10-bin reliability diagram,
every expected value worked by hand before any model uses it.

**Stage 3 is built**: the Predictor contract (`src/epl/predictors.py`), the split ledger
(`src/epl/ledger/`) and the Naive Baseline (`src/epl/benchmarks/naive.py`), issue #7 — one Predictor
walked over the whole Evaluation Window, stored, audited and scored at **0.22938 RPS** over 7,980
Fixtures and 952 Prediction Rounds.

**Stage 4 is built**: the vig removal (`src/epl/benchmarks/vig.py`), the Market Line and the Ceiling
Line (`src/epl/benchmarks/market.py`), issue #8.

**Stage 5 is built**: pyramid-wide Elo through an ordered logit (`src/epl/models/`, issue #9) — one
rating pool across E0–E3 folded in kickoff order (`elo.py`), the mapping from one edge to three
probabilities (`ordered_logit.py`), and the only place a hyperparameter may be fitted
(`burn_in.py`).

**Stage 6 is built**: the shared isotonic calibration layer (`src/epl/calibration.py`, issue #10) —
one step, fitted walk-forward on out-of-sample Predictions only, applied to every Predictor through
the contract by `epl.ledger.scoreboard`. Every metric is now reported twice.

**Stage 7 is built**: the Pundit backfill (`src/epl/pundits/`, issue #11) — nine MyFootballFacts
season pages fetched and parsed (`myfootballfacts.py`), reconciled with the corpus and frozen as
the committed `predictions.csv` (`dataset.py`), graded two ways (`grading.py`), and registered as
two named Predictors scored as-stated (`predictor.py`). 3,408 calls of a possible 3,420. The
scoreboard now reads:

```
pre-calibration
     predictor  fixtures    rps  brier  log_loss  accuracy    ece
   market_line      7980 0.1936 0.5684    0.9582    0.5471 0.0061
  ceiling_line      2660 0.1968 0.5717    0.9639    0.5498 0.0060
           elo      7980 0.1994 0.5810    0.9771    0.5380 0.0055
naive_baseline      7980 0.2294 0.6430    1.0642    0.4556 0.0061
     lawrenson      1896 0.3341 0.9810   16.9415    0.5095 0.3270
        sutton      1512 0.3343 1.0159   17.5435    0.4921 0.3386

post-calibration
     predictor  corrected    rps  brier  log_loss  accuracy    ece  correction
   market_line       7600 0.1945 0.5707    0.9931    0.5427 0.0124      0.0342
  ceiling_line       2280 0.1980 0.5755    0.9817    0.5445 0.0084      0.0328
           elo       7600 0.2004 0.5836    1.0128    0.5353 0.0097      0.0307
naive_baseline       7600 0.2309 0.6478    1.2269    0.4479 0.0161      0.0455
     lawrenson       1508 0.2374 0.6827    4.2683    0.5105 0.0792      0.3880
        sutton       1130 0.2473 0.7230    5.3558    0.5033 0.0988      0.3841
```

`epl.simulate` is still a documented shell, naming the issue that builds it.

Four things about stage 7 worth knowing before building on it:

- **The shared calibration layer gains a Pundit ~0.09 RPS where it cost the other four ~0.001.**
  That is the re-measurement stage 6 asked for by name, and it *confirms* stage 6 rather than
  overturning it: the layer was never broken, it had nothing to find. All four earlier Predictors
  arrive at a ten-bin error of about 0.006; a Pundit arrives at 0.33 and the same unchanged layer
  recovers most of it. **This is still not the Calibrated Pundit.** Issue #12 fits a different map,
  bucketed by predicted goal margin, and must not be collapsed into this one.
- **The as-stated number is worse than the floor, and that is the deliverable.** 0.334 against a
  Naive Baseline of 0.236 over the same Fixtures. On accuracy the same calls beat that floor
  0.5095 to 0.4388 and trail the market by four points. Both readings are published; either alone
  is an argument (ADR 0003). `python -m epl.pundits grades` is the lay pair beside it — 11.0% and
  9.1% exact scores, 50.9% and 49.2% correct Outcomes. Each Pundit's `note` carries three things
  onto the scoreboard and must keep carrying all three: the BBC as the origin, that 1,896 and
  1,512 Fixtures are not the board's 7,980 so the RPS is not comparable, and what the as-stated
  reading is.
- **Their `log_loss` of 16.9 and 17.5 is an artefact of the floor, not a measurement.** A one-hot
  Prediction that is wrong is clipped at `epl.metrics.LOG_LOSS_FLOOR` before the log, so the number
  is the miss rate times the floor. `epl.metrics.log_loss` names this exact case in its docstring.
  Do not report it, and do not "fix" it by raising the floor.
- **The build reads the result the page publishes and then throws it away.** 3,402 of the 3,406
  calls that carry one agree with Football-Data — two Fixtures were only ever listed as postponed,
  so they have no result to check — and that agreement is what confirms two spellings became the
  right two Clubs the right way round, which is the one thing no unit test reaches. The four that
  disagree are named in `tests/pundits/test_over_the_corpus.py`. Do not store the result:
  `predictions.csv` holds Fixture, Scoreline, Pundit and date, and nothing that knows an Outcome
  (ADR 0005).

Four things about stage 6 worth knowing before building on it:

- **Calibration makes every Predictor worse, and that is the deliverable.** It costs 0.0009–0.0015
  RPS and moves 3–5% of every Prediction's mass. Two effects, separated by tests rather than
  asserted: **knot resolution** (a knot per distinct quote, and Elo edges and market odds are nearly
  continuous, so most knots rest on one Fixture — a ten-band fit recovers 73% of Elo's loss and 52%
  of the market's) and **the corpus** (even coarse, both stay worse than raw; all four start at a
  ten-bin calibration error of about 0.006, so there is little left to find). A split half of the
  market's Fixtures says the same without the walk: the map improves the half it was fitted on by
  0.0017 RPS and costs the later half 0.0005. Do not "fix" this by dropping the pre-calibration
  column — it is the only reason the tax is visible. ADR 0006 has the table.
- **The correction points the right way; the noise is bigger.** Elo's draw quote at even Supremacy
  moves 30.2% → 29.3% against 27.6% observed, which is exactly the defect #9 handed the layer. So
  the diagnosis is "well-calibrated inputs", not "wired up wrong", and a Predictor that genuinely
  needs correcting — a Pundit scored as-stated, at issue #11 — should be measured again rather than
  assumed to behave like these four. **Stage 7 measured it**, and it does not behave like them: the
  same layer gains a Pundit about 0.09 RPS.
- **The layer stores nothing, and no forecast is ever calibrated.** A calibrated Prediction is a
  function of a stored Prediction *and* of Outcomes that happened after it, so it is derived at
  scoring time. No row in either store knows an Outcome, and that is what makes a leaked Prediction
  distinguishable from a recorded one (ADR 0005). Do not add a `calibrated_*` column to the ledger.
  The consequence matters at **issue #17**: a Prediction sealed for an unplayed Fixture is published
  raw and gains a calibrated form only once its round has been scored.
- **The first 380 Predictions of every track record are uncorrected**, because the map is fitted
  out-of-sample only and there is nothing behind them. `corrected` on the scoreboard says how many
  a fitted map reached; it is 7,600 of 7,980, not a bug.

Four things about stage 5 worth knowing before building on it:

- **Elo takes 0.030 of the 0.036 RPS the market takes out of the floor** — 84% of the available
  edge, from ratings alone. It is 0.0058 short of the market and 0.0008 above the README's ≤0.1986
  target, which was what #10 and #13 were for — and after stage 6 it is #13 alone. An Elo that
  *beat* the market would be a leak, and the corpus test says so out loud.
- **`epl.models.burn_in` is the only place anything is fitted**, and it cuts the corpus to the
  Burn-In Window before it walks a single match — so handing it all 26 Seasons is indistinguishable
  from handing it the five. 2000/01 warms the ratings and is not fitted on; the fit is scored on
  2001/02–2004/05's 1,520 Fixtures. What it found is frozen as literals in `elo.py` and re-derived
  by `tests/models/test_elo_over_the_corpus.py`.
- **The fit refuses a winner sitting on the wall of its own *coarse* grid** — checked after the
  first pass only, since a refinement pass is deliberately bounded by the box it was given. That is
  not decoration: the home-advantage grid originally stopped at 140 points, the answer sat on it,
  and the refinement then re-centred on the boundary and reported 155 as fitted. Widen the grid; a
  boundary is not an optimum.
- **Elo quotes draws slightly too often in every Supremacy bucket** — predicted 30.2%→14.5% against
  observed 27.6%→13.8%. That is frozen hyperparameters drifting exactly as ADR 0008 says they will,
  and it is what issue #10's shared calibration layer was for. Stage 6 built that layer, and it
  moves the even end 30.2%→29.3% — correctly, and not far enough, at a net cost of 0.0009 RPS. Both
  facts are pinned by tests so neither can be quietly forgotten.

Four things about stage 4 worth knowing before building on it:

- **The Ceiling Line's 0.1968 is not worse than the Market Line's 0.1936.** They are measured over
  different Fixtures. On the 2,660 they share, the Market Line scores 0.1981 and the Ceiling Line
  beats it by 0.0013 RPS. That caveat rides onto the scoreboard as the Ceiling Line's `note` and
  must not be dropped — the bare number reads as the opposite of what it means.
- **The Predictor contract grew three optional attributes**, documented above the accessors in
  `src/epl/predictors.py` and read through `predictors.also_sees`, `predictors.note` and
  `schema.covered`. None is on the Protocol (a Protocol member is required of everything claiming
  it, and almost no Predictor wants these):
  `covers(fixtures)` says which Fixtures a Predictor can speak to at all; `also_sees` claims extra
  Fixture columns, checked against `schema.PRIVILEGED_FIXTURE_COLUMNS`; `note` is a caveat the
  scoreboard prints. **Issue #11's Pundits will want the first and the third.**
- **A Predictor that covers nothing writes no file**, and `backfill` says so out loud rather than
  passing over it. That line is also what a broken input column looks like, so read it.
- **Four books in the corpus have an overround below one** — 2025/26 League One closing averages,
  as low as 0.955. `vig.is_book` excludes them; no Premier League Fixture and no pre-match book is
  affected. Do not "fix" this by loosening `as_book`: a book that pays out more than it takes is
  not a book, and that check is what would catch a genuinely shifted column.

Three things about stage 3 worth knowing before building on it:

- **A Predictor is handed `Evidence`, not an As-Of Instant.** Evidence is the corpus already cut at
  the instant, and it records what it hands over, so `inputs_seen` and `latest_input` on every
  stored row are a receipt. Do not add a Predictor that reads the corpus directly — that is the
  one way a leak can enter without an audit failing.
- **The scoreboard has no branch per Predictor and must not grow one.** Registering a Predictor is
  what puts it on the board; the ledger, the audits and the scoring are written once against the
  contract (spec, user story 16).
- **The row audit is two-tier on kickoff**, matching the As-Of rule: strictly-before where a
  kickoff time was recorded, and at-or-before where it was not. 313 Evaluation Window Fixtures are
  played on the very Tuesday or Friday they anchor to and are legitimately equal.

Two things about stage 2 worth knowing before building on it:

- **A Prediction Round's As-Of Instant is midnight at the start of its anchor day**, not the
  afternoon the market samples at. Nothing is played in between, so midnight withholds no data the
  market had — and it is the only choice checkable for the 437 Fixtures that kick off on their own
  anchor day with no recorded kickoff time. Football-Data records no kickoff time at all before
  2019/20 (7,220 of 9,880 Premier League Fixtures).
- **There are 1,189 Prediction Rounds, not the 1,332 the design recorded.** The anchor rule is
  stated as executable code and was kept; the unreproducible count was corrected. See
  [docs/DECISIONS.md](./docs/DECISIONS.md).

## What to build next

**Issue #12 — the Calibrated Pundit** is the other half of what #11 just started, and it has real
numbers to beat now rather than illustrative ones. Two things it must not do. It must not reuse the
shared layer: #12's map is bucketed by **predicted goal margin**, so that a 3-0 call is treated as
the stronger claim it is, and the shared layer sees only a one-hot input with no Scoreline in it.
And a Calibrated Pundit is a one-feature model, not a person (ADR 0003) — name it so nobody can
report "Sutton beat the model". The bar it has to clear is the generic layer's: 0.2374 for
Lawrenson and 0.2473 for Sutton, from 0.3341 and 0.3343 as stated. It also carries user story 34,
each Pundit's best and worst calls by calibration miss, which `epl.pundits.grading` has the rows
for.

**Issue #13 — Dixon-Coles** is now the only remaining way to close the 0.0058 gap to the market,
since the calibration layer turned out to cost rather than buy. It is also the one that unblocks
the Season Projection (goal difference, ADR 0007).

Also ready: **issue #18**, the deferred-v2 stubs (XGBoost, Golden Boot, API-Football). It has been
unblocked since stage 1, needs nothing from the ledger, and is small — pick it up when a stage
lands and there is no appetite to start the next one.

Check the graph before starting:

```
gh issue view <n> | sed -n '/## Blocked by/,$p'
```

```
conda env create -f environment.yml
conda activate epl-predictor
python -m epl.ingest fetch     # fill data/raw/ — 104 files, 26 Seasons x 4 tiers
python -m epl.ingest build     # write matches.csv (52,672) + odds_availability.csv
python -m epl.pundits fetch    # cache the nine MyFootballFacts season pages
python -m epl.pundits build    # re-freeze predictions.csv — 3,408 calls, cross-checked
python -m epl.pundits grades   # exact-score and correct-Outcome rates per Pundit and Season
python -m epl.ledger backfill  # walk every registered Predictor over the Evaluation Window
python -m epl.ledger scoreboard      # every metric twice, pre- and post-calibration
python -m epl.ledger reliability     # the 10-bin diagrams per Predictor, in both forms
python -m epl.ledger audit     # re-check both stores and the seal on outputs/live/
python -m epl.benchmarks overround   # the margin in each book, per Season and tier
python -m epl.benchmarks methods     # the three vig removals compared on one book
python -m epl.models fit       # re-derive the frozen hyperparameters on the Burn-In Window
python -m epl.models draws     # the draw rate against Supremacy, predicted and observed
python -m epl.models ratings   # the pool at a Season's first Prediction Round
pytest                         # add --run-network to also hit football-data.co.uk
```

`backfill` now takes about a minute: Elo rebuilds its pool at every round on purpose (see above).
