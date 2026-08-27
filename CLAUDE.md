# EPL Predictor — read this first

A calibrated Premier League forecasting system: match Outcome probabilities, a simulated final table,
and an honest three-way scoreboard against the betting market and public pundits.

**Before writing any code, read these in order:**

1. [CONTEXT.md](./CONTEXT.md) — the glossary. Use these exact terms in code and in conversation.
2. [docs/DECISIONS.md](./docs/DECISIONS.md) — every decision, the measured evidence behind it, and the open risks.
3. [docs/adr/](./docs/adr/) — ten ADRs explaining the choices that look wrong until you know why.
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
- **27 seasons are ingested and the 27th is in neither Window.** [ADR 0010](./docs/adr/0010-live-season-outside-both-windows.md)
- **Dixon-Coles is fitted two different ways** — MLE and Bayesian. [ADR 0007](./docs/adr/0007-mle-for-matches-bayesian-for-projections.md)
- **Dixon-Coles' likelihood lives apart from its fit**, in `epl.models.likelihood`, which knows
  nothing about optimisers, Evidence or Predictors. That is ADR 0007's "both paths share one
  likelihood function" made structural: issue #14's Bayesian fit imports it rather than restating
  it. Do not fold it into `dixon_coles.py`.
- **Dixon-Coles fits all four tiers too, and the reason is not Elo's.** Elo is zero-sum, so a
  rating survives a promotion by construction; nothing in this likelihood knows a division exists,
  and the tiers are joined only by the Clubs that changed tier inside the decay horizon. It was
  therefore measured rather than inherited — 0.20165 against a Premier-League-only 0.20382 on the
  Burn-In Window, with mean attack falling monotonically E0 → E3 at every checkpoint.
- **Dixon-Coles' low-score correction comes out near zero and is kept anyway.** The 1997 paper's
  −0.13 does not reproduce here: the fitted value wanders around zero and changes sign, and pinning
  it at zero costs 0.00011 RPS. It stays because it belongs to the shared likelihood rather than to
  this fit — deleting it would change the model, not simplify the code.
- **`python -m epl.models sequential` produces a number that must never be quoted as a score.**
  It is ADR 0002's promised diagnostic: every Fixture predicted from its round *and* from its own
  kickoff, so the cost of the weekly batch is measured rather than argued. Its rows carry the
  Outcome and are never written to either store. The answer is **+0.00001 RPS for Elo and +0.00003
  for Dixon-Coles** — the comparability the weekly batch buys is very nearly free, which is a
  happier result than ADR 0002 needed and does not license quoting the sequential column anywhere.
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
- **The three-way board's model column is still Elo, not Dixon-Coles.** `epl.pundits.report.OPPONENTS`
  is a *chosen* comparison of three named opponents rather than a view of the registry, so a new
  Predictor reaches the scoreboard and does not silently rewrite the argument ADR 0003 is making.
  Its docstring says so; do not "fix" it by reading the registry there.
- **Four Pundit-derived Predictors are registered, not two**, and the two named for people are
  the *worse* pair. `lawrenson` and `sutton` are the as-stated readings; `margin_map_lawrenson` and
  `margin_map_sutton` are the Calibrated Pundits over the same calls. The maps are deliberately
  **not** named after the forecasters — a Calibrated Pundit is a one-feature model, not a person,
  and "Sutton beat the model" is a sentence no output may support. [ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)
- **A Calibrated Pundit covers fewer Fixtures than the Pundit it is built from** — 1,856 of 1,896
  and 1,472 of 1,512. The missing 40 each are the opening calls of a record, which no map has a
  sample behind yet. That is `epl.pundits.margin.MINIMUM_SAMPLE` being visible rather than hidden;
  do not fill them in with a base rate.
- **The margin map's ten-bin error is 0.019, three times the other Predictors' 0.006**, and it is
  not a defect. The map has seven buckets, so it is coarse by construction; the comparison that
  matters is against the 0.327 its own Pundit arrives at.
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
- **`environment.yml` pins the BLAS provider to OpenBLAS**, and that pin is load-bearing. Without
  it conda selects MKL 2026, whose Intel-OpenMP threading layer cannot resolve its symbols against
  the `libiomp5md.dll` shim llvm-openmp 22 supplies, and *every LAPACK call aborts the interpreter*
  — `numpy.linalg`, `scipy.linalg`, L-BFGS-B and PyMC all die with `0xc06d007f`. It looks like an
  arbitrary preference and is the second place after `arviz <1` where a free version choice breaks
  the build rather than drifting. See docs/DECISIONS.md, "Added at stage 9".
- **The Bayesian fit refuses part of the parameter space, and the guard is load-bearing.**
  `epl.simulate.posterior.log_likelihood_at` returns `-inf` wherever Dixon-Coles' correction would
  turn a Scoreline probability negative. That is the model's real support, not a workaround: the
  shared likelihood's `CORRECTION_FLOOR` keeps the log-density smooth there while the gradient goes
  from 229 to 9.3e12, and a sampler handed that flies to infinity and returns a `nan` no divergence
  check catches. Deleting it makes the fit 55x slower *and* wrong, while looking merely slow.
- **The posterior's priors are deliberately far too wide to be sensible**, because a prior that
  shrinks makes it a different model from the MLE and ADR 0007 forbids exactly that. See
  `epl.simulate.posterior.Priors`; tightening them to something reasonable-looking moves a single
  Fixture's Home probability by 0.079.
- **The tiebreaker chain has two more steps than the Premier League's own regulation.** The
  competition goes points → goal difference → goals scored → play-off at a neutral ground, with no
  head-to-head step at all; `epl.simulate.table.TIEBREAKERS` puts head-to-head points and
  head-to-head away goals in between, because issue #15 asks for them by name. It is a strict
  refinement — it changes nothing the regulation decides and replaces some coin flips with a rule.
  Do not "correct" it to the regulation without reading the ticket. Measured: **no real final table
  in the 26 ingested Seasons has ever needed those two steps**, so they exist for the simulated
  tables, and every `Projection` reports its own `level_pairs` rather than assuming.
- **A `Slate` cannot express the result of a Fixture that has not been played**, and that is the
  point of the type rather than an oversight. Validation hands a projection a corpus containing
  every result it is supposed to be forecasting, so `Slate.of(played, remaining)` reads goals off
  the first frame and only the two Club columns off the second. Do not add a convenience that takes
  one frame and a mask.
- **A Season Projection has two seeds and both are recorded.** The sampler's is on `Diagnostics`,
  the walk's is on `Simulation`, and they are deliberately different numbers so no reader mistakes
  one for the other being reused. `Projection.describe()` prints both.
- **`epl.simulate.projection.draw_order` lays down whole copies of every draw rather than
  truncating a permutation.** The obvious version leaves some draws used three times and others
  once; and walking the draws in order would be worse still, because `fit` concatenates its chains,
  so the first quarter of a posterior belongs to one chain.
- **A Season the corpus does not hold raises `CheckpointError`, not `ProjectionError`**, from
  `slate_at` and `final_positions` as well as from `projection_rounds`. All three cut the corpus
  through `epl.simulate.checkpoints.season_fixtures`, written once so `PROJECTED_DIVISION` cannot
  quietly become the literal `"E0"` in one of them — and that error already means "a Season
  Projection was asked for where none can be taken". Do not give each module its own copy back.
- **A projection's reliability diagram counts `projections` where a Predictor's counts
  `predictions`**, and both are binned by the same `epl.metrics.diagram`. The shared middle is what
  makes 0.008 and 0.006 comparable; the different column name is what stops a Club-projection being
  read as a Prediction. Neither half is an accident.
- **The corpus holds 27 Seasons and the Evaluation Window is still 21.** `epl.windows.LAST_SEASON`
  is 2026 and `EVALUATION_WINDOW` is `range(2005, 2026)`; they are deliberately not the same edge.
  The Season in progress is `LIVE_SEASON`, ingested so a Predictor can see it and a Sealed Prediction
  can be joined to a result, and in **neither Window**: never fitted on, never backfilled, scored on
  its own board at `outputs/live_scoreboard.csv`. Folding it into the Evaluation Window would make
  the headline RPS grow every Saturday *and* would have `backfill` regenerate Predictions the live
  loop had sealed — ADR 0005's exact failure. `matches.csv` therefore grows weekly, which is why
  every fixed count in the corpus tests is over the 26 closed Seasons.
- **`epl.live.seal` has no branch per Predictor, and five of the nine say nothing.** `lawrenson`,
  `sutton`, `margin_map_*` and `ceiling_line` all answer `schema.covered` with nothing for an
  unplayed Fixture, for their own reasons, so stage 12's "a Pundit cannot be sealed" reaches the
  live loop as zero lines of code. `tests/live/test_live_loop_over_the_corpus.py` pins which four
  speak. Do not add a list of sealable Predictors.
- **A superseding Sealed Prediction is stamped later than the one it replaces, and its Evidence is
  cut later too**, so it may genuinely know more. That is uncomfortable next to ADR 0002 and it is
  right: stamping a correction at the round's original midnight would be a false claim about when it
  was made, in the one store whose whole value is that such claims are true. `seal --supersede`
  writes `2026-08-28.1.csv` beside `2026-08-28.csv` and refuses any row stamped at or before what
  the store already holds. It is for correcting a bug, never for refreshing a quote.
- **`epl.live.__main__.clock` is a function and not a `--now` flag**, on purpose. That value decides
  whether a round is inside its sealing window, so an operator who could name it could seal a round
  after its own kickoff and have the file say otherwise. It reads **`epl.ledger.live.uk_now`, not
  `pd.Timestamp.now`**, and that is the same rule from the other side: every instant it is compared
  against is a UK local reading off Football-Data, so a machine in another zone gets the window
  wrong without anybody choosing to — eight hours on the desktop this was built on, one hour of
  British Summer Time in a container defaulting to UTC. **Both directions fail silently and they
  are not equally bad.** A clock that runs *late* refuses a round as having kicked off, and the loop
  seals nothing; a clock that runs *early* — which is what UTC+8 did, reading 2026-08-28 00:17 where
  the UK read 2026-08-27 17:17 — defeats the `NOT_OPEN` end instead, sealing a round under a moment
  that has not happened and odds Football-Data has not sampled. The second is the dangerous one:
  sealing nothing is eventually visible, and a false instant in this store is not visible at all.
  `tests/live/test_unattended.py` pins the function and the absence of the flag. It pins the *zone*
  only on a machine outside the UK — on a UK machine the two implementations are observationally
  identical and no test can separate them, which is why the expectation there is derived from UTC.
- **`seal` exits 0 when there is nothing to seal, and that is not sloppiness about failure.**
  `epl.live.upcoming.NothingToSeal` is a subclass of `LiveError` covering the two silences that need
  nobody — no round inside its window, and a rolling file with no Premier League row — because those
  are most of the week and, until upcoming Fixtures have a source, *every* fire of the schedule. A
  job that goes red twice a week for a season is one whose owner stops reading it. Everything else
  still exits 1: a stale `LIVE_SEASON`, a round written but not committed, a round committed but not
  pushed. An upstream shape change stays an uncaught `IngestError` with its traceback on purpose —
  the rolling file's shape is `epl.ingest`'s to complain about, and catching it here would put
  "upstream changed" behind the same exit code as "it is Wednesday".
- **Refreshing a cached raw file archives the old bytes** into `superseded/` instead of replacing
  them, and **`Wimbledon`, `Milton Keynes Dons` and `AFC Wimbledon` are three separate Clubs**.
  Both are explained where they live — `src/epl/ingest/cache.py`, which both the Football-Data
  ingest and the Pundit fetch write through, and `src/epl/clubs/table.py`.
- **The Season pages are named in `PAGES` *and* discovered from an index, and both are correct.**
  The nine frozen Seasons are literals because they are over; the Season in progress cannot be,
  because the archive has used four slug conventions in eighteen Seasons and 2026/27 dropped the
  `for-` its four predecessors carried. `discover_pages` follows the index's own `rel="next"`
  rather than walking `page/2/`, `page/3/` until one 404s — a loop whose exit condition is an error
  asks upstream for a page it was never told exists.
- **`epl.pundits.live` lifts the backfill's size floor and keeps its ceiling**, and that asymmetry
  is the point. `MIN_CALLS = 360` is right for a complete Season and wrong for every live page
  there has ever been — 2026/27's held ten. The ceiling survives because a page scanned twice
  doubles whether or not the Season is over.
- **A live call the corpus cannot place is handed back, not dropped and not raised on.**
  `epl.pundits.dataset._locate` refuses that case and says in its own docstring that doing so is
  right for the backfill and wrong for #16. `live.build` filters and hands the rest to the frozen
  builder, so there is one implementation of what a listing becomes. Measured, `unplaced` is empty
  on every live page seen — which is why anything in it is worth reading.
- **`src/epl/v2/` is a package nothing imports, and that is its whole specification.** Three
  deferred features written down instead of built (decision 12): prose, named constants, no
  functions and no classes. It is Python rather than Markdown so that "the pipeline does not import
  it" is a thing a test can check, and `epl.v2.STUBS` names its members as strings so that importing
  the package executes nothing. Do not wire it in, do not add a helper to it, and do not collapse it
  into a docs page.

## Never do this

- Never edit anything under `data/raw/` — it is a byte-identical cache of upstream files.
- Never rewrite a file under `outputs/live/`. Supersede it — `python -m epl.live seal --supersede`
  writes a new revision at a new As-Of Instant. That directory is evidence, not output.
- Never put the Live Season in the Evaluation Window, and never backfill it. It is sealed only.
- Never tune a hyperparameter using data from outside the Burn-In Window (2000/01–2004/05).
- Never report accuracy as the headline metric. RPS is primary; accuracy is for lay explanation only.
- Never add a `--now` flag so a missed round can be sealed late, and never schedule
  `python -m epl.ledger backfill`. The first hands an operator the one lie this store exists to
  prevent; the second would regenerate Predictions the live loop had sealed (ADR 0005) and move the
  Evaluation Window's numbers on a timer.

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
two named Predictors scored as-stated (`predictor.py`). 3,408 calls of a possible 3,420.

**Stage 8 is built**: the Calibrated Pundit and the three-way scoreboard (`src/epl/pundits/margin.py`,
`calibrated.py`, `report.py`, issue #12) — a published Scoreline reduced to its predicted goal
margin and quoted what such a call has historically produced, fitted walk-forward on that Pundit's
past calls only, registered as two more Predictors and reported beside the as-stated reading.

**Stage 9 is built**: Dixon-Coles by maximum likelihood (`src/epl/models/likelihood.py`,
`dixon_coles.py`, issue #13) — one likelihood written once for both fits (ADR 0007), fitted at all
952 scored Prediction Rounds over a time-decayed sample of all four tiers, producing Scoreline
probabilities that collapse onto the same three Outcomes. **It scores 0.19752 RPS and clears the
README's ≤0.1986 target**, which Elo missed by 0.0008. ADR 0002's per-Fixture diagnostic is built
beside it (`epl.ledger.backtest.sequential`, `python -m epl.models sequential`). The scoreboard now
reads:

```
pre-calibration
           predictor  fixtures    rps  brier  log_loss  accuracy    ece
         market_line      7980 0.1936 0.5684    0.9582    0.5471 0.0061
        ceiling_line      2660 0.1968 0.5717    0.9639    0.5498 0.0060
         dixon_coles      7980 0.1975 0.5768    0.9707    0.5360 0.0080
                 elo      7980 0.1994 0.5810    0.9771    0.5380 0.0055
   margin_map_sutton      1472 0.2111 0.6045    1.0137    0.4993 0.0200
margin_map_lawrenson      1856 0.2127 0.6012    1.0092    0.5116 0.0195
      naive_baseline      7980 0.2294 0.6430    1.0642    0.4556 0.0061
           lawrenson      1896 0.3341 0.9810   16.9415    0.5095 0.3270
              sutton      1512 0.3343 1.0159   17.5435    0.4921 0.3386

post-calibration
           predictor  corrected    rps  brier  log_loss  accuracy    ece  correction
         market_line       7600 0.1945 0.5707    0.9931    0.5427 0.0124      0.0342
        ceiling_line       2280 0.1980 0.5755    0.9817    0.5445 0.0084      0.0328
         dixon_coles       7600 0.1979 0.5782    0.9964    0.5400 0.0102      0.0378
                 elo       7600 0.2004 0.5836    1.0128    0.5353 0.0097      0.0307
   margin_map_sutton       1091 0.2126 0.6081    1.1261    0.5037 0.0162      0.0274
margin_map_lawrenson       1468 0.2141 0.6043    1.1645    0.5110 0.0207      0.0317
      naive_baseline       7600 0.2309 0.6478    1.2269    0.4479 0.0161      0.0455
           lawrenson       1508 0.2374 0.6827    4.2683    0.5105 0.0792      0.3880
              sutton       1130 0.2473 0.7230    5.3558    0.5033 0.0988      0.3841
```

**Stage 10 is built**: the Bayesian Dixon-Coles posterior (`src/epl/simulate/posterior.py`,
`checkpoints.py`, issue #14) — the same likelihood sampled instead of maximised, handed to PyMC as
one opaque node so no second likelihood exists (ADR 0007), fitted only at Season Projection points.
The two fits agree on a real Fixture to **0.0090**, which is what licenses the split at all. One
posterior fit takes about **four minutes** against the MLE's 0.22 seconds.
`python -m epl.simulate checkpoints` and `python -m epl.simulate posterior`.

**Stage 11 is built**: the Monte Carlo Season Projection (`src/epl/simulate/table.py`,
`projection.py`, `validation.py`, issue #15) — the final league table and its full tiebreaker chain
in a module that has never heard of a posterior, a 10,000-Season walk over the draws with both seeds
recorded, and a validation across completed Seasons that asks where the real champion landed.
`python -m epl.simulate project` and `python -m epl.simulate validate`.

Validated across the 20 completed Evaluation Window Seasons at three checkpoints each: the eventual
champion was the projection's favourite **73%** of the time and in its top three **97%**, it gave
the champion a mean title probability of **0.581**, and its ten-bin calibration error is **0.008**
pooled — against about 0.006 for every match Predictor on the scoreboard.

**Stage 12 is built**: the BBC live Pundit spike (`src/epl/pundits/live.py`,
`myfootballfacts.discover_pages`, issue #16) — a spike, so the deliverable is a decision backed by
evidence, and the evidence changed the question. `python -m epl.pundits live`.

**The answer is that a Pundit cannot be part of a Sealed Prediction.** Not because the calls do not
exist in time — the BBC publishes them days before kickoff — but because the only source this
project is *permitted* to read has not transcribed them yet.

Seven things about stage 12 worth knowing before building on it:

- **The BBC is reachable and machine-readable, and is still unusable.** Re-tested 27 Aug 2026: it
  answers in 0.1s, the opaque article IDs resolve, and the pages carry `application/ld+json` and
  `window.__INITIAL_DATA__`. The blocker is `bbc.co.uk/robots.txt`, which forbids "scraping,
  crawling, or systematic extraction", "creating datasets from BBC content" and text and data
  mining — and disallows `ClaudeBot`, `Claude-Web` and `anthropic-ai` from the whole site. That
  answer does not change when a network does, which is why open risk 1 is closed rather than
  deferred. **Do not build a BBC fetcher.**
- **MyFootballFacts permits exactly what the BBC forbids** — `Allow: /` for the same agents, named
  individually — and has the one thing the BBC lacks: an index. So it is the source, and the BBC
  remains the *origin* named beside every call.
- **Its measured latency is the finding.** Against archived snapshots of the 2025/26 page, asking
  what a Prediction Round asks — are the next round's calls published yet? — the answer was **10 of
  10 on the season opener, then 0 of 10 two days before a round, and 0 of 10 eleven days before**.
  The live 2026/27 page agreed on 27 Aug 2026: one day before round two it held round one only,
  results already filled in. The archive transcribes a matchday *after* it is played.
- **The consequence for #17**: the live loop seals the models and the Market Line, and a Pundit's
  column on the three-way board is filled in retrospectively. This constrains the live loop only —
  the committed backfill is nine complete Seasons, and every published Pundit number comes from it.
- **The index links eighteen Season pages and the backfill uses nine.** 2009/10–2026/27
  consecutively; the eight before 2017/18 are ~3,000 more Lawrenson calls this project has never
  scored, on the same source in the same shape. Worth an issue, not in scope at #16.
- **All nine cached pages had drifted from upstream and every call still parses identically.**
  280–450 bytes different each, 3,408 calls unchanged. Refreshing one for real archived the old
  bytes into `superseded/` and the frozen dataset still rebuilt byte-for-byte. The HTML churns and
  the content does not, which is the evidence behind `predictions.csv` being committed rather than
  re-scraped.
- **`epl.pundits.live` has two measurements, not one, and only one can be taken today.** `lag` looks
  backwards and needs played matches, so the corpus answers it. `coverage` looks forwards and needs
  Fixtures that have not kicked off, which the corpus by definition lacks — those come from
  `fixtures.csv` at #17. Do not merge them: that would report "cannot tell yet" as `sealable=False`,
  which is the one answer this finding must not give by accident.
- **Open risk 2 half closed on the same day.** `mmz4281/2627/E0.csv` now exists and parses — 10
  rows, with `Avg*` odds. `fixtures.csv` still holds **no E0 rows**: five rows, one National League
  and four Spanish, one day before a Premier League round. Upcoming Fixtures are what #17 needs.

**Stage 13 is built**: the live loop (`src/epl/live/`, issue #17) — the rolling fixtures file turned
into the one Prediction Round that can be sealed right now (`upcoming.py`), every registered
Predictor run over it and the result sealed and committed (`seal.py`), and the Season in progress
scored retrospectively on its own board. `python -m epl.live upcoming | seal | score`.

**Ingesting 2026/27 changed nothing that is scored, and it was checked.** With the Season in progress
in the corpus (82 matches across four tiers, 10 of them Premier League), `python -m epl.ledger
backfill` rewrites all nine files **byte for byte identically** and the scoreboard above is
unchanged. The reason is the calendar — May to August, so no scored round's As-Of Instant reaches the
new Season — which is why the separation is free today and no reason to rely on it. The guard is
`EVALUATION_WINDOW` staying at `range(2005, 2026)`.

**The code is built and tested end to end; its input is not proven.** Open risk 2 is **documented
rather than closed**, and this is the thing to know before planning anything live:

- **`fixtures.csv` has never been seen carrying a Premier League row.** Two fetches — 21 Aug 2026
  (the evening round one kicked off) and 27 Aug 2026 (the day before round two) — held 3 and 5 rows
  and **no E0**. The second was two days stale: `Last-Modified` was Tue 25 Aug 09:59 GMT, and even
  that batch, generated three days before a Premier League round, carried none. Its forward horizon
  at generation is about two days, which is shorter than a Prediction Round's own window.
- **It is not that the file omits English football.** The first fetch carried an `E2` tie. It has
  simply never been seen carrying the one tier this project predicts. Two fetches cannot prove a
  negative, which is why this is measured rather than closed — `python -m epl.live upcoming` asks the
  question on any given day and writes nothing.
- **This puts issue #18's premise in doubt.** API-Football was deferred *because* "`fixtures.csv`
  already carries upcoming Fixtures with the Market Line". The stub should record the measurement,
  not the assumption.
- **The live Season Projection is blocked on the same input, and was not built.** A projection
  simulates every *remaining* Fixture of the campaign and `slate_at` says those must come from
  `fixtures.csv`; a two-day horizon cannot supply the rest of a season. Worth an issue.
- **The Pundit column on the live board is empty by design.** Filling it means folding a Season in
  progress into the committed, frozen `predictions.csv`, which is the one thing issue #11 froze it
  against. The right moment is when the archive's page for the Season is complete.
- **The loop is schedulable and is deliberately not scheduled.** Both write commands are idempotent
  inside a round, which is all a cron entry or a workflow needs; none is committed. There is nothing
  to seal yet, and a schedule that fetches upstream and pushes commits to this repository is the
  repository owner's call rather than a default.

**Stage 15 is built**: the schedule (`deploy/`, issue #19) — the live loop running unattended in a
container on a Raspberry Pi, fired by the Pi's own crontab. `deploy/run_live.sh seal --push`.

The two decisions #19 asked to be made deliberately were made by the repository owner on 27 Aug 2026
and are recorded in docs/DECISIONS.md, "The schedule, and where it runs":

- **A Pi in a container, not GitHub Actions.** Actions was weighed and rejected: `data/raw/` and
  `data/processed/` are gitignored and `score` rebuilds the match table from all 108 files, so every
  run would cold-fetch the whole corpus twice a week forever. The price accepted is real and is
  **open risk 6** — a round whose window passes while the Pi is off is lost, and `supersede` refuses
  a round after kickoff on purpose. Tolerable only because there is nothing to seal yet; revisit it
  the day `fixtures.csv` starts carrying Premier League rows.
- **An automated push is acceptable**, opt-in via `--push`, over a deploy key. A commit proves *when*
  to whoever can reach the machine holding it, and on a Pi that is nobody — so an unattended loop
  that does not push has not finished sealing anything. A failed push exits 1 and says `NOT PUSHED`,
  as loudly as a failed commit, because the file on disk looks identical either way.

Four things about stage 15 worth knowing:

- **The image is the environment; the repository is bind-mounted over it.** So `git pull` updates the
  code with no rebuild, and a sealed round lands in a real checkout rather than in a container layer
  that evaporates — which is not convenience, since evidence nobody can inspect is not evidence.
- **The Pi's other tenant is a venv and stays one.** A paper-trading project on the same box installs
  as prebuilt aarch64 wheels specifically to avoid a toolchain; this project needs conda-forge and a
  pinned BLAS provider (ADR 0009). The container exists to stop two package managers competing for
  one machine's system libraries, and for no other reason. The two share the crontab and a log-block
  convention and nothing else — deliberately not an interpreter, not a virtualenv, not a process.
- **18:30 is a retry and it still pushes.** A second fire inside a round finds it already sealed,
  writes nothing and adds no commit — and pushing is then the whole of what is left for it to do,
  which is the point: the run that seals a round is exactly the run that may have been unable to
  reach the network.
- **Nothing here schedules `backfill`.** It would regenerate Predictions the loop had sealed — ADR
  0005's exact failure — and move the Evaluation Window's numbers on a timer.

**Stage 14 is built**: the deferred-v2 stubs (`src/epl/v2/`, issue #18) — the XGBoost/ML layer, the
Golden Boot player model and the API-Football client, written down instead of built (decision 12).
Prose and named constants, no functions and no classes, imported by nothing. Deleting the directory
would break no import and move no number, and `tests/v2/test_stubs_are_unreachable.py` is what keeps
that true rather than assumed.

Three things about stage 14 worth knowing:

- **They are Python modules and not a docs page, because criterion 5 is only checkable that way.**
  "None of the stubs is imported or executed by the pipeline" is satisfied trivially and
  unfalsifiably by Markdown; under `src/epl/v2/` it is satisfied by a test that walks every import
  in `src/epl`. That test reads `from epl import v2` as well as `import epl.v2` — recording only the
  module half of a `from X import Y` would let the most natural spelling of the violation through
  the one test that exists to catch it.
- **`epl.v2.STUBS` names its members as strings and does not import them.** If `__init__` imported
  the three, then `import epl.v2` would execute all three, and the guarantee would rest on nothing
  importing the *package* either — a weaker claim and a harder one to check. Do not "tidy" it into
  imports.
- **`api_football.py` records a measurement, not a rationale.** Its stated reason for deferral is
  the premise stage 13 could not confirm, so it carries `FETCHES_MEASURED`, `PREMIER_LEAGUE_ROWS_SEEN
  = 0` and `WHAT_WOULD_REVIVE_IT` as data. A third fetch of `fixtures.csv` is an append to a tuple.
  Update it when `python -m epl.live upcoming` is run on a Friday, whichever way it comes out.

Seven things about stage 11 worth knowing before building on it:

- **The validation numbers in the docs came from a reduced run, and that is stated everywhere they
  appear.** 60 fits at `--draws 300 --tune 500 --chains 2` took 98 minutes; the same 60 at the
  shipped sampler settings is about nine hours, and all six checkpoints is nineteen. Do not quote
  the numbers as though they came from the full run, and do not quietly re-run at other settings and
  overwrite them without saying so.
- **The walk is not the expensive half and never was.** Ten thousand Seasons over half a Season of
  Fixtures and 4,000 draws takes **2.7 seconds**; the posterior fit in front of it took **537**. So
  re-walking one fit with a different seed, different bands or a sensitivity check is free, which is
  why `project` takes an already-fitted `posterior` and `simulate` is a separate function. Note that
  537 s is well above stage 10's measured 231 s at the first round of 2015/16 — budget from the
  higher number before planning a run of a hundred and twenty-six of them.
- **Drawing from the posterior is worth what ADR 0007 said it was, measured on real football.** At
  2011/12's first checkpoint, Manchester United's title probability is **0.853 from the posterior
  mean, 0.850 from the MLE and 0.772 from the draws**. The point estimate and the mean agree with
  each other and disagree with the honest answer — which is the whole argument, and the reason
  `Posterior.mean()` must never be what a projection runs on.
- **Points ties are routine; the head-to-head steps have never once been needed.** 24 of 26 Seasons
  had a pair level on points, 85 pairs in all — and goal difference or goals scored settled every
  single one. That is why `Finish.level_pairs` exists: a projection says how many of its ten
  thousand tables reached the lower half of the chain instead of assuming the answer.
- **The corpus knows who won and the projection must not.** `tests/simulate/test_projection_over_the_corpus.py`
  projects 2015/16 from Christmas, when every remaining result is sitting in the match table it was
  handed, and checks Leicester's title probability is between 0 and 0.5. It comes out **0.058**,
  behind Arsenal's 0.517. They won. That number is the worked example everywhere in the docs, and it
  is quoted rather than a comfortable one on purpose.
- **`epl.simulate.table` is the seam, and it must stay ignorant.** It takes Fixtures and goals and
  returns positions, so every step of the chain is exercised against hand-built leagues of three and
  four Clubs. The tests that carry the weight are the ones where two steps disagree. Do not give it
  a posterior, a Predictor or an As-Of Instant.
- **`validate` hands its caller every result the moment it exists**, and the command line writes the
  file on every one. A run of this length that returned nothing until the end loses everything to a
  single interruption — which happened once during the build, at 20 of 60 fits.

Six things about stage 10 worth knowing before building on it:

- **The priors are scaffolding and must stay too wide to do anything.** A perfectly reasonable
  strength prior of 0.5 pulls fitted attack to **0.65 times** the MLE's — Darlington from −0.645 to
  −0.127 — and moves one Fixture's Home probability by 0.079. That is textbook shrinkage and it
  is *wrong here*: `epl.models.dixon_coles` "regresses no Club to the mean and carries no prior", so
  a shrinking posterior is a different model, which is the one thing ADR 0007 exists to prevent.
  `strength_sigma = 2.0` puts every strength this project has ever fitted inside half a standard
  deviation. Do not "tighten these sensible defaults".
- **Attack and defence are built the same way on purpose, and the obvious version is wrong.** Both
  are zero-sum deviations from a `scoring_level` that carries the pyramid's goal rate. A plain
  Normal on each defence — the obvious way — constrains not just the spread of defence but its
  *mean*, which is that goal rate, and the MLE has no opinion about it. That asymmetry alone left
  defence at 0.86× the MLE while attack had recovered to 1.005.
- **`log_likelihood_at` refuses part of the parameter space, and deleting that guard breaks the fit
  in a way that looks like slowness.** Dixon-Coles' correction turns Scoreline probabilities
  negative beyond `rho ≈ 0.35`, where `CORRECTION_FLOOR` keeps the log-density smooth while putting
  `1/1e-12` into the gradient — 229 at `rho = 0.30`, **9.3e12** at 0.35. NUTS takes one step against
  that, overflows, and gets a `nan`, which never trips a divergence check because comparisons
  against `nan` are false. Measured: 1,023 leapfrog steps per draw against 7, and a posterior mean
  0.85 log-goals from the MLE. The clamp is right for L-BFGS-B and a trap for a sampler.
- **About 3.3% of draws are divergent, and that is a dozen fourth-tier Clubs rather than a defect.**
  The correlation between a Club's log weight and its posterior width is **−0.977**; Grimsby carries
  half a weighted match, a posterior sd of 1.18, and draws reaching an attack of 5.85. Narrowing the
  prior would remove them and bring the shrinkage back. `target_accept` is 0.95 because 0.99 could
  not finish a corpus-scale fit in twenty minutes.
- **Defence comes back at 0.875× the MLE and that is not shrinkage** — the Clubs carrying it are
  pushed *further* from zero than the MLE puts them, which is what a posterior mean does to a
  weakly-identified skewed log-rate. Same handful of Clubs as the divergences.
- **The posterior mean is not what a Season Projection runs on.** `Posterior.mean()` exists to be
  compared against the MLE; collapsing the draws to it throws away exactly the parameter uncertainty
  the expensive fit was run to capture, which is the 48%-versus-34% title probability in ADR 0007.

Seven things about stage 9 worth knowing before building on it:

- **The 0.0019 it takes out of Elo does not show up on accuracy** — 0.5360 against Elo's 0.5380,
  slightly *worse*. The two models pick nearly the same winners; the goals model is better
  calibrated about how sure it should be, which is exactly what RPS measures and accuracy does not.
  Do not report the accuracy column as though it disagreed with the headline.
- **`epl.models.likelihood` is the seam ADR 0007 asks for, and it is deliberately ignorant.** It
  holds `Sample`, `Decay`, `Strengths`, the log-likelihood with its analytic gradient, the
  correction and the Scoreline grid — and imports nothing from `dixon_coles` or from any Predictor.
  Issue #14 fits the same `Sample` and returns draws of the same `Strengths`.
- **A fitted `Strengths` is meaningless until it is centred.** Adding a constant to every attack
  *and* every defence changes no rate, so the likelihood is flat along one direction and a fit is
  an arbitrary point on a line. `fit` applies `centred()` on the way out; two fits are not
  comparable and no attack table is readable without it.
- **The gradient is analytic, and the test that keeps it true differences the function itself.**
  Two hundred parameters differenced is the difference between a five-minute backfill and most of
  a day. A derivative that drifted from its own function would converge slightly wrong and look
  entirely normal on the scoreboard.
- **The half-life is a region, not a number.** 322.5 days is the fit; anything from 270 to 480
  scores within 0.0001 RPS. The short end is what the data excludes — 60 days costs 0.007 RPS. And
  the weight floor beneath it is a stated tolerance, not a knob: five times it moves the Burn-In
  score by 0.00001.
- **The iteration ceiling is 10,000 and was set from measurement.** A fit takes a mean of 251
  L-BFGS-B iterations and a worst of 1,454 over a 136-round sample, but the tail is longer than
  that: at a ceiling of 2,000 the per-Fixture diagnostic found a cut that needed more, and the run
  died on it. A ceiling near the observed worst case turns a slow fit into a failed run. The
  tolerances were not touched — the *objective* tolerance is what terminates every fit, and
  loosening the gradient tolerance tenfold changes not one iteration count.
- **The backfill now takes about six minutes**, five of them Dixon-Coles refitting ~230 parameters
  at each of 952 rounds. `python -m epl.models fit` takes several minutes more, because a half-life
  can only be judged by predicting with it and every candidate walks the Burn-In rounds.

Five things about stage 8 worth knowing before building on it:

- **The cost of stating certainty is 0.1209 and 0.1235 RPS**, and that is the deliverable. Over the
  Fixtures every Predictor reached, the market scores 0.1943/0.1968, Elo 0.2016/0.2031, the
  Calibrated Pundit **0.2127/0.2111**, the floor 0.2356/0.2322 and the as-stated Pundit
  0.3335/0.3346. **Read fairly, the same calls beat the floor they were a tenth of a point below**
  — that sentence is what issue #12 existed to make true. And accuracy barely moves across the two
  readings (0.5102 → 0.5116, 0.4925 → 0.4993), so the 0.12 is the format of the question rather
  than a different set of opinions. `python -m epl.pundits three-way`.
- **A Calibrated Pundit is a Predictor, not a scoring step**, and that is the one structural
  difference from `epl.calibration`. Its map is fitted on matches that had *already kicked off* at
  the As-Of Instant and its input was published before it, so it can genuinely be quoted forward.
  It reads its history through `Evidence` like Elo, stores rows, and audits — `inputs_seen > 0`
  where a Pundit records 0. Do not move it to scoring time; the whole point is that it need not be.
- **Nothing chooses the buckets and nothing enforces monotonicity.** The margins each Pundit
  actually called are the buckets, and a bucket too thin to carry a rate merges with its neighbour
  nearer zero — so Lawrenson ends with `-3,-2 | -1 | 0 | 1 | 2 | 3,4` and Sutton with
  `-5,-4,-3,-2 | -1 | 0 | 1 | 2 | 3,4,5,6`, from the sample rather than from a cap. Both come out
  monotone in the Home rate at every step (+1 goes Home 42%/48%, +3 or better 83%/81%), and that is
  a measurement rather than a constraint. `MINIMUM_SAMPLE = 40` is stated, not fitted, because it
  *cannot* be fitted: ADR 0008 allows tuning only in the Burn-In Window and no Pundit published a
  call within a dozen years of it.
- **The shared layer's 0.09 RPS gain disappears once the margin map runs first** — it costs 0.0014
  and 0.0015, exactly what it costs Elo and the market, and ten-bin error falls 0.327 → 0.019 and
  0.338 → 0.020. That confirms stage 6 from the other direction rather than overturning it. The
  residual 0.019 against the other four's 0.006 is the map's seven buckets showing.
- **`epl.ledger.scoreboard.lines` was split out of `build` so the three-way comparison can cut the
  slate without cutting the calibration.** Each Predictor is calibrated over its whole track record
  and *then* scored on the shared Fixtures. Do not collapse it back: cutting first would give the
  Market Line a calibrated form fitted on a Pundit's 1,900 Fixtures rather than its own 7,980, a
  number that exists nowhere else. Narrow the comparison, never the Predictor (ADR 0001's rule).

Four things about stage 7 worth knowing before building on it:

- **The shared calibration layer gains a Pundit ~0.09 RPS where it cost the other four ~0.001.**
  That is the re-measurement stage 6 asked for by name, and it *confirms* stage 6 rather than
  overturning it: the layer was never broken, it had nothing to find. All four earlier Predictors
  arrive at a ten-bin error of about 0.006; a Pundit arrives at 0.33 and the same unchanged layer
  recovers most of it. **This is not the Calibrated Pundit**, which stage 8 built as a separate
  map bucketed by predicted goal margin. The two must not be collapsed into one: the shared layer
  sees a one-hot Prediction with no Scoreline left in it, and gets to 0.2374/0.2473 where reading
  the margin gets to 0.2127/0.2111.
- **The as-stated number is worse than the floor, and that is the deliverable.** 0.334 against a
  Naive Baseline of 0.236 over the same Fixtures. On accuracy the same calls beat that floor
  0.5095 to 0.4388 and trail the market by four points. Both readings are published; either alone
  is an argument (ADR 0003). `python -m epl.pundits grades` is the lay pair beside it — 11.0% and
  9.1% exact scores, 50.9% and 49.2% correct Outcomes. Each Pundit's `note` carries four things
  onto the scoreboard and must keep carrying all four: the BBC as the origin, that 1,896 and
  1,512 Fixtures are not the board's 7,980 so the RPS is not comparable, what the as-stated
  reading is, and which Predictor the fair reading is.
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

**Every ticket through #19 is built.** The modelling stages are done, the live loop is built, the
deferred features are written down, and the schedule that runs the loop is committed under `deploy/`.
What is left is one unproven input and three issues worth filing.

**The schedule was built knowing it will seal nothing, and that is not a contradiction.** #19's
fourth acceptance criterion names the case by name — "the *only* case until upcoming Fixtures have a
source" — so the ticket was written to be built despite the blocker rather than after it. The
schedule is installed now so that the day upstream starts listing Premier League Fixtures, the loop
is already watching. What it cannot do is tell anybody that day has come: a fire that finds no E0 row
is a quiet success by design, so the loop will simply start sealing rounds and nothing will announce
it. `deploy/logs/live_loop.log` is where that shows up.

**The one thing to re-check periodically**: run `python -m epl.live upcoming` on a Friday afternoon
of a Premier League round — or just read the log, which now asks twice a week. If `fixtures.csv`
carries E0 rows, the live half works and `seal` will write, commit and push the round. If it does
not, upcoming Premier League Fixtures still have no confirmed source and that is the blocker for
everything live — including the weekly Season Projection. Whatever it says, record the date and the
result in three places: beside the fetch table in docs/DECISIONS.md, in
`epl.v2.api_football.FETCHES_MEASURED`, and — if a Premier League row ever appears — in
`PREMIER_LEAGUE_ROWS_SEEN`, which is the number that decides whether that stub stays a stub. Three
fetches are recorded there as of 27 Aug 2026 and all three found nothing.

**#20 is filed and open: a Telegram bot** — seal and score notifications, and the board on demand.
Its first acceptance criterion is the gap #19 closed leaving open: the schedule now asks upstream
twice a week whether `fixtures.csv` carries a Premier League row, and a fire that finds none is a
quiet success, so *nothing announces the day the answer changes*. A bot is what would. Note the
constraints the ticket carries — it must be read-only with respect to both stores (a chat app is not
a second door into `outputs/live/`), and it must not quote a calibrated Live Season figure, the
sequential diagnostic, accuracy as a headline, or a Pundit without its `note`.

**Three issues worth filing, none of them in scope anywhere yet:**

- **A confirmed source of upcoming Premier League Fixtures.** The live loop is complete, scheduled
  and idle without one. API-Football is the named candidate and `epl.v2.api_football` is where the
  case for it lives — including the conditions that would revive it. #20 is partly blocked on it
  too, though the useful half of that ticket is not.
- **The live Season Projection.** `projection_rounds(..., live=True)` exists and `slate_at` cannot be
  fed: a projection needs every remaining Fixture of the campaign, and the rolling file's horizon is
  two days. Blocked on the same source as above.
- **Eight more Seasons of Pundit calls.** The archive's index links eighteen Season pages, 2009/10
  onward, where the backfill uses nine. The eight before 2017/18 are ~3,000 more Lawrenson calls on
  the same source in the same shape.

The model target is met, so there is no longer a pending question about accuracy.

Check the graph before starting:

```
gh issue view <n> | sed -n '/## Blocked by/,$p'
```

```
conda env create -f environment.yml
conda activate epl-predictor
python -m epl.ingest fetch     # fill data/raw/ — 108 files, 27 Seasons x 4 tiers
python -m epl.ingest build     # write matches.csv (52,672 closed + the Season in progress)
python -m epl.pundits fetch    # cache the nine MyFootballFacts season pages
python -m epl.pundits build    # re-freeze predictions.csv — 3,408 calls, cross-checked
python -m epl.pundits grades   # exact-score and correct-Outcome rates per Pundit and Season
python -m epl.pundits three-way      # the three-way board and the cost of stating certainty
python -m epl.pundits calls    # every call ranked by the miss its fair reading still had
python -m epl.pundits map      # what a call of each predicted goal margin is worth
python -m epl.pundits live     # the Season in progress: what the archive has, and how late
python -m epl.ledger backfill  # walk every registered Predictor over the Evaluation Window
python -m epl.ledger scoreboard      # every metric twice, pre- and post-calibration
python -m epl.ledger reliability     # the 10-bin diagrams per Predictor, in both forms
python -m epl.ledger audit     # re-check both stores and the seal on outputs/live/
python -m epl.benchmarks overround   # the margin in each book, per Season and tier
python -m epl.benchmarks methods     # the three vig removals compared on one book
python -m epl.models fit       # re-derive both frozen fits on the Burn-In Window
python -m epl.models draws     # the draw rate against Supremacy, predicted and observed
python -m epl.models ratings   # the Elo pool at a Season's first Prediction Round
python -m epl.models strengths # Dixon-Coles' attack and defence at the same instant
python -m epl.models sequential      # ADR 0002's diagnostic: per Fixture against per round
python -m epl.simulate checkpoints   # where a Season is projected from, and where it is not
python -m epl.simulate posterior     # fit one posterior, and read it beside the MLE
python -m epl.simulate project       # one Season Projection: title, Europe and relegation
python -m epl.simulate validate      # project completed Seasons; where the real champion landed
python -m epl.live upcoming    # what the rolling fixtures file holds, and what could be sealed
python -m epl.live seal        # predict the upcoming round, write it to outputs/live/, commit it
python -m epl.live seal --push # and push it — what makes it evidence off the machine that made it
python -m epl.live score       # ingest results, then score what has been sealed
deploy/run_live.sh seal --push # the same, in the container the Pi's crontab fires (deploy/README.md)
pytest                         # add --run-network to also hit football-data.co.uk
```

`backfill` now takes about six minutes, five of them Dixon-Coles: both models rebuild from cold at
every round on purpose (see above). `python -m epl.models sequential` is several times that again,
and `--seasons FIRST LAST` is how to want less of it.
