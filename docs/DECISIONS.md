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
| 15 | Dixon-Coles: MLE at all 1,332 Prediction Rounds, Bayesian only where a Season Projection is produced | [0007](./adr/0007-mle-for-matches-bayesian-for-projections.md) |
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

## Measured facts

Derived from the source data during design. Recorded so they need not be re-derived — but re-verify
before relying on any of them.

### Football-Data era boundaries (English tiers)

| From | What appears |
|---|---|
| 1993/94 | Result only. 1993/94–1994/95 have 462 fixtures (22 clubs) |
| 2000/01 | Match stats: shots, shots on target, corners, fouls, cards, referee. 100% populated |
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

### Other measurements

- **Prediction Rounds**: 1,332 across 2000/01–2025/26 (mean 51.2 per season, 7.42 fixtures per round).
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

## Open risks

1. **BBC live scraping is unproven.** `www.bbc.co.uk` was unreachable during design, article URLs are opaque IDs (`/sport/football/articles/cvg0e92ezz4o`, legacy `/sport/football/28859459`) and there is no index page. Needs a spike at stage 5. If it fails, live pundit data has no confirmed source — MyFootballFacts' update latency during a season is unknown.
2. **The live path is untested.** As of 20 Aug 2026, `fixtures.csv` held no E0 rows and `mmz4281/2627/E0.csv` did not exist. The 2026/27 season starts 21 Aug 2026; both should appear once matches begin. Verify before relying on the live loop.
3. **MyFootballFacts parseability is unverified.** Content correctness was confirmed — a 2025/26 result cross-checked exactly against Football-Data — but the HTML has not been parsed across all nine season pages.
4. **Cross-tier Elo has no burn-in before 2000/01**, so early ratings linking E0 to E3 will be unreliable. This sits inside the Burn-In Window so it should not reach scored results, but it is worth watching.
5. **Frozen hyperparameters will drift out of date** by the late Evaluation Window, given the measured decline in home advantage. Accepted deliberately; see ADR 0008.
