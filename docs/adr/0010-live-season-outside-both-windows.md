# Ingest the Season in progress, score it on its own board, and never backfill it

The live loop (issue #17) needs the current Season in the corpus for two reasons that have nothing to
do with each other. A Predictor has to *see* it — otherwise Elo and Dixon-Coles forecast round twenty
having last watched football in May — and a Sealed Prediction has to be *joinable to a result*, or it
can never be scored. So `epl.windows.LAST_SEASON` moves at the start of each campaign, and the
leakage protocol's own module is where that move is made.

The question that needed deciding is what the Evaluation Window does when it moves. It does nothing:
it stays at 2005/06–2025/26. The Season in progress is a third span, `epl.windows.LIVE_SEASON` —
ingested, never fitted on, never backfilled, and scored on its own board at
`outputs/live_scoreboard.csv`.

Two reasons, and the second is the serious one.

**A headline that changes shape every week is not a headline.** Folding the live Season into the
Evaluation Window would grow the scored slate by ten Fixtures each Saturday, so `0.1975 RPS` this
week and `0.1975 RPS` next week would be numbers over different populations. Everything this project
publishes is a comparison, and a comparison needs a fixed slate.

**A backfilled live Season is the exact failure ADR 0005 exists to prevent.** `python -m epl.ledger
backfill` walks the Evaluation Window and *regenerates* Predictions. Point it at a Season the live
loop has sealed and it produces a second Prediction for each of those Fixtures — from a corpus that
has since been backfilled with odds and results that did not exist when the sealed one was made. The
scoreboard would then be scoring the regenerated row beside the sealed one, and nothing in the code
would look wrong. Keeping the live Season out of the window is what makes that impossible rather than
merely discouraged.

Measured rather than argued: with 2026/27 ingested, `python -m epl.ledger backfill` rewrites all
nine Backtest Prediction files **byte for byte identically**, and the scoreboard prints the same
0.1936 / 0.1975 / 0.1994 / 0.2294 it printed at stage 9. The last Fixture of 2025/26 was played in
May and the first of 2026/27 in August, so no scored round's As-Of Instant reaches the new Season —
which is the reason the separation costs nothing today, and no reason at all to rely on it.

## Consequences

`matches.csv` grows every Saturday. Every fixed count in the corpus tests is therefore taken over the
26 *closed* Seasons — 52,672 matches, 42,792 below the Premier League, nine rows missing match
statistics — and the Season in progress is checked for being present and partial instead. A count
that included it would be a test that failed weekly and told nobody anything.

The live Season is scored pre-calibration only. The shared isotonic layer is fitted walk-forward on a
Predictor's own past Predictions and needs a track record behind it (ADR 0006); a Season in progress
has none, so a calibrated column there would be the raw one under another name.

`LIVE_SEASON` is defined as `LAST_SEASON` rather than as a second literal, so there is one place to
move and no way for the two to disagree. It is still a constant that can go stale, so
`epl.live.upcoming.to_predict` refuses a Season the corpus does not hold and refuses one it holds a
complete campaign of — a Season that is over has nothing upcoming, which is exactly what a stale
`LIVE_SEASON` looks like from the loop.
