# Split the Prediction ledger into a regenerable store and a sealed store

Two stores for one kind of record looks like duplication, so the reason needs recording. A Backtest
Prediction is reproducible — rerun the pipeline and it comes back identical — so storing it is a
convenience and it can be deleted and rebuilt freely. A Sealed Prediction is evidence, and evidence
that can be silently regenerated is not evidence.

The failure mode is concrete: Football-Data's current-season file grows weekly and backfills results
and odds into rows already published. Regenerating last Friday's live prediction a month later can
feed it odds that did not exist when the prediction was claimed to have been made, and nothing in the
code would flag it. The live accuracy log — the one artifact in this project that is worthless if not
trustworthy — would quietly become fiction.

`outputs/backtest/` is therefore regenerable, deterministic and gitignored. `outputs/live/` is
append-only: one file per Prediction Round, written before that round's first kickoff and committed,
so git history is itself the proof of when each Prediction existed.

## Consequences

Both stores share one row schema so scoring code never needs to know which it is reading. A test
asserts that no file in `outputs/live/` changes after its round's first kickoff. Correcting a genuine
bug in a sealed round means adding a superseding row with a new As-Of Instant, never editing history.
