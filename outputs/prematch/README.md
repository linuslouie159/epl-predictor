# Pre-Match Readings

What the models said about one Fixture roughly an hour before it kicked off, **after** its
Prediction Round had already been sealed.

One file per calendar day, named after the day the matches were played on. Written by
`python -m epl.live prematch`, which the Pi's crontab fires every half hour on a matchday, and read
by `python -m epl.bot notify prematch`, which sends one card per match.

## What this is not

**It is not the track record.** The forecasts this project is scored on live in `../live/`, are
stamped before their whole round kicked off, and are the only thing `python -m epl.live score` and
the scoreboard ever read. Nothing in this directory reaches either.

That separation is the entire reason this directory exists rather than these rows going into
`../live/`. A Reading is stamped *later* than the round it belongs to, and in the sealed store a
later As-Of Instant for the same Predictor and Fixture means one thing: a superseding revision
correcting a bug (ADR 0005). The scoreboard keeps the latest instant per Fixture. Putting Readings
there would quietly swap the honest before-the-round forecast for one taken after two of that
round's matches had already been played — on every Fixture, every week — and the live track record
would improve for a reason nobody could see.

## Why it is committed anyway

Because it cannot be regenerated. A Backtest Prediction is reproducible, which is what makes
`../backtest/` disposable and gitignored. A Reading was cut from a corpus that has grown since it
was taken, so re-running the same code tomorrow gives a different number and there is no way back
to this one. A file nobody can date proves nothing, so these are committed and pushed like the
sealed rounds beside them.

## The open question

Whether a Reading is actually *better* than the Prediction it was taken after is not answered here
and needs a season of them. This store exists so the question can be asked later. Until it is
answered, the sealed forecast remains the one that is quoted as the record — including in the
messages that carry a Reading, which say so every time.

See `CONTEXT.md` for the term, `docs/DECISIONS.md` ("The pre-match message, and a third store") for
the decision, and `src/epl/ledger/readings.py` for the rules.
