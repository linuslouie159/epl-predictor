# Run Elo across all four English tiers, not just the Premier League

A reader of an "EPL predictor" will wonder why the ingest pulls League Two results. The reason is
promoted clubs. Over 2001/02–2025/26, promoted clubs averaged 36.9 points against a league average
of 52.0, losing 51.5% of matches — but that penalty is not stable, swinging from 44.0 points for the
2010/11 intake to 19.7 for 2024/25. Any fixed starting rating is wrong by a drifting amount, and any
Premier-League-only prior gives all three promoted clubs the same rating on day one.

Ingesting E0–E3 lets a promoted club arrive with a rating it earned, keeps relegated clubs updating
instead of freezing, and removes the special case for yo-yo clubs entirely. Football-Data serves all
four tiers with an identical schema back to 2000/01, so the cost is one loop in the loader and
~43,000 additional matches — trivial for an O(n) online algorithm.

All four tiers were chosen over Championship-only deliberately: adding tiers later would invalidate
every stored backtest and force a full re-run, which is precisely the retrofit this avoids.

## Consequences

Elo ratings must be comparable across tiers connected only by promotion and relegation; this
resolves given 26 seasons of burn-in but the first seasons' cross-tier ratings should not be trusted.
Clubs promoted into League Two from the National League still start cold — accepted, as they are four
tiers from the Premier League. Football-Data's 552 rows per tier are regular season only, so
play-off matches do not contribute to ratings. Cross-source Aliases remain necessary only for
Premier League Clubs, since Understat, FBref and BBC do not cover the lower tiers.
