"""Benchmarks: the Market Line, the Ceiling Line and the Naive Baseline.

Built by issue #8.

The **Market Line** is the opponent: vig-removed implied probabilities from the market-average
*pre-match* odds, scoring ~0.1936 RPS. Pre-match rather than closing, against convention, because
closing odds absorb team news the model and the Pundits cannot have — scoring against them would
produce a loss that says nothing about model quality (ADR 0001).

The **Ceiling Line** is the same treatment of closing odds, available from 2019/20. Reference upper
bound only, always labelled as knowing more than we do. It is never the headline opponent.

The **Naive Baseline** is the floor: Outcome base rates with no knowledge of which Clubs are
playing, scoring ~0.2292 RPS. Anything that fails to beat it has no value. Its base rates are
estimated only from Seasons already seen, so even the floor is leak-free.

Vig removal implements normalisation, the power method and Shin's method behind one interface, with
**Shin as the default**. They differ by 0.0002 RPS, so the choice is near-immaterial for
benchmarking — but power and Shin correct favourite-longshot bias, which matters if value betting is
ever explored. The overround is reported alongside every Market Line so the removal can be
sanity-checked rather than trusted; it measures 1.0562 over the Evaluation Window.

Which Seasons carry a market at all is recorded by the ingest, not re-derived here — see
`epl.ingest.season_coverage`.
"""

__all__: list[str] = []
