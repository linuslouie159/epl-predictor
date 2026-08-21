"""Rebuild the canonical Club table and its Aliases from the raw cache.

Tooling, not pipeline code — nothing in the ingest or the models calls this. It exists so the
committed ``clubs.csv`` and ``aliases.csv`` have stated provenance and can be regenerated rather
than hand-maintained:

    python -m epl.clubs.build

The mapping below is the judgement: which of Football-Data's spellings is which Club, and what to
call that Club. Everything else is derived. The build **fails** if the raw cache contains a
spelling the mapping does not cover, or if the mapping names a Club the cache never fielded — so
the table cannot drift away from the data it describes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epl.clubs.table import (
    Club,
    ClubResolver,
    load_aliases,
    load_clubs,
    write_aliases,
    write_clubs,
)
from epl.ingest import SOURCE, club_names_in_raw_cache
from epl.paths import processed_dir

# Football-Data's spelling -> (slug, display name).
FOOTBALL_DATA = {
    "AFC Wimbledon": ("afc_wimbledon", "AFC Wimbledon"),
    "Accrington": ("accrington", "Accrington Stanley"),
    "Aldershot": ("aldershot", "Aldershot Town"),
    "Arsenal": ("arsenal", "Arsenal"),
    "Aston Villa": ("aston_villa", "Aston Villa"),
    "Barnet": ("barnet", "Barnet"),
    "Barnsley": ("barnsley", "Barnsley"),
    "Barrow": ("barrow", "Barrow"),
    "Birmingham": ("birmingham", "Birmingham City"),
    "Blackburn": ("blackburn", "Blackburn Rovers"),
    "Blackpool": ("blackpool", "Blackpool"),
    "Bolton": ("bolton", "Bolton Wanderers"),
    "Boston": ("boston", "Boston United"),
    "Bournemouth": ("bournemouth", "Bournemouth"),
    "Bradford": ("bradford", "Bradford City"),
    "Brentford": ("brentford", "Brentford"),
    "Brighton": ("brighton", "Brighton & Hove Albion"),
    "Bristol City": ("bristol_city", "Bristol City"),
    "Bristol Rvs": ("bristol_rovers", "Bristol Rovers"),
    "Bromley": ("bromley", "Bromley"),
    "Burnley": ("burnley", "Burnley"),
    "Burton": ("burton", "Burton Albion"),
    "Bury": ("bury", "Bury"),
    "Cambridge": ("cambridge", "Cambridge United"),
    "Cardiff": ("cardiff", "Cardiff City"),
    "Carlisle": ("carlisle", "Carlisle United"),
    "Charlton": ("charlton", "Charlton Athletic"),
    "Chelsea": ("chelsea", "Chelsea"),
    "Cheltenham": ("cheltenham", "Cheltenham Town"),
    "Chester": ("chester", "Chester City"),
    "Chesterfield": ("chesterfield", "Chesterfield"),
    "Colchester": ("colchester", "Colchester United"),
    "Coventry": ("coventry", "Coventry City"),
    "Crawley Town": ("crawley", "Crawley Town"),
    "Crewe": ("crewe", "Crewe Alexandra"),
    "Crystal Palace": ("crystal_palace", "Crystal Palace"),
    "Dag and Red": ("dagenham_redbridge", "Dagenham & Redbridge"),
    "Darlington": ("darlington", "Darlington"),
    "Derby": ("derby", "Derby County"),
    "Doncaster": ("doncaster", "Doncaster Rovers"),
    "Everton": ("everton", "Everton"),
    "Exeter": ("exeter", "Exeter City"),
    "Fleetwood Town": ("fleetwood", "Fleetwood Town"),
    "Forest Green": ("forest_green", "Forest Green Rovers"),
    "Fulham": ("fulham", "Fulham"),
    "Gillingham": ("gillingham", "Gillingham"),
    "Grimsby": ("grimsby", "Grimsby Town"),
    "Halifax": ("halifax", "Halifax Town"),
    "Harrogate": ("harrogate", "Harrogate Town"),
    "Hartlepool": ("hartlepool", "Hartlepool United"),
    "Hereford": ("hereford", "Hereford United"),
    "Huddersfield": ("huddersfield", "Huddersfield Town"),
    "Hull": ("hull", "Hull City"),
    "Ipswich": ("ipswich", "Ipswich Town"),
    "Kidderminster": ("kidderminster", "Kidderminster Harriers"),
    "Leeds": ("leeds", "Leeds United"),
    "Leicester": ("leicester", "Leicester City"),
    "Leyton Orient": ("leyton_orient", "Leyton Orient"),
    "Lincoln": ("lincoln", "Lincoln City"),
    "Liverpool": ("liverpool", "Liverpool"),
    "Luton": ("luton", "Luton Town"),
    "Macclesfield": ("macclesfield", "Macclesfield Town"),
    "Man City": ("man_city", "Manchester City"),
    "Man United": ("man_united", "Manchester United"),
    "Mansfield": ("mansfield", "Mansfield Town"),
    "Middlesbrough": ("middlesbrough", "Middlesbrough"),
    "Millwall": ("millwall", "Millwall"),
    "Milton Keynes Dons": ("mk_dons", "Milton Keynes Dons"),
    "Morecambe": ("morecambe", "Morecambe"),
    "Newcastle": ("newcastle", "Newcastle United"),
    "Newport County": ("newport_county", "Newport County"),
    "Northampton": ("northampton", "Northampton Town"),
    "Norwich": ("norwich", "Norwich City"),
    "Nott'm Forest": ("nottm_forest", "Nottingham Forest"),
    "Notts County": ("notts_county", "Notts County"),
    "Oldham": ("oldham", "Oldham Athletic"),
    "Oxford": ("oxford", "Oxford United"),
    "Peterboro": ("peterborough", "Peterborough United"),
    "Plymouth": ("plymouth", "Plymouth Argyle"),
    "Port Vale": ("port_vale", "Port Vale"),
    "Portsmouth": ("portsmouth", "Portsmouth"),
    "Preston": ("preston", "Preston North End"),
    "QPR": ("qpr", "Queens Park Rangers"),
    "Reading": ("reading", "Reading"),
    "Rochdale": ("rochdale", "Rochdale"),
    "Rotherham": ("rotherham", "Rotherham United"),
    "Rushden & D": ("rushden_diamonds", "Rushden & Diamonds"),
    "Salford": ("salford", "Salford City"),
    "Scunthorpe": ("scunthorpe", "Scunthorpe United"),
    "Sheffield United": ("sheffield_united", "Sheffield United"),
    "Sheffield Weds": ("sheffield_wednesday", "Sheffield Wednesday"),
    "Shrewsbury": ("shrewsbury", "Shrewsbury Town"),
    "Southampton": ("southampton", "Southampton"),
    "Southend": ("southend", "Southend United"),
    "Stevenage": ("stevenage", "Stevenage"),
    "Stockport": ("stockport", "Stockport County"),
    "Stoke": ("stoke", "Stoke City"),
    "Sunderland": ("sunderland", "Sunderland"),
    "Sutton": ("sutton", "Sutton United"),
    "Swansea": ("swansea", "Swansea City"),
    "Swindon": ("swindon", "Swindon Town"),
    "Torquay": ("torquay", "Torquay United"),
    "Tottenham": ("tottenham", "Tottenham Hotspur"),
    "Tranmere": ("tranmere", "Tranmere Rovers"),
    "Walsall": ("walsall", "Walsall"),
    "Watford": ("watford", "Watford"),
    "West Brom": ("west_brom", "West Bromwich Albion"),
    "West Ham": ("west_ham", "West Ham United"),
    "Wigan": ("wigan", "Wigan Athletic"),
    "Wimbledon": ("wimbledon", "Wimbledon"),
    "Wolves": ("wolves", "Wolverhampton Wanderers"),
    "Wrexham": ("wrexham", "Wrexham"),
    "Wycombe": ("wycombe", "Wycombe Wanderers"),
    "Yeovil": ("yeovil", "Yeovil Town"),
    "York": ("york", "York City"),
}

# Football-Data's rolling fixtures.csv spells some Clubs more fully than its Season files do.
# Observed, not guessed: an unobserved spelling is left to raise when it first appears.
FIXTURES_VARIANTS = {
    "Sheffield Wed": "sheffield_wednesday",
    "Bradford City": "bradford",
}


def clubs_and_aliases() -> tuple[list[Club], list[tuple[str, str, str]]]:
    """The Club table and Alias rows the mapping above implies."""
    clubs = {slug: Club(slug, name) for slug, name in FOOTBALL_DATA.values()}
    aliases = [(SOURCE, alias, slug) for alias, (slug, _) in FOOTBALL_DATA.items()]
    aliases += [(SOURCE, alias, slug) for alias, slug in FIXTURES_VARIANTS.items()]

    unknown = sorted({slug for _, _, slug in aliases} - set(clubs))
    if unknown:
        raise SystemExit(f"alias variants point at unknown slugs: {unknown}")
    return list(clubs.values()), aliases


def check_against_cache() -> None:
    """Fail if the mapping and the raw cache disagree in either direction."""
    observed = set(club_names_in_raw_cache())
    if not observed:
        raise SystemExit("raw cache is empty; run `python -m epl.ingest fetch` first")

    unmapped = sorted(observed - set(FOOTBALL_DATA))
    if unmapped:
        raise SystemExit(f"spellings in the cache with no Club: {unmapped}")

    unused = sorted(set(FOOTBALL_DATA) - observed)
    if unused:
        raise SystemExit(f"Clubs mapped but never seen in the cache: {unused}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.clubs.build", description=__doc__)
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="rebuild without a populated data/raw/, losing the guard that the mapping and the "
        "cache agree",
    )
    parser.add_argument(
        "--teamname-replacements",
        type=Path,
        default=None,
        help="where to write soccerdata's teamname_replacements.json "
        "(default: data/processed/teamname_replacements.json)",
    )
    args = parser.parse_args(argv)

    if not args.skip_cache_check:
        check_against_cache()

    clubs, aliases = clubs_and_aliases()
    write_clubs(clubs)
    write_aliases(aliases)
    print(f"{len(clubs)} Clubs, {len(aliases)} Aliases")

    # Decision 5: the Club table is the one authority, exported to soccerdata's format rather than
    # maintained twice. soccerdata itself is only needed for Understat and FBref.
    export_path = args.teamname_replacements or processed_dir() / "teamname_replacements.json"
    written = ClubResolver(load_clubs(), load_aliases()).export_teamname_replacements(
        export_path, SOURCE
    )
    print(f"teamname replacements -> {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
