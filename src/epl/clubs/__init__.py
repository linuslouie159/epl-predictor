"""Clubs: the canonical Club table and Alias resolution.

A Club is identified by a slug that is stable across Seasons, tiers and sources. An Alias is one
source's spelling of it. Aliases are data, held in ``aliases.csv``; they never appear in model code.
"""

from epl.clubs.table import (
    AliasTableError,
    Club,
    ClubResolver,
    UnknownAliasError,
    aliases_path,
    clubs_path,
    load_aliases,
    load_clubs,
)

__all__ = [
    "AliasTableError",
    "Club",
    "ClubResolver",
    "UnknownAliasError",
    "aliases_path",
    "clubs_path",
    "load_aliases",
    "load_clubs",
]
