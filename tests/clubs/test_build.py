"""The Club table builder — the provenance of the committed ``clubs.csv`` and ``aliases.csv``."""

from __future__ import annotations

import json

import pytest

from epl.clubs import build
from epl.clubs.table import aliases_path, clubs_path, load_aliases, load_clubs


class TestMapping:
    def test_every_mapped_spelling_yields_a_club_and_an_alias(self) -> None:
        clubs, aliases = build.clubs_and_aliases()
        assert len(clubs) == len(build.FOOTBALL_DATA)
        assert len(aliases) == (
            len(build.FOOTBALL_DATA)
            + len(build.FIXTURES_VARIANTS)
            + len(build.MYFOOTBALLFACTS)
        )

    def test_a_second_source_adds_spellings_and_never_a_club(self) -> None:
        """Only Football-Data covers the whole pyramid, so only it can introduce a Club."""
        clubs, _ = build.clubs_and_aliases()
        assert len(clubs) == len(build.FOOTBALL_DATA)

    @pytest.mark.parametrize("mapping", ["FIXTURES_VARIANTS", "MYFOOTBALLFACTS"])
    def test_the_extra_spellings_point_at_real_clubs(self, mapping: str) -> None:
        clubs, _ = build.clubs_and_aliases()
        slugs = {club.slug for club in clubs}
        assert set(getattr(build, mapping).values()) <= slugs

    def test_the_pundit_source_name_still_matches_the_module_that_owns_it(self) -> None:
        """`build.PUNDIT_SOURCE` is written out rather than imported, because importing it would
        register two Predictors as a side effect of rebuilding the Club table. Written out, it can
        drift — so the check is made here, where importing the package is harmless."""
        from epl.pundits.myfootballfacts import SOURCE as owned_by_the_parser

        assert build.PUNDIT_SOURCE == owned_by_the_parser

    def test_the_pundit_spellings_cover_the_frozen_nine_seasons_and_the_live_one(self) -> None:
        """MyFootballFacts publishes no lower tier, so every spelling here is a Premier League
        Club and a lower-tier one would be a spelling nothing asks for.

        The nine frozen Seasons — 2017/18-2025/26 — field 32 Clubs between them. Coventry and Hull
        are the other two: both were promoted for 2026/27 and appear on the live page, which is the
        ordinary way a Club arrives at this table (issue #16). The count is asserted rather than
        the names so that a *third* arriving unnoticed still fails.
        """
        assert len(set(build.MYFOOTBALLFACTS.values())) == 34
        assert {"coventry", "hull"} <= set(build.MYFOOTBALLFACTS.values())

    def test_slugs_are_unique(self) -> None:
        slugs = [slug for slug, _ in build.FOOTBALL_DATA.values()]
        assert len(slugs) == len(set(slugs))

    def test_display_names_are_unique(self) -> None:
        names = [name for _, name in build.FOOTBALL_DATA.values()]
        assert len(names) == len(set(names))

    def test_the_wimbledon_lineage_stays_three_separate_clubs(self) -> None:
        """Contested identity, and neither successor ever reached the Premier League."""
        slugs = {slug for slug, _ in build.FOOTBALL_DATA.values()}
        assert {"wimbledon", "mk_dons", "afc_wimbledon"} <= slugs


class TestReproducesTheCommittedTable:
    """The committed CSVs must be exactly what the builder produces, or provenance is a fiction."""

    def test_clubs_csv_is_up_to_date(self, tmp_path) -> None:
        clubs, _ = build.clubs_and_aliases()
        from epl.clubs.table import write_clubs

        rebuilt = write_clubs(clubs, tmp_path / "clubs.csv")
        assert rebuilt.read_text(encoding="utf-8") == clubs_path().read_text(encoding="utf-8")

    def test_aliases_csv_is_up_to_date(self, tmp_path) -> None:
        _, aliases = build.clubs_and_aliases()
        from epl.clubs.table import write_aliases

        rebuilt = write_aliases(aliases, tmp_path / "aliases.csv")
        assert rebuilt.read_text(encoding="utf-8") == aliases_path().read_text(encoding="utf-8")


class TestCacheCheck:
    def test_refuses_to_run_against_an_empty_cache(self, project_root) -> None:
        with pytest.raises(SystemExit, match="raw cache is empty"):
            build.check_against_cache()

    def test_reports_a_spelling_the_mapping_does_not_cover(self, project_root, monkeypatch) -> None:
        monkeypatch.setattr(build, "club_names_in_raw_cache", lambda: ["Arsenal", "Woking"])
        with pytest.raises(SystemExit, match="Woking"):
            build.check_against_cache()

    def test_reports_a_club_the_cache_never_fielded(self, project_root, monkeypatch) -> None:
        """A Club nobody played is as much a defect as a spelling nobody mapped."""
        monkeypatch.setattr(build, "club_names_in_raw_cache", lambda: ["Arsenal"])
        with pytest.raises(SystemExit, match="never seen in the cache"):
            build.check_against_cache()


class TestCli:
    def test_writes_both_tables_and_the_soccerdata_export(
        self, project_root, tmp_path, monkeypatch, capsys
    ) -> None:
        """Decision 5: the Club table is exported to teamname_replacements.json, not kept twice."""
        monkeypatch.setattr(build, "write_clubs", lambda clubs: clubs_path())
        monkeypatch.setattr(build, "write_aliases", lambda aliases: aliases_path())

        export = tmp_path / "teamname_replacements.json"
        assert build.main(["--skip-cache-check", "--teamname-replacements", str(export)]) == 0

        payload = json.loads(export.read_text(encoding="utf-8"))
        assert payload["Manchester United"] == ["Man United"]
        assert payload["Sheffield Wednesday"] == ["Sheffield Wed", "Sheffield Weds"]
        assert "teamname replacements" in capsys.readouterr().out

    def test_the_export_covers_every_club_whose_spelling_differs(
        self, project_root, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(build, "write_clubs", lambda clubs: clubs_path())
        monkeypatch.setattr(build, "write_aliases", lambda aliases: aliases_path())

        export = tmp_path / "out.json"
        build.main(["--skip-cache-check", "--teamname-replacements", str(export)])
        payload = json.loads(export.read_text(encoding="utf-8"))

        by_slug = dict(zip(load_clubs()["slug"], load_clubs()["name"], strict=True))
        # One source at a time: the export is soccerdata's file for Football-Data, and the Alias
        # table now also holds MyFootballFacts' spellings for the Pundit backfill (issue #11).
        differing = {
            by_slug[row.slug]
            for row in load_aliases().itertuples()
            if row.source == build.SOURCE and row.alias != by_slug[row.slug]
        }
        assert set(payload) == differing
