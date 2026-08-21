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
        assert len(aliases) == len(build.FOOTBALL_DATA) + len(build.FIXTURES_VARIANTS)

    def test_the_fixtures_variants_point_at_real_clubs(self) -> None:
        clubs, _ = build.clubs_and_aliases()
        slugs = {club.slug for club in clubs}
        assert set(build.FIXTURES_VARIANTS.values()) <= slugs

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
        differing = {
            by_slug[row.slug]
            for row in load_aliases().itertuples()
            if row.alias != by_slug[row.slug]
        }
        assert set(payload) == differing
