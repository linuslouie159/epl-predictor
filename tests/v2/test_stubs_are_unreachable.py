"""What a stub must be, and what it must not become (issue #18).

Decision 12 deferred three features and asked for a written explanation of each instead of an
implementation. A written explanation decays in two directions, and both are structural, so both
are checked here rather than trusted to review.

It decays **into code**: someone adds a helper "while they are in there", something imports it, and
a deferred feature is now a half-built one that the pipeline depends on. Acceptance criterion 5 is
that no stub is imported or executed by the pipeline, and the only way to know that is to read
every import in ``src/epl`` — which is what :func:`_internal_imports` does.

It decays **into an excuse**: the prose drifts until it says a feature was deferred without saying
what it would take to pick it up, which is exactly the rediscovery the ticket exists to prevent.
Criterion 4 asks for the entry price, so every stub carries it as ``WHAT_IT_NEEDS`` — a named
constant a test can hold onto, rather than a paragraph that can quietly lose its last sentence.

This mirrors ``tests/metrics/test_module_contract.py``, which checks the other direction of the
same kind of claim: there, that the metrics cannot see a Predictor.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

import epl
from epl import v2

#: The stub modules, named by :mod:`epl.v2` itself so the package and this test cannot disagree
#: about how many there are.
STUB_NAMES = v2.STUBS

#: Where the package being guarded lives, as an import prefix.
V2_PACKAGE = "epl.v2"


#: The directory the stubs live in. Files under it are exempt from the sweep below, which is about
#: what the *pipeline* reaches for. A stub importing a sibling stub is a different failure, and
#: ``test_it_imports_nothing_from_the_project`` is where that one lands.
V2_DIR = Path(v2.__file__).parent


def _source_files() -> list[Path]:
    """Every module in the installed package, stubs included."""
    return sorted(Path(epl.__file__).parent.rglob("*.py"))


def _pipeline_files() -> list[Path]:
    """Every module that is *not* a stub — the half of the package that does the work."""
    return [path for path in _source_files() if V2_DIR not in path.parents]


def _internal_imports(path: Path) -> set[str]:
    """The ``epl.*`` modules that one file imports.

    Read from the syntax tree rather than by importing, because importing would answer a different
    question — what is reachable at runtime today — and the claim is about the source.
    """
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # Both spellings, because ``from epl import v2`` binds the submodule just as surely as
            # ``import epl.v2`` does, and recording only the module half would miss it.
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in imported if name == "epl" or name.startswith("epl.")}


def _stub(name: str) -> ModuleType:
    return importlib.import_module(f"{V2_PACKAGE}.{name}")


class TestNothingInThePipelineCanReachAStub:
    def test_no_module_outside_v2_imports_v2(self) -> None:
        """Criterion 5. The whole point of a stub is that deleting it would break nothing."""
        offenders = {}
        for path in _pipeline_files():
            names = _internal_imports(path)
            reaching = sorted(
                n for n in names if n == V2_PACKAGE or n.startswith(f"{V2_PACKAGE}.")
            )
            if reaching:
                offenders[path.name] = reaching
        assert offenders == {}, offenders

    def test_the_package_itself_imports_nothing(self) -> None:
        """``import epl.v2`` must cost nothing and run nothing, so it imports no stub either."""
        assert _internal_imports(Path(v2.__file__)) == set()

    def test_there_is_something_to_check(self) -> None:
        """Guards the sweep above: a glob that found no files would pass by vacuum."""
        assert len(_pipeline_files()) >= 40
        assert len(STUB_NAMES) == 3
        assert sorted(p.stem for p in _source_files() if V2_DIR in p.parents) == sorted(
            ["__init__", *STUB_NAMES]
        )

    def test_the_sweep_would_notice_an_import(self) -> None:
        """The load-bearing test above passes today because nothing imports a stub.

        It would also pass if ``_internal_imports`` quietly stopped finding imports, and that
        failure is invisible — so the detector is pointed at a file known to import ``epl.v2``.
        This test module is one.
        """
        assert V2_PACKAGE in _internal_imports(Path(__file__))


class TestAStubHoldsNoImplementation:
    """Decision 12: three stubs, *no implementation*. Prose and named constants only."""

    @pytest.mark.parametrize("name", STUB_NAMES)
    def test_it_defines_no_function_and_no_class(self, name: str) -> None:
        module = _stub(name)
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        defined = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        assert defined == [], defined

    @pytest.mark.parametrize("name", STUB_NAMES)
    def test_it_imports_nothing_from_the_project(self, name: str) -> None:
        """A stub that reached for a real module would be the first line of an implementation."""
        module = _stub(name)
        assert _internal_imports(Path(module.__file__ or "")) == set()


class TestAStubSaysWhatItWouldTakeToPickItUp:
    """Criterion 4: what it *needs*, not merely that it was deferred."""

    @pytest.mark.parametrize("name", STUB_NAMES)
    def test_it_explains_itself_at_length(self, name: str) -> None:
        docstring = _stub(name).__doc__
        assert docstring is not None
        assert len(docstring) >= 400, len(docstring)

    @pytest.mark.parametrize("name", STUB_NAMES)
    def test_it_names_its_entry_price(self, name: str) -> None:
        needs = _stub(name).WHAT_IT_NEEDS
        assert isinstance(needs, tuple)
        assert len(needs) >= 2
        assert all(isinstance(item, str) and item.strip() for item in needs)

    def test_the_ml_stub_names_the_features_it_would_want(self) -> None:
        """Criterion 1 asks for the features by name, so they are a list and not a gesture."""
        features = _stub("xgboost_layer").FEATURES
        assert len(features) >= 5
        assert all(isinstance(item, str) and item.strip() for item in features)

    def test_the_api_football_stub_records_the_measurement_not_the_assumption(self) -> None:
        """Criterion 3, as stage 13 left it.

        The stub's premise — that ``fixtures.csv`` already carries upcoming Fixtures with the
        Market Line — is the one thing stage 13 could not confirm. So the stub carries the fetches
        that failed to confirm it, and the conditions that would revive the client.
        """
        stub = _stub("api_football")
        assert len(stub.FETCHES_MEASURED) >= 2
        assert len(stub.WHAT_WOULD_REVIVE_IT) >= 2
        assert all(len(fetch) == 4 for fetch in stub.FETCHES_MEASURED)

    def test_the_headline_count_agrees_with_the_fetches_behind_it(self) -> None:
        """``PREMIER_LEAGUE_ROWS_SEEN`` is derived data that no function may derive.

        A stub holds no implementation, so the count cannot be a ``sum()`` over the fetches — it is
        written out, and writing it out is how it comes to disagree with them. Appending a fetch and
        leaving the count alone is the realistic mistake, and this is what catches it.

        Note what this does *not* assert: that the count is still zero. When a Premier League row
        finally appears, updating both together should pass, and what happens next is a decision
        about the stub rather than a test failure to route around.
        """
        stub = _stub("api_football")
        assert stub.PREMIER_LEAGUE_ROWS_SEEN == sum(
            premier_league_rows for _, _, _, premier_league_rows in stub.FETCHES_MEASURED
        )
