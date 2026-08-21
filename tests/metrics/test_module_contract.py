"""What the metrics module must not know.

"Metrics take a Prediction and an Outcome and know nothing about which Predictor produced them,
so no comparison can ever be apples-to-oranges" (issue #6). That is a structural claim, so it is
checked structurally rather than trusted: a metric that could tell an Elo output from a Market
Line output could be tuned, however unintentionally, to flatter one of them.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from epl import metrics

#: Every module that produces Predictions. The metrics must be unable to reach any of them.
PREDICTOR_MODULES = ("epl.models", "epl.benchmarks", "epl.pundits", "epl.simulate")

#: Words that would mean a metric had been handed an identity along with the numbers.
IDENTIFYING = ("predictor", "model", "pundit", "market", "source", "season", "fixture_id")

#: The public functions, as opposed to the constants and types also in ``__all__``.
PUBLIC_FUNCTIONS = sorted(
    name
    for name in metrics.__all__
    if callable(getattr(metrics, name)) and not inspect.isclass(getattr(metrics, name))
)

PERFECT = [1.0, 0.0, 0.0]
MAXIMALLY_WRONG = [0.0, 0.0, 1.0]
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def _imported_modules() -> set[str]:
    imported: set[str] = set()
    for path in Path(inspect.getfile(metrics)).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


class TestTheMetricsCannotSeeAPredictor:
    @pytest.mark.parametrize("module", PREDICTOR_MODULES)
    def test_it_imports_no_module_that_produces_predictions(self, module: str) -> None:
        assert module not in _imported_modules()

    def test_it_imports_nothing_from_the_project_but_itself(self) -> None:
        internal = {name for name in _imported_modules() if name.startswith("epl")}
        assert all(name.startswith("epl.metrics") for name in internal), internal

    def test_there_is_something_to_check(self) -> None:
        """Guards the filter below: an empty parametrize would pass by vacuum."""
        assert len(PUBLIC_FUNCTIONS) >= 10

    @pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
    def test_no_public_function_accepts_an_identity(self, name: str) -> None:
        parameters = inspect.signature(getattr(metrics, name)).parameters
        leaked = [p for p in parameters if any(word in p.lower() for word in IDENTIFYING)]
        assert not leaked, f"{name} accepts {leaked}"

    def test_two_predictors_agreeing_on_a_fixture_are_scored_identically(self) -> None:
        """The whole point: the same numbers in, the same score out, whoever produced them."""
        assert metrics.score([PERFECT], ["H"]) == metrics.score([[1.0, 0.0, 0.0]], ["H"])


class TestTheThreeKnownPredictions:
    """Issue #6: a perfect Prediction, a maximally wrong one and a uniform one all return known
    values. Every number below is worked by hand; see tests/metrics/test_scores.py for the
    arithmetic."""

    @pytest.mark.parametrize(
        ("prediction", "rps", "brier", "log_loss", "accuracy"),
        [
            (PERFECT, 0.0, 0.0, 0.0, 1.0),
            (MAXIMALLY_WRONG, 1.0, 2.0, 34.538776394910684, 0.0),
            (UNIFORM, 5 / 18, 2 / 3, 1.0986122886681098, 1 / 3),
        ],
        ids=["perfect", "maximally-wrong", "uniform"],
    )
    def test_scores_a_home_win(
        self, prediction: list[float], rps: float, brier: float, log_loss: float, accuracy: float
    ) -> None:
        card = metrics.score([prediction], ["H"])
        assert card.rps == pytest.approx(rps)
        assert card.brier == pytest.approx(brier)
        assert card.log_loss == pytest.approx(log_loss)
        assert card.accuracy == pytest.approx(accuracy)

    def test_the_perfect_and_the_worst_bracket_every_other_prediction(self) -> None:
        middling = metrics.rps([0.4, 0.35, 0.25], "H")
        assert metrics.rps(PERFECT, "H") < middling < metrics.rps(MAXIMALLY_WRONG, "H")
