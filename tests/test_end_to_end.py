import os
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(PROJECT_DIR / "src"))

from builder import USQLModelBuilder
from optimizer import QueryOptimizer
from query_engine import QueryEngine


class EndToEndTests(unittest.TestCase):
    # LOAD paths in fixture programs are resolved against process CWD.
    # Keep CWD at project root so "./module.csv" works in all tests.
    @classmethod
    def setUpClass(cls) -> None:
        cls._original_cwd = os.getcwd()
        os.chdir(PROJECT_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._original_cwd)

    # Helpers
    def _build(self, filename: str) -> object:
        source = (FIXTURE_DIR / filename).read_text(encoding="utf-8")
        return USQLModelBuilder().build(source)

    def _interp(self, filename: str) -> str:
        return repr(QueryEngine().run(self._build(filename)))

    def _opt(self, filename: str) -> str:
        return QueryOptimizer().run(self._build(filename))

    # === Interpreter end-to-end tests ===

    def test_interpreter_test_usql(self) -> None:
        result = self._interp("test.usql")
        self.assertEqual(
            result,
            "mc, mt, cr;\n"
            "CS3052, Computational Complexity, 15;\n"
            "cs3101, Databases, 15;",
        )

    def test_interpreter_trivial_usql(self) -> None:
        result = self._interp("trivial.usql")
        self.assertEqual(
            result,
            "mc, mt;\n"
            "CS2001, Foundations of Computation;\n"
            "CS3050, Logic and Reasoning;\n"
            "CS3052, Computational Complexity;\n"
            "cs3101, Databases;",
        )

    def test_interpreter_pushdown_proj_usql(self) -> None:
        result = self._interp("pushdown_proj.usql")
        self.assertEqual(
            result,
            "lmc, lmt;\n"
            "CS2001, Foundations of Computation;\n"
            "CS3050, Logic and Reasoning;\n"
            "CS3052, Computational Complexity;\n"
            "cs3101, Databases;",
        )

    def test_interpreter_pushdown_join_usql(self) -> None:
        result = self._interp("pushdown_join.usql")
        self.assertEqual(
            result,
            "lmc, lmt, lms, lcr, rmc, rmt, rms, rcr;\n"
            "CS3052, Computational Complexity, 2, 15, CS3050, Logic and Reasoning, 1, 15;\n"
            "CS3052, Computational Complexity, 2, 15, CS3052, Computational Complexity, 2, 15;\n"
            "CS3052, Computational Complexity, 2, 15, cs3101, Databases, 2, 15;\n"
            "cs3101, Databases, 2, 15, CS3050, Logic and Reasoning, 1, 15;\n"
            "cs3101, Databases, 2, 15, CS3052, Computational Complexity, 2, 15;\n"
            "cs3101, Databases, 2, 15, cs3101, Databases, 2, 15;",
        )

    # === Optimizer end-to-end tests ===
    def test_optimizer_test_usql(self) -> None:
        result = self._opt("test.usql")
        self.assertEqual(
            result,
            "Initial inlined query: π[mc, mt, cr](σ[ms=2](module))\n"
            "π[mc, mt, cr](σ[ms=2](module))",
        )

    def test_optimizer_trivial_usql(self) -> None:
        result = self._opt("trivial.usql")
        self.assertEqual(
            result,
            "Initial inlined query: (π[mc, mt](module) ∪ π[mc, mt](module))\n"
            "Trivial simplification: π[mc, mt](module)\n"
            "π[mc, mt](module)",
        )

    def test_optimizer_pushdown_proj_usql(self) -> None:
        result = self._opt("pushdown_proj.usql")
        self.assertEqual(
            result,
            "Initial inlined query: π[lmc, lmt](ρ[lmc, lmt, lms, lcr](module))\n"
            "Projection pushdown through rename: ρ[lmc, lmt](π[mc, mt](module))\n"
            "ρ[lmc, lmt](π[mc, mt](module))",
        )

    def test_optimizer_pushdown_join_usql(self) -> None:
        result = self._opt("pushdown_join.usql")
        self.assertEqual(
            result,
            "Initial inlined query: σ[lcr=rcr](σ[rcr=15](σ[lms=2]("
            "(ρ[lmc, lmt, lms, lcr](module) ⋈ ρ[rmc, rmt, rms, rcr](module)))))\n"
            "Selection pushdown through join: σ[lcr=rcr](σ[rcr=15]("
            "(σ[lms=2](ρ[lmc, lmt, lms, lcr](module)) ⋈ ρ[rmc, rmt, rms, rcr](module))))\n"
            "Selection pushdown through rename: σ[lcr=rcr](σ[rcr=15]("
            "(ρ[lmc, lmt, lms, lcr](σ[ms=2](module)) ⋈ ρ[rmc, rmt, rms, rcr](module))))\n"
            "Selection pushdown through join: σ[lcr=rcr]("
            "(ρ[lmc, lmt, lms, lcr](σ[ms=2](module)) ⋈ σ[rcr=15](ρ[rmc, rmt, rms, rcr](module))))\n"
            "Selection pushdown through rename: σ[lcr=rcr]("
            "(ρ[lmc, lmt, lms, lcr](σ[ms=2](module)) ⋈ ρ[rmc, rmt, rms, rcr](σ[cr=15](module))))\n"
            "σ[lcr=rcr]((ρ[lmc, lmt, lms, lcr](σ[ms=2](module)) ⋈ ρ[rmc, rmt, rms, rcr](σ[cr=15](module))))",
        )


if __name__ == "__main__":
    unittest.main()
