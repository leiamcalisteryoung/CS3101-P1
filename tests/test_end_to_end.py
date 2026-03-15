import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from builder import USQLModelBuilder
from optimizer import QueryOptimizer
from query_engine import QueryEngine


class EndToEndTests(unittest.TestCase):
    # LOAD statements in .usql files use relative paths (e.g. "./module.csv"),
    # which resolve against the process CWD.  We pin CWD to the project root for
    # the duration of this class so every test can find the data files.
    @classmethod
    def setUpClass(cls) -> None:
        cls._original_cwd = os.getcwd()
        os.chdir(ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._original_cwd)

    # Helpers
    def _build(self, filename: str) -> object:
        source = (ROOT / filename).read_text(encoding="utf-8")
        return USQLModelBuilder().build(source)

    def _interp(self, filename: str) -> str:
        return repr(QueryEngine().run(self._build(filename)))

    def _opt(self, filename: str) -> str:
        return QueryOptimizer().run(self._build(filename))

    # === Interpreter end-to-end tests ===

    def test_interpreter_test_usql(self) -> None:
        # Given test program: select semester-2 modules and project onto mc, mt, cr.
        # This is the canonical test.usql supplied with the assignment.
        result = self._interp("test.usql")
        self.assertEqual(
            result,
            "mc, mt, cr;\n"
            "CS3052, Computational Complexity, 15;\n"
            "cs3101, Databases, 15;",
        )

    def test_interpreter_trivial_usql(self) -> None:
        # Union of a projection with itself should produce deduplicated rows of
        # all four modules projected onto mc and mt.
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
        # Rename then project: result should be all modules with renamed mc/mt
        # attributes, identical rows to trivial.usql but with lmc/lmt names.
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
        # Self-join with selection predicates: only rows where both sides have the
        # same credit value (lcr=rcr), right side has credits=15 (rcr=15), and
        # left side is semester 2 (lms=2).
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
        # test.usql: projection and selection are already on the base relation,
        # so no further rewrites are possible — only the initial inlined form.
        result = self._opt("test.usql")
        self.assertEqual(
            result,
            "Initial inlined query: π[mc, mt, cr](σ[ms=2](module))",
        )

    def test_optimizer_trivial_usql(self) -> None:
        # trivial.usql: union of identical sub-expressions triggers the
        # trivial-simplification rule (X ∪ X → X).
        result = self._opt("trivial.usql")
        self.assertEqual(
            result,
            "Initial inlined query: (π[mc, mt](module) ∪ π[mc, mt](module))\n"
            "Trivial simplification: π[mc, mt](module)",
        )

    def test_optimizer_pushdown_proj_usql(self) -> None:
        # pushdown_proj.usql: projection through rename is pushed down so the
        # projection operates on the raw base relation before renaming.
        result = self._opt("pushdown_proj.usql")
        self.assertEqual(
            result,
            "Initial inlined query: π[lmc, lmt](ρ[lmc, lmt, lms, lcr](module))\n"
            "Projection pushdown through rename: ρ[lmc, lmt](π[mc, mt](module))",
        )

    def test_optimizer_pushdown_join_usql(self) -> None:
        # pushdown_join.usql: three stacked selections are pushed down through a
        # join and through rename nodes, reducing the number of tuples processed
        # at each operator as early as possible.
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
            "(ρ[lmc, lmt, lms, lcr](σ[ms=2](module)) ⋈ ρ[rmc, rmt, rms, rcr](σ[cr=15](module))))",
        )

    # === Optimizer + interpreter equivalence: optimized query yields same output ===
    def test_optimizer_and_interpreter_agree_on_test_usql(self) -> None:
        # Verifies that running the optimizer does not change the final result
        # produced by the interpreter for the test.usql program.
        unoptimized_result = self._interp("test.usql")
        optimization_output = self._opt("test.usql")
        optimized_query = optimization_output.splitlines()[-1]
        optimized_result = repr(QueryEngine().run(self._build("test.usql"), optimized_query))
        self.assertEqual(unoptimized_result, optimized_result)


if __name__ == "__main__":
    unittest.main()
