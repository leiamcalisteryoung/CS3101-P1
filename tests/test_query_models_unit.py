import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from query_models import (
    AndPredicate,
    AttrEqAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    EmptyQuery,
    JoinQuery,
    LetQuery,
    ProjectQuery,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    format_query_expr,
    format_predicate,
    inline_final_query,
    predicate_attributes,
)


class QueryModelsUnitTests(unittest.TestCase):
    # === Predicate attribute extraction tests ===
    def test_predicate_attributes_for_attr_eq_attr(self) -> None:
        # Verifies attribute extraction for A1=A2 leaf predicates.
        pred = AttrEqAttrPredicate(left_attr="lcr", right_attr="rcr")
        self.assertEqual(predicate_attributes(pred), {"lcr", "rcr"})

    def test_predicate_attributes_for_attr_eq_const(self) -> None:
        # Verifies attribute extraction for A=c leaf predicates.
        pred = AttrEqConstPredicate(attr="ms", value=2)
        self.assertEqual(predicate_attributes(pred), {"ms"})

    def test_predicate_attributes_for_and(self) -> None:
        # Verifies recursive attribute extraction across conjunction predicates.
        pred = AndPredicate(
            left=AttrEqConstPredicate(attr="ms", value=2),
            right=AttrEqAttrPredicate(left_attr="lcr", right_attr="rcr"),
        )
        self.assertEqual(predicate_attributes(pred), {"ms", "lcr", "rcr"})

    # === Query expression pretty-printing tests ===
    def test_format_predicate_attr_eq_attr(self) -> None:
        # Verifies pretty-printing for A1=A2 predicates.
        pred = AttrEqAttrPredicate(left_attr="lcr", right_attr="rcr")
        self.assertEqual(format_predicate(pred), "lcr=rcr")

    def test_format_predicate_attr_eq_const(self) -> None:
        # Verifies pretty-printing for integer constants in A=c predicates.
        pred = AttrEqConstPredicate(attr="ms", value=2)
        self.assertEqual(format_predicate(pred), "ms=2")

    def test_format_predicate_and(self) -> None:
        # Verifies pretty-printing for conjunction predicates with parentheses.
        pred = AndPredicate(
            left=AttrEqConstPredicate(attr="ms", value=2),
            right=AttrEqAttrPredicate(left_attr="lcr", right_attr="rcr"),
        )
        self.assertEqual(format_predicate(pred), "(ms=2 ∧ lcr=rcr)")

    def test_format_query_expr_leaf_and_empty(self) -> None:
        # Verifies pretty-printing for relation-variable and empty-relation leaves.
        self.assertEqual(format_query_expr(RelVarQuery(name="module")), "module")
        self.assertEqual(format_query_expr(EmptyQuery()), "∅")

    def test_format_query_expr_composite(self) -> None:
        # Verifies pretty-printing across nested relational operators.
        expr = ProjectQuery(
            source=SelectQuery(
                source=RenameQuery(
                    source=JoinQuery(
                        left=RelVarQuery(name="leftmod"),
                        right=RelVarQuery(name="rightmod"),
                    ),
                    new_attributes=["lmc", "lmt", "lms", "lcr", "rmc", "rmt", "rms", "rcr"],
                ),
                predicate=AttrEqConstPredicate(attr="lms", value=2),
            ),
            attributes=["lmc", "lmt"],
        )
        self.assertEqual(
            format_query_expr(expr),
            "π[lmc, lmt](σ[lms=2](ρ[lmc, lmt, lms, lcr, rmc, rmt, rms, rcr]((leftmod ⋈ rightmod))))",
        )

    def test_format_query_expr_union_and_difference(self) -> None:
        # Verifies pretty-printing for binary set operators.
        union_expr = UnionQuery(left=RelVarQuery(name="a"), right=RelVarQuery(name="b"))
        diff_expr = DifferenceQuery(left=RelVarQuery(name="a"), right=RelVarQuery(name="b"))
        self.assertEqual(format_query_expr(union_expr), "(a ∪ b)")
        self.assertEqual(format_query_expr(diff_expr), "(a − b)")

    # === Final query inlining tests ===
    def test_inline_final_query_inlines_let_binding(self) -> None:
        # Verifies LET-bound relvars are substituted in the final query expression.
        queries = [
            LetQuery(
                target_relvar="x",
                query=ProjectQuery(source=RelVarQuery(name="module"), attributes=["mc", "mt"]),
            ),
            RelVarQuery(name="x"),
        ]
        self.assertEqual(
            inline_final_query(queries),
            ProjectQuery(source=RelVarQuery(name="module"), attributes=["mc", "mt"]),
        )

    def test_inline_final_query_keeps_base_relvar_leaf(self) -> None:
        # Verifies non-LET relvars remain in the final query expression.
        queries = [RelVarQuery(name="module")]
        self.assertEqual(inline_final_query(queries), RelVarQuery(name="module"))

    def test_inline_final_query_detects_cycle(self) -> None:
        # Verifies cyclic LET dependencies are rejected with a clear error.
        queries = [
            LetQuery(target_relvar="x", query=RelVarQuery(name="y")),
            LetQuery(target_relvar="y", query=RelVarQuery(name="x")),
            RelVarQuery(name="x"),
        ]
        with self.assertRaisesRegex(ValueError, "Cyclic LET dependency detected"):
            inline_final_query(queries)

    def test_inline_final_query_raises_on_empty_program(self) -> None:
        # Verifies empty query programs are rejected.
        with self.assertRaisesRegex(ValueError, "Program has no queries to optimize"):
            inline_final_query([])

if __name__ == "__main__":
    unittest.main()
