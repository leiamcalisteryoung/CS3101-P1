import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from builder import ProgramState
from cost_optimizer import CostBasedJoinOptimizer
from models import Attribute, Domain, GType, Relation, RelVar
from query_models import (
    AndPredicate,
    AttrEqAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    EmptyQuery,
    JoinQuery,
    ProjectQuery,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
)


class CostOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        int_domain = Domain(name="IntD", gtype=GType.INT)

        a = Attribute(name="a", domain=int_domain)
        b = Attribute(name="b", domain=int_domain)
        c = Attribute(name="c", domain=int_domain)

        rel_r = Relation(name="r", attributes=[a, b])
        rel_s = Relation(name="s", attributes=[b, c])
        rel_t = Relation(name="t", attributes=[c])
        rel_u = Relation(name="u", attributes=[a])

        self.state = ProgramState(
            relvars={
                "r": RelVar(relation=rel_r, tuples=[{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 3, "b": 2}]),
                "s": RelVar(relation=rel_s, tuples=[{"b": 1, "c": 10}, {"b": 2, "c": 20}, {"b": 2, "c": 30}]),
                "t": RelVar(relation=rel_t, tuples=[{"c": 10}, {"c": 20}, {"c": 40}]),
                "u": RelVar(relation=rel_u, tuples=[{"a": 1}, {"a": 2}]),
            }
        )
        self.optimizer = CostBasedJoinOptimizer(self.state)

    # === Binary join tree enumeration tests ===
    def test_enumerate_binary_join_trees_arity_3(self) -> None:
        # Verifies 3-operand join enumeration returns 2 parenthesizations.
        trees = self.optimizer._enumerate_binary_join_trees([RelVarQuery("r"), RelVarQuery("s"), RelVarQuery("t")])
        self.assertEqual(len(trees), 2)

    def test_enumerate_binary_join_trees_arity_4(self) -> None:
        # Verifies 4-operand join enumeration returns 5 parenthesizations.
        trees = self.optimizer._enumerate_binary_join_trees(
            [RelVarQuery("r"), RelVarQuery("s"), RelVarQuery("t"), RelVarQuery("u")]
        )
        self.assertEqual(len(trees), 5)

    def test_enumerate_binary_join_trees_arity_2(self) -> None:
        # Verifies a flat join with 2 operands returns empty list (no alternative trees).
        trees = self.optimizer._enumerate_binary_join_trees([RelVarQuery("r"), RelVarQuery("s")])
        self.assertEqual(trees, [])

    # === Cost-based join application tests ===
    def test_apply_cost_based_join_reorder_returns_none_for_simple_join(self) -> None:
        # Verifies no reordering is attempted when node has no join child.
        expr = JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s"))
        rewritten, rule = self.optimizer._apply_cost_based_join_reorder(expr)
        self.assertIsNone(rewritten)
        self.assertIsNone(rule)

    def test_apply_cost_based_join_reorder_returns_cheaper_tree_arity_3(self) -> None:
        # Verifies for (r⋈s)⋈t the cheapest tree is r⋈(s⋈t).
        # With the current data, s⋈t yields 3 rows, while r⋈s yields 4.5 rows.
        # So doing s⋈t first gives the cheaper overall 3-way plan.
        inner_join = JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s"))
        expr = JoinQuery(left=inner_join, right=RelVarQuery("t"))
        rewritten, rule = self.optimizer._apply_cost_based_join_reorder(expr)
        expected = JoinQuery(
            left=RelVarQuery("r"),
            right=JoinQuery(left=RelVarQuery("s"), right=RelVarQuery("t")),
        )
        self.assertIsNotNone(rewritten)
        self.assertEqual(rewritten, expected)
        self.assertEqual(rule, "Cost-based join reordering")


    def test_apply_cost_based_join_reorder_returns_cheaper_tree_arity_4(self) -> None:
        # Verifies for ((r⋈s)⋈(u⋈t)) the cheapest tree is (((r⋈u)⋈s)⋈t).
        # - r⋈u -> 2 rows(best early reduction)
        # - r⋈s -> 4.5 rows
        # - u⋈t -> 6 rows (cartesian product, so bad to do early)
        # That makes (((r⋈u)⋈s)⋈t) the best option
        inner_join_left = JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s"))
        inner_join_right = JoinQuery(left=RelVarQuery("u"), right=RelVarQuery("t"))
        expr = JoinQuery(left=inner_join_left, right=inner_join_right)
        rewritten, rule = self.optimizer._apply_cost_based_join_reorder(expr)
        expected = JoinQuery(
            left=JoinQuery(
                left=JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("u")),
                right=RelVarQuery("s"),
            ),
            right=RelVarQuery("t"),
        )
        self.assertIsNotNone(rewritten)
        self.assertEqual(rewritten, expected)
        self.assertEqual(rule, "Cost-based join reordering")


    def test_apply_cost_based_join_reorder_returns_cheaper_tree_deeply_nested(self) -> None:
        # Verifies bottom-up behavior on (((r⋈s)⋈t)⋈((u⋈t)⋈s)).
        # The two children start unoptimized and must both be rewritten first:
        # - ((r⋈s)⋈t)   -> (r⋈(s⋈t))
        # - ((u⋈t)⋈s)   -> (u⋈(t⋈s))
        # Then the parent treats (s⋈t) and (t⋈s) as atomic rewritten children.
        # The cheapest top-level reorder is (((r⋈u)⋈(s⋈t))⋈(t⋈s)).
        left_child = JoinQuery(
            left=JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s")),
            right=RelVarQuery("t"),
        )
        right_child = JoinQuery(
            left=JoinQuery(left=RelVarQuery("u"), right=RelVarQuery("t")),
            right=RelVarQuery("s"),
        )
        expr = JoinQuery(left=left_child, right=right_child)

        rewritten, steps = self.optimizer.optimize(expr)
        expected = JoinQuery(
            left=JoinQuery(
                left=JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("u")),
                right=JoinQuery(left=RelVarQuery("s"), right=RelVarQuery("t")),
            ),
            right=JoinQuery(left=RelVarQuery("t"), right=RelVarQuery("s")),
        )

        self.assertEqual(rewritten, expected)
        self.assertEqual(
            steps,
            [
                "Cost-based join reordering: (r ⋈ (s ⋈ t))",
                "Cost-based join reordering: (u ⋈ (t ⋈ s))",
                "Cost-based join reordering: (((r ⋈ u) ⋈ (s ⋈ t)) ⋈ (t ⋈ s))",
            ],
        )


    # === Cost estimation tests ===
    def test_estimate_stats_relvar(self) -> None:
        # Verifies base-relation stats and cost are derived from loaded tuples.
        rows, attrs, distinct, cost = self.optimizer._estimate_stats_and_cost(RelVarQuery("r"))
        self.assertEqual(rows, 3.0)
        self.assertEqual(attrs, ["a", "b"])
        self.assertEqual(distinct["a"], 3.0)
        self.assertEqual(cost, 3.0)

    def test_estimate_stats_empty(self) -> None:
        # Verifies empty relation stats are zeros.
        rows, attrs, distinct, cost = self.optimizer._estimate_stats_and_cost(EmptyQuery())
        self.assertEqual(rows, 0.0)
        self.assertEqual(attrs, [])
        self.assertEqual(distinct, {})
        self.assertEqual(cost, 0.0)

    def test_estimate_stats_select_attr_eq_const(self) -> None:
        # Verifies selection cardinality uses n_r / V(A,r).
        expr = SelectQuery(source=RelVarQuery("r"), predicate=AttrEqConstPredicate("b", 1))
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(attrs, ["a", "b"])
        self.assertAlmostEqual(rows, 1.5)

    def test_estimate_stats_select_attr_eq_attr(self) -> None:
        # Verifies selection cardinality uses n_r / max(V(A,r),V(B,r)).
        expr = SelectQuery(source=RelVarQuery("r"), predicate=AttrEqAttrPredicate("a", "b"))
        rows, _, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertAlmostEqual(rows, 1.0)

    def test_estimate_stats_select_and_predicate(self) -> None:
        # Verifies conjunction selection uses (s1*s2)/n_r composition.
        pred = AndPredicate(
            left=AttrEqConstPredicate("b", 1),
            right=AttrEqConstPredicate("a", 1),
        )
        expr = SelectQuery(source=RelVarQuery("r"), predicate=pred)
        rows, _, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertAlmostEqual(rows, 0.5)

    def test_estimate_stats_project(self) -> None:
        # Verifies projection cardinality uses max V(attr) capped by n(r).
        expr = ProjectQuery(source=RelVarQuery("r"), attributes=["a", "b"])
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(attrs, ["a", "b"])
        self.assertEqual(rows, 3.0)

    def test_estimate_stats_rename(self) -> None:
        # Verifies rename remaps attribute names while preserving row estimate.
        expr = RenameQuery(source=RelVarQuery("r"), new_attributes=["x", "y"])
        rows, attrs, distinct, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(rows, 3.0)
        self.assertEqual(attrs, ["x", "y"])
        self.assertIn("x", distinct)
        self.assertIn("y", distinct)

    def test_estimate_stats_rename_arity_mismatch_raises(self) -> None:
        # Verifies rename stats reject mismatched rename arity.
        expr = RenameQuery(source=RelVarQuery("r"), new_attributes=["x"])
        with self.assertRaisesRegex(ValueError, "Rename arity mismatch"):
            self.optimizer._estimate_stats_and_cost(expr)

    def test_estimate_stats_union(self) -> None:
        # Verifies union cardinality sums input cardinalities.
        expr = UnionQuery(left=RelVarQuery("u"), right=RelVarQuery("u"))
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(rows, 4.0)
        self.assertEqual(attrs, ["a"])

    def test_estimate_stats_difference(self) -> None:
        # Verifies difference cardinality uses min(nr, ns) per current estimator.
        expr = DifferenceQuery(left=RelVarQuery("r"), right=RelVarQuery("r"))
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(rows, 3.0)
        self.assertEqual(attrs, ["a", "b"])

    def test_estimate_stats_join_no_shared_attrs(self) -> None:
        # Verifies join with disjoint schemas estimates cartesian product cardinality.
        expr = JoinQuery(left=RelVarQuery("u"), right=RelVarQuery("t"))
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertEqual(rows, 6.0)
        self.assertEqual(attrs, ["a", "c"])

    def test_estimate_stats_join_with_shared_attrs(self) -> None:
        # Verifies join with shared attrs uses min(nr*ns/V(A,r), nr*ns/V(A,s)).
        expr = JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s"))
        rows, attrs, _, _ = self.optimizer._estimate_stats_and_cost(expr)
        self.assertAlmostEqual(rows, 4.5)
        self.assertEqual(attrs, ["a", "b", "c"])

    def test_estimate_plan_cost(self) -> None:
        # Verifies plan cost helper returns cumulative cost from recursive estimator.
        expr = JoinQuery(left=RelVarQuery("r"), right=RelVarQuery("s"))
        cost = self.optimizer._estimate_plan_cost(expr)
        self.assertGreater(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
