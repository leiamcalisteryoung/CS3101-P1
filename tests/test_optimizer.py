import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost_optimizer import CostBasedJoinOptimizer

from builder import ProgramState
from models import Attribute, Domain, GType, Relation, RelVar
from optimizer import QueryOptimizer
from query_engine import QueryEngine
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


class QueryOptimizerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        # Sets up a common optimizer and sample program state for testing.
        self.optimizer = QueryOptimizer()

        int_domain = Domain(name="IntD", gtype=GType.INT)
        str_domain = Domain(name="StrD", gtype=GType.STRING)

        a1 = Attribute(name="a", domain=int_domain)
        b1 = Attribute(name="b", domain=int_domain)
        c1 = Attribute(name="c", domain=str_domain)
        d1 = Attribute(name="d", domain=str_domain)

        left_rel = Relation(name="left", attributes=[a1, b1])
        right_rel = Relation(name="right", attributes=[b1, c1, d1])
        base_rel = Relation(name="base", attributes=[a1, b1, c1])

        state = ProgramState(
            relvars={
                "r": RelVar(relation=base_rel, tuples=[{"a": 1, "b": 1, "c": "x"}]),
                "s": RelVar(relation=left_rel, tuples=[{"a": 1, "b": 1}]),
                "t": RelVar(relation=right_rel, tuples=[{"b": 1, "c": "x", "d": "extra"}]),
            },
            queries=[],
        )
        self.state = state
        self.optimizer._state = state
        self.engine = QueryEngine()

    # === Trivial simplification tests ===
    def test_trivial_union_idempotence(self) -> None:
        # Verifies r ∪ r simplifies to r.
        expr = UnionQuery(left=RelVarQuery("r"), right=RelVarQuery("r"))
        rewritten = self.optimizer._apply_trivial_simplification(expr)
        self.assertEqual(rewritten, RelVarQuery("r"))

    def test_trivial_join_with_empty(self) -> None:
        # Verifies r ⋈ ∅ simplifies to ∅.
        expr = JoinQuery(left=RelVarQuery("r"), right=EmptyQuery())
        rewritten = self.optimizer._apply_trivial_simplification(expr)
        self.assertEqual(rewritten, EmptyQuery())

    def test_trivial_difference_self(self) -> None:
        # Verifies r − r simplifies to ∅.
        expr = DifferenceQuery(left=RelVarQuery("r"), right=RelVarQuery("r"))
        rewritten = self.optimizer._apply_trivial_simplification(expr)
        self.assertEqual(rewritten, EmptyQuery())

    # === Unary equivalence tests ===
    def test_unary_selection_merge(self) -> None:
        # Verifies nested selections merge into one conjunction predicate.
        inner = SelectQuery(
            source=RelVarQuery("r"),
            predicate=AttrEqConstPredicate(attr="a", operator="=", value=1),
        )
        expr = SelectQuery(
            source=inner,
            predicate=AttrEqConstPredicate(attr="b", operator="=", value=1),
        )
        rewritten, rule = self.optimizer._apply_unary_equivalences(expr)
        self.assertEqual(rule, "Selection conjunction merge")
        self.assertIsInstance(rewritten, SelectQuery)
        self.assertIsInstance(rewritten.predicate, AndPredicate)

    def test_unary_projection_merge(self) -> None:
        # Verifies nested projections collapse to outer projection attributes.
        inner = ProjectQuery(source=RelVarQuery("r"), attributes=["a", "b"])
        expr = ProjectQuery(source=inner, attributes=["a"])
        rewritten, rule = self.optimizer._apply_unary_equivalences(expr)
        self.assertEqual(rule, "Nested projection merge")
        self.assertEqual(rewritten, ProjectQuery(source=RelVarQuery("r"), attributes=["a"]))

    def test_unary_rename_merge(self) -> None:
        # Verifies nested renames collapse to outer rename list.
        inner = RenameQuery(source=RelVarQuery("r"), new_attributes=["x", "y", "z"])
        expr = RenameQuery(source=inner, new_attributes=["a1", "b1", "c1"])
        rewritten, rule = self.optimizer._apply_unary_equivalences(expr)
        self.assertEqual(rule, "Nested rename merge")
        self.assertEqual(rewritten, RenameQuery(source=RelVarQuery("r"), new_attributes=["a1", "b1", "c1"]))

    # === Selection pushdown tests ===
    def test_selection_pushdown_through_projection(self) -> None:
        # Verifies selection pushdown through projection rule shape.
        expr = SelectQuery(
            source=ProjectQuery(source=RelVarQuery("r"), attributes=["a", "b"]),
            predicate=AttrEqConstPredicate(attr="a", operator="=", value=1),
        )
        rewritten, rule = self.optimizer._apply_selection_pushdown(expr)
        self.assertEqual(rule, "Selection pushdown through projection")
        self.assertIsInstance(rewritten, ProjectQuery)
        self.assertIsInstance(rewritten.source, SelectQuery)

    def test_selection_pushdown_through_rename_remaps_attrs(self) -> None:
        # Verifies selection pushdown through rename maps predicate back to source names.
        expr = SelectQuery(
            source=RenameQuery(source=RelVarQuery("r"), new_attributes=["x", "y", "z"]),
            predicate=AttrEqConstPredicate(attr="x", operator="=", value=1),
        )
        rewritten, rule = self.optimizer._apply_selection_pushdown(expr)
        self.assertEqual(rule, "Selection pushdown through rename")
        self.assertIsInstance(rewritten, RenameQuery)
        self.assertIsInstance(rewritten.source, SelectQuery)
        self.assertEqual(
            rewritten.source.predicate,
            AttrEqConstPredicate(attr="a", operator="=", value=1),
        )

    def test_selection_pushdown_through_join_splits_predicates(self) -> None:
        # Verifies join pushdown keeps cross predicate outside and pushes side-local predicates.
        left = RenameQuery(source=RelVarQuery("s"), new_attributes=["la", "lb"])
        right = RenameQuery(source=RelVarQuery("t"), new_attributes=["rb", "rc"])
        predicate = AndPredicate(
            left=AttrEqConstPredicate(attr="la", operator="=", value=1),
            right=AttrEqAttrPredicate(left_attr="lb", operator="=", right_attr="rb"),
        )
        expr = SelectQuery(source=JoinQuery(left=left, right=right), predicate=predicate)
        rewritten, rule = self.optimizer._apply_selection_pushdown(expr)
        self.assertEqual(rule, "Selection pushdown through join")
        self.assertIsInstance(rewritten, SelectQuery)
        self.assertIsInstance(rewritten.source, JoinQuery)
        self.assertIsInstance(rewritten.source.left, SelectQuery)

    # === Projection pushdown tests ===
    def test_projection_pushdown_through_rename(self) -> None:
        # Verifies projection pushdown through rename adjusts attribute names correctly.
        expr = ProjectQuery(
            source=RenameQuery(source=RelVarQuery("r"), new_attributes=["x", "y", "z"]),
            attributes=["x", "y"],
        )
        rewritten, rule = self.optimizer._apply_projection_pushdown(expr)
        self.assertEqual(rule, "Projection pushdown through rename")
        self.assertIsInstance(rewritten, RenameQuery)
        self.assertIsInstance(rewritten.source, ProjectQuery)
        self.assertEqual(rewritten.source.attributes, ["a", "b"])

    def test_projection_pushdown_through_union(self) -> None:
        # Verifies projection distributes through union.
        expr = ProjectQuery(
            source=UnionQuery(left=RelVarQuery("r"), right=RelVarQuery("r")),
            attributes=["a", "b"],
        )
        rewritten, rule = self.optimizer._apply_projection_pushdown(expr)
        self.assertEqual(rule, "Projection pushdown through union")
        self.assertIsInstance(rewritten, UnionQuery)
        self.assertIsInstance(rewritten.left, ProjectQuery)

    def test_projection_pushdown_through_join(self) -> None:
        # Verifies projection pushdown through join introduces inner projections when useful.
        expr = ProjectQuery(
            source=JoinQuery(left=RelVarQuery("s"), right=RelVarQuery("t")),
            attributes=["a", "c"],
        )
        rewritten, rule = self.optimizer._apply_projection_pushdown(expr)
        self.assertEqual(rule, "Projection pushdown through join")
        self.assertIsInstance(rewritten, ProjectQuery)
        self.assertIsInstance(rewritten.source, JoinQuery)

    def test_projection_pushdown_through_join_noop_returns_none(self) -> None:
        # Verifies join projection pushdown skips no-op rewrites.
        expr = ProjectQuery(
            source=JoinQuery(left=RelVarQuery("s"), right=RelVarQuery("t")),
            attributes=["a", "b", "c", "d"],
        )
        rewritten, rule = self.optimizer._apply_projection_pushdown(expr)
        self.assertIsNone(rewritten)
        self.assertIsNone(rule)

    # === Helper method tests ===
    def test_flatten_conjunction(self) -> None:
        # Verifies conjunction flattening returns atomic predicates in order.
        pred = AndPredicate(
            left=AttrEqConstPredicate(attr="a", operator="=", value=1),
            right=AndPredicate(
                left=AttrEqConstPredicate(attr="b", operator="=", value=2),
                right=AttrEqAttrPredicate(left_attr="a", operator="=", right_attr="b"),
            ),
        )
        flat = QueryOptimizer._flatten_conjunction(pred)
        self.assertEqual(len(flat), 3)

    def test_build_conjunction(self) -> None:
        # Verifies conjunction builder produces a nested AndPredicate tree.
        preds = [
            AttrEqConstPredicate(attr="a", operator="=", value=1),
            AttrEqConstPredicate(attr="b", operator="=", value=2),
        ]
        result = QueryOptimizer._build_conjunction(preds)
        self.assertIsInstance(result, AndPredicate)

    def test_output_attributes_for_join(self) -> None:
        # Verifies output-attribute inference for natural join ordering.
        attrs = self.optimizer._output_attributes(JoinQuery(left=RelVarQuery("s"), right=RelVarQuery("t")))
        self.assertEqual(attrs, ["a", "b", "c", "d"])

    def test_output_attributes_unknown_relvar_raises(self) -> None:
        # Verifies output-attribute inference fails for unknown relation variables.
        with self.assertRaisesRegex(ValueError, "Unknown relvar"):
            self.optimizer._output_attributes(RelVarQuery("missing"))

    def test_rename_predicate_attributes_recursive(self) -> None:
        # Verifies predicate attribute renaming handles nested conjunctions.
        pred = AndPredicate(
            left=AttrEqConstPredicate(attr="x", operator="=", value=1),
            right=AttrEqAttrPredicate(left_attr="x", operator="=", right_attr="y"),
        )
        renamed = self.optimizer._rename_predicate_attributes(pred, {"x": "a", "y": "b"})
        self.assertEqual(
            renamed,
            AndPredicate(
                left=AttrEqConstPredicate(attr="a", operator="=", value=1),
                right=AttrEqAttrPredicate(left_attr="a", operator="=", right_attr="b"),
            ),
        )

    # === Node rewrite application tests ===
    def test_apply_node_rewrite_rules_priority(self) -> None:
        # Verifies node rewrite priority picks trivial simplification before other rules.
        expr = UnionQuery(left=RelVarQuery("r"), right=RelVarQuery("r"))
        rewritten, rule = self.optimizer._apply_node_rewrite_rules(expr)
        self.assertEqual(rule, "Trivial simplification")
        self.assertEqual(rewritten, RelVarQuery("r"))

    def test_rewrite_once_bottom_up_rewrites_child_first(self) -> None:
        # Verifies bottom-up pass rewrites child before attempting parent rewrite.
        child = UnionQuery(left=RelVarQuery("r"), right=RelVarQuery("r"))
        expr = SelectQuery(
            source=child,
            predicate=AttrEqConstPredicate(attr="a", operator="=", value=1),
        )
        rewritten, rule = self.optimizer._rewrite_once_bottom_up(expr)
        self.assertEqual(rule, "Trivial simplification")
        self.assertIsInstance(rewritten, SelectQuery)
        self.assertEqual(rewritten.source, RelVarQuery("r"))