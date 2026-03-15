import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from builder import ProgramState
from models import Attribute, Domain, GType, Relation, RelVar
from query_engine import QueryEngine
from query_models import (
    AttrEqAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    JoinQuery,
    LetQuery,
    ProjectQuery,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
)


class QueryEngineUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        # Sets up a common query engine and sample program state for testing.
        self.engine = QueryEngine()
        self.int_domain = Domain(name="IntD", gtype=GType.INT)
        self.str_domain = Domain(name="StrD", gtype=GType.STRING)

        self.attr_id = Attribute(name="id", domain=self.int_domain)
        self.attr_name = Attribute(name="name", domain=self.str_domain)
        self.attr_grp = Attribute(name="grp", domain=self.int_domain)
        self.attr_score = Attribute(name="score", domain=self.int_domain)

        self.rel_r = Relation(name="r", attributes=[self.attr_id, self.attr_name, self.attr_grp])
        self.rel_s = Relation(name="s", attributes=[self.attr_grp, self.attr_score])
        self.rel_u = Relation(name="u", attributes=[self.attr_id, self.attr_name])

        self.r = RelVar(
            relation=self.rel_r,
            tuples=[
                {"id": 1, "name": "alpha", "grp": 1},
                {"id": 2, "name": "beta", "grp": 1},
                {"id": 3, "name": "gamma", "grp": 2},
            ],
        )
        self.s = RelVar(
            relation=self.rel_s,
            tuples=[
                {"grp": 1, "score": 10},
                {"grp": 1, "score": 20},
                {"grp": 2, "score": 30},
            ],
        )
        self.u = RelVar(
            relation=self.rel_u,
            tuples=[
                {"id": 1, "name": "alpha"},
                {"id": 2, "name": "beta"},
                {"id": 2, "name": "beta"},
            ],
        )
        self.u2 = RelVar(
            relation=self.rel_u,
            tuples=[
                {"id": 2, "name": "beta"},
                {"id": 3, "name": "gamma"},
            ],
        )

        self.state = ProgramState(
            relvars={
                "r": self.r,
                "s": self.s,
                "u": self.u,
                "u2": self.u2,
            }
        )

    # === Query evaluation tests (_eval_query) ===
    def test_eval_query_relvar_leaf(self) -> None:
        # Verifies RelVarQuery resolves directly from ProgramState.
        result = self.engine._eval_query(RelVarQuery(name="r"), self.state)
        self.assertIs(result, self.r)

    def test_eval_query_let_assignment(self) -> None:
        # Verifies LET evaluates nested query, stores it in state, and returns it.
        query = LetQuery(
            target_relvar="x",
            query=ProjectQuery(source=RelVarQuery(name="r"), attributes=["id", "name"]),
        )
        result = self.engine._eval_query(query, self.state)
        self.assertIn("x", self.state.relvars)
        self.assertEqual(result.relation.attr_names(), ["id", "name"])

    def test_eval_query_select_attr_eq_const(self) -> None:
        # Verifies SELECT with A=c filters rows correctly.
        query = SelectQuery(source=RelVarQuery(name="r"), predicate=AttrEqConstPredicate(attr="grp", value=1))
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(len(result.tuples), 2)
        self.assertTrue(all(row["grp"] == 1 for row in result.tuples))

    def test_eval_query_select_attr_eq_attr(self) -> None:
        # Verifies SELECT with A1=A2 filters rows correctly.
        query = SelectQuery(source=RelVarQuery(name="r"), predicate=AttrEqAttrPredicate(left_attr="id", right_attr="grp"))
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.tuples, [{"id": 1, "name": "alpha", "grp": 1}])

    def test_eval_query_project_dedupes(self) -> None:
        # Verifies PROJECT removes duplicate projected tuples.
        query = ProjectQuery(source=RelVarQuery(name="u"), attributes=["id", "name"])
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.relation.attr_names(), ["id", "name"])
        self.assertEqual(
            result.tuples,
            [
                {"id": 1, "name": "alpha"},
                {"id": 2, "name": "beta"},
            ],
        )

    def test_eval_query_union(self) -> None:
        # Verifies UNION combines rows and removes duplicates.
        query = UnionQuery(left=RelVarQuery(name="u"), right=RelVarQuery(name="u2"))
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.relation.attr_names(), ["id", "name"])
        self.assertEqual(
            result.tuples,
            [
                {"id": 1, "name": "alpha"},
                {"id": 2, "name": "beta"},
                {"id": 3, "name": "gamma"},
            ],
        )

    def test_eval_query_difference(self) -> None:
        # Verifies DIFFERENCE removes right-side rows from left-side rows.
        query = DifferenceQuery(left=RelVarQuery(name="u"), right=RelVarQuery(name="u2"))
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.tuples, [{"id": 1, "name": "alpha"}])

    def test_eval_query_join(self) -> None:
        # Verifies natural JOIN matches on shared attributes and builds joined schema.
        query = JoinQuery(left=RelVarQuery(name="r"), right=RelVarQuery(name="s"))
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.relation.attr_names(), ["id", "name", "grp", "score"])
        self.assertEqual(len(result.tuples), 5)

    def test_eval_query_rename(self) -> None:
        # Verifies RENAME updates schema and tuple keys while preserving values.
        query = RenameQuery(source=RelVarQuery(name="r"), new_attributes=["rid", "rname", "rgrp"])
        result = self.engine._eval_query(query, self.state)
        self.assertEqual(result.relation.attr_names(), ["rid", "rname", "rgrp"])
        self.assertEqual(result.tuples[0], {"rid": 1, "rname": "alpha", "rgrp": 1})

    def test_eval_query_unsupported_type_raises(self) -> None:
        # Verifies unsupported query nodes raise a clear ValueError.
        with self.assertRaisesRegex(ValueError, "Unsupported query type"):
            self.engine._eval_query(object(), self.state)  # type: ignore[arg-type]

    # === Overall run() method tests ===
    def test_run_executes_sequence_and_returns_final(self) -> None:
        # Verifies run() executes all queries in order and returns final result.
        self.state.queries = [
            LetQuery(target_relvar="x", query=ProjectQuery(source=RelVarQuery(name="r"), attributes=["id", "grp"])),
            SelectQuery(source=RelVarQuery(name="x"), predicate=AttrEqConstPredicate(attr="grp", value=1)),
        ]
        result = self.engine.run(self.state)
        self.assertEqual(result.relation.attr_names(), ["id", "grp"])
        self.assertEqual(result.tuples, [{"id": 1, "grp": 1}, {"id": 2, "grp": 1}])

    def test_run_with_only_let_still_returns_last_result(self) -> None:
        # Verifies run() returns the evaluated LET value when program ends with LET.
        self.state.queries = [
            LetQuery(target_relvar="x", query=ProjectQuery(source=RelVarQuery(name="r"), attributes=["id"]))
        ]
        result = self.engine.run(self.state)
        self.assertEqual(result.relation.attr_names(), ["id"])
        self.assertEqual(result.tuples, [{"id": 1}, {"id": 2}, {"id": 3}])

    def test_run_raises_on_empty_queries(self) -> None:
        # Verifies run() rejects empty programs.
        self.state.queries = []
        with self.assertRaisesRegex(ValueError, "No queries to execute"):
            self.engine.run(self.state)

    # === Helper method tests ===
    def test_get_relvar_success(self) -> None:
        # Verifies _get_relvar returns an existing relation variable.
        result = QueryEngine._get_relvar(self.state, "r")
        self.assertIs(result, self.r)

    def test_get_relvar_unknown_raises(self) -> None:
        # Verifies _get_relvar raises on unknown relation variable names.
        with self.assertRaisesRegex(ValueError, "Unknown relvar"):
            QueryEngine._get_relvar(self.state, "missing")

    def test_dedupe_rows(self) -> None:
        # Verifies _dedupe_rows removes duplicates.
        rows = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
            {"id": 1, "name": "a"},
            {"id": 3, "name": "c"},
            {"id": 2, "name": "b"},
        ]
        deduped = QueryEngine._dedupe_rows(rows, ["id", "name"])
        self.assertEqual(
            deduped,
            [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
                {"id": 3, "name": "c"},
            ],
        )

    def test_assert_schema_compatible_success(self) -> None:
        # Verifies compatible schemas pass without exceptions.
        QueryEngine._assert_schema_compatible(self.u, self.u2)

    def test_assert_schema_compatible_arity_mismatch_raises(self) -> None:
        # Verifies schema compatibility rejects different arities.
        rel_short = Relation(name="short", attributes=[self.attr_id])
        short_relvar = RelVar(relation=rel_short, tuples=[{"id": 1}])
        with self.assertRaisesRegex(ValueError, "same arity"):
            QueryEngine._assert_schema_compatible(self.u, short_relvar)

    def test_assert_schema_compatible_name_or_type_mismatch_raises(self) -> None:
        # Verifies schema compatibility rejects mismatched attribute names or types.
        other_attr = Attribute(name="other_name", domain=self.str_domain)
        mismatched_relation = Relation(name="m", attributes=[self.attr_id, other_attr])
        mismatched_relvar = RelVar(relation=mismatched_relation, tuples=[{"id": 1, "other_name": "x"}])
        with self.assertRaisesRegex(ValueError, "matching attribute names and types"):
            QueryEngine._assert_schema_compatible(self.u, mismatched_relvar)


if __name__ == "__main__":
    unittest.main()
