import sys
import tempfile
import unittest
from pathlib import Path

from lark.exceptions import UnexpectedInput

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from builder import USQLModelBuilder
from parser import parse_usql


class ParserBuilderTests(unittest.TestCase):
    # Builds a minimal valid USQL program string that loads from the provided CSV path.
    @staticmethod
    def _program_with_csv(csv_path: str, query_block: str) -> str:
        return f"""
DOMAIN IdD IS Int;
DOMAIN NameD IS String;

TYPE id AS IdD;
TYPE name AS NameD;

RELATION users WITH id, name;
LOAD \"{csv_path}\" INTO users;

{query_block}
""".strip()

    # === Parser tests (minimal as builder tests are more in-depth)===
    def test_parse_usql_valid_program(self) -> None:
        # Verifies parser accepts a valid minimal program and returns 'start' root.
        source = (
            "DOMAIN IdD IS Int; "
            "TYPE id AS IdD; "
            "RELATION users WITH id; "
            "LOAD \"./module.csv\" INTO users; "
            "PROJECT users ON id;"
        )
        tree = parse_usql(source)
        self.assertEqual(tree.data, "start")

    def test_parse_usql_invalid_program_raises(self) -> None:
        # Verifies parser rejects syntactically invalid USQL.
        source = "DOMAIN IdD Int; PROJECT users ON id;"
        with self.assertRaises(UnexpectedInput):
            parse_usql(source)

    # === Builder tests ===
    def test_builder_happy_path_builds_state(self) -> None:
        # Verifies builder creates domains/types/relations/relvars/queries from a valid program.
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            csv_path = temp_dir / "users.csv"
            csv_path.write_text("1,Alice\n2,Bob\n", encoding="utf-8")

            source = self._program_with_csv(
                str(csv_path),
                "PROJECT users ON id, name;",
            )

            state = USQLModelBuilder().build(source)

            self.assertIn("IdD", state.domains)
            self.assertIn("id", state.types)
            self.assertIn("users", state.relations)
            self.assertIn("users", state.relvars)
            self.assertEqual(len(state.relvars["users"].tuples), 2)
            self.assertEqual(len(state.queries), 1)

    def test_builder_type_unknown_domain_raises(self) -> None:
        # Verifies builder rejects TYPE declarations that reference undeclared DOMAINs.
        source = (
            "DOMAIN IdD IS Int; "
            "TYPE id AS MissingDomain; "
            "RELATION users WITH id; "
            "LOAD \"./module.csv\" INTO users; "
            "PROJECT users ON id;"
        )
        with self.assertRaisesRegex(ValueError, "references unknown DOMAIN"):
            USQLModelBuilder().build(source)

    def test_builder_relation_unknown_type_raises(self) -> None:
        # Verifies builder rejects RELATION attributes that were not declared as TYPEs.
        source = (
            "DOMAIN IdD IS Int; "
            "RELATION users WITH id; "
            "LOAD \"./module.csv\" INTO users; "
            "PROJECT users ON id;"
        )
        with self.assertRaisesRegex(ValueError, "uses unknown TYPE attribute"):
            USQLModelBuilder().build(source)

    def test_builder_load_unknown_relation_raises(self) -> None:
        # Verifies builder rejects LOAD declarations for undeclared relations.
        source = (
            "DOMAIN IdD IS Int; "
            "TYPE id AS IdD; "
            "LOAD \"./module.csv\" INTO users; "
            "PROJECT users ON id;"
        )
        with self.assertRaisesRegex(ValueError, "unknown RELATION"):
            USQLModelBuilder().build(source)

    def test_builder_csv_arity_mismatch_raises(self) -> None:
        # Verifies builder detects CSV rows with incorrect number of columns.
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            csv_path = temp_dir / "users.csv"
            csv_path.write_text("1,Alice,Extra\n", encoding="utf-8")

            source = self._program_with_csv(
                str(csv_path),
                "PROJECT users ON id, name;",
            )

            with self.assertRaisesRegex(ValueError, "expected 2"):
                USQLModelBuilder().build(source)

    def test_builder_non_integer_value_raises(self) -> None:
        # Verifies builder enforces integer coercion for Int-typed attributes.
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            csv_path = temp_dir / "users.csv"
            csv_path.write_text("abc,Alice\n", encoding="utf-8")

            source = self._program_with_csv(
                str(csv_path),
                "PROJECT users ON id, name;",
            )

            with self.assertRaisesRegex(ValueError, "non-integer value"):
                USQLModelBuilder().build(source)



if __name__ == "__main__":
    unittest.main()
