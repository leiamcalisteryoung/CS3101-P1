from builder import ProgramState
from models import Attribute, Relation, RelVar
from query_models import (
    AndPredicate,
    OrPredicate,
    AttrOpAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    JoinQuery,
    LetQuery,
    ProjectQuery,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    Predicate,
    Query,
    QueryExpr
)


class QueryEngine:
    # Execute all parsed queries in order and return the final query result.
    def run(self, state: ProgramState) -> RelVar:
        if not state.queries:
            raise ValueError("No queries to execute.")

        # each query is executed in order, with the state updated as we go
        result: RelVar | None = None
        for query in state.queries:
            result = self._eval_query(query, state)

        if result is None:
            raise ValueError("Query execution produced no result.")

        return result

    # Evaluate a single query
    def _eval_query(self, query: Query, state: ProgramState) -> RelVar:
        # Relvar leaf node resolves directly from program state.
        if isinstance(query, RelVarQuery):
            return self._get_relvar(state, query.name)
        
        # LET queries recursively evaluate their nested query and assign the result to a relvar in the state
        if isinstance(query, LetQuery):
            value = self._eval_query(query.query, state)
            state.relvars[query.target_relvar] = value
            return value

        # Select queries: filter source tuples by predicate
        if isinstance(query, SelectQuery):
            source = self._eval_query(query.source, state)
            return self._select(source, query.predicate)

        # Project queries: keep only selected attributes of source
        if isinstance(query, ProjectQuery):
            source = self._eval_query(query.source, state)
            return self._project(source, query.attributes)

        # Union queries: set union of two source expressions
        if isinstance(query, UnionQuery):
            left = self._eval_query(query.left, state)
            right = self._eval_query(query.right, state)
            return self._union(left, right)

        # Difference queries: set difference of two source expressions
        if isinstance(query, DifferenceQuery):
            left = self._eval_query(query.left, state)
            right = self._eval_query(query.right, state)
            return self._difference(left, right)

        # Join queries: natural join of two source expressions
        if isinstance(query, JoinQuery):
            left = self._eval_query(query.left, state)
            right = self._eval_query(query.right, state)
            return self._join(left, right)

        # Rename queries: change attribute names of a source expression
        if isinstance(query, RenameQuery):
            source = self._eval_query(query.source, state)
            return self._rename(source, query.new_attributes)

        raise ValueError(f"Unsupported query type: {type(query).__name__}")

    # Filter rows of source relvar by predicate, keeping the same relation schema.
    def _select(self, source: RelVar, predicate: Predicate) -> RelVar:
        # resulting tuples that satisfy the predicate
        result_rows: list[dict[str, int | str]] = []

        # for each tuple, add to result if it satisfies the predicate
        for row in source.tuples:
            if self._predicate_holds(row, predicate):
                result_rows.append(dict(row))

        return RelVar(relation=source.relation, tuples=result_rows)

    # Evaluate whether one tuple satisfies a predicate tree.
    def _predicate_holds(self, row: dict[str, int | str], predicate: Predicate) -> bool:
        if isinstance(predicate, AttrOpAttrPredicate):
            if predicate.left_attr not in row or predicate.right_attr not in row:
                raise ValueError("SELECT predicate references missing attribute.")
            return self._apply_comparison(
                row[predicate.left_attr],
                predicate.operator,
                row[predicate.right_attr],
            )

        if isinstance(predicate, AttrEqConstPredicate):
            if predicate.attr not in row:
                raise ValueError("SELECT predicate references missing attribute.")
            return self._apply_comparison(row[predicate.attr], predicate.operator, predicate.value)

        # recursively evaluate AND and OR predicates
        if isinstance(predicate, AndPredicate):
            return self._predicate_holds(row, predicate.left) and self._predicate_holds(row, predicate.right)

        if isinstance(predicate, OrPredicate):
            return self._predicate_holds(row, predicate.left) or self._predicate_holds(row, predicate.right)

        raise ValueError(f"Unsupported predicate type: {type(predicate).__name__}")

    # Apply one comparison operator to two scalar values.
    @staticmethod
    def _apply_comparison(left: int | str, operator: str, right: int | str) -> bool:
        try:
            if operator == "=":
                return left == right
            if operator == "!=":
                return left != right
            if operator == "<":
                return left < right
            if operator == "<=":
                return left <= right
            if operator == ">":
                return left > right
            if operator == ">=":
                return left >= right
        except TypeError as exc:
            raise ValueError(f"Invalid comparison between values '{left}' and '{right}'.") from exc

        raise ValueError(f"Unsupported comparison operator '{operator}'.")
    
    # Keep only selected columns and remove duplicate tuples
    def _project(self, source: RelVar, attr_names: list[str]) -> RelVar:
        # Check that all projected attribute names exist in the source relation
        for name in attr_names:
            if source.relation.get_attr(name) is None:
                raise ValueError(f"PROJECT references unknown attribute '{name}'.")

        # Create the projected relation schema based on the selected attribute names
        projected_attrs: list[Attribute] = [source.relation.get_attr(name) for name in attr_names]
        projected_relation = Relation(name=source.relation.name, attributes=projected_attrs)

        # For each tuple, create a new tuple with only the projected attributes
        projected_rows: list[dict[str, int | str]] = []
        for row in source.tuples:
            projected_rows.append({name: row[name] for name in attr_names})

        # remove duplicates from projeccted rows and return the new relvar
        projected_rows = self._dedupe_rows(projected_rows, attr_names)
        return RelVar(relation=projected_relation, tuples=projected_rows)
    
    # Set union of two relvars with the same schema, removing duplicates.
    def _union(self, left: RelVar, right: RelVar) -> RelVar:
        # Ensure the two relvars have compatible schemas
        self._assert_schema_compatible(left, right)

        # Create the union of the two sets then remove duplicates and return
        attrs = left.relation.attr_names()
        rows = [dict(r) for r in left.tuples] + [dict(r) for r in right.tuples]
        rows = self._dedupe_rows(rows, attrs)
        return RelVar(relation=left.relation, tuples=rows)

    # Set difference of two relvars with the same schema
    def _difference(self, left: RelVar, right: RelVar) -> RelVar:
        # Ensure the two relvars have compatible schemas
        self._assert_schema_compatible(left, right)
        
        # Compute signatures for all right-hand rows for fast lookup
        right_sigs = {right.row_signature(row) for row in right.tuples}

        # For each tuple in the left relvar, keep if it is not in the right relvar
        result_rows: list[dict[str, int | str]] = []
        for row in left.tuples:
            if left.row_signature(row) not in right_sigs:
                result_rows.append(dict(row))

        return RelVar(relation=left.relation, tuples=result_rows)

    # Natural join of two relvars on shared attribute names
    def _join(self, left: RelVar, right: RelVar) -> RelVar:
        # Get the list of shared attribute names
        left_attrs = left.relation.attr_names()
        right_attrs = right.relation.attr_names()
        shared = [name for name in left_attrs if name in right_attrs]
        
        # Create the joined relation schema: all left attributes plus right-only attributes
        right_only = [name for name in right_attrs if name not in left_attrs]
        joined_attrs: list[Attribute] = list(left.relation.attributes) + [
            attr for attr in right.relation.attributes if attr.name in right_only
        ]
        joined_relation = Relation(
            name=f"{left.relation.name}_join_{right.relation.name}",
            attributes=joined_attrs,
        )

        # for each pair of tuples, if they match on all shared attributes, combine them into a joined tuple
        result_rows: list[dict[str, int | str]] = []
        for left_row in left.tuples:
            for right_row in right.tuples:
                if all(left_row[name] == right_row[name] for name in shared):
                    # copy left row and add right-only attributes to create the joined row
                    new_row = dict(left_row)
                    for name in right_only:
                        new_row[name] = right_row[name]
                    result_rows.append(new_row)

        # remove duplicates from joined rows and return the new relvar
        result_rows = self._dedupe_rows(result_rows, joined_relation.attr_names())
        return RelVar(relation=joined_relation, tuples=result_rows)

    # Renames all attributes of the source relvar to the new names provided (arity must stay the same).
    def _rename(self, source: RelVar, new_attr_names: list[str]) -> RelVar:
        old_attrs = source.relation.attributes
        
        # Ensure the number of new attribute names matches the number of existing attributes
        if len(old_attrs) != len(new_attr_names):
            raise ValueError("RENAME must provide exactly one name per existing attribute.")

        # Create a list of new Attribute objects with the new names but the same domains as the old attributes
        renamed_attrs: list[Attribute] = []
        for old_attr, new_name in zip(old_attrs, new_attr_names):
            renamed_attrs.append(Attribute(name=new_name, domain=old_attr.domain))

        # Create the new Relation schema for the renamed relvar
        renamed_relation = Relation(name=source.relation.name, attributes=renamed_attrs)

        # For each tuple in the source relvar, create a new tuple with keys renamed according to the new attribute names
        result_rows: list[dict[str, int | str]] = []
        for row in source.tuples:
            new_row: dict[str, int | str] = {}
            for old_attr, new_name in zip(old_attrs, new_attr_names):
                new_row[new_name] = row[old_attr.name]
            result_rows.append(new_row)

        return RelVar(relation=renamed_relation, tuples=result_rows)
    

    # Helper method to get relvar from state by name with error handling.
    @staticmethod
    def _get_relvar(state: ProgramState, name: str) -> RelVar:
        relvar = state.relvars.get(name)
        if relvar is None:
            raise ValueError(f"Unknown relvar '{name}'.")
        return relvar

    # Helper method to remove duplicate tuples, preserving order
    @staticmethod
    def _dedupe_rows(rows: list[dict[str, int | str]], attrs: list[str]) -> list[dict[str, int | str]]:
        seen: set[tuple[int | str, ...]] = set()
        unique_rows: list[dict[str, int | str]] = []
        for row in rows:
            sig = tuple(row[attr] for attr in attrs)
            if sig not in seen:
                seen.add(sig)
                unique_rows.append(row)
        return unique_rows

    # Helper method to check that two relvars have compatible schemas for UNION/DIFFERENCE
    @staticmethod
    def _assert_schema_compatible(left: RelVar, right: RelVar) -> None:
        left_attrs = left.relation.attributes
        right_attrs = right.relation.attributes

        if len(left_attrs) != len(right_attrs):
            raise ValueError(f"UNION/DIFFERENCE requires relations with the same arity.")

        for la, ra in zip(left_attrs, right_attrs):
            if la.name != ra.name or la.domain.gtype != ra.domain.gtype:
                raise ValueError(f"UNION/DIFFERENCE requires matching attribute names and types.")
