from lark import Token, Tree
import ast

from query_models import (
    AndPredicate,
    OrPredicate,
    AttrEqAttrPredicate,
    AttrEqConstPredicate,
    RelVarQuery,
    DifferenceQuery,
    JoinQuery,
    LetQuery,
    ProjectQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    Query
)

class QueryBuilder:
    # Build a query object from a CST query node.
    def build_query(self, query_node: Tree) -> Query:
        if not isinstance(query_node, Tree) or query_node.data != "query" or not query_node.children:
            raise ValueError("Expected rule 'query'.")

        # Get the child node, which tells us the query type
        query_type = query_node.children[0]
        if not isinstance(query_type, Tree):
            raise ValueError("Malformed query node.")

        # Let queries: LET r BE q
        if query_type.data == "let_query":
            target_relvar = self._name_from_node(query_type.children[0])
            nested_query_node = query_type.children[1]
            if not isinstance(nested_query_node, Tree) or nested_query_node.data != "query":
                raise ValueError("LET query must contain a nested query.")
            # recursively build the nested query and return a LetQuery object
            return LetQuery(target_relvar=target_relvar, query=self.build_query(nested_query_node))

        # Select queries: SELECT r WHERE p
        if query_type.data == "select_query":
            relvar = self._name_from_node(query_type.children[0])
            predicate_node = query_type.children[1]
            # build the predicate from the theta node and return a SelectQuery object
            return SelectQuery(source=RelVarQuery(name=relvar), predicate=self._build_predicate(predicate_node))

        # Project queries: PROJECT r ON A1, ..., An
        if query_type.data == "project_query":
            relvar = self._name_from_node(query_type.children[0])
            attrs = self._attr_names(query_type.children[1])
            return ProjectQuery(source=RelVarQuery(name=relvar), attributes=attrs)

        # Union queries: UNION r AND r
        if query_type.data == "union_query":
            left = self._name_from_node(query_type.children[0])
            right = self._name_from_node(query_type.children[1])
            return UnionQuery(left=RelVarQuery(name=left), right=RelVarQuery(name=right))

        # Difference queries: DIFFERENCE r AND r
        if query_type.data == "difference_query":
            left = self._name_from_node(query_type.children[0])
            right = self._name_from_node(query_type.children[1])
            return DifferenceQuery(left=RelVarQuery(name=left), right=RelVarQuery(name=right))

        # Join queries: JOIN r AND r
        if query_type.data == "join_query":
            left = self._name_from_node(query_type.children[0])
            right = self._name_from_node(query_type.children[1])
            return JoinQuery(left=RelVarQuery(name=left), right=RelVarQuery(name=right))

        # Rename queries: RENAME r ON A1, ..., An
        if query_type.data == "rename_query":
            relvar = self._name_from_node(query_type.children[0])
            attrs = self._attr_names(query_type.children[1])
            return RenameQuery(source=RelVarQuery(name=relvar), new_attributes=attrs)

        raise ValueError(f"Unsupported query rule '{query_type.data}'.")

    # Build a predicate object from a predicate expression node.
    def _build_predicate(self, node: Tree):
        if not isinstance(node, Tree):
            raise ValueError("Malformed predicate expression.")

        # These are wrappers for comparison predicates of the form attr COMP attr/const
        if node.data == "predicate":
            return self._build_predicate(node.children[0])

        # For OR and AND predicates, recursively build the left and right subpredicates and combine them.
        if node.data == "or_pred":
            parts = [self._build_predicate(child) for child in node.children if isinstance(child, Tree)]
            if len(parts) != 2:
                raise ValueError("Malformed OR predicate.")
            return OrPredicate(left=parts[0], right=parts[1])

        if node.data == "and_pred":
            parts = [self._build_predicate(child) for child in node.children if isinstance(child, Tree)]
            if len(parts) != 2:
                raise ValueError("Malformed AND predicate.")
            return AndPredicate(left=parts[0], right=parts[1])

        # Base case: comparison predicates
        if node.data == "comparison":
            if len(node.children) != 3:
                raise ValueError("Expected comparison with left operand, operator, and right operand.")
            
            # Extract the left attribute, operator, and right operand (attribute or constant)
            left_attr = self._name_from_node(node.children[0])
            operator_token = node.children[1]
            right = node.children[2]

            if not isinstance(operator_token, Token) or operator_token.type != "COMP":
                raise ValueError("Malformed comparison operator in predicate.")

            operator = operator_token.value

            # Case attributes on both sides: A1 COMP A2
            if isinstance(right, Tree) and right.data == "attr":
                right_attr = self._name_from_node(right)
                return AttrEqAttrPredicate(
                    left_attr=left_attr,
                    operator=operator,
                    right_attr=right_attr,
                )

            # Case constant on the right side: A COMP c
            if isinstance(right, Tree) and right.data == "const":
                constant_token = right.children[0]
                if not isinstance(constant_token, Token):
                    raise ValueError("Malformed constant token in predicate.")

                if constant_token.type == "INT":
                    value: int | str = int(constant_token.value)
                elif constant_token.type == "ESCAPED_STRING":
                    value = ast.literal_eval(constant_token.value)
                else:
                    value = constant_token.value

                return AttrEqConstPredicate(
                    attr=left_attr,
                    operator=operator,
                    value=value,
                )

            raise ValueError("Malformed comparison predicate.")

        # If grammar inlines single-child layers, recurse into first child tree.
        tree_children = [child for child in node.children if isinstance(child, Tree)]
        if len(tree_children) == 1:
            return self._build_predicate(tree_children[0])

        raise ValueError(f"Unsupported predicate rule '{node.data}'.")

    # Helper method to extract attribute names from an attrlist node
    @staticmethod
    def _attr_names(node: Tree) -> list[str]:
        if not isinstance(node, Tree) or node.data != "attrlist":
            raise ValueError("Expected rule 'attrlist'.")

        names: list[str] = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "attr":
                names.append(QueryBuilder._name_from_node(child))
        return names

    # Helper method to get the name string from a node
    @staticmethod
    def _name_from_node(node: Tree) -> str:
        if not isinstance(node, Tree) or not node.children:
            raise ValueError("Expected a non-empty rule node.")

        token = node.children[0]
        if isinstance(token, Token):
            return token.value
        raise ValueError("Malformed name rule.")
