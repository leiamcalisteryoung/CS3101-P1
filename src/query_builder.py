from lark import Token, Tree

from query_models import (
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
            theta_node = query_type.children[1]
            # build the predicate from the theta node and return a SelectQuery object
            return SelectQuery(source=RelVarQuery(name=relvar), predicate=self._build_predicate(theta_node))

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

    # Build a predicate object from a theta node.
    def _build_predicate(self, theta_node: Tree):
        if not isinstance(theta_node, Tree) or theta_node.data != "theta" or len(theta_node.children) != 2:
            raise ValueError("Expected rule 'theta' with two operands.")

        # Get the left attribute name and the right operand
        left_attr = self._name_from_node(theta_node.children[0])
        right = theta_node.children[1]

        # If right operand is an attribute, return an AttrEqAttrPredicate
        if isinstance(right, Tree) and right.data == "attr":
            return AttrEqAttrPredicate(left_attr=left_attr, right_attr=self._name_from_node(right))

        # If right operand is a constant, get the constant value and return an AttrEqConstPredicate
        if isinstance(right, Tree) and right.data == "const":
            constant_token = right.children[0]
            if not isinstance(constant_token, Token):
                raise ValueError("Malformed constant token in theta predicate.")
            if constant_token.type == "INT":
                return AttrEqConstPredicate(attr=left_attr, value=int(constant_token.value))
            
            return AttrEqConstPredicate(attr=left_attr, value=constant_token.value)

        raise ValueError("Malformed theta predicate.")

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
