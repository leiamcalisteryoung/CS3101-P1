from dataclasses import dataclass


@dataclass
class AttrEqAttrPredicate:
    """Predicate comparing two attributes with an operator.
    Supported operators: =, !=, <, <=, >, >=
    """
    left_attr: str
    operator: str
    right_attr: str


@dataclass
class AttrEqConstPredicate:
    """Predicate comparing an attribute to a constant with an operator.
    Supported operators: =, !=, <, <=, >, >=
    """
    attr: str
    operator: str
    value: int | str


@dataclass
class AndPredicate:
    """Conjunction of two predicates (θ1 ∧ θ2) (internal optimizer-only node)"""
    left: "Predicate"
    right: "Predicate"


@dataclass
class OrPredicate:
    """Disjunction of two predicates (θ1 ∨ θ2)"""
    left: "Predicate"
    right: "Predicate"


# Leaf node: reference to a relation variable name.
@dataclass
class RelVarQuery:
    name: str


@dataclass
class EmptyQuery:
    """Internal optimizer-only empty relation node"""
    pass


@dataclass
class LetQuery:
    """Assigns a query result to a relvar
    LET r BE q"""
    target_relvar: str
    query: "Query"


@dataclass
class SelectQuery:
    """Selects tuples from a source query that satisfy a predicate.
    SELECT r WHERE p"""
    source: "QueryExpr"
    predicate: "Predicate"


@dataclass
class ProjectQuery:
    """Projects attributes from a source query.
    PROJECT r ON A1, ..., An"""
    source: "QueryExpr"
    attributes: list[str]


@dataclass
class UnionQuery:
    """Forms the set union of two source queries.
    UNION r AND r"""
    left: "QueryExpr"
    right: "QueryExpr"


@dataclass
class DifferenceQuery:
    """Forms the set difference of two source queries.
    DIFFERENCE r AND r"""
    left: "QueryExpr"
    right: "QueryExpr"


@dataclass
class JoinQuery:
    """Forms the natural join of two source queries.
    JOIN r AND r"""
    left: "QueryExpr"
    right: "QueryExpr"


@dataclass
class RenameQuery:
    """Renames attributes of a source query.
    RENAME r AS A1, ..., An"""
    source: "QueryExpr"
    new_attributes: list[str]


# Type aliases
Predicate = (
    AttrEqAttrPredicate
    | AttrEqConstPredicate
    | AndPredicate
    | OrPredicate
)
QueryExpr = EmptyQuery | RelVarQuery | SelectQuery | ProjectQuery | UnionQuery | DifferenceQuery | JoinQuery | RenameQuery
Query = LetQuery | QueryExpr


# Build one large expression tree by inlining LET-defined relvars in query order.
def inline_final_query(queries: list[Query]) -> QueryExpr:
    if not queries:
        raise ValueError("Program has no queries to optimize.")

    # Map relvar names to their inlined expression trees from LET statements.
    bindings: dict[str, QueryExpr] = {}

    # Build bindings from preceding LET queries in execution order.
    for query in queries[:-1]:
        if isinstance(query, LetQuery):
            bindings[query.target_relvar] = _inline_query(
                query.query,
                bindings,
                active=set(),
            )

    # Return the final query with all LET-defined relvars substituted.
    final_query = queries[-1]
    return _inline_query(final_query, bindings, active=set())


# Recursively substitute LET-defined relvars with their expression trees.
def _inline_query(
    query: Query,
    bindings: dict[str, QueryExpr],
    active: set[str],
) -> QueryExpr:
    if isinstance(query, EmptyQuery):
        return EmptyQuery()

    if isinstance(query, RelVarQuery):
        # 'active' tracks the current inlining path (DFS stack).
        # If we revisit the same relvar while it's active, we found a LET cycle.
        if query.name in active:
            raise ValueError(f"Cyclic LET dependency detected at relvar '{query.name}'.")

        if query.name in bindings:
            return _inline_query(bindings[query.name], bindings, active | {query.name})

        # Base relation / loaded relvar leaf.
        return RelVarQuery(name=query.name)

    if isinstance(query, SelectQuery):
        source = _inline_query(query.source, bindings, active)
        return SelectQuery(source=source, predicate=query.predicate)

    if isinstance(query, ProjectQuery):
        source = _inline_query(query.source, bindings, active)
        return ProjectQuery(source=source, attributes=list(query.attributes))

    if isinstance(query, UnionQuery):
        left = _inline_query(query.left, bindings, active)
        right = _inline_query(query.right, bindings, active)
        return UnionQuery(left=left, right=right)

    if isinstance(query, DifferenceQuery):
        left = _inline_query(query.left, bindings, active)
        right = _inline_query(query.right, bindings, active)
        return DifferenceQuery(left=left, right=right)

    if isinstance(query, JoinQuery):
        left = _inline_query(query.left, bindings, active)
        right = _inline_query(query.right, bindings, active)
        return JoinQuery(left=left, right=right)

    if isinstance(query, RenameQuery):
        source = _inline_query(query.source, bindings, active)
        return RenameQuery(source=source, new_attributes=list(query.new_attributes))

    if isinstance(query, LetQuery):
        rhs = _inline_query(query.query, bindings, active)
        bindings[query.target_relvar] = rhs
        return rhs

    raise ValueError(f"Unsupported query node type: {type(query).__name__}")


# Pretty-printer for query expression trees.
def format_query_expr(expr: QueryExpr) -> str:
    if isinstance(expr, EmptyQuery):
        return "∅"

    if isinstance(expr, RelVarQuery):
        return expr.name

    if isinstance(expr, SelectQuery):
        pred = format_predicate(expr.predicate)
        return f"σ[{pred}]({format_query_expr(expr.source)})"

    if isinstance(expr, ProjectQuery):
        attrs = ", ".join(expr.attributes)
        return f"π[{attrs}]({format_query_expr(expr.source)})"

    if isinstance(expr, UnionQuery):
        return f"({format_query_expr(expr.left)} ∪ {format_query_expr(expr.right)})"

    if isinstance(expr, DifferenceQuery):
        return f"({format_query_expr(expr.left)} − {format_query_expr(expr.right)})"

    if isinstance(expr, JoinQuery):
        return f"({format_query_expr(expr.left)} ⋈ {format_query_expr(expr.right)})"

    if isinstance(expr, RenameQuery):
        attrs = ", ".join(expr.new_attributes)
        return f"ρ[{attrs}]({format_query_expr(expr.source)})"

    raise ValueError(f"Unsupported expression node type: {type(expr).__name__}")

# Pretty-printer for predicates.
def format_predicate(predicate: Predicate) -> str:
    if isinstance(predicate, AndPredicate):
        return f"({format_predicate(predicate.left)} ∧ {format_predicate(predicate.right)})"

    if isinstance(predicate, OrPredicate):
        return f"({format_predicate(predicate.left)} ∨ {format_predicate(predicate.right)})"

    if isinstance(predicate, AttrEqAttrPredicate):
        return f"{predicate.left_attr}{predicate.operator}{predicate.right_attr}"

    if isinstance(predicate, AttrEqConstPredicate):
        value = repr(predicate.value) if isinstance(predicate.value, str) else str(predicate.value)
        return f"{predicate.attr}{predicate.operator}{value}"

    raise ValueError(f"Unsupported predicate type: {type(predicate).__name__}")

# Gets the set of attribute names referenced in a predicate (for optimizer use).
def predicate_attributes(predicate: Predicate) -> set[str]:
    if isinstance(predicate, AttrEqAttrPredicate):
        return {predicate.left_attr, predicate.right_attr}

    if isinstance(predicate, AttrEqConstPredicate):
        return {predicate.attr}

    if isinstance(predicate, AndPredicate):
        return predicate_attributes(predicate.left) | predicate_attributes(predicate.right)

    if isinstance(predicate, OrPredicate):
        return predicate_attributes(predicate.left) | predicate_attributes(predicate.right)

    raise ValueError(f"Unsupported predicate type: {type(predicate).__name__}")
