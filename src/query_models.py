from dataclasses import dataclass


@dataclass
class AttrEqAttrPredicate:
    """Predicate comparing two attributes (A1 = A2)"""
    left_attr: str
    right_attr: str


@dataclass
class AttrEqConstPredicate:
    """Predicate comparing an attribute to a constant (A1 = c)"""
    attr: str
    value: int | str


# Leaf node: reference to a relation variable name.
@dataclass
class RelVarQuery:
    name: str


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
Predicate = AttrEqAttrPredicate | AttrEqConstPredicate
QueryExpr = RelVarQuery | SelectQuery | ProjectQuery | UnionQuery | DifferenceQuery | JoinQuery | RenameQuery
Query = LetQuery | QueryExpr
