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

@dataclass
class LetQuery:
    """Assigns a query result to a relvar
    LET r BE q"""
    target_relvar: str
    query: "Query"


@dataclass
class SelectQuery:
    """Selects tuples from a relvar that satisfy a predicate
    SELECT r WHERE p"""
    relvar: str
    predicate: "Predicate"


@dataclass
class ProjectQuery:
    """Projects attributes from a relvar
    PROJECT r ON A1, ..., An"""
    relvar: str
    attributes: list[str]


@dataclass
class UnionQuery:
    """Forms the set union of two relvars
    UNION r and r"""
    left_relvar: str
    right_relvar: str


@dataclass
class DifferenceQuery:
    """Forms the set difference of two relvars
    DIFFERENCE r and r"""
    left_relvar: str
    right_relvar: str


@dataclass
class JoinQuery:
    """Forms the natural join of two relvars
    JOIN r and r"""
    left_relvar: str
    right_relvar: str


@dataclass
class RenameQuery:
    """Renames attributes of a relvar
    RENAME r ON A1, ..., An"""
    relvar: str
    new_attributes: list[str]


# Type aliases
Predicate = AttrEqAttrPredicate | AttrEqConstPredicate
Query = LetQuery | SelectQuery | ProjectQuery | UnionQuery | DifferenceQuery | JoinQuery | RenameQuery
