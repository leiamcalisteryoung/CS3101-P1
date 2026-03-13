from enum import Enum
from dataclasses import dataclass, field


class GType(Enum):
    """The two primitive ground types supported by USQL."""
    INT = "Int"
    STRING = "String"


@dataclass
class Domain:
    """
    Represents a DOMAIN declaration: DOMAIN D IS t
    e.g. DOMAIN ModCode IS String -> Domain(name='ModCode', gtype=GType.STRING)
    """
    name: str
    gtype: GType


@dataclass
class Attribute:
    """
    Represents a TYPE declaration: TYPE A AS D
    e.g. TYPE mc AS ModCode -> Attribute(name='mc', domain=Domain('ModCode', GType.STRING))
    """
    name: str
    domain: Domain


@dataclass
class Relation:
    """
    Represents a RELATION declaration: RELATION r WITH A1, ..., An
    This is just the structure/schema — no data yet.
    e.g. RELATION module WITH mc, mt, ms, cr -> Relation(name='module', attributes=[Attribute(name='mc', domain=Domain(name='ModCode', gtype=GType.STRING)), ...])
    """
    name: str
    attributes: list["Attribute"] = field(default_factory=list)

    # get attribute names
    def attr_names(self) -> list[str]:
        return [a.name for a in self.attributes]

    # get attribute by name
    def get_attr(self, name: str) -> "Attribute | None":
        for a in self.attributes:
            if a.name == name:
                return a
        return None


@dataclass
class RelVar:
    """
    A relation variable: a Relation plus its current set of tuples.
    Created by LOAD (reading a CSV) or by LET (assigning a query result).
    Each tuple is a dict mapping attribute name to its value.
    This is what all query operators receive and return.
    """
    relation: Relation
    tuples: list[dict[str, int | str]] = field(default_factory=list)

    # return a hashable tuple of all values in this row, in attribute order
    def row_signature(self, row: dict[str, int | str]) -> tuple[int | str, ...]:
        return tuple(row[attr] for attr in self.relation.attr_names())

    # print the CSV formatted relvar
    def __repr__(self) -> str:
        lines = [", ".join(self.relation.attr_names()) + ";"]
        for t in self.tuples:
            row = ", ".join(str(t[a]) for a in self.relation.attr_names())
            lines.append(row + ";")
        return "\n".join(lines)
