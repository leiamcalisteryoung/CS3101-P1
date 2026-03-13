import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path

from lark import Token, Tree

from models import Attribute, Domain, GType, Relation, RelVar
from parser import parse_usql


@dataclass
class ProgramState:
    """
    In-memory state produced before query evaluation.
    Maps names to their definitions and loaded data for fast lookup.
    """
    domains: dict[str, Domain] = field(default_factory=dict)
    types: dict[str, Attribute] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    relvars: dict[str, RelVar] = field(default_factory=dict)
    loaded_from: dict[str, str] = field(default_factory=dict)


class USQLModelBuilder:
    def build(self, source: str) -> ProgramState:
        # Parse source text into a concrete syntax tree (CST).
        tree = parse_usql(source)
        if tree.data != "start":
            raise ValueError("Expected parse tree root 'start'.")

        program = tree.children[0]
        if not isinstance(program, Tree) or program.data != "program":
            raise ValueError("Expected 'program' node under 'start'.")

        state = ProgramState()

        # USQL requires declarations in this order:
        # DOMAIN* TYPE* RELATION* LOAD* QUERY+
        # We build the state up to LOAD declarations, then go to query evaluation
        for node in program.children:
            if not isinstance(node, Tree):
                continue

            if node.data == "domain":
                self._handle_domain(node, state)
            elif node.data == "type":
                self._handle_type(node, state)
            elif node.data == "relation":
                self._handle_relation(node, state)
            elif node.data == "load":
                self._handle_load(node, state)
            elif node.data == "query":
                # TODO: Query evaluation
                break

        return state

    # DOMAIN D IS Int|String;
    def _handle_domain(self, node: Tree, state: ProgramState) -> None:
        # get domain name
        domain_name = self._name_from_node(node.children[0])
        
        # get ground type and convert to GType enum
        gtype_token = node.children[1]
        if not isinstance(gtype_token, Token):
            raise ValueError("Expected GTYPE token in DOMAIN declaration.")
        gtype_text = gtype_token.value
        if gtype_text == "Int":
            gtype = GType.INT
        elif gtype_text == "String":
            gtype = GType.STRING
        else:
            raise ValueError(f"Unsupported ground type: {gtype_text}")

        # add domain to state
        state.domains[domain_name] = Domain(name=domain_name, gtype=gtype)

    # TYPE a AS D; resolves attribute name to a previously declared domain
    def _handle_type(self, node: Tree, state: ProgramState) -> None:
        # get attribute name and domain name
        attr_name = self._name_from_node(node.children[0])
        domain_name = self._name_from_node(node.children[1])

        # find Domain in state; throw error if not found
        domain = state.domains.get(domain_name)
        if domain is None:
            raise ValueError(f"TYPE {attr_name} references unknown DOMAIN '{domain_name}'.")

        # add attribute to state
        state.types[attr_name] = Attribute(name=attr_name, domain=domain)

    # RELATION r WITH a1, a2, ...; resolves each attribute to a previously declared attribute
    def _handle_relation(self, node: Tree, state: ProgramState) -> None:
        # get relation name and attribute names
        relation_name = self._name_from_node(node.children[0])

        attrlist_node = node.children[1]
        if not isinstance(attrlist_node, Tree) or attrlist_node.data != "attrlist":
            raise ValueError("Expected rule 'attrlist'.")

        attr_names: list[str] = []
        for child in attrlist_node.children:
            if isinstance(child, Tree) and child.data == "attr" and child.children:
                attr_names.append(self._name_from_node(child))

        # resolve each attribute name to an Attribute object in state; throw error if any not found
        relation_attrs: list[Attribute] = []
        for attr_name in attr_names:
            attr = state.types.get(attr_name)
            if attr is None:
                raise ValueError(
                    f"RELATION {relation_name} uses unknown TYPE attribute '{attr_name}'."
                )
            relation_attrs.append(attr)

        # add relation to state
        state.relations[relation_name] = Relation(name=relation_name, attributes=relation_attrs)

    # LOAD "path.csv" INTO r; materializes a RelVar for relation r.
    def _handle_load(self, node: Tree, state: ProgramState) -> None:
        # get path and resolve it from the current working directory
        path_node = node.children[0]
        if not isinstance(path_node, Tree) or path_node.data != "path" or not path_node.children:
            raise ValueError("Expected rule 'path'.")

        # path node contains an ESCAPED_STRING token, e.g. "./module.csv"
        raw_path = ast.literal_eval(self._name_from_node(path_node))
        csv_path = Path(raw_path).resolve()

        # get relation name
        relation_name = self._name_from_node(node.children[1])

        # find Relation in state; throw error if not found
        relation = state.relations.get(relation_name)
        if relation is None:
            raise ValueError(f"LOAD references unknown RELATION '{relation_name}'.")

        # read CSV, coerce values according to relation schema, and add relvar and loaded_from to state
        tuples = self._load_csv_as_tuples(csv_path, relation)
        state.relvars[relation_name] = RelVar(relation=relation, tuples=tuples)
        state.loaded_from[relation_name] = str(csv_path)

    # Helper methods for parsing and loading
    def _load_csv_as_tuples(self, csv_path: Path, relation: Relation) -> list[dict[str, int | str]]:
        # each row becomes a dict mapping attribute name to its value (int or str)
        tuples: list[dict[str, int | str]] = []

        # open the CSV file and read each row
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            expected_arity = len(relation.attributes)

            for row_idx, row in enumerate(reader, start=1):
                # Ensure row width matches relation schema arity.
                if len(row) != expected_arity:
                    raise ValueError(
                        f"CSV row {row_idx} in '{csv_path}' has {len(row)} values; "
                        f"expected {expected_arity}."
                    )

                # create a dict for this tuple
                tuple_dict: dict[str, int | str] = {}

                # zip gives a tuple of (Attribute, value) pairs;
                for attr, value in zip(relation.attributes, row):
                    # if the domain is Int, convert value to an int; otherwise keep as str
                    if attr.domain.gtype == GType.INT:
                        try:
                            tuple_dict[attr.name] = int(value)
                        except ValueError:
                            raise ValueError(
                                f"CSV row {row_idx} in '{csv_path}' has non-integer value '{value}' "
                                f"for attribute '{attr.name}' declared as Int."
                            )
                    else:
                        tuple_dict[attr.name] = value

                tuples.append(tuple_dict)

        return tuples
    
    # Helper method to get the name string from a node
    @staticmethod
    def _name_from_node(node: Tree) -> str:
        if not isinstance(node, Tree) or not node.children:
            raise ValueError("Expected a non-empty rule node.")

        token = node.children[0]
        if isinstance(token, Token):
            return token.value
        raise ValueError("Malformed name rule.")
