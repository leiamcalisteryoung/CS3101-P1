from lark import Lark

usql_grammar = """
    start:program
    
    lend: ";"
    
    dname:  CNAME
    attr:   CNAME
    relvar: CNAME
    path:   ESCAPED_STRING

    const: INT | ESCAPED_STRING

    attrlist: attr ("," attr)*

    program: domain* type* relation* load* query+
    
    GTYPE: "String" | "Int"
    domain: "DOMAIN" dname "IS" GTYPE lend

    type: "TYPE" attr "AS" dname lend

    relation: "RELATION" relvar "WITH" attrlist lend

    load: "LOAD" path "INTO" relvar lend

    theta: attr "=" attr
         | attr "=" const

    query: let_query
         | select_query
         | project_query
         | union_query
         | difference_query
         | join_query
         | rename_query

    let_query: "LET" relvar "BE" query
    select_query: "SELECT" relvar "WHERE" theta lend
    project_query: "PROJECT" relvar "ON" attrlist lend
    union_query: "UNION" relvar "AND" relvar lend
    difference_query: "DIFFERENCE" relvar "AND" relvar lend
    join_query: "JOIN" relvar "AND" relvar lend
    rename_query: "RENAME" relvar "ON" attrlist lend


    %import common.UCASE_LETTER
    %import common.CNAME
    %import common.WS
    %import common.ESCAPED_STRING
    %import common.INT
    %ignore WS
  """

l = Lark(usql_grammar)


def parse_usql(source: str):
     return l.parse(source)

