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
    
    gtype: "String" | "Int"
    domain: "DOMAIN" dname "IS" gtype lend

    type: "TYPE" attr "AS" dname lend

    relation: "RELATION" relvar "WITH" attrlist lend

    load: "LOAD" path "INTO" relvar lend

    theta: attr "=" attr
         | attr "=" const

    query: "LET" relvar "BE" query
         | "SELECT" relvar "WHERE" theta lend
         | "PROJECT" relvar "ON" attrlist lend
         | "UNION" relvar "AND" relvar lend
         | "DIFFERENCE" relvar "AND" relvar lend
         | "JOIN" relvar "AND" relvar lend
         | "RENAME" relvar "ON" attrlist lend


    %import common.UCASE_LETTER
    %import common.CNAME
    %import common.WS
    %import common.ESCAPED_STRING
    %import common.INT
    %ignore WS
  """

l = Lark(usql_grammar)

