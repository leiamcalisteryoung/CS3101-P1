from builder import ProgramState
from query_models import (
    format_query_expr,
    inline_final_query,
)

class QueryOptimizer:
    # Main entry point
    def run(self, state: ProgramState) -> str:
        inlined = inline_final_query(state.queries)
        return f"Initial inlined query: {format_query_expr(inlined)}"
