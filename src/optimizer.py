from builder import ProgramState
from query_models import (
    AndPredicate,
    DifferenceQuery,
    EmptyQuery,
    JoinQuery,
    Predicate,
    ProjectQuery,
    QueryExpr,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    format_predicate,
    format_query_expr,
    inline_final_query,
    predicate_attributes,
)


class QueryOptimizer:
    # Main entry point
    def run(self, state: ProgramState) -> str:
        # Get the final query expression with all LETs inlined
        expr = inline_final_query(state.queries)
        steps: list[str] = [f"Initial inlined query: {format_query_expr(expr)}"]

        # Repeatedly apply one rewrite per pass until no rule applies.
        while True:
            rewritten, rule_name = self._rewrite_once_bottom_up(expr)
            if rule_name is None:
                break # no more rewrites, we're done

            # Update expression and record step.
            expr = rewritten
            steps.append(f"{rule_name}: {format_query_expr(expr)}")

        return "\n".join(steps)

    # Apply first applicable rewrite rule in a bottom-up traversal, returning the rewritten query and the name of the rule applied (or None if no rewrite).
    def _rewrite_once_bottom_up(self, expr: QueryExpr) -> tuple[QueryExpr, str | None]:
        # Match on query type and recursively rewrite children
        # If any child rewrites, return a new query node with the rewritten child and the rule_name
        if isinstance(expr, SelectQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return SelectQuery(source=new_source, predicate=expr.predicate), rule_name

        elif isinstance(expr, ProjectQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return ProjectQuery(source=new_source, attributes=list(expr.attributes)), rule_name

        elif isinstance(expr, RenameQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return RenameQuery(source=new_source, new_attributes=list(expr.new_attributes)), rule_name

        elif isinstance(expr, UnionQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return UnionQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return UnionQuery(left=expr.left, right=new_right), rule_name

        elif isinstance(expr, DifferenceQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return DifferenceQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return DifferenceQuery(left=expr.left, right=new_right), rule_name

        elif isinstance(expr, JoinQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return JoinQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return JoinQuery(left=expr.left, right=new_right), rule_name

        # Base case: no child changed, try all node-level rewrites here.
        rewritten, rule_name = self._apply_node_rewrite_rules(expr)
        if rewritten is not None:
            return rewritten, rule_name

        # No rewrite applied at this node
        return expr, None

    # Try all rewrites for one node in priority order, returning the first successful rewrite and its rule name (or None if no rewrite applies).
    def _apply_node_rewrite_rules(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # 1) Try trivial simplifications first.
        rewritten = self._apply_trivial_simplification(expr)
        if rewritten is not None and rewritten != expr:
            return rewritten, "Trivial simplification"

        # 2) Unary equivalences
        rewritten, rule_name = self._apply_unary_equivalences(expr)
        if rewritten is not None:
            return rewritten, rule_name

        # No rewrite applied to this node
        return None, None

    # Trivial simplifications that don't require knowing the relation contents, just the query structure.
    def _apply_trivial_simplification(self, expr: QueryExpr) -> QueryExpr | None:
        # r ∪ r ≡ r
        if isinstance(expr, UnionQuery) and expr.left == expr.right:
            return expr.left

        # r ▷◁ r ≡ r
        if isinstance(expr, JoinQuery) and expr.left == expr.right:
            return expr.left

        # r − r ≡ ∅
        if isinstance(expr, DifferenceQuery) and expr.left == expr.right:
            return EmptyQuery()

        # r ∪ ∅ ≡ r,  ∅ ∪ r ≡ r
        if isinstance(expr, UnionQuery):
            if isinstance(expr.left, EmptyQuery):
                return expr.right
            if isinstance(expr.right, EmptyQuery):
                return expr.left

        # r ▷◁ ∅ ≡ ∅,  ∅ ▷◁ r ≡ ∅, r ▷◁θ ∅ ≡ ∅, ∅ ▷◁θ r ≡ ∅
        if isinstance(expr, JoinQuery):
            if isinstance(expr.left, EmptyQuery) or isinstance(expr.right, EmptyQuery):
                return EmptyQuery()

        # r − ∅ ≡ r,  ∅ − r ≡ ∅
        if isinstance(expr, DifferenceQuery):
            if isinstance(expr.right, EmptyQuery):
                return expr.left
            if isinstance(expr.left, EmptyQuery):
                return EmptyQuery()

        # πA(∅) ≡ ∅
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        # σθ(∅) ≡ ∅
        if isinstance(expr, SelectQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        # ρN(∅) ≡ ∅
        if isinstance(expr, RenameQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        return None
    
    # Apply unary equivalence rewrites. Returns the rewritten query and the name of the rule applied (or None if no rewrite applies).
    def _apply_unary_equivalences(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # σθ1 (σθ2 (r)) ≡ σθ1∧θ2 (r)
        if isinstance(expr, SelectQuery) and isinstance(expr.source, SelectQuery):
            merged = SelectQuery(
                source=expr.source.source,
                predicate=AndPredicate(left=expr.predicate, right=expr.source.predicate),
            )
            return merged, "Selection conjunction merge"
        
        # σθ1 (σθ2 (r)) ≡ σθ2 (σθ1 (r)) is pointless to apply because the predicates will be combined anyway

        # πA1 (πA2 (r)) ≡ πA1 (r)
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, ProjectQuery):
            merged = ProjectQuery(
                source=expr.source.source,
                attributes=[attr for attr in expr.attributes],
            )
            return merged, "Nested projection merge"
        
        # ρa1 (ρa2 (r)) ≡ ρa3 (r) (USQL only has generalised renaming so a3 = a1 here)
        if isinstance(expr, RenameQuery) and isinstance(expr.source, RenameQuery):
            merged = RenameQuery(
                source=expr.source.source,
                new_attributes=[attr for attr in expr.new_attributes],
            )
            return merged, "Nested rename merge"

        return None, None
