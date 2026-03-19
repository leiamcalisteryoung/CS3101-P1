from itertools import permutations

from builder import ProgramState
from query_models import (
    AndPredicate,
    OrPredicate,
    AttrOpAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    EmptyQuery,
    JoinQuery,
    Predicate,
    ProjectQuery,
    QueryExpr,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    format_query_expr,
)


class CostBasedJoinOptimizer:
    """Phase 2 optimizer: cost-based optimization of join chains.

    This class focuses only on join-chain reordering decisions.
    It runs as a separate phase after heuristic rewrites are finished.
    """

    def __init__(self, state: ProgramState) -> None:
        self._state = state

    # Public entry for phase 2.
    # Runs one bottom-up sweep and returns (rewritten_expr, phase2_steps)
    # where phase2_steps is a list of "Rule: expression" lines.
    def optimize(self, expr: QueryExpr) -> tuple[QueryExpr, list[str]]:
        if isinstance(expr, SelectQuery):
            new_source, steps = self.optimize(expr.source)
            return SelectQuery(source=new_source, predicate=expr.predicate), steps

        if isinstance(expr, ProjectQuery):
            new_source, steps = self.optimize(expr.source)
            return ProjectQuery(source=new_source, attributes=list(expr.attributes)), steps

        if isinstance(expr, RenameQuery):
            new_source, steps = self.optimize(expr.source)
            return RenameQuery(source=new_source, new_attributes=list(expr.new_attributes)), steps

        if isinstance(expr, UnionQuery):
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)
            return UnionQuery(left=new_left, right=new_right), (left_steps + right_steps)

        if isinstance(expr, DifferenceQuery):
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)
            return DifferenceQuery(left=new_left, right=new_right), (left_steps + right_steps)

        if isinstance(expr, JoinQuery):
            # 1) Optimize deeper joins first.
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)

            candidate = JoinQuery(left=new_left, right=new_right)
            steps = left_steps + right_steps

            # 2) Then optimize this join node using local cost-based alternatives.
            rewritten, rule_name = self._apply_cost_based_join_reorder(candidate)
            if rewritten is not None:
                steps.append(f"{rule_name}: {format_query_expr(rewritten)}")
                return rewritten, steps

            return candidate, steps

        # Leaf nodes (RelVarQuery / EmptyQuery) have no join-chain work.
        return expr, []

    # Decide local join reordering via cost comparison.
    def _apply_cost_based_join_reorder(self, expr: JoinQuery) -> tuple[QueryExpr | None, str | None]:
        # Get the join frontier operands for this node, if it matches a recognized shape.
        left_is_join = isinstance(expr.left, JoinQuery)
        right_is_join = isinstance(expr.right, JoinQuery)

        # If neither side is a join, return early with no change.
        if not left_is_join and not right_is_join:
            return None, None

        # Get the frontier operands for this node.
        if left_is_join and right_is_join:
            frontier_operands = [expr.left.left, expr.left.right, expr.right.left, expr.right.right]
        elif left_is_join:
            frontier_operands = [expr.left.left, expr.left.right, expr.right]
        else:
            frontier_operands = [expr.left, expr.right.left, expr.right.right]

        # Enumerate all binary join trees for all operand permutations.
        # We keep generation order and use it as deterministic tie-break.
        candidates: list[QueryExpr] = []
        for perm in permutations(frontier_operands):
            candidates.extend(self._enumerate_binary_join_trees(list(perm)))

        # Calculate cost for each candidate and pick the best.
        scored: list[tuple[float, int, QueryExpr]] = []
        for index, candidate_expr in enumerate(candidates):
            candidate_cost = self._estimate_plan_cost(candidate_expr)
            scored.append((candidate_cost, index, candidate_expr))

        # Sort by cost, then by generation order as tie-breaker.
        scored.sort(key=lambda item: (item[0], item[1]))

        # Get best cost and return the corresponding expression if it's different from the input.
        _, _, best_expr = scored[0]
        if best_expr != expr:
            return best_expr, f"Cost-based join reordering"

        return None, None

    # Enumerate all binary join parenthesizations over an ordered operand list of 3 or 4.
    def _enumerate_binary_join_trees(self, ordered_operands: list[QueryExpr]) -> list[QueryExpr]:
        arity = len(ordered_operands)

        if arity == 3:
            a, b, c = ordered_operands
            return [
                JoinQuery(left=JoinQuery(left=a, right=b), right=c),
                JoinQuery(left=a, right=JoinQuery(left=b, right=c)),
            ]

        if arity == 4:
            a, b, c, d = ordered_operands
            return [
                JoinQuery(left=JoinQuery(left=JoinQuery(left=a, right=b), right=c), right=d),
                JoinQuery(left=JoinQuery(left=a, right=JoinQuery(left=b, right=c)), right=d),
                JoinQuery(left=JoinQuery(left=a, right=b), right=JoinQuery(left=c, right=d)),
                JoinQuery(left=a, right=JoinQuery(left=JoinQuery(left=b, right=c), right=d)),
                JoinQuery(left=a, right=JoinQuery(left=b, right=JoinQuery(left=c, right=d))),
            ]

        return []
    
    # Estimates the total cost of executing a query plan by calling the recursive inner method.
    def _estimate_plan_cost(self, expr: QueryExpr) -> float:
        _, _, _, cumulative_cost = self._estimate_stats_and_cost(expr)
        return cumulative_cost

    # Estimate both output statistics and cumulative plan cost in one bottom-up traversal.
    #
    # Returns (rows, attrs, distinct, cost):
    #   rows     -> estimated output cardinality of expr
    #   attrs    -> output attribute list of expr
    #   distinct -> per-attribute distinct count estimates V(A, expr)
    #   cost     -> cumulative execution cost estimate for expr
    def _estimate_stats_and_cost(self, expr: QueryExpr) -> tuple[float, list[str], dict[str, float], float]:
        # Base relation leaf: exact statistics from loaded data.
        if isinstance(expr, RelVarQuery):
            relvar = self._state.relvars.get(expr.name)
            if relvar is None:
                raise ValueError(f"Unknown relvar '{expr.name}' while estimating cost.")

            attrs = relvar.relation.attr_names()
            rows = float(len(relvar.tuples))
            v: dict[str, float] = {}
            for attr in attrs:
                distinct_vals = {row[attr] for row in relvar.tuples}
                v[attr] = float(len(distinct_vals))

            return rows, attrs, v, rows

        if isinstance(expr, EmptyQuery):
            return 0.0, [], {}, 0.0

        # Selection: estimate selectivity from predicates.
        if isinstance(expr, SelectQuery):
            source_rows, source_attrs, source_v, source_cost = self._estimate_stats_and_cost(expr.source)
            selected_rows = self._estimate_selected_rows(expr.predicate, source_rows, source_v)
            rows = min(source_rows, selected_rows)

            # Ensure disctinct counts do not exceed output cardinality.
            v = {attr: min(source_v.get(attr, rows), rows) for attr in source_attrs}
            return rows, list(source_attrs), v, source_cost

        # Projection: estimate cardinality as V(A, r).
        if isinstance(expr, ProjectQuery):
            source_rows, _, source_v, source_cost = self._estimate_stats_and_cost(expr.source)
            attrs = [a for a in expr.attributes]

            # V(A, r) = max_{a in A} V(a, r), then cap by n(r).
            projected_distinct = max((source_v.get(attr, 1.0) for attr in attrs), default=1.0)
            rows = min(source_rows, projected_distinct)
            v = {attr: min(source_v.get(attr, rows), rows) for attr in attrs}
            return rows, attrs, v, source_cost

        # Rename: preserve rows/distinct counts, just remap attribute names.
        if isinstance(expr, RenameQuery):
            source_rows, source_attrs, source_v, source_cost = self._estimate_stats_and_cost(expr.source)
            old_attrs = source_attrs
            new_attrs = list(expr.new_attributes)

            if len(old_attrs) != len(new_attrs):
                raise ValueError("Rename arity mismatch while estimating cost.")

            rows = source_rows
            v: dict[str, float] = {}
            for old_attr, new_attr in zip(old_attrs, new_attrs):
                v[new_attr] = min(source_v.get(old_attr, rows), rows)

            return rows, new_attrs, v, source_cost

        # Union: sum of input cardinalities.
        if isinstance(expr, UnionQuery):
            left_rows, left_attrs, left_v, left_cost = self._estimate_stats_and_cost(expr.left)
            right_rows, _, right_v, right_cost = self._estimate_stats_and_cost(expr.right)

            rows = left_rows + right_rows
            attrs = list(left_attrs)
            v: dict[str, float] = {}
            for attr in attrs:
                v[attr] = min(rows, left_v.get(attr, 0.0) + right_v.get(attr, 0.0))

            return rows, attrs, v, left_cost + right_cost + rows

        # Difference: minimum of input cardinalities
        if isinstance(expr, DifferenceQuery):
            left_rows, left_attrs, left_v, left_cost = self._estimate_stats_and_cost(expr.left)
            right_rows, _, _, right_cost = self._estimate_stats_and_cost(expr.right)

            rows = min(left_rows, right_rows)
            attrs = list(left_attrs)
            v = {attr: min(left_v.get(attr, rows), rows) for attr in attrs}

            return rows, attrs, v, left_cost + right_cost + rows

        # Join: cartesian product if no shared attributes; otherwise use
        # min(nr*ns / V(A,r), nr*ns / V(A,s)) over shared attributes.
        if isinstance(expr, JoinQuery):
            nr, left_attrs_list, left_v, left_cost = self._estimate_stats_and_cost(expr.left)
            ns, right_attrs_list, right_v, right_cost = self._estimate_stats_and_cost(expr.right)

            left_attrs = set(left_attrs_list)
            right_attrs = set(right_attrs_list)
            shared_attrs = [a for a in left_attrs_list if a in right_attrs]

            if not shared_attrs:
                # R ∩ S = ∅  -> cartesian product
                join_rows = nr * ns
            else:
                # R ∩ S ≠ ∅  -> use min over shared attributes of (nr*ns / V(A,r)) and (nr*ns / V(A,s)).
                candidates: list[float] = []
                for attr in shared_attrs:
                    v_ar = left_v.get(attr, 1.0)
                    v_as = right_v.get(attr, 1.0)
                    candidates.append(min((nr * ns) / v_ar, (nr * ns) / v_as))
                join_rows = min(candidates) if candidates else (nr * ns)

            # Build output attribute list in natural-join order: left attrs + right-only attrs.
            right_only = [a for a in right_attrs_list if a not in left_attrs]
            out_attrs = list(left_attrs_list) + right_only

            # Estimate output distinct counts:
            # - shared attrs: cannot exceed min(V_left, V_right, join_rows)
            # - left-only attrs: cannot exceed min(V_left, join_rows)
            # - right-only attrs: cannot exceed min(V_right, join_rows)
            out_v: dict[str, float] = {}
            for attr in out_attrs:
                if attr in left_attrs and attr in right_attrs:
                    out_v[attr] = min(join_rows, left_v.get(attr, join_rows), right_v.get(attr, join_rows))
                elif attr in left_attrs:
                    out_v[attr] = min(join_rows, left_v.get(attr, join_rows))
                else:
                    out_v[attr] = min(join_rows, right_v.get(attr, join_rows))

            # Join contributes its intermediate result size as work, in addition to child costs.
            return join_rows, out_attrs, out_v, left_cost + right_cost + join_rows

        raise ValueError(f"Unsupported query node type while estimating cost: {type(expr).__name__}")

    # Estimate selected cardinality s for a predicate over a source with n_r rows.
    def _estimate_selected_rows(
        self,
        predicate: Predicate,
        rows: float,
        distinct: dict[str, float],
    ) -> float:
        if rows <= 0.0:
            return 0.0

        if isinstance(predicate, AttrOpAttrPredicate):
            if predicate.operator == "=":
                # Case A1=A2 use n_r / max(V(A1), V(A2))
                v_left = distinct.get(predicate.left_attr, 1.0)
                v_right = distinct.get(predicate.right_attr, 1.0)
                return rows / max(v_left, v_right)

            if predicate.operator == "!=":
                # Case A1!=A2: n_r - n_r/max(V(A1), V(A2))
                v_left = distinct.get(predicate.left_attr, 1.0)
                v_right = distinct.get(predicate.right_attr, 1.0)
                eq_rows = rows / max(v_left, v_right)
                return rows - eq_rows

            # Default to rows/3 for other operators.
            return rows / 3.0

        if isinstance(predicate, AttrEqConstPredicate):
            # Case A=c: n_r / V(A, r)
            if predicate.operator == "=":
                v_attr = distinct.get(predicate.attr, 1.0)
                return rows / v_attr

            # Case A!=c: n_r * (1 - (1 / V(A, r)))
            if predicate.operator == "!=":
                v_attr = distinct.get(predicate.attr, 1.0)
                return rows * (1.0 - (1.0 / v_attr))

            # Default to rows/3 for other operators.
            return rows / 3.0

        # σθ1∧···∧θn (r): nr ×Πi∈[1,n]si /nnr
        # For binary AND predicates this becomes (s1 * s2) / n_r.
        if isinstance(predicate, AndPredicate):
            left_rows = self._estimate_selected_rows(predicate.left, rows, distinct)
            right_rows = self._estimate_selected_rows(predicate.right, rows, distinct)
            return (left_rows * right_rows) / rows

        # σ(θ1 ∨ θ2): nr ×(1−Πi∈[1,n](1−si /nr ))
        # For binary OR predicates this becomes n_r * (1 - ((1 - (s1 / n_r)) * (1 - (s2 / n_r))))).
        if isinstance(predicate, OrPredicate):
            left_rows = self._estimate_selected_rows(predicate.left, rows, distinct)
            right_rows = self._estimate_selected_rows(predicate.right, rows, distinct)
            return rows * (1.0 - ((1.0 - (left_rows / rows)) * (1.0 - (right_rows / rows))))

        return rows
